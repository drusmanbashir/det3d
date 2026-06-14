# %%
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Hybrid scratch script (REPL / # %% cells):
# - det3d DataManagerDualDet* for train/val dataloaders
# - original MONAI LUNA16 RetinaNetDetector training loop
import ipdb

from utilz.imageviewers import ImageBBoxViewer
tr = ipdb.set_trace

import gc
import time
from pathlib import Path

import numpy as np
import torch
from det3d.configs.parser import ConfigMakerDet
from det3d.detection.visualize_image import visualize_one_xy_slice_in_3d_image
from det3d.detection.warmup_scheduler import GradualWarmupScheduler
from det3d.managers.data import DataManagerDualDet, DataManagerDualDetBTfms
from fran.managers import Project
from torch.utils.tensorboard import SummaryWriter
from utilz.stringz import ast_literal_eval

import monai
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
# from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from det3d.detection.retinanet_detector2 import RetinaNetDetector2 
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.data import box_utils
from monai.networks.nets import resnet
from monai.utils import set_determinism

BOX_KEY = "bbox"
LABEL_KEY = "label"
VAL_PATCH_SIZE = [512, 512, 208]

project_title = "lidc"
plan_id = 1
batch_size = 8
batch_tfms = True
debug = False
verbose = False
fold = 0
model_path = "/s/agent_rw/tmp/luna16_lidc_dm_hybrid/detector.pt"
tfevent_path = "/s/agent_rw/tmp/luna16_lidc_dm_hybrid/tfevents"
max_epochs = 400
val_interval = 5
w_cls = 1.0


def normalize_plan_modes_for_det_pipeline(configs):
    for key in ("plan_train", "plan_valid", "plan_test"):
        plan = configs[key]
        if plan["mode"] in {"det", "lbd"}:
            plan["mode"] = "lbd"


def setup_det_dataloaders(
    project_title,
    configs,
    batch_size=None,
    batch_tfms=True,
    debug=False,
    train_indices=None,
    val_indices=None,
    val_sampling=1.0,
):
    normalize_plan_modes_for_det_pipeline(configs)
    plan = configs["plan_train"]
    if batch_size is not None:
        configs["dataset_params"]["batch_size"] = int(batch_size)
        plan["batch_size"] = int(batch_size)
    dm_class = DataManagerDualDetBTfms if batch_tfms else DataManagerDualDet
    dm = dm_class(
        project_title=project_title,
        configs=configs,
        batch_size=int(plan["batch_size"]),
        cache_rate=configs["dataset_params"].get("cache_rate", 0.0),
        device=configs["dataset_params"].get("device", "cuda"),
        ds_type=configs["dataset_params"].get("ds_type"),
        train_indices=train_indices,
        val_indices=val_indices,
        val_sampling=val_sampling,
        debug=debug,
        batch_tfms=batch_tfms,
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    return dm


def dm_train_batch_to_detector(batch, device):
    images = batch["image"].to(device)
    targets = [
        {
            LABEL_KEY: batch[LABEL_KEY][i].to(device),
            BOX_KEY: batch[BOX_KEY][i].to(device),
        }
        for i in range(images.shape[0])
    ]
    return images, targets


def dm_val_batch_items(batch):
    if isinstance(batch, list):
        return batch
    return [batch]


def val_patch_size_from_plan(plan):
    val_patch_size = plan.get("val_patch_size", VAL_PATCH_SIZE)
    if isinstance(val_patch_size, str):
        val_patch_size = ast_literal_eval(val_patch_size)
    return val_patch_size


if __name__ == "__main__":
#SECTION:-------------------- setup --------------------------------------------------------------------------------------

    set_determinism(seed=0)

    amp = torch.cuda.is_available()
    monai.config.print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

    project = Project(project_title)
    config_maker = ConfigMakerDet(project)
    config_maker.setup(plan_id)
    configs = config_maker.configs
    configs["dataset_params"]["fold"] = fold
    plan = configs["plan_train"]
    val_patch_size = val_patch_size_from_plan(plan)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(tfevent_path).mkdir(parents=True, exist_ok=True)

# %%
    dm = setup_det_dataloaders(
        project_title=project_title,
        configs=configs,
        batch_size=batch_size,
        batch_tfms=batch_tfms,
        debug=debug,
    )
    train_manager = dm.train_manager
    valid_manager = dm.valid_manager
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    train_ds = train_manager.ds

    spf = int(plan["samples_per_file"])
    print(
        f"DataManager train={type(train_manager).__name__} n={len(train_ds)} "
        f"valid={type(valid_manager).__name__} n={len(valid_manager.ds)}"
    )
    print(
        f"batch_size={train_manager.batch_size} samples_per_file={spf} "
        f"effective_batch_size={train_manager.effective_batch_size} "
        f"dl.batch_size={train_loader.batch_size} "
        f"-> patches/step={train_manager.batch_size}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**level for level in range(len(plan["returned_layers"]) + 1)],
        base_anchor_shapes=plan["base_anchor_shapes"],
    )

    conv1_t_size = [max(7, 2 * stride + 1) for stride in plan["conv1_t_stride"]]
    backbone = resnet.ResNet(
        block=resnet.ResNetBottleneck,
        layers=[3, 4, 6, 3],
        block_inplanes=resnet.get_inplanes(),
        n_input_channels=int(plan["n_input_channels"]),
        conv1_t_stride=plan["conv1_t_stride"],
        conv1_t_size=conv1_t_size,
    )
    feature_extractor = resnet_fpn_feature_extractor(
        backbone=backbone,
        spatial_dims=int(plan["spatial_dims"]),
        pretrained_backbone=False,
        trainable_backbone_layers=None,
        returned_layers=plan["returned_layers"],
    )
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    size_divisible = [
        step * 2 * 2 ** max(plan["returned_layers"]) for step in feature_extractor.body.conv1.stride
    ]
    net = torch.jit.script(
        RetinaNet(
            spatial_dims=int(plan["spatial_dims"]),
            num_classes=len(plan["fg_labels"]),
            num_anchors=num_anchors,
            feature_extractor=feature_extractor,
            size_divisible=size_divisible,
        )
    )

    detector = RetinaNetDetector2(network=net, anchor_generator=anchor_generator, debug=verbose).to(
        device
    )

    detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
    detector.set_hard_negative_sampler(
        batch_size_per_image=64,
        positive_fraction=float(plan["balanced_sampler_pos_fraction"]),
        pool_size=20,
        min_neg=16,
    )
    detector.set_target_keys(box_key=BOX_KEY, label_key=LABEL_KEY)

    detector.set_box_selector_parameters(
        score_thresh=float(plan["score_thresh"]),
        topk_candidates_per_level=1000,
        nms_thresh=float(plan["nms_thresh"]),
        detections_per_img=100,
    )
    detector.set_sliding_window_inferer(
        roi_size=val_patch_size,
        overlap=0.25,
        sw_batch_size=1,
        mode="constant",
        device=str(device),
    )

# %%
    optimizer = torch.optim.SGD(
        detector.network.parameters(),
        float(plan["lr"]),
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True,
    )
    after_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    scheduler_warmup = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=10, after_scheduler=after_scheduler
    )
    scaler = torch.amp.GradScaler("cuda") if amp else None
    optimizer.zero_grad()
    optimizer.step()

    tensorboard_writer = SummaryWriter(tfevent_path)

    coco_metric = COCOMetric(classes=["nodule"], iou_list=[0.1], max_detection=[100])
    best_val_epoch_metric = 0.0
    best_val_epoch = -1
    epoch_len = len(train_ds) // train_loader.batch_size
    
# %%
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        detector.train()
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_box_reg_loss = 0
        step = 0
        start_time = time.time()
        scheduler_warmup.step()

        for batch_data in train_loader:
            if train_manager.transforms_batch is not None:
                batch_data = train_manager.transforms_batch(batch_data)

            step += 1
            inputs, targets = dm_train_batch_to_detector(batch_data, device)

            for param in detector.network.parameters():
                param.grad = None

            if amp and (scaler is not None):
                with torch.amp.autocast("cuda"):
                    outputs = detector(inputs, targets)
                    loss = w_cls * outputs[detector.cls_key] + outputs[detector.box_reg_key]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = detector(inputs, targets)
                loss = w_cls * outputs[detector.cls_key] + outputs[detector.box_reg_key]
                loss.backward()
                optimizer.step()

            epoch_loss += loss.detach().item()
            epoch_cls_loss += outputs[detector.cls_key].detach().item()
            epoch_box_reg_loss += outputs[detector.box_reg_key].detach().item()
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            tensorboard_writer.add_scalar("train_loss", loss.detach().item(), epoch_len * epoch + step)

        end_time = time.time()
        print(f"Training time: {end_time-start_time}s")
        del inputs, batch_data
        torch.cuda.empty_cache()
        gc.collect()

        epoch_loss /= step
        epoch_cls_loss /= step
        epoch_box_reg_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        tensorboard_writer.add_scalar("avg_train_loss", epoch_loss, epoch + 1)
        tensorboard_writer.add_scalar("avg_train_cls_loss", epoch_cls_loss, epoch + 1)
        tensorboard_writer.add_scalar("avg_train_box_reg_loss", epoch_box_reg_loss, epoch + 1)
        tensorboard_writer.add_scalar("train_lr", optimizer.param_groups[0]["lr"], epoch + 1)

        torch.jit.save(detector.network, model_path[:-3] + "_last.pt")
        print("saved last model")

        if (epoch + 1) % val_interval == 0:
            detector.eval()
            val_outputs_all = []
            val_targets_all = []
            start_time = time.time()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_items = dm_val_batch_items(val_batch)
                    use_inferer = not all(
                        [
                            val_data_i["image"][0, ...].numel() < np.prod(val_patch_size)
                            for val_data_i in val_items
                        ]
                    )
                    val_inputs = [val_data_i.pop("image").to(device) for val_data_i in val_items]

                    if amp:
                        with torch.autocast("cuda"):
                            val_outputs = detector(val_inputs, use_inferer=use_inferer)
                    else:
                        val_outputs = detector(val_inputs, use_inferer=use_inferer)

                    val_outputs_all += val_outputs
                    val_targets_all += val_items

            end_time = time.time()
            print(f"Validation time: {end_time-start_time}s")

            draw_img = visualize_one_xy_slice_in_3d_image(
                gt_boxes=val_items[0][detector.target_box_key].cpu().detach().numpy(),
                image=val_inputs[0][0, ...].cpu().detach().numpy(),
                pred_boxes=val_outputs[0][detector.target_box_key].cpu().detach().numpy(),
            )
            tensorboard_writer.add_image("val_img_xy", draw_img.transpose([2, 1, 0]), epoch + 1)

            del val_inputs
            torch.cuda.empty_cache()
            results_metric = matching_batch(
                iou_fn=box_utils.box_iou,
                iou_thresholds=coco_metric.iou_thresholds,
                pred_boxes=[
                    val_data_i[detector.target_box_key].cpu().detach().numpy()
                    for val_data_i in val_outputs_all
                ],
                pred_classes=[
                    val_data_i[detector.target_label_key].cpu().detach().numpy()
                    for val_data_i in val_outputs_all
                ],
                pred_scores=[
                    val_data_i[detector.pred_score_key].cpu().detach().numpy()
                    for val_data_i in val_outputs_all
                ],
                gt_boxes=[
                    val_data_i[detector.target_box_key].cpu().detach().numpy()
                    for val_data_i in val_targets_all
                ],
                gt_classes=[
                    val_data_i[detector.target_label_key].cpu().detach().numpy()
                    for val_data_i in val_targets_all
                ],
            )
            val_epoch_metric_dict = coco_metric(results_metric)[0]
            print(val_epoch_metric_dict)

            for k in val_epoch_metric_dict.keys():
                tensorboard_writer.add_scalar("val_" + k, val_epoch_metric_dict[k], epoch + 1)
            val_epoch_metric = val_epoch_metric_dict.values()
            val_epoch_metric = sum(val_epoch_metric) / len(val_epoch_metric)
            tensorboard_writer.add_scalar("val_metric", val_epoch_metric, epoch + 1)

            if val_epoch_metric > best_val_epoch_metric:
                best_val_epoch_metric = val_epoch_metric
                best_val_epoch = epoch + 1
                torch.jit.save(detector.network, model_path)
                print("saved new best metric model")
            print(
                "current epoch: {} current metric: {:.4f} "
                "best metric: {:.4f} at epoch {}".format(
                    epoch + 1, val_epoch_metric, best_val_epoch_metric, best_val_epoch
                )
            )

    print(f"train completed, best_metric: {best_val_epoch_metric:.4f} " f"at epoch: {best_val_epoch}")
    tensorboard_writer.close()

# %%
    input_images = 
    targets = None
    use_inferer = False
# %%  # T:block_start|RetinaNetDetector2.forward
#SECTION:-------------------- forward--------------------------------------------------------------------------------------  # T:block_meta|RetinaNetDetector2.forward
    if detector.training and isinstance(input_images, Tensor):  # T:self_ref|if self.training and isinstance(input_images, Tensor):
        pass  # T:early_return|    return forward_train_batched(self, input_images, targets)
    forward_result = super().forward(input_images, targets, use_inferer=use_inferer)  # T:return|return super().forward(input_images, targets, use_inferer=use_inferer)
    # end PythonMethodScratch  # T:block_end|RetinaNetDetector2.forward

# %%
    input_images = input_images
    targets = targets
    use_inferer = use_inferer
# %%  # T:block_start|RetinaNetDetector2.forward
#SECTION:-------------------- forward--------------------------------------------------------------------------------------  # T:block_meta|RetinaNetDetector2.forward
    if detector.training and isinstance(input_images, Tensor):  # T:self_ref|if self.training and isinstance(input_images, Tensor):
        pass  # T:early_return|    return forward_train_batched(self, input_images, targets)
    forward_result = super().forward(input_images, targets, use_inferer=use_inferer)  # T:return|return super().forward(input_images, targets, use_inferer=use_inferer)
    # end PythonMethodScratch  # T:block_end|RetinaNetDetector2.forward
# %%
