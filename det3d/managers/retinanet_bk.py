import gc

import numpy as np
import torch
from det3d.detection.retinanet_detector2 import RetinaNetDetector2
from det3d.detection.retinanet_train import forward_train_batched
from det3d.evaluation.coco import compute_coco_metrics
from det3d.transforms.warmup_scheduler import GradualWarmupScheduler
from fran.configs.helpers import is_excel_None
from fran.managers.project import Project
from lightning.pytorch import LightningModule
from utilz.stringz import ast_literal_eval
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.networks.nets import resnet

class RetinaNetManager(LightningModule):
    def __init__(self, project_title, configs, lr=None, sync_dist=False):
        super().__init__()
        self.sync_dist = sync_dist
        self.project = Project(project_title)
        self.save_hyperparameters("project_title", "configs", "lr")
        self.configs = configs
        self.plan = configs["plan_train"]
        self.lr = float(lr if lr is not None else self.plan["lr"])
        self.w_cls = float(self.plan.get("w_cls", 1.0))
        self.class_names = [self.plan.get("class_name", "nodule")]
        self.val_outputs_all = []
        self.val_targets_all = []
        self.scheduler_warmup = None
        val_patch_size = self.plan.get("val_patch_size", [512, 512, 208])
        if isinstance(val_patch_size, str):
            val_patch_size = ast_literal_eval(val_patch_size)
        self.val_patch_size = [int(v) for v in val_patch_size]
        plan = self.plan
        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=[2 ** level for level in range(len(plan["returned_layers"]) + 1)],
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
            step * 2 * 2 ** max(plan["returned_layers"])
            for step in feature_extractor.body.conv1.stride
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
        self.detector = RetinaNetDetector2(
            network=net, anchor_generator=anchor_generator, debug=False
        )
        self.detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
        self.detector.set_hard_negative_sampler(
            batch_size_per_image=64,
            positive_fraction=float(plan["balanced_sampler_pos_fraction"]),
            pool_size=20,
            min_neg=16,
        )
        self.detector.set_target_keys(box_key="bbox", label_key="label")
        self.detector.set_box_selector_parameters(
            score_thresh=float(plan["score_thresh"]),
            topk_candidates_per_level=1000,
            nms_thresh=float(plan["nms_thresh"]),
            detections_per_img=100,
        )
        self.detector.set_sliding_window_inferer(
            roi_size=self.val_patch_size,
            overlap=0.25,
            sw_batch_size=1,
            mode="constant",
            device="cpu",
        )

    def _image_batch_tensor(self, batch):
        image = batch["image"].to(self.device)
        if image.dim() == 4:
            image = image.unsqueeze(0)
        return image

    def _targets_from_batch(self, batch):
        box_key = self.detector.target_box_key
        label_key = self.detector.target_label_key
        boxes = batch[box_key]
        labels = batch[label_key]
        if isinstance(boxes, list):
            return [
                {
                    label_key: torch.as_tensor(label, device=self.device).reshape(-1),
                    box_key: torch.as_tensor(box, device=self.device).reshape(-1, 6),
                }
                for label, box in zip(labels, boxes)
            ]
        box = torch.as_tensor(boxes, device=self.device).reshape(-1, 6)
        label = torch.as_tensor(labels, device=self.device).reshape(-1)
        return [{label_key: label, box_key: box}]

    def train_images(self, batch):
        return self._image_batch_tensor(batch)

    def train_targets(self, batch):
        return self._targets_from_batch(batch)

    def train_head_loss_dict(self, images, targets):
        return forward_train_batched(self.detector, images, targets)

    def train_total_loss(self, outputs):
        cls_loss = outputs[self.detector.cls_key]
        box_loss = outputs[self.detector.box_reg_key]
        total = self.w_cls * cls_loss + box_loss
        return total, cls_loss, box_loss

    def training_step(self, batch, batch_idx):
        self.detector.train()
        images = self.train_images(batch)
        targets = self.train_targets(batch)
        outputs = self.train_head_loss_dict(images, targets)
        loss, cls_loss, box_loss = self.train_total_loss(outputs)
        self.log("train0_loss", loss, prog_bar=True, sync_dist=self.sync_dist)
        self.log("train0_cls_loss", cls_loss, sync_dist=self.sync_dist)
        self.log("train0_box_reg_loss", box_loss, sync_dist=self.sync_dist)
        return loss

    def _infer_device(self):
        return next(self.detector.parameters()).device

    def _val_inputs_from_batch(self, batch):
        device = self._infer_device()
        spatial_dims = self.detector.spatial_dims
        images = batch["image"].to(device)
        val_inputs = []
        for i in range(images.shape[0]):
            img = images[i]
            if img.dim() == spatial_dims + 2 and int(img.shape[0]) == 1:
                img = img.squeeze(0)
            val_inputs.append(img.contiguous())
        return val_inputs

    def _use_sliding_window_inferer(self, val_inputs):
        patch_voxels = int(np.prod(self.val_patch_size))
        return not all(item[0, ...].numel() < patch_voxels for item in val_inputs)

    def on_validation_epoch_start(self):
        self.val_outputs_all = []
        self.val_targets_all = []

    def val_inputs(self, batch):
        return self._val_inputs_from_batch(batch)

    def val_targets(self, batch):
        return self._targets_from_batch(batch)

    def val_use_inferer(self, val_inputs):
        return self._use_sliding_window_inferer(val_inputs)

    def val_forward(self, val_inputs, use_inferer=None):
        if use_inferer is None:
            use_inferer = self._use_sliding_window_inferer(val_inputs)
        device = self._infer_device()
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast("cuda"):
                    return self.detector(val_inputs, use_inferer=use_inferer)
            return self.detector(val_inputs, use_inferer=use_inferer)

    def validation_step(self, batch, batch_idx):
        self.detector.eval()
        val_inputs = self.val_inputs(batch)
        val_targets = self.val_targets(batch)
        val_outputs = self.val_forward(val_inputs)
        self.val_outputs_all.extend(val_outputs)
        self.val_targets_all.extend(val_targets)

    def on_validation_epoch_end(self):
        if len(self.val_outputs_all) == 0:
            return
        metrics = compute_coco_metrics(
            self.detector,
            self.val_outputs_all,
            self.val_targets_all,
            self.class_names,
        )
        metric_vals = list(metrics.values())
        val_metric = sum(metric_vals) / len(metric_vals)
        for key, value in metrics.items():
            self.log(f"val0_{key}", value, sync_dist=self.sync_dist)
        self.log("val0_metric", val_metric, prog_bar=True, sync_dist=self.sync_dist)
        del self.val_outputs_all, self.val_targets_all
        torch.cuda.empty_cache()
        gc.collect()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.detector.network.parameters(),
            self.lr,
            momentum=0.9,
            weight_decay=3e-5,
            nesterov=True,
        )
        after_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
        self.scheduler_warmup = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=10, after_scheduler=after_scheduler
        )
        return optimizer

    def on_fit_start(self):
        device = self._infer_device()
        self.detector.to(device)
        self.detector.set_sliding_window_inferer(
            roi_size=self.val_patch_size,
            overlap=0.25,
            sw_batch_size=1,
            mode="constant",
            device=str(device),
        )

    def on_train_epoch_start(self):
        if self.scheduler_warmup is not None:
            self.scheduler_warmup.step()
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False)
