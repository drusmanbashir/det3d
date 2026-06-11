import gc

import numpy as np
import torch
from det3d.detection.retinanet_train import forward_train_batched
from det3d.evaluation.coco import compute_coco_metrics
from det3d.transforms.warmup_scheduler import GradualWarmupScheduler
from fran.configs.helpers import is_excel_None
from fran.managers.project import Project
from lightning.pytorch import LightningModule
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.networks.nets import resnet

VAL_PATCH_SIZE = [512, 512, 208]


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
        self.detector = RetinaNetDetector(
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
            roi_size=VAL_PATCH_SIZE,
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

    def training_step(self, batch, batch_idx):
        self.detector.train()
        images = self._image_batch_tensor(batch)
        targets = self._targets_from_batch(batch)
        outputs = forward_train_batched(self.detector, images, targets)
        loss = self.w_cls * outputs[self.detector.cls_key] + outputs[self.detector.box_reg_key]
        self.log("train0_loss", loss, prog_bar=True, sync_dist=self.sync_dist)
        self.log("train0_cls_loss", outputs[self.detector.cls_key], sync_dist=self.sync_dist)
        self.log(
            "train0_box_reg_loss",
            outputs[self.detector.box_reg_key],
            sync_dist=self.sync_dist,
        )
        return loss

    def on_validation_epoch_start(self):
        self.val_outputs_all = []
        self.val_targets_all = []

    def validation_step(self, batch, batch_idx):
        self.detector.eval()
        images = batch["image"].to(self.device)
        val_targets = self._targets_from_batch(batch)
        if images.dim() == 5:
            val_inputs = [images[0]]
        else:
            val_inputs = [images]
        use_inferer = val_inputs[0][0, ...].numel() >= int(np.prod(VAL_PATCH_SIZE))
        with torch.no_grad():
            val_outputs = self.detector(val_inputs, use_inferer=use_inferer)
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
        self.detector.to(self.device)

    def on_train_epoch_start(self):
        if self.scheduler_warmup is not None:
            self.scheduler_warmup.step()
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False)
