import numpy as np
import torch
from det3d.architectures.create_detector import INFER_OVERLAP, create_detector_from_conf
from det3d.detection.nndet_train import maybe_store_batch_grid_preds
from det3d.detection.retinanet_train import forward_train_batched
from det3d.evaluation.coco import compute_coco_metrics
from det3d.managers.det_schedule import configure_detection_optimizers
from fran.managers.project import Project
from lightning.pytorch import LightningModule


class RetinaNetManager(LightningModule):
    def __init__(self, project_title, configs, lr=None, sync_dist=False):
        super().__init__()
        self.sync_dist = sync_dist
        self.project = Project(project_title)
        self.save_hyperparameters("project_title", "configs", "lr")
        self.configs = configs
        self.plan = configs["plan_train"]
        self.lr = float(lr if lr is not None else self.configs["model_params"]["lr"])
        self.w_cls = float(self.plan["w_cls"])
        self.w_reg = float(self.plan["w_reg"])
        self.class_names = [self.plan["class_name"]]
        self.val_outputs_all = []
        self.val_targets_all = []
        self.scheduler_warmup = None
        self.detector, self.val_patch_size = create_detector_from_conf(configs)

    def _targets_from_batch(self, batch):
        box_key = self.detector.target_box_key
        label_key = self.detector.target_label_key
        boxes = batch[box_key]
        labels = batch[label_key]
        return [
            {
                label_key: torch.as_tensor(label, device=self.device).reshape(-1),
                box_key: torch.as_tensor(box, device=self.device).reshape(-1, 6),
            }
            for label, box in zip(labels, boxes)
        ]

    def _val_inputs_from_batch(self, batch):
        images = batch["image"]
        return [images[i].contiguous() for i in range(images.shape[0])]

    def _use_sliding_window_inferer(self, val_inputs):
        patch_voxels = int(np.prod(self.val_patch_size))
        return not all(item[0, ...].numel() < patch_voxels for item in val_inputs)

    def training_step(self, batch, batch_idx):
        outputs = forward_train_batched(
            self.detector, batch["image"], self._targets_from_batch(batch)
        )
        cls_loss = outputs[self.detector.cls_key]
        box_loss = outputs[self.detector.box_reg_key]
        loss = self.w_cls * cls_loss + self.w_reg * box_loss
        self.log("train0_loss", loss, prog_bar=True, sync_dist=self.sync_dist)
        self.log("train0_cls_loss", cls_loss, sync_dist=self.sync_dist)
        self.log("train0_box_reg_loss", box_loss, sync_dist=self.sync_dist)
        return loss

    def on_validation_epoch_start(self):
        self.val_outputs_all = []
        self.val_targets_all = []

    def validation_step(self, batch, batch_idx):
        val_inputs = self._val_inputs_from_batch(batch)
        val_targets = self._targets_from_batch(batch)
        val_outputs = self.detector(
            val_inputs, use_inferer=self._use_sliding_window_inferer(val_inputs)
        )
        maybe_store_batch_grid_preds(self, batch, val_outputs)
        self.val_outputs_all.extend(val_outputs)
        self.val_targets_all.extend(val_targets)

    def on_validation_epoch_end(self):
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

    def configure_optimizers(self):
        holder = {}
        result = configure_detection_optimizers(
            self.detector.network.parameters(), self.plan, self.lr, holder
        )
        self.scheduler_warmup = holder["scheduler_warmup"]
        return result

    def on_fit_start(self):
        plan = self.plan
        self.detector.set_sliding_window_inferer(
            roi_size=self.val_patch_size,
            overlap=float(plan.get("infer_overlap", INFER_OVERLAP)),
            sw_batch_size=int(plan.get("infer_sw_batch_size", 1)),
            mode="constant",
            device=str(self.device),
        )

    def on_train_epoch_start(self):
        if self.scheduler_warmup is not None:
            self.scheduler_warmup.step()
