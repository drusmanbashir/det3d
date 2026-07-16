import numpy as np
import torch
from monai.apps.detection.utils.detector_utils import check_training_targets
from monai.apps.detection.utils.predict_utils import ensure_dict_value_to_list_

from det3d.architectures.create_detector import INFER_OVERLAP, create_detector_from_conf
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
        self.model_params = configs["model_params"]
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
        label_to_idx = {int(v): i for i, v in enumerate(self.plan["fg_labels"])}
        out = []
        for label, box in zip(labels, boxes):
            box_tensor = torch.as_tensor(box, device=self.device).reshape(-1, 6)
            cls_raw = torch.as_tensor(label, device=self.device).reshape(-1)[: box_tensor.shape[0]]
            mapped = torch.tensor(
                [label_to_idx[int(v.item())] for v in cls_raw],
                dtype=torch.long,
                device=self.device,
            )
            out.append({label_key: mapped, box_key: box_tensor})
        return out

    def _val_inputs_from_batch(self, batch):
        images = batch["image"]
        return [images[i].contiguous() for i in range(images.shape[0])]

    def _use_sliding_window_inferer(self, val_inputs):
        patch_voxels = int(np.prod(self.val_patch_size))
        return not all(item[0, ...].numel() < patch_voxels for item in val_inputs)

    def _store_batch_grid_preds(self, batch, preds):
        batch["pred"] = [
            {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in p.items()
            }
            for p in preds
        ]

    def maybe_store_grid_preds(self, batch):
        if not (hasattr(self.trainer, "store_preds") and self.trainer.store_preds):
            return
        with torch.no_grad():
            val_inputs = self._val_inputs_from_batch(batch)
            was_training = self.detector.training
            self.detector.eval()
            val_outputs = self.detector(
                val_inputs, use_inferer=self._use_sliding_window_inferer(val_inputs)
            )
            if was_training:
                self.detector.train()
            self._store_batch_grid_preds(batch, val_outputs)

    def _forward_network_head(self, images):
        dtype = next(self.detector.network.parameters()).dtype
        if images.dtype != dtype:
            images = images.to(dtype=dtype)
        head_outputs = self.detector.network(images)
        if isinstance(head_outputs, (tuple, list)):
            head_outputs = {
                self.detector.cls_key: head_outputs[: len(head_outputs) // 2],
                self.detector.box_reg_key: head_outputs[len(head_outputs) // 2 :],
            }
        else:
            ensure_dict_value_to_list_(head_outputs)
        return head_outputs

    def _build_train_anchors(self, images, head_outputs):
        self.detector.generate_anchors(images, head_outputs)
        num_anchor_locs_per_level = [
            x.shape[2:].numel() for x in head_outputs[self.detector.cls_key]
        ]
        for key in (self.detector.cls_key, self.detector.box_reg_key):
            head_outputs[key] = self.detector._reshape_maps(head_outputs[key])
        return head_outputs, num_anchor_locs_per_level

    def _forward_train_batched(self, images, targets):
        """Training forward on DM-prebatched (B,C,D,H,W); skips RetinaNetDetector.preprocess_images."""
        targets = check_training_targets(
            images,
            targets,
            self.detector.spatial_dims,
            self.detector.target_label_key,
            self.detector.target_box_key,
        )
        self.detector._check_detector_training_components()
        head_outputs = self._forward_network_head(images)
        head_outputs, num_anchor_locs_per_level = self._build_train_anchors(
            images, head_outputs
        )
        return self.detector.compute_loss(
            head_outputs, targets, self.detector.anchors, num_anchor_locs_per_level
        )

    def training_step(self, batch, batch_idx):
        outputs = self._forward_train_batched(
            batch["image"], self._targets_from_batch(batch)
        )
        cls_loss = outputs[self.detector.cls_key]
        box_loss = outputs[self.detector.box_reg_key]
        loss = self.w_cls * cls_loss + self.w_reg * box_loss
        self.log("train0_loss", loss, prog_bar=True, sync_dist=self.sync_dist)
        self.log("train0_cls_loss", cls_loss, sync_dist=self.sync_dist)
        self.log("train0_box_reg_loss", box_loss, sync_dist=self.sync_dist)
        self.maybe_store_grid_preds(batch)
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
        self._store_batch_grid_preds(batch, val_outputs)
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
            self.detector.network.parameters(),
            self.plan,
            self.lr,
            holder,
            monitor=self.plan["scheduler_monitor"],
        )
        self.scheduler_warmup = holder["scheduler_warmup"]
        return result
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
