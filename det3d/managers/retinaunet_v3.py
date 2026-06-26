"""RetinaUNet v3 Lightning manager — self-contained (TrainerDet → detector_factory → here)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.apps.detection.utils.detector_utils import check_training_targets
from monai.apps.detection.utils.predict_utils import ensure_dict_value_to_list_
from monai.losses import DiceLoss
from monai.utils.enums import LossReduction

from det3d.detection.retinaunet_v3 import INFER_OVERLAP, create_retinaunet_v3_detector
from det3d.evaluation.coco import compute_coco_metrics
from det3d.transforms.warmup_scheduler import GradualWarmupScheduler
from fran.managers.project import Project
from lightning.pytorch import LightningModule


class RetinaUNetSegLoss(nn.Module):
    def __init__(self, lambda_dice=0.5, lambda_ce=0.5):
        super().__init__()
        self.lambda_dice = float(lambda_dice)
        self.lambda_ce = float(lambda_ce)
        self.loss_dict = {}
        self.dice = DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            reduction=LossReduction.MEAN,
        )
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, seg_logits, lm_batch):
        if isinstance(lm_batch, list):
            stacked = torch.stack([torch.as_tensor(item) for item in lm_batch], dim=0)
        else:
            stacked = lm_batch
        if stacked.dim() == 4:
            stacked = stacked.unsqueeze(1)
        if stacked.shape[1] != 1:
            stacked = stacked[:, :1]
        target = (stacked > 0).long().to(device=seg_logits.device)
        if target.shape[-3:] != seg_logits.shape[-3:]:
            target = F.interpolate(
                target.float(),
                size=seg_logits.shape[-3:],
                mode="nearest",
            ).long()
        loss_ce = self.cross_entropy(seg_logits, target[:, 0])
        loss_dice = self.dice(seg_logits, target)
        total = self.lambda_dice * loss_dice + self.lambda_ce * loss_ce
        self.loss_dict = {
            "loss_ce": loss_ce.detach(),
            "loss_dice": loss_dice.detach(),
        }
        return total


def _compute_seg_dice(val_pred_seg, val_target_lm):
    dices = []
    for pred, target in zip(val_pred_seg, val_target_lm):
        pred_fg = (pred > 0).reshape(-1).float()
        if isinstance(target, list):
            target = target[0]
        target_t = torch.as_tensor(target)
        if target_t.dim() == 4:
            target_t = target_t[0]
        tgt_fg = (target_t > 0).reshape(-1).float()
        inter = (pred_fg * tgt_fg).sum()
        denom = pred_fg.sum() + tgt_fg.sum()
        if denom > 0:
            dices.append(float((2.0 * inter / denom).item()))
        else:
            dices.append(1.0)
    if not dices:
        return 0.0
    return sum(dices) / len(dices)


class RetinaUNetManagerV3(LightningModule):
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
        self.val_pred_seg = []
        self.val_target_lm = []
        self.scheduler_warmup = None
        self.batch_size = 1
        self.detector, self.val_patch_size = create_retinaunet_v3_detector(configs)
        lp = configs["loss_params"]
        self.seg_loss_fnc = RetinaUNetSegLoss(
            lambda_dice=float(lp["lambda_dice"]),
            lambda_ce=float(lp["lambda_ce"]),
        )

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

    def _forward_network_head(self, images):
        # dtype = next(self.detector.network.parameters()).dtype
        # if images.dtype != dtype:
        #     images = images.to(dtype=dtype)
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

    def _forward_train_joint(self, images, targets, lm_batch):
        targets = check_training_targets(
            images,
            targets,
            self.detector.spatial_dims,
            self.detector.target_label_key,
            self.detector.target_box_key,
        )
        self.detector._check_detector_training_components()
        head_outputs = self._forward_network_head(images)
        seg_key = self.detector.network.seg_key
        seg_logits = head_outputs.pop(seg_key)
        if isinstance(seg_logits, list):
            seg_logits = seg_logits[0]
        head_outputs, num_anchor_locs_per_level = self._build_train_anchors(
            images, head_outputs
        )
        det_losses = self.detector.compute_loss(
            head_outputs, targets, self.detector.anchors, num_anchor_locs_per_level
        )
        seg_total = self.seg_loss_fnc(seg_logits, lm_batch)
        cls_key = self.detector.cls_key
        box_key = self.detector.box_reg_key
        cls_loss = det_losses[cls_key]
        box_loss = det_losses[box_key]
        total = self.w_cls * cls_loss + self.w_reg * box_loss + seg_total
        return {
            "loss": total,
            cls_key: cls_loss.detach(),
            box_key: box_loss.detach(),
            "loss_ce": self.seg_loss_fnc.loss_dict["loss_ce"],
            "loss_dice": self.seg_loss_fnc.loss_dict["loss_dice"],
        }

    def training_step(self, batch, batch_idx):
        self.batch_size = batch["image"].shape[0]
        outputs = self._forward_train_joint(
            batch["image"],
            self._targets_from_batch(batch),
            batch["lm"],
        )
        loss = outputs["loss"]
        logger_dict = {f"train0_{k}": outputs[k] for k in outputs if k != "loss"}
        logger_dict["train0_loss"] = loss
        self.log_dict(
            logger_dict,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=self.sync_dist,
            prog_bar=True,
        )
        return loss

    def on_validation_epoch_start(self):
        self.val_outputs_all = []
        self.val_targets_all = []
        self.val_pred_seg = []
        self.val_target_lm = []

    def validation_step(self, batch, batch_idx):
        val_inputs = self._val_inputs_from_batch(batch)
        val_targets = self._targets_from_batch(batch)
        val_outputs = self.detector(
            val_inputs, use_inferer=self._use_sliding_window_inferer(val_inputs)
        )
        self._store_batch_grid_preds(batch, val_outputs)
        with torch.no_grad():
            out = self.detector.network(batch["image"])
        post = self.detector.network.segmenter.postprocess_for_inference(
            {"seg_logits": out[self.detector.network.seg_key]}
        )
        for i, pred in enumerate(batch["pred"]):
            pred["pred_seg"] = post["pred_seg"][i].detach().cpu()
        self.val_outputs_all.extend(val_outputs)
        self.val_targets_all.extend(val_targets)
        for i in range(batch["image"].shape[0]):
            self.val_pred_seg.append(post["pred_seg"][i, 1].detach().cpu())
            lm = batch["lm"][i]
            if isinstance(lm, list):
                self.val_target_lm.append(lm[0])
            else:
                self.val_target_lm.append(lm.detach().cpu())

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
        dice = _compute_seg_dice(self.val_pred_seg, self.val_target_lm)
        self.log("val0_seg_dice", dice, prog_bar=True, sync_dist=self.sync_dist)

    def configure_optimizers(self):
        plan = self.plan
        optimizer = torch.optim.SGD(
            self.detector.network.parameters(),
            self.lr,
            momentum=float(plan.get("momentum", 0.9)),
            weight_decay=float(plan.get("weight_decay", 3e-5)),
            nesterov=bool(plan.get("nesterov", True)),
        )
        lr_schedule = str(plan.get("lr_schedule", "epoch_step")).lower()
        if lr_schedule == "poly_iter":
            warm = int(plan.get("warm_iterations", 4000))
            gamma = float(plan.get("poly_gamma", 0.9))
            max_iter = int(plan.get("max_iterations", 125000))
            warm_lr = float(plan.get("warm_lr", 1e-6))

            def lr_lambda(step: int):
                if step < warm:
                    return max(warm_lr / self.lr, step / max(warm, 1))
                progress = (step - warm) / max(max_iter - warm, 1)
                return max((1.0 - progress) ** gamma, 0.0)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            self.scheduler_warmup = None
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        after_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
        self.scheduler_warmup = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=10, after_scheduler=after_scheduler
        )
        return optimizer

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
