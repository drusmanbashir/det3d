"""Detection + segmentation losses (FRAN evaluation/losses pattern)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss
from monai.utils.enums import LossReduction


def _detach_for_logging(value):
    if torch.is_tensor(value):
        return value.detach()
    return value


def lm_to_seg_target(lm_batch):
    """Instance label map batch -> binary class indices [B, 1, D, H, W]."""
    if isinstance(lm_batch, list):
        stacked = torch.stack(
            [torch.as_tensor(item) for item in lm_batch],
            dim=0,
        )
    else:
        stacked = lm_batch
    if stacked.dim() == 4:
        stacked = stacked.unsqueeze(1)
    if stacked.shape[1] != 1:
        stacked = stacked[:, :1]
    target = (stacked > 0).long()
    return target


class RetinaUNetSegLoss(nn.Module):
    """MONAI Dice + CE on joint RetinaUNet seg logits (fg/bg)."""

    def __init__(
        self,
        lambda_dice=0.5,
        lambda_ce=0.5,
        include_background=False,
        fg_classes=1,
    ):
        super().__init__()
        self.lambda_dice = float(lambda_dice)
        self.lambda_ce = float(lambda_ce)
        self.include_background = bool(include_background)
        self.fg_classes = int(fg_classes)
        self.loss_dict = {}
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=True,
            softmax=True,
            reduction=LossReduction.MEAN,
        )
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, seg_logits, lm_batch, use_mask=False):
        target = lm_to_seg_target(lm_batch).to(device=seg_logits.device)
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
            "loss": _detach_for_logging(total),
            "loss_ce": _detach_for_logging(loss_ce),
            "loss_dice": _detach_for_logging(loss_dice),
        }
        return total


def combine_det_seg_loss_dict(det_losses, seg_total, seg_loss_dict, detector, plan):
    cls_key = detector.cls_key
    box_key = detector.box_reg_key
    w_cls = float(plan["w_cls"])
    w_reg = float(plan["w_reg"])
    cls_loss = det_losses[cls_key]
    box_loss = det_losses[box_key]
    total = w_cls * cls_loss + w_reg * box_loss + seg_total
    out = {
        "loss": total,
        cls_key: _detach_for_logging(cls_loss),
        box_key: _detach_for_logging(box_loss),
        "loss_ce": seg_loss_dict["loss_ce"],
        "loss_dice": seg_loss_dict["loss_dice"],
    }
    return out
