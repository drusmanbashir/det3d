"""Segmentation validation metrics."""

import torch


def compute_seg_dice(val_pred_seg, val_target_lm):
    """Mean foreground Dice over accumulated val batches."""
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
