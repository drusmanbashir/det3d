"""RetinaUNetManager helpers — nnDet V001 build, batch bridge, box coords, inference."""

from __future__ import annotations

import sys
from copy import deepcopy

import numpy as np
import torch

from det3d.configs.nndet_bridge import NNDET_ROOT, load_nndet_train_cfgs
from det3d.utils.tensor import to_numpy


def ensure_nndet_importable():
    # AI
    if str(NNDET_ROOT) not in sys.path:
        sys.path.insert(0, str(NNDET_ROOT))
    import nndet.compat  # noqa: F401
    import omegaconf  # noqa: F401
    import loguru  # noqa: F401
    import pytorch_lightning  # noqa: F401


def plan_anchors_from_det3d(plan_train):
    # AI
    if "nndet_anchors" in plan_train:
        return deepcopy(plan_train["nndet_anchors"])
    shapes = plan_train["base_anchor_shapes"]
    n_levels = len(plan_train["decoder_levels"])
    while len(shapes) < n_levels:
        shapes = shapes + [shapes[-1]]
    zsizes = tuple(int(s[2]) for s in shapes[:n_levels])
    sizes = tuple(max(int(s[0]), int(s[1])) for s in shapes[:n_levels])
    return {
        "stride": 1,
        "aspect_ratios": (0.5, 1.0, 2.0),
        "sizes": sizes,
        "zsizes": zsizes,
    }


def plan_architecture_from_det3d(plan_train):
    # AI
    from det3d.detection.retinaunet_network import _plan_arch

    arch = _plan_arch(plan_train)
    n_fg = len(plan_train["fg_labels"])
    arch["classifier_classes"] = n_fg
    arch["seg_classes"] = 1
    arch["score_thresh"] = float(plan_train["score_thresh"])
    arch["nms_thresh"] = float(plan_train["nms_thresh"])
    arch["detections_per_img"] = int(plan_train["detections_per_img"])
    arch["topk_candidates"] = int(plan_train.get("topk_candidates_per_level", 1000))
    arch["remove_small_boxes"] = float(plan_train.get("remove_small_boxes", 0.0))
    return arch


def plan_from_det3d(plan_train):
    # AI
    architecture = plan_architecture_from_det3d(plan_train)
    if "nndet_anchors" in plan_train:
        architecture["seg_classes"] = len(plan_train["fg_labels"])
    plan = {
        "architecture": architecture,
        "anchors": plan_anchors_from_det3d(plan_train),
        "patch_size": [int(v) for v in plan_train["patch_size"]],
    }
    return plan


def apply_det3d_plan_to_nndet_model_cfg(model_cfg, plan_train):
    # AI
    model_cfg = deepcopy(model_cfg)
    model_cfg["matcher_kwargs"]["num_candidates"] = int(
        plan_train["matcher_num_candidates"]
    )
    model_cfg["matcher_kwargs"]["center_in_gt"] = bool(
        plan_train["matcher_center_in_gt"]
    )
    model_cfg["head_sampler_kwargs"]["batch_size_per_image"] = int(
        plan_train["sampler_batch_size_per_image"]
    )
    model_cfg["head_sampler_kwargs"]["positive_fraction"] = float(
        plan_train["balanced_sampler_pos_fraction"]
    )
    model_cfg["head_sampler_kwargs"]["pool_size"] = int(
        plan_train["sampler_pool_size"]
    )
    model_cfg["head_sampler_kwargs"]["min_neg"] = int(plan_train["sampler_min_neg"])
    return model_cfg


def build_nndet_retinaunet_module(configs):
    # AI
    ensure_nndet_importable()
    from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001

    plan_train = configs["plan_train"]
    model_cfg, trainer_cfg = load_nndet_train_cfgs()
    model_cfg = apply_det3d_plan_to_nndet_model_cfg(model_cfg, plan_train)
    plan = plan_from_det3d(plan_train)
    module = RetinaUNetV001(
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        plan=plan,
    )
    return module, plan


def to_nndet(boxes: torch.Tensor) -> torch.Tensor:  #AI
    """Convert exclusive xyzxyz boxes to nnDetection xyxyzz with padded minima."""
    native = boxes.reshape(-1, 6)[:, [0, 1, 3, 4, 2, 5]]
    native[:, [0, 1, 4]] -= 1
    return native


def from_nndet(boxes: torch.Tensor) -> torch.Tensor:  #AI
    """Convert xyxyzz to exclusive xyzxyz, preserving a clamped zero minimum.

    Native zero is ambiguous after clamping, so a det3d minimum of 1 round-trips to 0.
    """
    canonical = boxes.reshape(-1, 6)[:, [0, 1, 4, 2, 3, 5]]
    minima = canonical[:, :3]
    canonical[:, :3] = torch.where(minima > 0, minima + 1, 0)
    return canonical


def offset_nndet_xyxyzz_boxes(boxes, origin):
    # AI
    """Shift nnDet xyxyzz tile-local boxes to parent-volume coords."""
    if boxes.numel() == 0:
        return boxes
    shifted = boxes.clone()
    ox, oy, oz = (float(v) for v in origin)
    shifted[:, 0] += ox
    shifted[:, 1] += oy
    shifted[:, 2] += ox
    shifted[:, 3] += oy
    shifted[:, 4] += oz
    shifted[:, 5] += oz
    return shifted


def nndet_target_classes(target_boxes, target_classes_raw, fg_labels):
    # AI
    label_to_idx = {int(v): i for i, v in enumerate(fg_labels)}
    out = []
    for boxes, cls in zip(target_boxes, target_classes_raw):
        cls = torch.as_tensor(cls, dtype=torch.long).reshape(-1)
        n = int(boxes.shape[0])
        cls = cls[:n]
        mapped = torch.tensor(
            [label_to_idx[int(v.item())] for v in cls],
            dtype=torch.long,
            device=boxes.device,
        )
        out.append(mapped)
    return out


def det3d_batch_to_nndet(
    batch,
    fg_labels,
    *,
    seg_key="lm",
):
    # AI
    data = batch["image"]
    n = int(data.shape[0])
    target_boxes = []
    target_classes_raw = []
    for i in range(n):
        target_boxes.append(to_nndet(batch["bbox"][i]))
        label = torch.as_tensor(batch["label"][i], dtype=torch.long).reshape(-1)
        target_classes_raw.append(label)
    target_classes = nndet_target_classes(target_boxes, target_classes_raw, fg_labels)
    lm = batch[seg_key].squeeze(1).long()
    label_to_seg = {int(label): index + 1 for index, label in enumerate(fg_labels)}
    target_seg = torch.zeros_like(lm)
    for label, mapped in label_to_seg.items():
        target_seg[lm == label] = mapped
    out = {
        "data": data,
        "target_boxes": target_boxes,
        "target_classes": target_classes,
        "target_seg": target_seg,
    }
    return out


def fast_nndet_batch_to_device(nb, device):
    # AI
    nb["data"] = nb["data"].to(device)
    nb["target_boxes"] = [b.to(device) for b in nb["target_boxes"]]
    nb["target_classes"] = [c.to(device) for c in nb["target_classes"]]
    nb["target_seg"] = nb["target_seg"].to(device)
    return nb


NNDET_LOSS_LOG_NAMES = {
    "cls": "cls_loss",
    "reg": "box_reg_loss",
    "seg_ce": "loss_ce",
    "seg_dice": "loss_dice",
}


def nndet_loss_log_key(prefix, loss_key):
    # AI
    return f"{prefix}_{NNDET_LOSS_LOG_NAMES.get(loss_key, loss_key)}"


def log_nndet_det3d_step_losses(pl_module, losses, prefix, sync_dist=False):
    # AI
    total = sum(losses.values())
    pl_module.log(
        f"{prefix}_loss",
        total,
        prog_bar=(prefix == "train0"),
        on_step=True,
        on_epoch=True,
        sync_dist=sync_dist,
    )
    for key, val in losses.items():
        pl_module.log(
            nndet_loss_log_key(prefix, key),
            val,
            on_step=(prefix == "train0"),
            on_epoch=True,
            sync_dist=sync_dist,
        )
    return total


def nndet_pred_to_vis(
    pred, box_key="bbox", label_key="label", score_key="label_scores"
):
    # AI
    boxes = pred["pred_boxes"][0]
    labels = pred["pred_labels"][0]
    scores = pred["pred_scores"][0]
    out = {box_key: boxes, label_key: labels, score_key: scores}
    if "pred_seg" in pred:
        seg = pred["pred_seg"]
        out["pred_seg"] = seg[0] if seg.dim() > 3 else seg
    return out
