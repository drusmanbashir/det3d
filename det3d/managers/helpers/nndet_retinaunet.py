"""RetinaUNetManager helpers — nnDet V001 build, batch bridge, box coords, inference."""

from __future__ import annotations

import sys
from copy import deepcopy

import numpy as np
import torch

from det3d.configs.nndet_bridge import NNDET_ROOT, build_nndet_trainer_cfg, load_nndet_train_cfgs
from det3d.utils.tensor import to_numpy
from utilz.helpers import pp


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


def plan_from_det3d(plan_train, plan_path=None):
    # AI
    if plan_path is not None:
        raise ValueError(
            "nnDet plan pickle is not read at RetinaUNet build; use plan_path=None"
        )
    return {
        "architecture": plan_architecture_from_det3d(plan_train),
        "anchors": plan_anchors_from_det3d(plan_train),
        "patch_size": [int(v) for v in plan_train["patch_size"]],
    }


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
    model_cfg, _ = load_nndet_train_cfgs()
    model_cfg = apply_det3d_plan_to_nndet_model_cfg(model_cfg, plan_train)
    trainer_cfg = build_nndet_trainer_cfg(configs)
    trainer_cfg["max_num_epochs"] = int(configs["model_params"]["max_epochs"])
    plan = plan_from_det3d(plan_train, plan_path=None)
    module = RetinaUNetV001(
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        plan=plan,
    )
    return module, plan


def xyzxyz_exclusive_to_nndet(box):
    # AI
    b = torch.as_tensor(box, dtype=torch.float32).reshape(-1)
    out = torch.empty(6, dtype=b.dtype, device=b.device)
    out[0] = b[0]
    out[1] = b[1]
    out[2] = b[3]
    out[3] = b[4]
    out[4] = b[2]
    out[5] = b[5]
    return out


def xyzxyz_exclusive_batch_to_nndet(boxes):
    # AI
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.numel() == 0:
        return boxes.reshape(0, 6)
    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)
    rows = [xyzxyz_exclusive_to_nndet(boxes[i]) for i in range(boxes.shape[0])]
    return torch.stack(rows, 0)


def disk_bbox_to_nndet_xyxyzz(disk_bbox: torch.Tensor) -> torch.Tensor:
    # AI
    out = xyzxyz_exclusive_batch_to_nndet(disk_bbox)
    if out.numel():
        out[:, [0, 1, 4]] -= 1
    return out


def nndet_xyxyzz_to_xyzxyz(box):
    # AI
    """nnDet xyxyzz -> xyzxyz axis reorder only (no +/-1)."""
    b = torch.as_tensor(box, dtype=torch.float32).reshape(-1)
    out = torch.empty(6, dtype=b.dtype, device=b.device)
    out[0] = b[0]
    out[1] = b[1]
    out[2] = b[4]
    out[3] = b[2]
    out[4] = b[3]
    out[5] = b[5]
    return out


def nndet_batch_to_xyzxyz(boxes):
    # AI
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.numel() == 0:
        return boxes.reshape(0, 6)
    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)
    rows = [nndet_xyxyzz_to_xyzxyz(boxes[i]) for i in range(boxes.shape[0])]
    return torch.stack(rows, 0)


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
    use_disk_box_plug=True,
):
    # AI
    # nnDet head expects xyxyzz tile-local boxes. Upstream contract: batch["bbox"] is
    # xyzxyz in patch/crop voxel space (LMG → LBD sidecar bbox_xyzxyz → DataManager).
    # disk_bbox_to_nndet_xyxyzz / xyzxyz_exclusive_batch_to_nndet complete the bridge.
    data = batch["image"]
    n = int(data.shape[0])
    target_boxes = []
    target_classes_raw = []
    box_to_nndet = (
        disk_bbox_to_nndet_xyxyzz
        if use_disk_box_plug
        else xyzxyz_exclusive_batch_to_nndet
    )
    for i in range(n):
        box = batch["bbox"][i]
        target_boxes.append(box_to_nndet(box))
        label = torch.as_tensor(batch["label"][i], dtype=torch.long).reshape(-1)
        target_classes_raw.append(label)
    target_classes = nndet_target_classes(target_boxes, target_classes_raw, fg_labels)
    lm = batch[seg_key]
    target_seg = lm.squeeze(1).long()
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


# %%
if __name__ == "__main__":
    RUN_NAME = "LIDCA-GYRO"
    CKPT = "/s/fran_storage/checkpoints/lidca/lidc/LIDCA-GYRO/checkpoints/last.ckpt"

# %%
# SECTION:--- load plan_train + pickle (checkpoint hyperparams) ---
    import torch
    from fran.inference.helpers import load_params
    from nndet.io.load import load_pickle

    params = load_params(RUN_NAME)
    plan_train = params["configs"]["plan_train"]
    plan_path = params["configs"]["nndet_plan_path"]
    pkl_plan = load_pickle(plan_path)
    print("plan_train decoder_levels", plan_train["decoder_levels"])
    print("pickle  decoder_levels", pkl_plan["architecture"]["decoder_levels"])
    print("pickle path", plan_path)

    pp(pkl_plan.keys())
# %%
# SECTION:--- BEFORE override (current production behaviour) ---
    def _apply_before(plan, plan_train):
        plan = deepcopy(plan)
        plan["patch_size"] = [int(v) for v in plan_train["patch_size"]]
        arch = plan["architecture"]
        arch["classifier_classes"] = len(plan_train["fg_labels"])
        arch["seg_classes"] = 1
        return plan

    plan_before = _apply_before(pkl_plan, plan_train)
    print("BEFORE merged decoder_levels", plan_before["architecture"]["decoder_levels"])

# %%
# SECTION:--- AFTER override (proposed apply_det3d_overrides_to_nndet_plan) ---
    from det3d.detection.retinaunet_network import _plan_arch

    def _plan_arch_scratch(plan_train):
        arch = _plan_arch(plan_train)
        arch["classifier_classes"] = len(plan_train["fg_labels"])
        arch["seg_classes"] = 1
        arch["score_thresh"] = float(plan_train["score_thresh"])
        arch["nms_thresh"] = float(plan_train["nms_thresh"])
        arch["detections_per_img"] = int(plan_train["detections_per_img"])
        arch["topk_candidates"] = int(plan_train["topk_candidates_per_level"])
        arch["remove_small_boxes"] = float(plan_train["remove_small_boxes"])
        return arch

    def _apply_after(plan, plan_train):
        plan = deepcopy(plan)
        plan["patch_size"] = [int(v) for v in plan_train["patch_size"]]
        plan["architecture"] = _plan_arch_scratch(plan_train)
        plan["anchors"] = plan_anchors_from_det3d(plan_train)
        return plan

    plan_after = _apply_after(pkl_plan, plan_train)
    print("AFTER  merged decoder_levels", plan_after["architecture"]["decoder_levels"])
    print("AFTER  cls_out channels (estimate)", len(plan_train["fg_labels"]), "fg labels")

# %%
# SECTION:--- strict checkpoint load: BEFORE should fail, AFTER should pass ---
    from det3d.configs.nndet_bridge import load_nndet_train_cfgs
    from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001

    model_cfg, trainer_cfg = load_nndet_train_cfgs()

    def _try_strict(plan, label):
        mod = RetinaUNetV001(model_cfg=model_cfg, trainer_cfg=trainer_cfg, plan=plan)
        sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
        net_sd = {
            k.replace("nndet_module.model.", ""): v
            for k, v in sd.items()
            if k.startswith("nndet_module.model.")
        }
        try:
            mod.model.load_state_dict(net_sd, strict=True)
            print(label, "strict load OK")
        except RuntimeError as exc:
            print(label, "strict load FAIL:", str(exc).split("\n")[0])

    _try_strict(plan_before, "BEFORE")
    _try_strict(plan_after, "AFTER")
# end PythonMethodScratch
