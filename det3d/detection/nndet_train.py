"""det3d batch → nnDetection RetinaUNetV001 train_step bridge."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from det3d.utils.tensor import plain_tensor as _plain_tensor, to_numpy
from utilz.stringz import ast_literal_eval

NNDET_ROOT = Path("/home/ub/code/nnDetection")
NNDET_TRAIN_CFG = NNDET_ROOT / "nndet/conf/train/v001.yaml"


def ensure_nndet_importable():
    if str(NNDET_ROOT) not in sys.path:
        sys.path.insert(0, str(NNDET_ROOT))
    import nndet.compat  # noqa: F401
    import omegaconf  # noqa: F401
    import loguru  # noqa: F401
    import pytorch_lightning  # noqa: F401


def _center_crop_starts(full_shape, patch_size):
    patch_size = tuple(int(v) for v in patch_size)
    return tuple(max(0, (int(full) - int(ps)) // 2) for full, ps in zip(full_shape, patch_size))


def _center_crop_spatial(x, patch_size, starts=None):
    patch_size = tuple(int(v) for v in patch_size)
    spatial = x.shape[-3:]
    if starts is None:
        starts = _center_crop_starts(spatial, patch_size)
    slices = tuple(slice(st, st + ps) for st, ps in zip(starts, patch_size))
    if x.dim() == 5:
        return x[(..., *slices)], starts
    if x.dim() == 4:
        return x[(slice(None), *slices)], starts
    if x.dim() == 3:
        return x[slices], starts
    raise ValueError(f"expected 3D, 4D, or 5D spatial tensor, got {x.shape}")


def _crop_boxes_to_patch(boxes, starts, patch_size):
    from monai.data.box_utils import clip_boxes_to_image

    if boxes.numel() == 0:
        return boxes
    patch_size = tuple(int(v) for v in patch_size)
    starts = torch.tensor(starts, device=boxes.device, dtype=boxes.dtype)
    shifted = boxes.clone()
    for i in range(3):
        shifted[:, i] -= starts[i]
        shifted[:, i + 3] -= starts[i]
    clipped, _ = clip_boxes_to_image(shifted, patch_size, remove_empty=True)
    return clipped


def forward_patch_size_from_configs(configs):
    fps = configs["model_params"].get("nndet_forward_patch_size")
    if fps is None:
        fps = configs["plan_train"]["patch_size"]
    if fps is None:
        return None
    if isinstance(fps, str):
        fps = ast_literal_eval(fps)
    return [int(v) for v in fps]


def _lm_seg_volume(lm_item):
  #AI
    t = torch.as_tensor(lm_item).long()
    while t.dim() > 3 and int(t.shape[0]) == 1:
        t = t.squeeze(0)
    if t.dim() == 4:
        t = t[0]
    return t


# Training bridge: det3d bbox sidecar (xyzxyz exclusive) -> nnDet xyxyzz for train_step.
# nnDet GT uses instances_to_boxes() min-1 / max+1 padding; see
# nnDetection/nndet/io/transforms/instances.py :: instances_to_boxes (~127-129).
# Do not apply inverse +/-1 on model predictions — keep pred_box in nnDet xyxyzz.


def nndet_xyxyzz_to_xyzxyz(box):
    """nnDet xyxyzz -> xyzxyz axis reorder only (no +/-1). For MONAI layout after inference."""
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
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.numel() == 0:
        return boxes.reshape(0, 6)
    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)
    rows = [nndet_xyxyzz_to_xyzxyz(boxes[i]) for i in range(boxes.shape[0])]
    return torch.stack(rows, 0)


def xyzxyz_exclusive_to_nndet(box):
    """det3d xyzxyz half-open sidecar -> nnDet xyxyzz (training targets only)."""
    b = torch.as_tensor(box, dtype=torch.float32).reshape(-1)
    out = torch.empty(6, dtype=b.dtype, device=b.device)
    out[0] = b[0] 
    out[1] = b[1]
    out[2] = b[3]      # x2: exclusive upper -> nnDet max+1
    out[3] = b[4]      # y2
    out[4] = b[2]
    out[5] = b[5]      # z2
    return out


def xyzxyz_exclusive_batch_to_nndet(boxes):
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.numel() == 0:
        return boxes.reshape(0, 6)
    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)
    rows = [xyzxyz_exclusive_to_nndet(boxes[i]) for i in range(boxes.shape[0])]
    return torch.stack(rows, 0)


def _nndet_target_classes(target_boxes, target_classes_raw, fg_labels):
  #AI
    """Map semantic box labels to nnDetection fg indices (0..K-1)."""
    label_to_idx = {int(v): i for i, v in enumerate(fg_labels)}
    out = []
    for boxes, cls in zip(target_boxes, target_classes_raw):
        cls = torch.as_tensor(cls, dtype=torch.long).reshape(-1)
        n = int(boxes.shape[0])
        cls = cls[:n]
        device = boxes.device if boxes.numel() else cls.device
        mapped = torch.tensor(
            [label_to_idx[int(v.item())] for v in cls],
            dtype=torch.long,
            device=device,
        )
        out.append(mapped)
    return out


def det3d_batch_to_nndet(batch, forward_patch_size=None, seg_key="lm", fg_labels=None):
  #AI
    """det3d collate → nnDetection train_step targets (center crop, box remap, fg_labels)."""
    data = batch["image"]
    crop_starts = None
    if forward_patch_size is not None:
        forward_patch_size = tuple(int(v) for v in forward_patch_size)
        spatial = tuple(int(v) for v in data.shape[-3:])
        if any(s > p for s, p in zip(spatial, forward_patch_size)):
            data, crop_starts = _center_crop_spatial(data, forward_patch_size)

    lm_src = batch[seg_key]
    n = int(data.shape[0])

    target_seg_list = []
    target_boxes = []
    target_classes_raw = []

    for i in range(n):
        lm_item = lm_src[i] if isinstance(lm_src, list) else lm_src[i]
        lm_vol = _lm_seg_volume(lm_item)
        if crop_starts is not None:
            lm_vol, _ = _center_crop_spatial(lm_vol, forward_patch_size, crop_starts)

        target_seg_list.append(lm_vol.long())

        box = batch["bbox"][i]
        if crop_starts is not None:
            box = _crop_boxes_to_patch(box, crop_starts, forward_patch_size)
        box = xyzxyz_exclusive_batch_to_nndet(box)
        target_boxes.append(box)

        label = torch.as_tensor(batch["label"][i], dtype=torch.long).reshape(-1)
        target_classes_raw.append(label)

    target_classes = _nndet_target_classes(
        target_boxes, target_classes_raw, fg_labels
    )

    target_seg = torch.stack(target_seg_list, 0)
    out = {
        "data": data,
        "target_boxes": target_boxes,
        "target_classes": target_classes,
        "target_seg": target_seg,
    }
    return out


def load_nndet_train_cfgs(cfg_path=NNDET_TRAIN_CFG):
    with open(cfg_path) as f:
        train_cfg = yaml.safe_load(f)
    return deepcopy(train_cfg["model_cfg"]), deepcopy(train_cfg["trainer_cfg"])


def plan_anchors_from_det3d(plan_train):
    shapes = plan_train.get("base_anchor_shapes")
    if shapes is None:
        shapes = [[6, 8, 4], [8, 6, 5], [10, 10, 6]]
    n_levels = len(plan_train.get("decoder_levels", (1, 2, 3, 4)))
    if isinstance(n_levels, str):
        n_levels = ast_literal_eval(n_levels)
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
    from det3d.detection.retinaunet_network import _plan_arch

    arch = _plan_arch(plan_train)
    n_fg = len(plan_train["fg_labels"])
    arch["classifier_classes"] = n_fg
    arch["seg_classes"] = n_fg
    arch["score_thresh"] = float(plan_train.get("score_thresh", 0.02))
    arch["nms_thresh"] = float(plan_train.get("nms_thresh", 0.22))
    arch["detections_per_img"] = int(plan_train.get("detections_per_img", 25))
    arch["topk_candidates"] = int(plan_train.get("topk_candidates_per_level", 1000))
    arch["remove_small_boxes"] = float(plan_train.get("remove_small_boxes", 0.01))
    return arch


def plan_from_det3d(plan_train, plan_path=None):
    if plan_path is not None:
        ensure_nndet_importable()
        from nndet.io.load import load_pickle

        return load_pickle(plan_path)
    fps = plan_train.get("nndet_forward_patch_size")
    patch_size = [int(v) for v in (fps if fps is not None else plan_train["patch_size"])]
    return {
        "architecture": plan_architecture_from_det3d(plan_train),
        "anchors": plan_anchors_from_det3d(plan_train),
        "patch_size": patch_size,
    }


def apply_det3d_plan_to_nndet_model_cfg(model_cfg, plan_train):
    model_cfg = deepcopy(model_cfg)
    model_cfg["matcher_kwargs"]["num_candidates"] = int(
        plan_train.get("matcher_num_candidates", 4)
    )
    model_cfg["matcher_kwargs"]["center_in_gt"] = bool(
        plan_train.get("matcher_center_in_gt", False)
    )
    model_cfg["head_sampler_kwargs"]["batch_size_per_image"] = int(
        plan_train.get("sampler_batch_size_per_image", 32)
    )
    model_cfg["head_sampler_kwargs"]["positive_fraction"] = float(
        plan_train.get("balanced_sampler_pos_fraction", 0.33)
    )
    model_cfg["head_sampler_kwargs"]["pool_size"] = int(
        plan_train.get("sampler_pool_size", 20)
    )
    model_cfg["head_sampler_kwargs"]["min_neg"] = int(
        plan_train.get("sampler_min_neg", 1)
    )
    return model_cfg


def build_nndet_retinaunet_module(configs, num_train_batches):
    ensure_nndet_importable()
    from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001

    plan_train = configs["plan_train"]
    plan_path = configs["model_params"].get("nndet_plan_path")
    model_cfg, trainer_cfg = load_nndet_train_cfgs()
    model_cfg = apply_det3d_plan_to_nndet_model_cfg(model_cfg, plan_train)
    trainer_cfg["num_train_batches_per_epoch"] = int(num_train_batches)
    trainer_cfg["max_num_epochs"] = int(configs["model_params"].get("max_epochs", 600))
    plan = plan_from_det3d(plan_train, plan_path=plan_path)
    module = RetinaUNetV001(
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        plan=plan,
    )
    return module, plan


def offset_nndet_xyxyzz_boxes(boxes, origin):
    """Shift nnDet xyxyzz tile-local boxes to parent-volume coords.

    Pair shift per axis: cols 0,2 += ox; 1,3 += oy; 4,5 += oz (not xyzxyz pairs).
    Layout matches nnDet train_step / postprocess_detections output.
    """
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


def nndet_predict_plan_for_inference(nndet_plan, plan_train):
  #AI
    arch = nndet_plan["architecture"]
    plan = deepcopy(nndet_plan)
    plan["network_dim"] = 3
    plan["batch_size"] = 1
    plan["transpose_backward"] = [0, 1, 2]
    plan["inference_plan"] = {
        "model_score_thresh": float(plan_train["score_thresh"]),
        "model_detections_per_image": int(arch["detections_per_img"]),
        "model_topk": int(arch["topk_candidates"]),
        "remove_small_boxes": float(arch["remove_small_boxes"]),
    }
    return plan


def nndet_identity_properties(spatial_shape, spacing):
  #AI
    import numpy as np

    shape = tuple(int(v) for v in spatial_shape)
    spacing = [float(v) for v in spacing]
    crop_bbox = [[0, s] for s in shape]
    out = {
        "transpose_backward": [0, 1, 2],
        "original_spacing": spacing,
        "spacing_after_resampling": spacing,
        "crop_bbox": crop_bbox,
        "size_after_cropping": list(shape),
        "original_size_of_raw_data": list(shape),
        "itk_origin": [0.0, 0.0, 0.0],
        "itk_spacing": spacing,
        "itk_direction": np.eye(3).reshape(-1).tolist(),
    }
    return out


@torch.no_grad()
def nndet_predict_case(net, nndet_plan, plan_train, image, device, overlap=0.25, do_seg=True, num_tta=0):
  #AI
    """Full-volume nnDetection predict_case (BoxEnsemblerSelective + SegmentationEnsembler)."""
    ensure_nndet_importable()
    from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001

    image = _plain_tensor(image)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    case_np = to_numpy(image).astype(np.float32)
    spacing = plan_train["spacing"]
    properties = nndet_identity_properties(case_np.shape[1:], spacing)
    plan = nndet_predict_plan_for_inference(nndet_plan, plan_train)
    predictor = RetinaUNetV001.get_predictor(
        plan=plan,
        models=[net],
        num_tta_transforms=int(num_tta),
        do_seg=do_seg,
        device=str(device),
        overlap=float(overlap),
    )
    result = predictor.predict_case(
        {"data": case_np},
        properties,
        save_dir=None,
        case_id=None,
        restore=False,
    )
    boxes = result["boxes"]
    out = {
        "pred_boxes": [torch.as_tensor(boxes["pred_boxes"], device=device)],
        "pred_scores": [torch.as_tensor(boxes["pred_scores"], device=device)],
        "pred_labels": [torch.as_tensor(boxes["pred_labels"], device=device)],
    }
    if do_seg:
        seg = result["seg"]["pred_seg"]
        if torch.is_tensor(seg):
            seg_label = seg.to(dtype=torch.uint8)
        else:
            seg_label = torch.as_tensor(seg, dtype=torch.uint8)
        out["pred_seg_label"] = seg_label
    return out


def nndet_pred_to_vis(pred, box_key="bbox", label_key="label", score_key="label_scores"):
    boxes = pred["pred_boxes"][0]
    labels = pred["pred_labels"][0]
    scores = pred["pred_scores"][0]
    out = {box_key: boxes, label_key: labels, score_key: scores}
    if "pred_seg" in pred:
        seg = pred["pred_seg"]
        out["pred_seg"] = seg[0] if seg.dim() > 3 else seg
    return out


def nndet_batch_pred_to_vis_list(pred):
    """One vis pred dict per batch item from nnDetection train_step / inference output."""
    n = len(pred["pred_boxes"])
    vis_preds = []
    for b in range(n):
        item = {
            "pred_boxes": [pred["pred_boxes"][b]],
            "pred_labels": [pred["pred_labels"][b]],
            "pred_scores": [pred["pred_scores"][b]],
        }
        if "pred_seg" in pred:
            item["pred_seg"] = pred["pred_seg"][b : b + 1]
        vis = nndet_pred_to_vis(item)
        vis_preds.append(
            {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in vis.items()
            }
        )
    return vis_preds


def maybe_store_batch_grid_preds(pl_module, batch, preds):
    """Stash validation preds on batch for wandb det grid (no extra forward)."""
    if isinstance(preds, list):
        batch["pred"] = [
            {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in p.items()
            }
            for p in preds
        ]
        return
    batch["pred"] = nndet_batch_pred_to_vis_list(preds)


def nndet_val_targets_from_batch(batch, seg_key="lm"):
    target_boxes = list(batch["bbox"])
    target_classes = list(batch["label"])
    target_seg = None
    if seg_key in batch:
        seg_source = batch[seg_key]
        if isinstance(seg_source, list):
            target_seg = torch.stack(seg_source, 0)
        else:
            target_seg = seg_source
        if target_seg.dim() == 5:
            target_seg = target_seg[:, 0]
    return {
        "target_boxes": target_boxes,
        "target_classes": target_classes,
        "target_seg": target_seg,
    }
