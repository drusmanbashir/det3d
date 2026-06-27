"""det3d batch → nnDetection RetinaUNetV001 train_step bridge."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from det3d.utils.tensor import plain_tensor as _plain_tensor, sanitize_for_numpy, sanitize_tensor_for_numpy, to_numpy

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
    return tuple(
        max(0, (int(full) - int(ps)) // 2) for full, ps in zip(full_shape, patch_size)
    )


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


def _lm_seg_volume(lm_item):
    # AI
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
    out[2] = b[3]  # x2: exclusive upper -> nnDet max+1
    out[3] = b[4]  # y2
    out[4] = b[2]
    out[5] = b[5]  # z2
    return out


def xyzxyz_exclusive_batch_to_nndet(boxes):
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.numel() == 0:
        return boxes.reshape(0, 6)
    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)
    rows = [xyzxyz_exclusive_to_nndet(boxes[i]) for i in range(boxes.shape[0])]
    return torch.stack(rows, 0)


def disk_bbox_to_nndet_xyxyzz(disk_bbox: torch.Tensor) -> torch.Tensor:
    """MONAI xyzxyz (post aug) → nnDet xyxyzz matching ``instances_to_boxes`` lower −1."""
    out = xyzxyz_exclusive_batch_to_nndet(disk_bbox)
    if out.numel():
        out[:, [0, 1, 4]] -= 1
    return out


def det3d_semantic_target_seg_from_batch(
    batch_pre: dict,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Map aug'd lm instance ids → semantic seg (same rule as Instances2Segmentation)."""
    ensure_nndet_importable()
    from nndet.io.transforms.instances import instances_to_segmentation

    target = batch_pre["target"]
    semantic = torch.zeros_like(target)
    for i in range(int(target.shape[0])):
        instances_to_segmentation(
            target[i, 0],
            batch_pre["instance_mapping"][i],
            add_background=True,
            out=semantic[i, 0],
        )
    if device is not None:
        semantic = semantic.to(device)
    return semantic[:, 0]


def _nndet_target_classes(target_boxes, target_classes_raw, fg_labels):
    # AI
    """Map semantic box labels to nnDetection fg indices (0..K-1)."""
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


def _instance_mapping_for_item(lm_item, labels, *, fg_labels, instances=None):
    # AI
    """Map lm instance ids to nnDet fg class indices."""
    if instances is not None:
        label_to_idx = {int(v): i for i, v in enumerate(fg_labels)}
        mapping = {}
        for key, semantic in instances.items():
            mapping[str(key)] = label_to_idx[int(semantic)]
        return mapping
    vol = _lm_seg_volume(lm_item)
    inst = vol.unique(sorted=True)
    inst = inst[inst > 0].tolist()
    lbl = torch.as_tensor(labels).reshape(-1).long()
    label_to_idx = {int(v): i for i, v in enumerate(fg_labels)}
    mapping = {}
    for j, iid in enumerate(sorted(int(i) for i in inst)):
        if j < lbl.numel():
            mapping[str(iid)] = label_to_idx[int(lbl[j].item())]
        else:
            mapping[str(iid)] = 0
    return mapping


def det3d_batch_to_pre_trafo_input(batch, patch_size, fg_labels):
    # AI
    """det3d collate batch -> nnDet pre_trafo input: data, target, instance_mapping."""
    data = batch["image"].float()
    patch_size = tuple(int(v) for v in patch_size)
    spatial = tuple(int(v) for v in data.shape[-3:])
    crop_starts = None
    if any(s > p for s, p in zip(spatial, patch_size)):
        data, crop_starts = _center_crop_spatial(data, patch_size)

    lm_src = batch["lm"]
    n = int(data.shape[0])
    targets = []
    mappings = []
    instances_batch = batch["instances"]
    labels = batch["label"]
    for i in range(n):
        vol = _lm_seg_volume(lm_src[i])
        if crop_starts is not None:
            vol, _ = _center_crop_spatial(vol, patch_size, crop_starts)
        targets.append(vol.float().unsqueeze(0).unsqueeze(0))
        mappings.append(
            _instance_mapping_for_item(
                lm_src[i], labels[i], instances=instances_batch[i], fg_labels=fg_labels
            )
        )

    batch_pre = {
        "data": data,
        "target": torch.cat(targets, 0),
        "instance_mapping": mappings,
    }
    return batch_pre


def det3d_batch_to_nndet(
    batch,
    fg_labels,
    *,
    seg_key="lm",
    use_disk_box_plug=True,
):
    # AI
    """det3d collate → nnDetection train_step targets (disk boxes + semantic seg)."""
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

    target_classes = _nndet_target_classes(target_boxes, target_classes_raw, fg_labels)
    lm = batch[seg_key]
    target_seg = lm.squeeze(1).long()
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
    from det3d.detection.retinaunet_network import _plan_arch

    arch = _plan_arch(plan_train)
    n_fg = len(plan_train["fg_labels"])
    arch["classifier_classes"] = n_fg
    arch["seg_classes"] = 1
    arch["score_thresh"] = float(plan_train["score_thresh"])
    arch["nms_thresh"] = float(plan_train["nms_thresh"])
    arch["detections_per_img"] = int(plan_train["detections_per_img"])
    arch["topk_candidates"] = int(plan_train["topk_candidates_per_level"])
    arch["remove_small_boxes"] = float(plan_train.get("remove_small_boxes", 0.01))
    return arch


def apply_det3d_overrides_to_nndet_plan(plan, plan_train):
    plan = deepcopy(plan)
    plan["patch_size"] = [int(v) for v in plan_train["patch_size"]]
    arch = plan["architecture"]
    arch["classifier_classes"] = len(plan_train["fg_labels"])
    arch["seg_classes"] = 1
    return plan


def plan_from_det3d(plan_train, plan_path=None):
    if plan_path is not None:
        ensure_nndet_importable()
        from nndet.io.load import load_pickle

        plan = load_pickle(plan_path)
    else:
        plan = {
            "architecture": plan_architecture_from_det3d(plan_train),
            "anchors": plan_anchors_from_det3d(plan_train),
            "patch_size": [int(v) for v in plan_train["patch_size"]],
        }
    return apply_det3d_overrides_to_nndet_plan(plan, plan_train)


def apply_det3d_plan_to_nndet_model_cfg(model_cfg, plan_train):
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


def build_nndet_retinaunet_module(configs, num_train_batches):
    ensure_nndet_importable()
    from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001

    from det3d.configs.parser import resolve_nndet_plan_path

    plan_train = configs["plan_train"]
    plan_path = resolve_nndet_plan_path(
        configs["mnemonic"], Path(configs["configurations_dir"])
    )
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    model_cfg, trainer_cfg = load_nndet_train_cfgs()
    model_cfg = apply_det3d_plan_to_nndet_model_cfg(model_cfg, plan_train)
    trainer_cfg["num_train_batches_per_epoch"] = int(num_train_batches)
    plan = plan_from_det3d(plan_train, plan_path=str(plan_path))
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
    # AI
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
    # AI
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
def nndet_predict_case(
    net, nndet_plan, plan_train, image, device, overlap=0.25, do_seg=True, num_tta=0
):
    # AI
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


def nndet_pred_to_vis(
    pred, box_key="bbox", label_key="label", score_key="label_scores"
):
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
                k: sanitize_tensor_for_numpy(v) if isinstance(v, torch.Tensor) else v
                for k, v in vis.items()
            }
        )
    return vis_preds


def native_nndet_batch_to_wandb_grid_batch(batch_pt, *, keys=None):
    # AI
    """Native post-pre_trafo batch -> det3d wandb grid batch (image/bbox/label/lm)."""
    grid_batch = {
        "image": batch_pt["data"],
        "lm": batch_pt["target"],
        "bbox": [nndet_batch_to_xyzxyz(boxes) for boxes in batch_pt["boxes"]],
        "label": [cls.long() + 1 for cls in batch_pt["classes"]],
    }
    if keys is not None:
        grid_batch["keys"] = keys
    return grid_batch


def patch_module_for_native_wandb_grid(module):
    # AI
    """Wrap native RetinaUNet validation_step to stash preds for wandb grid."""
    import types

    def validation_step(self, batch, batch_idx):
        keys = batch["keys"]
        with torch.no_grad():
            batch_pt = self.pre_trafo(**batch)
            targets = {
                "target_boxes": batch_pt["boxes"],
                "target_classes": batch_pt["classes"],
                "target_seg": batch_pt["target"][:, 0],
            }
            losses, prediction = self.model.train_step(
                images=batch_pt["data"],
                targets=targets,
                evaluation=True,
                batch_num=batch_idx,
            )
            loss = sum(losses.values())
        prediction = sanitize_for_numpy(prediction)
        targets = sanitize_for_numpy(targets)
        self.evaluation_step(prediction=prediction, targets=targets)
        grid_batch = native_nndet_batch_to_wandb_grid_batch(batch_pt, keys=keys)
        maybe_store_batch_grid_preds(self, grid_batch, prediction)
        return {
            "loss": loss.detach().item(),
            **{key: l.detach().item() for key, l in losses.items()},
        }

    module.validation_step = types.MethodType(validation_step, module)
    module._nndet_wandb_grid_val_batches = []
    return module


def maybe_store_batch_grid_preds(pl_module, batch, preds):
    """Stash validation preds on batch for wandb det grid (no extra forward)."""
    if isinstance(preds, list):
        batch["pred"] = [
            {
                k: sanitize_tensor_for_numpy(v) if isinstance(v, torch.Tensor) else v
                for k, v in p.items()
            }
            for p in preds
        ]
    else:
        batch["pred"] = nndet_batch_pred_to_vis_list(preds)
    pl_module._nndet_wandb_grid_val_batches.append(batch)

