"""Scratch: det3d DataManager dataloaders + full nnDetection RetinaUNet pathway.

Run # %% cells in order. Stage map + det3d vs nnDetection comparison:
  det3d/extra/trainer_nndet.md

Data stages (0–3): det3d only. Model stages (4+): import ~/code/nnDetection.

Stages 4+ prereqs (dl env):
  pip install omegaconf hydra-core pytorch-lightning loguru
  Shims auto-patch torch._six and pytorch_lightning.core.memory (PL 2.x).

Optional: set PLAN_PATH to an nnDetection plan.pkl to skip synthesized plan.
If GPU tight during scratch, use batch_size=1 from stage 1.
"""

from __future__ import annotations

import sys
import types
from copy import deepcopy
from pathlib import Path

import torch
import yaml

NNDET_ROOT = Path("/home/ub/code/nnDetection")
NNDET_TRAIN_CFG = NNDET_ROOT / "nndet/conf/train/v001.yaml"
PLAN_PATH = None  # e.g. "/path/to/D3V001_3d.pkl"
SCRATCH_BATCH_SIZE = 1  # nndet anchor matching is VRAM-heavy; raise after stage 6 OK
SCRATCH_PATCH_SIZE = None  # DM/shard patch — needs matching preprocessed src_* folder; leave None
NNDET_FORWARD_PATCH_SIZE = [128, 128, 64]  # center-crop loaded patch before nnDet only; lowers VRAM
SCRATCH_VAL_PATCH_SIZE = None  # e.g. [256, 256, 104] for stage 8; None = plan default
SCRATCH_INSTANCES_JSON = None  # optional bboxes/*.json with "instances" for stage 3 pre_trafo


def apply_batch_transforms(manager, batch):
    tfm = manager.transforms_batch
    if tfm is None:
        return batch
    return tfm(batch)


def clear_cuda_scratch():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_scratch_memory_overrides(conf, batch_size=None, patch_size=None, val_patch_size=None):
    from fran.configs.helpers import make_src_dims_from_patch_size

    if batch_size is not None:
        conf["dataset_params"]["batch_size"] = int(batch_size)
    if patch_size is not None:
        patch_size = [int(v) for v in patch_size]
        for key in ("plan_train", "plan_valid", "plan_test"):
            plan = conf[key]
            plan["patch_size"] = patch_size
            plan["patch_dim0"] = patch_size[0]
            plan["patch_dim1"] = patch_size[1]
            plan["src_dims"] = make_src_dims_from_patch_size(patch_size)
    if val_patch_size is not None:
        val_patch_size = [int(v) for v in val_patch_size]
        conf["model_params"]["val_patch_size"] = val_patch_size
        conf["plan_train"]["val_patch_size"] = val_patch_size
    return conf


def _nndet_import_shim():
    if "torch._six" not in sys.modules:
        torch_six = types.ModuleType("torch._six")
        torch_six.string_classes = (str,)
        sys.modules["torch._six"] = torch_six


def _lightning_import_shim():
    """nnDetection targets PL 1.x; dl has Lightning 2.x."""
    mem_key = "pytorch_lightning.core.memory"
    existing = sys.modules.get(mem_key)
    if existing is not None and hasattr(existing, "ModelSummary"):
        return
    try:
        from pytorch_lightning.core.memory import ModelSummary  # noqa: F401
        return
    except ImportError:
        pass
    from pytorch_lightning.utilities.model_summary import ModelSummary

    import pytorch_lightning as pl

    core = pl.core
    memory = types.ModuleType(mem_key)
    memory.__spec__ = None
    memory.ModelSummary = ModelSummary
    sys.modules[mem_key] = memory
    core.memory = memory


def _ensure_nndet_importable():
    _nndet_import_shim()
    _lightning_import_shim()
    if str(NNDET_ROOT) not in sys.path:
        sys.path.insert(0, str(NNDET_ROOT))
    try:
        import omegaconf  # noqa: F401
        import loguru  # noqa: F401
        import pytorch_lightning  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "nnDetection stages need omegaconf + loguru + pytorch-lightning. "
            "In dl: pip install omegaconf hydra-core pytorch-lightning loguru"
        ) from exc


def normalize_plan_modes_for_det_pipeline(configs):
    for key in ("plan_train", "plan_valid", "plan_test"):
        plan = configs[key]
        if plan["mode"] in {"det", "lbd"}:
            plan["mode"] = "lbd"


def setup_det_dataloaders(
    project_title,
    configs,
    batch_size=None,
    batch_tfms=True,
    debug=False,
    train_indices=None,
    val_indices=None,
    val_sampling=1.0,
):
    from det3d.managers.data import DataManagerDualDet, DataManagerDualDetBTfms

    normalize_plan_modes_for_det_pipeline(configs)
    if batch_size is not None:
        configs["dataset_params"]["batch_size"] = int(batch_size)
    dm_class = DataManagerDualDetBTfms if batch_tfms else DataManagerDualDet
    dm = dm_class(
        project_title=project_title,
        configs=configs,
        batch_size=int(configs["dataset_params"]["batch_size"]),
        cache_rate=configs["dataset_params"].get("cache_rate", 0.0),
        device=configs["dataset_params"].get("device", "cuda"),
        ds_type=configs["dataset_params"].get("ds_type"),
        train_indices=train_indices,
        val_indices=val_indices,
        val_sampling=val_sampling,
        debug=debug,
        batch_tfms=batch_tfms,
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    return dm


def inspect_det_batch(batch):
    out = {"keys": list(batch.keys())}
    out["image"] = tuple(batch["image"].shape)
    out["n_items"] = len(batch["bbox"])
    out["bbox_counts"] = [int(b.shape[0]) for b in batch["bbox"]]
    out["label_counts"] = [int(l.shape[0]) for l in batch["label"]]
    if "lm" in batch:
        lm = batch["lm"]
        if isinstance(lm, list):
            out["lm"] = [tuple(torch.as_tensor(x).shape) for x in lm]
        else:
            out["lm"] = tuple(lm.shape)
    if "mask" in batch:
        out["mask"] = [tuple(m.shape) for m in batch["mask"]]
    return out


def _lm_to_instance_volume(lm_item):
    t = torch.as_tensor(lm_item).long()
    while t.dim() > 3 and int(t.shape[0]) == 1:
        t = t.squeeze(0)
    if t.dim() == 4:
        t = t[0]
    return t


def print_det3d_batch_for_pre_trafo(batch):
    lines = ["=== det3d DM batch (feeds pre_trafo input builder) ===", f"keys={list(batch.keys())}"]
    lines.append(f"image shape={tuple(batch['image'].shape)} dtype={batch['image'].dtype}")
    n = int(batch["image"].shape[0])
    for i in range(n):
        box = batch["bbox"][i]
        lbl = batch["label"][i]
        lines.append(f"[{i}] sidecar bbox shape={tuple(box.shape)}")
        if box.numel():
            lines.append(f"    bbox xyzxyz=\n{box.detach().cpu().numpy()}")
        lines.append(f"    label={lbl.detach().cpu().tolist()}")
        if "lm" in batch:
            lm = batch["lm"][i] if isinstance(batch["lm"], list) else batch["lm"][i]
            vol = _lm_to_instance_volume(lm)
            inst = vol.unique(sorted=True)
            inst = inst[inst > 0].tolist()
            lines.append(f"    lm shape={tuple(vol.shape)} instance_ids={inst}")
    print("\n".join(lines))


def print_pre_trafo_state(label, batch):
    lines = [f"=== {label} ==="]
    if "data" in batch:
        lines.append(
            f"data shape={tuple(batch['data'].shape)} dtype={batch['data'].dtype}"
        )
    if "target" in batch:
        t = batch["target"]
        t0 = t[0, 0] if t.dim() >= 4 else t
        inst = t0.long().unique(sorted=True)
        inst = inst[inst > 0].tolist()
        lines.append(f"target shape={tuple(t.shape)} instance_ids={inst}")
        lines.append(f"target unique values={t0.unique(sorted=True).tolist()}")
    if "instance_mapping" in batch:
        lines.append(f"instance_mapping={batch['instance_mapping']}")
    if "present_instances" in batch:
        lines.append(
            "present_instances="
            f"{[[int(v) for v in x] for x in batch['present_instances']]}"
        )
    if "boxes" in batch:
        for i, boxes in enumerate(batch["boxes"]):
            cls = batch["classes"][i] if "classes" in batch else None
            bnp = boxes.detach().cpu().numpy()
            lines.append(f"boxes[{i}] shape={tuple(boxes.shape)}")
            lines.append(f"    {bnp}")
            lines.append(f"classes[{i}]={cls.detach().cpu().tolist() if cls is not None else None}")
    print("\n".join(lines))


def _load_instances_json(path):
    import json

    payload = json.loads(Path(path).read_text())
    if "instances" not in payload:
        raise KeyError(f"missing 'instances' in {path}")
    return payload["instances"]


def _instance_mapping_for_item(lm_item, labels, instances_json=None):
    if instances_json is not None:
        return _load_instances_json(instances_json)
    vol = _lm_to_instance_volume(lm_item)
    inst = vol.unique(sorted=True)
    inst = inst[inst > 0].tolist()
    lbl = torch.as_tensor(labels).reshape(-1).long()
    mapping = {}
    for j, iid in enumerate(sorted(int(i) for i in inst)):
        if j < lbl.numel():
            mapping[str(iid)] = int(lbl[j].item())
        else:
            mapping[str(iid)] = 0
    if not mapping:
        print("pre_trafo: no instance ids in lm; empty mapping")
    elif instances_json is None:
        print(
            "pre_trafo: no SCRATCH_INSTANCES_JSON — mapping from lm ids + sidecar "
            f"label order: {mapping}"
        )
    return mapping


def det3d_batch_to_pre_trafo_input(
    batch,
    forward_patch_size=NNDET_FORWARD_PATCH_SIZE,
    instances_json=SCRATCH_INSTANCES_JSON,
):
    """det3d collate batch -> nnDet pre_trafo input: data, target, instance_mapping."""
    data = batch["image"].float()
    crop_starts = None
    if forward_patch_size is not None:
        forward_patch_size = tuple(int(v) for v in forward_patch_size)
        spatial = tuple(int(v) for v in data.shape[-3:])
        if any(s > p for s, p in zip(spatial, forward_patch_size)):
            data, crop_starts = _center_crop_spatial(data, forward_patch_size)

    if "lm" not in batch:
        raise KeyError(
            "batch['lm'] required for pre_trafo inspect — set arch=retinaunet / uses_lm_seg"
        )

    lm_list = batch["lm"]
    n = int(data.shape[0])
    targets = []
    mappings = []
    for i in range(n):
        lm_item = lm_list[i] if isinstance(lm_list, list) else lm_list[i]
        vol = _lm_to_instance_volume(lm_item)
        if crop_starts is not None and forward_patch_size is not None:
            vol, _ = _center_crop_spatial(vol, forward_patch_size, crop_starts)
        targets.append(vol.float().unsqueeze(0).unsqueeze(0))
        mappings.append(
            _instance_mapping_for_item(lm_item, batch["label"][i], instances_json)
        )

    return {
        "data": data,
        "target": torch.cat(targets, 0),
        "instance_mapping": mappings,
    }, crop_starts


def run_pre_trafo_stepped(batch_pre, sidecar_boxes=None):
    """Apply FindInstances -> Instances2Boxes -> Instances2Segmentation with prints."""
    _ensure_nndet_importable()
    from nndet.io.transforms.instances import (
        FindInstances,
        Instances2Boxes,
        Instances2Segmentation,
    )

    find_instances = FindInstances(instance_key="target", save_key="present_instances")
    instances2boxes = Instances2Boxes(
        instance_key="target",
        map_key="instance_mapping",
        box_key="boxes",
        class_key="classes",
        present_instances="present_instances",
    )
    instances2seg = Instances2Segmentation(
        instance_key="target",
        map_key="instance_mapping",
        present_instances="present_instances",
    )

    print_pre_trafo_state("before pre_trafo (nnDet input)", batch_pre)

    batch_find = find_instances(**batch_pre)
    print_pre_trafo_state("after FindInstances", batch_find)

    batch_boxes = instances2boxes(**batch_find)
    print_pre_trafo_state("after Instances2Boxes", batch_boxes)

    batch_post = instances2seg(**batch_boxes)
    print_pre_trafo_state("after Instances2Segmentation (pre_trafo done)", batch_post)

    if sidecar_boxes is not None:
        for i, (sb, tb) in enumerate(zip(sidecar_boxes, batch_post["boxes"])):
            sbnp = sb.detach().cpu().numpy()
            tbnp = tb.detach().cpu().numpy()
            print(f"[{i}] sidecar vs Instances2Boxes match atol=2: {sbnp.shape} vs {tbnp.shape}")
            if sbnp.size and tbnp.size:
                print(f"    sidecar=\n{sbnp}\n    pre_trafo=\n{tbnp}")

    return batch_post


def pre_trafo_train_step_targets(batch_post):
    return {
        "target_boxes": batch_post["boxes"],
        "target_classes": batch_post["classes"],
        "target_seg": batch_post["target"][:, 0],
    }


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


def det3d_batch_to_nndet(batch, forward_patch_size=NNDET_FORWARD_PATCH_SIZE):
    """det3d collate → nnDetection train_step targets (skip pre_trafo)."""
    data = batch["image"]
    crop_starts = None
    if forward_patch_size is not None:
        forward_patch_size = tuple(int(v) for v in forward_patch_size)
        data, crop_starts = _center_crop_spatial(data, forward_patch_size)
    target_boxes = []
    for b in batch["bbox"]:
        box = b
        if crop_starts is not None:
            box = _crop_boxes_to_patch(box, crop_starts, forward_patch_size)
        target_boxes.append(box)
    target_classes = list(batch["label"])
    if crop_starts is not None:
        target_classes = [
            cls[: boxes.shape[0]]
            for cls, boxes in zip(target_classes, target_boxes)
        ]
    if "lm" in batch:
        lm_list = batch["lm"]
        segs = []
        for i in range(int(data.shape[0])):
            lm_item = lm_list[i] if isinstance(lm_list, list) else lm_list[i]
            vol = _lm_to_instance_volume(lm_item)
            if crop_starts is not None:
                vol, _ = _center_crop_spatial(vol, forward_patch_size, crop_starts)
            segs.append(vol)
        target_seg = torch.stack(segs, 0)
        target_seg = (target_seg > 0).long()
    elif "mask" in batch:
        target_seg = torch.stack(batch["mask"], 0)
        if target_seg.dim() == 5:
            target_seg = target_seg[:, 0]
        if crop_starts is not None:
            target_seg, _ = _center_crop_spatial(target_seg, forward_patch_size, crop_starts)
    else:
        target_seg = torch.zeros(data.shape[0], *data.shape[2:], device=data.device)
    return {
        "data": data,
        "target_boxes": target_boxes,
        "target_classes": target_classes,
        "target_seg": target_seg,
    }


def load_nndet_train_cfgs(cfg_path=NNDET_TRAIN_CFG):
    with open(cfg_path) as f:
        train_cfg = yaml.safe_load(f)
    return deepcopy(train_cfg["model_cfg"]), deepcopy(train_cfg["trainer_cfg"])


def plan_anchors_from_det3d(plan_train):
    shapes = plan_train.get("base_anchor_shapes")
    if shapes is None:
        shapes = [[6, 8, 4], [8, 6, 5], [10, 10, 6]]
    n_levels = len(plan_train.get("decoder_levels", (1, 2, 3, 4)))
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
    arch["detections_per_img"] = int(plan_train.get("detections_per_img", 100))
    arch["topk_candidates"] = int(plan_train.get("topk_candidates_per_level", 1000))
    arch["remove_small_boxes"] = float(plan_train.get("remove_small_boxes", 0.01))
    return arch


def plan_from_det3d(plan_train, plan_path=None):
    if plan_path is not None:
        _ensure_nndet_importable()
        from nndet.io.load import load_pickle

        return load_pickle(plan_path)
    return {
        "architecture": plan_architecture_from_det3d(plan_train),
        "anchors": plan_anchors_from_det3d(plan_train),
        "patch_size": [int(v) for v in plan_train["patch_size"]],
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


def build_nndet_retinaunet_module(plan, model_cfg, trainer_cfg, num_train_batches):
    _ensure_nndet_importable()
    from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001

    trainer_cfg = deepcopy(trainer_cfg)
    trainer_cfg["num_train_batches_per_epoch"] = int(num_train_batches)
    return RetinaUNetV001(
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        plan=plan,
    )


if __name__ == "__main__":
#SECTION:-------------------- stage 0 — setup --------------------------------------------------------------------------------------
# Same as det3d/extra/trainer.py and TrainerDet.fit preamble.
# Sources: det3d/configs/parser.py (ConfigMakerDet), fran/managers/project.py (Project)
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers import Project
    from utilz.helpers import pp

    project_title = "lidc"
    plan_id = 1
    conf_fold = 0

    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = C.configs
    conf["dataset_params"]["fold"] = conf_fold
    apply_scratch_memory_overrides(
        conf,
        batch_size=SCRATCH_BATCH_SIZE,
        patch_size=SCRATCH_PATCH_SIZE,
        val_patch_size=SCRATCH_VAL_PATCH_SIZE,
    )
    pp(conf["plan_train"])

#SECTION:-------------------- stage 1 — dataloaders (det3d only) -------------------------------------------------------------------
# Mirrors TrainerDet.init_dm → DataManagerDualDetBTfms (not nnDetection Datamodule).
# Sources: det3d/trainers/trainerdet.py (init_dm, normalize_plan_modes),
#   det3d/managers/data/batch_tfms.py (DataManagerDetLBDBTfms),
#   det3d/detection/luna16_training_dm_hybrid.py (setup_det_dataloaders pattern)
# nnDetection equivalent (unused here): nndet/io/datamodule/bg_module.py
# %%
    batch_size = SCRATCH_BATCH_SIZE
    batch_tfms = True
    debug_ = False
    train_indices = None
    val_indices = None
    val_sampling = 1.0

    D = setup_det_dataloaders(
        project_title,
        conf,
        batch_size=batch_size,
        batch_tfms=batch_tfms,
        debug=debug_,
        train_indices=train_indices,
        val_indices=val_indices,
        val_sampling=val_sampling,
    )
    tmt = D.train_manager
    tmv = D.valid_manager
    tmt.setup()
    tmv.setup()
    train_dl = tmt.dl
    val_dl = tmv.dl
    print(f"train: {tmt}")
    print(f"valid: {tmv}")

#SECTION:-------------------- stage 2 — inspect batch -------------------------------------------------------------------------------
# Same as trainer.py TS block: collate → optional GpuTail batch transforms.
# Sources: det3d/managers/data/collate.py (lbd_det_collate → image/bbox/label),
#   det3d/transforms/gpu_det.py (BatchItemCompose / GpuTail)
# %%
    train_batch = next(iter(train_dl))
    train_batch = apply_batch_transforms(tmt, train_batch)
    print(inspect_det_batch(train_batch))

#SECTION:-------------------- stage 3 — pre_trafo stepped + adapter compare ---------------------------
# Step each nnDet pre_trafo transform on det3d batch; then compare skip-pre_trafo adapter.
# Set SCRATCH_INSTANCES_JSON to bboxes/*.json with "instances" when available.
# Sources: nndet/io/transforms/instances.py (FindInstances, Instances2Boxes, Instances2Segmentation)
# %%
    device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    print_det3d_batch_for_pre_trafo(train_batch)
    batch_pre, _crop_starts = det3d_batch_to_pre_trafo_input(
        train_batch,
        forward_patch_size=NNDET_FORWARD_PATCH_SIZE,
        instances_json=SCRATCH_INSTANCES_JSON,
    )
    batch_post = run_pre_trafo_stepped(batch_pre, sidecar_boxes=train_batch["bbox"])
    targets_pre_trafo = pre_trafo_train_step_targets(batch_post)

    print("\n=== train_step targets from pre_trafo ===")
    for i in range(len(targets_pre_trafo["target_boxes"])):
        seg = targets_pre_trafo["target_seg"][i]
        print(
            f"[{i}] boxes {tuple(targets_pre_trafo['target_boxes'][i].shape)} "
            f"classes {targets_pre_trafo['target_classes'][i].tolist()} "
            f"target_seg unique {seg.unique(sorted=True).tolist()}"
        )

    print("\n=== det3d adapter (skip pre_trafo — current trainer path) ===")
    nndet_batch = det3d_batch_to_nndet(train_batch)
    print("forward patch", NNDET_FORWARD_PATCH_SIZE, "data", tuple(nndet_batch["data"].shape))
    for i in range(len(nndet_batch["target_boxes"])):
        seg = nndet_batch["target_seg"][i]
        print(
            f"[{i}] boxes {tuple(nndet_batch['target_boxes'][i].shape)} "
            f"classes {nndet_batch['target_classes'][i].tolist()} "
            f"target_seg unique {seg.unique(sorted=True).tolist()}"
        )

#SECTION:-------------------- stage 4 — nndet plan + cfg --------------------------------------------------------------------------
# Plan: synthesize from excel plan_train via det3d/detection/retinaunet_network.py (_plan_arch),
#   or load native plan.pkl (nndet/planning/experiment/v001.py + BoxC002 output).
# Cfg: nnDetection/nndet/conf/train/v001.yaml (model_cfg, trainer_cfg).
# det3d path uses create_detector_from_conf in det3d/architectures/create_detector.py instead.
# %%
    plan_train = conf["plan_train"]
    plan = plan_from_det3d(plan_train, plan_path=PLAN_PATH)
    model_cfg, trainer_cfg = load_nndet_train_cfgs()
    model_cfg = apply_det3d_plan_to_nndet_model_cfg(model_cfg, plan_train)
    print("architecture keys", sorted(plan["architecture"].keys()))
    print("anchors", plan["anchors"])

#SECTION:-------------------- stage 5 — build nnDetection model ----------------------------------------------------------------------
# nnDetection RetinaUNetV001 → BaseRetinaNet via from_config_plan.
# Sources: nndet/ptmodule/retinaunet/v001.py, nndet/ptmodule/retinaunet/base.py (from_config_plan)
# trainer.py equivalent: RetinaNetManager + create_detector_from_conf (MONAI RetinaNetDetector2)
# %%
    clear_cuda_scratch()
    module = build_nndet_retinaunet_module(
        plan=plan,
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        num_train_batches=len(train_dl),
    )
    net = module.model.to(device).train()
    n_params = sum(p.numel() for p in net.parameters())
    print(type(net).__name__, "params", n_params)

#SECTION:-------------------- stage 6 — loss forward + backward -------------------------------------------------------------------
# Forward + backward in one cell: notebook re-runs of separate forward/backward cells OOM easily.
# nnDetection: BaseRetinaNet.train_step (nndet/core/retina.py) — cls, reg, seg losses.
# trainer.py equivalent: forward_train_batched (det3d/detection/retinanet_train.py)
# %%
    clear_cuda_scratch()
    net.train()
    with torch.autocast("cuda", enabled=device.type == "cuda"):
        losses, _ = net.train_step(
            images=nndet_batch["data"],
            targets={
                "target_boxes": nndet_batch["target_boxes"],
                "target_classes": nndet_batch["target_classes"],
                "target_seg": nndet_batch["target_seg"],
            },
            evaluation=False,
            batch_num=0,
        )
    print({k: float(v) for k, v in losses.items()})
    opt_cfgs = module.configure_optimizers()
    optimizer = opt_cfgs[0][0]
    scheduler = opt_cfgs[1]["scheduler"]
    optimizer.zero_grad(set_to_none=True)
    loss = sum(losses.values())
    loss.backward()
    optimizer.step()
    scheduler.step()
    print("loss", float(loss.detach()), "lr", optimizer.param_groups[0]["lr"])
    clear_cuda_scratch()

#SECTION:-------------------- stage 7 — (merged into stage 6) -----------------------------------------------------------------------
# Kept for notebook cell index stability. Re-run stage 6 instead.
# %%
    pass

#SECTION:-------------------- stage 8 — val forward --------------------------------------------------------------------------------
# nnDetection: validation_step body — train_step(evaluation=True) + postprocess.
# Val DL loads full volumes (not LBD patches) — too big / odd shapes for raw train_step.
# Smoke test: reuse one train patch batch. Full val = sliding window (trainer.py N.val_forward).
# %%
    clear_cuda_scratch()
    net.eval()
    val_batch = apply_batch_transforms(tmt, next(iter(train_dl)))
    val_nndet = det3d_batch_to_nndet(val_batch)
    with torch.no_grad():
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            val_losses, _ = net.train_step(
                images=val_nndet["data"],
                targets={
                    "target_boxes": val_nndet["target_boxes"],
                    "target_classes": val_nndet["target_classes"],
                    "target_seg": val_nndet["target_seg"],
                },
                evaluation=False,
                batch_num=0,
            )
    print("val losses", {k: float(v) for k, v in val_losses.items()})
    # evaluation=True postprocess needs nnDetection GPU NMS build (nms_fn is None otherwise)
    # predictions = net.postprocess_for_inference(...)  # wire after nndet install with CUDA ops

#SECTION:-------------------- stage 9 — manual epoch loop (optional) --------------------------------------------------------------
# Conceptual equivalent of TrainerDet.fit() / trainer.py Tm.fit().
# nnDetection native: scripts/train.py → pl.Trainer.fit(module, datamodule).
# %%
    for epoch in range(2):
        net.train()
        for batch_idx, batch in enumerate(train_dl):
            batch = tmt.transforms_batch(batch)
            nb = det3d_batch_to_nndet(batch)
            optimizer.zero_grad()
            step_losses, _ = net.train_step(
                images=nb["data"],
                targets={
                    "target_boxes": nb["target_boxes"],
                    "target_classes": nb["target_classes"],
                    "target_seg": nb["target_seg"],
                },
                evaluation=False,
                batch_num=batch_idx,
            )
            total = sum(step_losses.values())
            total.backward()
            optimizer.step()
            scheduler.step()
        print(f"epoch {epoch} last loss {float(total):.4f}")
# %%
