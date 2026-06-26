"""Disk-box approval gate — post-augmentation parity vs nnDet Instances2Boxes.

Production train path (``DataManagerDetLBDBTfms``):

1. **Disk:** sidecar ``bbox`` through ``BoxToWorld → ToPoints → GpuTail`` (affine,
   point warp, ``ToBoxes``, resize/pad, ``BoxClip``).
2. **Reference:** same crop + aug on ``image``/``lm``; derive xyxyzz boxes via
   ``FindInstances → Instances2Boxes`` (nnDet pre_trafo box step only).

If transformed disk boxes match reference → **approve load-from-disk** at train time
(skip ``Instances2Boxes``). Disk boxes use ``disk_bbox_to_nndet_xyxyzz`` at the
nnDet boundary (lower xyz mins −1 to match ``instances_to_boxes``).

Run one ``# %%`` cell at a time. See ``NATIVE_NNDET_HANDOFF.md`` § disk-box gate.
"""
from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from det3d.detection.nndet_train import (
    _instance_mapping_for_item,
    _lm_seg_volume,
    det3d_semantic_target_seg_from_batch,
    disk_bbox_to_nndet_xyxyzz,
    ensure_nndet_importable,
    xyzxyz_exclusive_batch_to_nndet,
)
from det3d.extra.nndet_parity_cp0_4 import (
    BOXES_ATOL,
    CASE_ID,
    DET3D_PLAN_ID,
    DET3D_PROJECT,
    LBD_FOLDER,
    PARITY_SEED,
)
from det3d.managers.data.batch_tfms import DataManagerDetLBDBTfms
from det3d.managers.data.collate import lbd_det_collate_train
from det3d.utils.bbox_sidecar import bbox_sidecar_path, load_detection_sidecar

BOX_COUNT_MUST_MATCH = True


def _fg_labels_from_manifest() -> list[int]:
    from utilz.fileio import load_json

    manifest = load_json(LBD_FOLDER / "manifest.json")
    labels_all = manifest["labels_all"]
    return [int(v) for v in labels_all if int(v) != 0] or [0]


def setup_parity_dm(
    *,
    batch_size: int = 1,
    debug: bool = True,
    device: str | None = None,
):
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers.project import Project

    P = Project(DET3D_PROJECT)
    C = ConfigMakerDet(P)
    C.setup(DET3D_PLAN_ID)
    conf = deepcopy(C.configs)
    conf["dataset_params"]["fold"] = 0
    if device is not None:
        conf["dataset_params"]["device"] = device
    dm = DataManagerDetLBDBTfms(
        P,
        conf,
        batch_size=batch_size,
        split="train",
        debug=debug,
    )
    dm.prepare_data()
    dm.setup()
    return dm


def find_case_idx(dm, case_id: str) -> int:
    return next(i for i, row in enumerate(dm.data) if row["case_id"] == case_id)


def _apply_item_key(dici, key: str, tfm):
    out = tfm(dici)
    if key == "Rtr":
        return out[0]
    return out


def run_disk_box_pipeline(dm, case_idx: int, seed: int) -> dict:
    """Full item keys + GpuTail batch tfms; returns collated batch after aug."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dici = dict(dm.data[case_idx])
    tfms = dm.transforms_dict
    item_keys = [k.strip() for k in dm.keys.split(",") if k.strip()]
    for key in item_keys:
        dici = _apply_item_key(dici, key, tfms[key])

    batch = lbd_det_collate_train([[dici]])
    if dm.transforms_batch is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        batch = dm.transforms_batch(batch)
    return batch


def load_sidecar_instances(case_id: str) -> dict | None:
    _boxes, _labels, instances = load_detection_sidecar(
        bbox_sidecar_path(LBD_FOLDER / "bboxes", case_id)
    )
    return instances


def reference_boxes_nndet(
    lm_tensor: torch.Tensor,
    label: torch.Tensor,
    instances: dict | None,
    fg_labels: list[int],
):
    ensure_nndet_importable()
    from nndet.io.transforms.instances import FindInstances, Instances2Boxes

    vol = _lm_seg_volume(lm_tensor)
    target = vol.float().unsqueeze(0).unsqueeze(0)
    mapping = _instance_mapping_for_item(
        vol, label, instances=instances, fg_labels=fg_labels
    )
    batch_pre = {
        "target": target,
        "instance_mapping": [mapping],
    }
    find = FindInstances(instance_key="target", save_key="present_instances")
    batch_find = find(**batch_pre)
    i2b = Instances2Boxes(
        instance_key="target",
        map_key="instance_mapping",
        box_key="boxes",
        class_key="classes",
        present_instances="present_instances",
    )
    batch_boxes = i2b(**batch_find)
    return (
        batch_boxes["boxes"][0],
        batch_boxes["classes"][0],
        mapping,
        batch_find["present_instances"][0],
    )


def _map_disk_labels(label: torch.Tensor, n_boxes: int, fg_labels: list[int]) -> torch.Tensor:
    label_to_idx = {int(v): i for i, v in enumerate(fg_labels)}
    lab = torch.as_tensor(label, dtype=torch.long).reshape(-1)
    return torch.tensor(
        [label_to_idx[int(v.item())] for v in lab[:n_boxes]],
        dtype=torch.long,
    )


def compare_disk_vs_reference(
    disk_bbox: torch.Tensor,
    disk_label: torch.Tensor,
    ref_boxes: torch.Tensor,
    ref_classes: torch.Tensor,
    fg_labels: list[int],
    *,
    boxes_atol: float = BOXES_ATOL,
    count_must_match: bool = BOX_COUNT_MUST_MATCH,
) -> bool:
    disk_xyxyzz = disk_bbox_to_nndet_xyxyzz(disk_bbox)
    ref_xyxyzz = ref_boxes.detach().cpu().float()
    disk_xyxyzz = disk_xyxyzz.detach().cpu().float()

    n_disk = int(disk_xyxyzz.shape[0])
    n_ref = int(ref_xyxyzz.shape[0])
    print(f"n_boxes disk={n_disk} ref(lm Instances2Boxes)={n_ref}")

    if n_disk != n_ref:
        msg = "box count mismatch after full aug chain"
        print(f"FAIL {msg}")
        if count_must_match:
            return False
        n_cmp = min(n_disk, n_ref)
    else:
        n_cmp = n_disk

    if n_cmp == 0:
        print("both empty — approve")
        return True

    disk_cls = _map_disk_labels(disk_label, n_cmp, fg_labels)
    ref_cls = ref_classes.detach().cpu().long()[:n_cmp]
    approved = True
    for i in range(n_cmp):
        diff = float(torch.max(torch.abs(disk_xyxyzz[i] - ref_xyxyzz[i])).item())
        cls_match = int(disk_cls[i].item()) == int(ref_cls[i].item())
        print(
            f"  box {i} max_diff={diff:.4f} cls disk/ref="
            f"{int(disk_cls[i].item())}/{int(ref_cls[i].item())} match={cls_match}"
        )
        if diff > boxes_atol:
            print(f"    FAIL coord atol={boxes_atol}")
            approved = False
        if not cls_match:
            print("    FAIL class")
            approved = False

    if n_disk != n_ref and count_must_match:
        approved = False

    return approved


def gate_disk_boxes_post_aug(
    dm,
    case_id: str,
    seed: int,
    fg_labels: list[int],
    *,
    boxes_atol: float = BOXES_ATOL,
) -> bool:
    case_idx = find_case_idx(dm, case_id)
    instances = load_sidecar_instances(case_id)
    if instances is None:
        print("note: sidecar has no instances — mapping from lm ids + labels")
    batch = run_disk_box_pipeline(dm, case_idx, seed)
    disk_bbox = batch["bbox"][0]
    disk_label = batch["label"][0]
    lm = batch["lm"][0]

    ref_boxes, ref_classes, mapping, present = reference_boxes_nndet(
        lm, disk_label, instances, fg_labels
    )
    print("instance_mapping", mapping)
    print("present_instances", [int(v) for v in present.tolist()])
    print("disk bbox xyzxyz\n", disk_bbox.detach().cpu().numpy())
    print("ref bbox xyxyzz\n", ref_boxes.detach().cpu().numpy())

    approved = compare_disk_vs_reference(
        disk_bbox,
        disk_label,
        ref_boxes,
        ref_classes,
        fg_labels,
        boxes_atol=boxes_atol,
    )
    print("DISK_BOX_GATE", "APPROVED" if approved else "REJECTED")
    return approved


# %%
#SECTION:--- config ---
    print("CASE_ID", CASE_ID)
    print("PARITY_SEED", PARITY_SEED)
    print("BOXES_ATOL", BOXES_ATOL)
    FG_LABELS = _fg_labels_from_manifest()
    print("FG_LABELS", FG_LABELS)
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("DEVICE", DEVICE)

# %%
#SECTION:--- setup DataManagerDetLBDBTfms (production box + GpuTail path) ---
    dm = setup_parity_dm(batch_size=1, debug=True, device=DEVICE)
    case_idx = find_case_idx(dm, CASE_ID)
    print(dm)
    print("item keys", dm.keys)
    print("batch keys", dm.keys_tr_batch)
    print("case", dm.data[case_idx]["case_id"])

# %%
#SECTION:--- stepped item transforms (inspect) ---
    case_idx = find_case_idx(dm, CASE_ID)
    torch.manual_seed(PARITY_SEED)
    np.random.seed(PARITY_SEED)
    random.seed(PARITY_SEED)
    dici = dict(dm.data[case_idx])
    tfms = dm.transforms_dict
    for key in [k.strip() for k in dm.keys.split(",") if k.strip()]:
        dici = _apply_item_key(dici, key, tfms[key])
        if key in ("L2", "ToPoints", "Norm"):
            print(key, "image", tuple(dici["image"].shape), "bbox", tuple(dici["bbox"].shape))

# %%
#SECTION:--- Gate — disk boxes after full aug vs lm Instances2Boxes ---
    APPROVED = gate_disk_boxes_post_aug(
        dm, CASE_ID, PARITY_SEED, FG_LABELS, boxes_atol=BOXES_ATOL
    )
    if not APPROVED:
        print(
            "REJECTED: do not use skip-Instances2Boxes fast path for this case/config; "
            "fix sidecar↔lm or use RetinaUNetManagerV2 / materialized pre_trafo targets"
        )
    else:
        print(
            "APPROVED: disk bbox safe through DM aug chain; train may use transformed "
            "sidecar boxes without Instances2Boxes (still need semantic target_seg)"
        )

# %%
#SECTION:--- optional — dataloader one batch sanity ---
    torch.manual_seed(PARITY_SEED)
    np.random.seed(PARITY_SEED)
    dl_batch = next(iter(dm.dl))
    if dm.transforms_batch is not None:
        dl_batch = dm.transforms_batch(dl_batch)
    print("dl batch image", tuple(dl_batch["image"].shape))
    print("dl n_boxes", int(dl_batch["bbox"][0].shape[0]))
