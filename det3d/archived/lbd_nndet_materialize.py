"""LBD folder -> nnDetection imagesTr materialization (det3d-owned stage 1)."""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

DUSTING_THRESHOLD = 3.0


def _spacing_from_lm_meta(lm) -> np.ndarray:
    meta = lm.meta
    affine = meta["affine"]
    if torch.is_tensor(affine):
        affine = affine.detach().cpu().numpy()
    spacing = np.array(
        [float(abs(affine[0, 0])), float(abs(affine[1, 1])), float(abs(affine[2, 2]))],
        dtype=np.float64,
    )
    return spacing


def lm_pt_to_instance_seg(lm, dusting_threshold=DUSTING_THRESHOLD):
    #AI
    """lm MetaTensor/tensor -> instance seg (label_cc ids) + instance→class mapping (all 0)."""
    import SimpleITK as sitk

    from det3d.geometry.lmg import DetectionLabelMapGeometryPT

    while lm.dim() > 3:
        lm = lm.squeeze(0)
    L = DetectionLabelMapGeometryPT(lm, ignore_labels=[], compute_feret=False)
    L.dust(float(dusting_threshold))
    seg = sitk.GetArrayFromImage(L.li_cc_sitk).astype(np.int32)
    lm_np = torch.as_tensor(lm).long().numpy()
    if seg.shape != lm_np.shape:
        seg = np.transpose(seg, (2, 1, 0))
    mapping = {}
    for _, row in L.nbrhoods.iterrows():
        mapping[str(int(row["label_cc"]))] = 0
    return seg, mapping, L


def lbd_lm_to_sidecar_tensors(lm, dusting_threshold=DUSTING_THRESHOLD):
    #AI
    """Regenerate det3d xyzxyz sidecar boxes/labels from lm (DetectionLabelMapGeometryPT)."""
    from det3d.geometry.lmg import DetectionLabelMapGeometryPT

    while lm.dim() > 3:
        lm = lm.squeeze(0)
    L = DetectionLabelMapGeometryPT(lm, ignore_labels=[], compute_feret=False)
    L.dust(float(dusting_threshold))
    rec = L.to_voxel_detection_records("xyzxyz")
    boxes = rec["box"]
    labels = rec["label"]
    if len(boxes) == 0:
        box_t = torch.zeros((0, 6), dtype=torch.float32)
        label_t = torch.zeros((0,), dtype=torch.long)
    else:
        box_t = torch.stack(boxes)
        label_t = torch.tensor(labels, dtype=torch.long)
    return box_t, label_t, L


def instance_seg_to_nndet_boxes_pkl(seg: np.ndarray, instance_mapping: dict) -> dict:
    #AI
    """Instance seg -> nnDetection {case}_boxes.pkl (instances_to_boxes_np xyxyzz padding)."""
    from nndet.io.transforms.instances import instances_to_boxes_np

    boxes, instance_idx = instances_to_boxes_np(seg[np.newaxis, ...], dim=3)
    inst_list = [int(x) for x in instance_idx.tolist()]
    labels = [int(instance_mapping[str(i)]) for i in inst_list]
    payload = {
        "boxes": boxes,
        "instances": inst_list,
        "labels": labels,
    }
    return payload


def nndet_boxes_pkl_to_seg_mask(seg_shape, boxes_pkl: dict) -> np.ndarray:
    #AI
    """Reverse check: paint instance ids from pkl onto empty volume (not full lm recovery)."""
    seg = np.zeros(seg_shape, dtype=np.int32)
    boxes = np.asarray(boxes_pkl["boxes"], dtype=np.float64)
    instances = boxes_pkl["instances"]
    for inst_id, box in zip(instances, boxes):
        x1, y1, x2, y2, z1, z2 = [int(round(v)) for v in box.tolist()]
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        z1 = max(z1, 0)
        x2 = min(x2, seg_shape[0])
        y2 = min(y2, seg_shape[1])
        z2 = min(z2, seg_shape[2])
        seg[x1:x2, y1:y2, z1:z2] = int(inst_id)
    return seg


def lbd_case_to_nndet_arrays(lbd_folder: Path, case_id: str, dusting_threshold=DUSTING_THRESHOLD):
    #AI
    """One LBD case -> image [1,D,H,W], instance seg [D,H,W], properties, boxes.pkl."""
    from det3d.inference.hybrid_lbd import load_lbd_pt

    lbd_folder = Path(lbd_folder)
    img = load_lbd_pt(lbd_folder / "images" / f"{case_id}.pt")
    lm = load_lbd_pt(lbd_folder / "lms" / f"{case_id}.pt")
    while img.dim() > 3:
        img = img.squeeze(0)
    seg, mapping, _L = lm_pt_to_instance_seg(lm, dusting_threshold=dusting_threshold)
    data = img.detach().cpu().numpy().astype(np.float32)
    if data.ndim == 3:
        data = data[np.newaxis, ...]
    spacing = _spacing_from_lm_meta(lm)
    properties = OrderedDict(
        {
            "original_size_of_raw_data": np.array(data.shape[1:], dtype=np.int32),
            "original_spacing": spacing.copy(),
            "spacing_after_resampling": spacing.copy(),
            "size_after_cropping": np.array(data.shape[1:], dtype=np.int32),
            "size_after_resampling": np.array(data.shape[1:], dtype=np.int32),
            "instances": dict(mapping),
            "classes": np.unique(seg),
            "use_nonzero_mask_for_norm": False,
            "list_of_data_files": [str(lbd_folder / "images" / f"{case_id}.pt")],
            "seg_file": str(lbd_folder / "lms" / f"{case_id}.pt"),
        }
    )
    boxes_pkl = instance_seg_to_nndet_boxes_pkl(seg, mapping)
    return data, seg, properties, boxes_pkl


def verify_lbd_format_roundtrip(
    lbd_folder: Path,
    case_ids: Sequence[str],
    dusting_threshold=DUSTING_THRESHOLD,
    sidecar_atol: float = 1.0,
) -> dict:
    #AI
    """
    Confirm lm <-> json sidecar <-> nndet boxes.pkl interchangeability.

    - lm -> LMG xyzxyz matches existing bboxes/*.json (det3d sidecar)
    - lm -> instance seg -> boxes.pkl round-trips via instances_to_boxes_np
    - pkl instances are exactly the fg ids in instance seg
    """
    from det3d.inference.hybrid_lbd import load_lbd_pt
    from det3d.utils.bbox_sidecar import bbox_sidecar_path, load_detection_sidecar
    from nndet.io.transforms.instances import instances_to_boxes_np

    lbd_folder = Path(lbd_folder)
    report = {}
    for case_id in case_ids:
        lm = load_lbd_pt(lbd_folder / "lms" / f"{case_id}.pt")
        regen_boxes, _regen_labels, _ = lbd_lm_to_sidecar_tensors(
            lm, dusting_threshold=dusting_threshold
        )
        sidecar_boxes, _sidecar_labels, _ = load_detection_sidecar(
            bbox_sidecar_path(lbd_folder / "bboxes", case_id)
        )
        sidecar_t = (
            torch.stack(sidecar_boxes)
            if sidecar_boxes
            else torch.zeros((0, 6), dtype=torch.float32)
        )
        lmg_ok = regen_boxes.shape == sidecar_t.shape and (
            regen_boxes.numel() == 0 or torch.allclose(regen_boxes, sidecar_t, atol=sidecar_atol)
        )
        seg, mapping, _ = lm_pt_to_instance_seg(lm, dusting_threshold=dusting_threshold)
        boxes_pkl = instance_seg_to_nndet_boxes_pkl(seg, mapping)
        boxes_from_seg, _ = instances_to_boxes_np(seg[np.newaxis, ...], dim=3)
        pkl_roundtrip = np.allclose(boxes_pkl["boxes"], boxes_from_seg, atol=0.5)
        inst_ok = sorted(boxes_pkl["instances"]) == sorted(
            int(x) for x in np.unique(seg) if x > 0
        )
        painted = nndet_boxes_pkl_to_seg_mask(seg.shape, boxes_pkl)
        painted_ids = set(int(x) for x in np.unique(painted) if x > 0)
        reverse_ok = painted_ids == set(boxes_pkl["instances"])
        report[case_id] = {
            "lmg_vs_sidecar": lmg_ok,
            "seg_pkl_roundtrip": bool(pkl_roundtrip),
            "instance_ids_ok": inst_ok,
            "pkl_paint_reverse": reverse_ok,
            "n_instances": len(boxes_pkl["instances"]),
        }
    return report


def select_fg_case_ids(
    lbd_folder: Path,
    n_cases: int,
    case_ids: Optional[Sequence[str]] = None,
) -> list[str]:
    #AI
    import pandas as pd

    lbd_folder = Path(lbd_folder)
    if case_ids is not None:
        return [str(c) for c in case_ids]
    df = pd.read_csv(lbd_folder / "dataset_details.csv")
    rows = df[(df["has_fg"]) & (~df["bbox_empty"])].sort_values("case_id")
    return rows["case_id"].head(int(n_cases)).astype(str).tolist()


def _symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    if src.is_file():
        os.symlink(src, dst)


def materialize_lbd_nndet_task(
    lbd_folder: Path,
    scratch_det_data: Path,
    task_name: str,
    plan_id: str,
    plan_src: Path,
    dataset_json_src: Path,
    case_ids: Optional[Sequence[str]] = None,
    n_cases: int = 16,
    dusting_threshold: float = DUSTING_THRESHOLD,
    train_as_val: bool = True,
) -> dict:
    #AI
    """Write nnDetection preprocessed tree from LBD folder; return paths + verify report."""
    from nndet.io.load import save_pickle

    lbd_folder = Path(lbd_folder)
    scratch_det_data = Path(scratch_det_data)
    case_ids = select_fg_case_ids(lbd_folder, n_cases=n_cases, case_ids=case_ids)

    verify = verify_lbd_format_roundtrip(lbd_folder, case_ids, dusting_threshold=dusting_threshold)
    critical = ("seg_pkl_roundtrip", "instance_ids_ok", "pkl_paint_reverse")
    failed = [cid for cid, row in verify.items() if not all(row[k] for k in critical)]
    if failed:
        raise RuntimeError(f"LBD seg/pkl roundtrip failed for {failed}: {verify}")
    sidecar_drift = [cid for cid, row in verify.items() if not row["lmg_vs_sidecar"]]
    if sidecar_drift:
        print(f"sidecar drift vs lm (using lm truth): {sidecar_drift}")

    task_dir = scratch_det_data / task_name
    preprocessed = task_dir / "preprocessed"
    images_tr = preprocessed / plan_id / "imagesTr"
    images_tr.mkdir(parents=True, exist_ok=True)

    _symlink_or_copy(plan_src, preprocessed / f"{plan_id}.pkl")
    dataset_meta = json.loads(dataset_json_src.read_text())
    dataset_meta["task"] = task_name
    (task_dir / "dataset.json").write_text(json.dumps(dataset_meta, indent=2))

    for case_id in case_ids:
        data, seg, properties, boxes_pkl = lbd_case_to_nndet_arrays(
            lbd_folder, case_id, dusting_threshold=dusting_threshold
        )
        np.save(images_tr / f"{case_id}.npy", data)
        np.save(images_tr / f"{case_id}_seg.npy", seg[np.newaxis, ...].astype(np.int32))
        save_pickle(properties, images_tr / f"{case_id}.pkl")
        save_pickle(boxes_pkl, images_tr / f"{case_id}_boxes.pkl")

    split = {"train": list(case_ids), "val": list(case_ids) if train_as_val else []}
    splits = [split]
    save_pickle(splits, preprocessed / "splits_final.pkl")

    out = {
        "task_dir": task_dir,
        "images_tr": images_tr,
        "case_ids": case_ids,
        "verify": verify,
        "sidecar_drift": sidecar_drift,
        "splits": splits,
    }
    return out
