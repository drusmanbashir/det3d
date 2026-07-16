"""CP-spatial-boxes — per-checkpoint disk vs pre_trafo oracle + dual-pipeline gate.

Layer 1: ``run_pipeline_to_checkpoint`` at L2/Rtr/Norm/Affine/ResizePC/BoxClip/full.
Layer 2: instance-id / IoU aligned compare (``compare_instance_aligned``).
Layer 3: ``DataManagerDetLBDPreTrafoBTfms`` vs ``DataManagerDetLBDBTfms`` at nnDet boundary.

Run one ``# %%`` cell at a time. See ``NATIVE_NNDET_HANDOFF.md`` § CP-spatial-boxes.
"""
from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from det3d.detection.nndet_train import det3d_batch_to_nndet
from det3d.archived.nndet_parity_cp0_4 import (
    CASE_ID,
    DET3D_PLAN_ID,
    DET3D_PROJECT,
    LBD_FOLDER,
    PARITY_SEED,
)
from det3d.archived.nndet_parity_disk_boxes_post_aug import (
    _apply_item_key,
    compare_instance_aligned,
    find_case_idx,
    load_sidecar_instances,
    pre_trafo_oracle_boxes,
    setup_parity_dm,
)
from det3d.archived.nndet_parity_pre_trafo import (
    DATA_ATOL,
    pre_trafo_train_targets,
    run_pre_trafo_stepped,
)
from det3d.managers.data.batch_tfms import DataManagerDetLBDBTfms
from det3d.managers.data.collate import lbd_det_collate_train, lbd_det_collate_train_pre_trafo
from det3d.managers.data.main import DataManagerDetLBD
from det3d.managers.data.pre_trafo import DataManagerDetLBDPreTrafoBTfms
from det3d.utils.bbox_sidecar import bbox_sidecar_path, load_detection_sidecar

SPATIAL_BOXES_ATOL = 2.0
CPU_KEYS_TR = "Ld,Rtr,L2,E,Norm,F1,F2,Affine,ResizePC,BoxClip,IntensityTfms"
ITEM_KEYS_BT = "Ld,Rtr,L2,E,Norm"
GPU_BATCH_STOPS = {
    "Affine": 3,
    "ResizePC": 4,
    "BoxClip": 5,
    "IntensityTfms": 6,
    "full": 6,
}
CHECKPOINTS = ("L2", "Rtr", "Norm", "Affine", "ResizePC", "BoxClip", "full")


def _fg_labels_from_manifest() -> list[int]:
    from utilz.fileio import load_json

    manifest = load_json(LBD_FOLDER / "manifest.json")
    labels_all = manifest["labels_all"]
    return [int(v) for v in labels_all if int(v) != 0] or [0]


def collate_with_instances(item_batch, instances, *, pre_trafo: bool = False) -> dict:
    if pre_trafo:
        batch = lbd_det_collate_train_pre_trafo(item_batch)
    else:
        batch = lbd_det_collate_train(item_batch)
    batch["instances"] = [instances]
    return batch


def _apply_gpu_batch_steps(batch, gpu_tail, n_steps: int, seed: int) -> dict:
    if n_steps <= 0:
        return batch
    from monai.transforms import Compose

    from det3d.transforms.gpu_det import BatchItemCompose

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    partial = BatchItemCompose(
        Compose(gpu_tail.tfms.transforms[:n_steps]),
        image_key=gpu_tail.image_key,
        box_key=gpu_tail.box_key,
        label_key=gpu_tail.label_key,
        point_key=gpu_tail.point_key,
        mask_key=gpu_tail.mask_key,
        lm_key=gpu_tail.lm_key,
    )
    return partial(batch)


def _item_keys_until(stop: str, keys_csv: str) -> list[str]:
    stop = "IntensityTfms" if stop == "full" else stop
    keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
    out = []
    for key in keys:
        out.append(key)
        if key == stop:
            break
    if stop in ("Ld", "Rtr") and "L2" not in out and "L2" in keys:
        idx = keys.index("L2")
        out = keys[: idx + 1]
    return out


def run_pipeline_to_checkpoint(
    dm,
    case_idx: int,
    seed: int,
    stop_after_key: str,
    *,
    gpu_tail: bool = False,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dici = dict(dm.data[case_idx])
    case_id = dici["case_id"]
    instances = load_sidecar_instances(case_id)

    stop = "IntensityTfms" if stop_after_key == "full" else stop_after_key

    if gpu_tail:
        item_keys = _item_keys_until(stop, dm.keys)
        for key in item_keys:
            dici = _apply_item_key(dici, key, dm.transforms_dict[key])
        batch = collate_with_instances([[dici]], instances)
        n_batch = GPU_BATCH_STOPS.get(stop_after_key, 0)
        if n_batch > 0:
            batch = _apply_gpu_batch_steps(batch, dm.GpuTail, n_batch, seed)
    else:
        keys = _item_keys_until(stop, CPU_KEYS_TR)
        for key in keys:
            dici = _apply_item_key(dici, key, dm.transforms_dict[key])
        batch = collate_with_instances([[dici]], instances)

    return batch


def setup_cpu_dm(*, batch_size: int = 1, debug: bool = True, device: str | None = None):
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers.project import Project

    P = Project(DET3D_PROJECT)
    C = ConfigMakerDet(P)
    C.setup(DET3D_PLAN_ID)
    conf = deepcopy(C.configs)
    conf["dataset_params"]["fold"] = 0
    if device is not None:
        conf["dataset_params"]["device"] = device
    dm = DataManagerDetLBD(P, conf, batch_size=batch_size, split="train", debug=debug)
    dm.prepare_data()
    dm.setup()
    return dm


def setup_pretrafo_btfms_dm(*, batch_size: int = 1, debug: bool = True, device: str | None = None):
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers.project import Project

    P = Project(DET3D_PROJECT)
    C = ConfigMakerDet(P)
    C.setup(DET3D_PLAN_ID)
    conf = deepcopy(C.configs)
    conf["dataset_params"]["fold"] = 0
    if device is not None:
        conf["dataset_params"]["device"] = device
    dm = DataManagerDetLBDPreTrafoBTfms(
        P, conf, batch_size=batch_size, split="train", debug=debug
    )
    dm.prepare_data()
    dm.setup()
    return dm


def gate_checkpoint_sample(
    dm,
    case_idx: int,
    seed: int,
    checkpoint: str,
    fg_labels: list[int],
    *,
    gpu_tail: bool = False,
    boxes_atol: float = SPATIAL_BOXES_ATOL,
) -> dict:
    batch = run_pipeline_to_checkpoint(
        dm, case_idx, seed, checkpoint, gpu_tail=gpu_tail
    )
    instances = batch["instances"][0]
    ref_boxes, ref_classes, present = pre_trafo_oracle_boxes(
        batch, fg_labels, forward_patch_size=dm.plan["patch_size"]
    )
    return compare_instance_aligned(
        batch["bbox"][0],
        batch["label"][0],
        ref_boxes,
        ref_classes,
        instances,
        present,
        fg_labels,
        boxes_atol=boxes_atol,
    )


def _sample_passes_gate(result: dict, *, min_matched_frac: float = 0.95) -> bool:
    if result["high_iou_bad_corner"]:
        return False
    n_ref = result["n_pairs"] + result["unmatched_pre"]
    if n_ref == 0:
        return result["unmatched_disk"] == 0
    matched_frac = result["matched_ok"] / n_ref
    return matched_frac >= min_matched_frac and result["coord_drift"] == 0


def sweep_checkpoint(
    dm,
    case_ids: list[str],
    seeds: list[int],
    checkpoint: str,
    fg_labels: list[int],
    *,
    gpu_tail: bool = False,
    boxes_atol: float = SPATIAL_BOXES_ATOL,
) -> dict:
    ok = 0
    total = 0
    unmatched_disk = 0
    unmatched_pre = 0
    coord_drift = 0
    ordering_only = 0
    max_diffs = []

    for case_id in case_ids:
        case_idx = find_case_idx(dm, case_id)
        for seed in seeds:
            total += 1
            result = gate_checkpoint_sample(
                dm,
                case_idx,
                seed,
                checkpoint,
                fg_labels,
                gpu_tail=gpu_tail,
                boxes_atol=boxes_atol,
            )
            if _sample_passes_gate(result):
                ok += 1
            unmatched_disk += result["unmatched_disk"]
            unmatched_pre += result["unmatched_pre"]
            coord_drift += result["coord_drift"]
            if result["ordering_only"]:
                ordering_only += 1
            max_diffs.append(result["max_diff"])

    return {
        "checkpoint": checkpoint,
        "gpu_tail": gpu_tail,
        "ok": ok,
        "total": total,
        "unmatched_disk": unmatched_disk,
        "unmatched_pre": unmatched_pre,
        "coord_drift": coord_drift,
        "ordering_only": ordering_only,
        "max_diff_p50": float(np.median(max_diffs)) if max_diffs else 0.0,
    }


def stress_sweep(
    dm,
    n_samples: int,
    seed_base: int,
    checkpoint: str,
    fg_labels: list[int],
    *,
    gpu_tail: bool = False,
    affine_p: float = 0.5,
) -> dict:
    n_cases = len(dm.data)
    ok = 0
    unmatched_disk = 0
    unmatched_pre = 0
    coord_drift = 0
    max_diffs = []

    affine3d = dm.configs["affine3d"]
    orig_p = float(affine3d["p"])
    affine3d["p"] = affine_p
    if gpu_tail:
        dm.GpuTail.tfms.transforms[2].rand_affine.prob = affine_p
    else:
        dm.transforms_dict["Affine"].rand_affine.prob = affine_p

    for i in range(n_samples):
        case_idx = i % n_cases
        seed = seed_base + i
        result = gate_checkpoint_sample(
            dm,
            case_idx,
            seed,
            checkpoint,
            fg_labels,
            gpu_tail=gpu_tail,
        )
        if _sample_passes_gate(result):
            ok += 1
        unmatched_disk += result["unmatched_disk"]
        unmatched_pre += result["unmatched_pre"]
        coord_drift += result["coord_drift"]
        max_diffs.append(result["max_diff"])

    affine3d["p"] = orig_p
    if gpu_tail:
        dm.GpuTail.tfms.transforms[2].rand_affine.prob = orig_p
    else:
        dm.transforms_dict["Affine"].rand_affine.prob = orig_p

    return {
        "checkpoint": checkpoint,
        "gpu_tail": gpu_tail,
        "affine_p": affine_p,
        "ok": ok,
        "total": n_samples,
        "unmatched_disk": unmatched_disk,
        "unmatched_pre": unmatched_pre,
        "coord_drift": coord_drift,
        "max_diff_p50": float(np.median(max_diffs)) if max_diffs else 0.0,
    }


def run_pretrafo_btfms_batch(dm, case_idx: int, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dici = dict(dm.data[case_idx])
    instances = load_sidecar_instances(dici["case_id"])
    item_keys = [k.strip() for k in dm.keys.split(",") if k.strip()]
    for key in item_keys:
        dici = _apply_item_key(dici, key, dm.transforms_dict[key])
    batch = collate_with_instances([[dici]], instances, pre_trafo=True)
    if dm.transforms_batch is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        batch = dm.transforms_batch(batch)
    return batch


def compare_nndet_target_boxes(
    disk_boxes: torch.Tensor,
    disk_classes: torch.Tensor,
    pre_boxes: torch.Tensor,
    pre_classes: torch.Tensor,
    instances: dict | None,
    present_instances: torch.Tensor,
    fg_labels: list[int],
    *,
    boxes_atol: float = SPATIAL_BOXES_ATOL,
) -> dict:
    from det3d.archived.nndet_parity_disk_boxes_post_aug import (
        match_instance_aligned_pairs,
    )

    disk_xyxyzz = disk_boxes.detach().cpu().float()
    ref_xyxyzz = pre_boxes.detach().cpu().float()
    n_disk = int(disk_xyxyzz.shape[0])
    n_ref = min(int(ref_xyxyzz.shape[0]), int(pre_classes.shape[0]))
    ref_xyxyzz = ref_xyxyzz[:n_ref]
    ref_cls = pre_classes.detach().cpu().long()[:n_ref]
    disk_cls = disk_classes.detach().cpu().long()[:n_disk]

    if n_disk == 0 and n_ref == 0:
        return {
            "matched_ok": 0,
            "n_pairs": 0,
            "unmatched_disk": 0,
            "unmatched_pre": 0,
            "ordering_only": False,
            "coord_drift": 0,
            "max_diff": 0.0,
            "high_iou_bad_corner": 0,
            "approved": True,
        }

    pairs, used_disk, used_ref = match_instance_aligned_pairs(
        disk_xyxyzz,
        disk_cls,
        ref_xyxyzz,
        ref_cls,
        instances,
        present_instances,
    )
    matched_ok = 0
    coord_drift = 0
    max_diff = 0.0
    high_iou_bad_corner = 0
    for disk_idx, ref_idx, via in pairs:
        diff = float(torch.max(torch.abs(disk_xyxyzz[disk_idx] - ref_xyxyzz[ref_idx])).item())
        max_diff = max(max_diff, diff)
        cls_match = int(disk_cls[disk_idx].item()) == int(ref_cls[ref_idx].item())
        if diff <= boxes_atol and cls_match:
            matched_ok += 1
        else:
            coord_drift += 1
    unmatched_disk = n_disk - len(used_disk)
    unmatched_pre = n_ref - len(used_ref)
    return {
        "matched_ok": matched_ok,
        "n_pairs": len(pairs),
        "unmatched_disk": unmatched_disk,
        "unmatched_pre": unmatched_pre,
        "ordering_only": False,
        "coord_drift": coord_drift,
        "max_diff": max_diff,
        "high_iou_bad_corner": high_iou_bad_corner,
        "approved": high_iou_bad_corner == 0 and coord_drift == 0,
    }


def dual_pipeline_oracle(
    case_id: str,
    seed: int,
    fg_labels: list[int],
    *,
    boxes_atol: float = SPATIAL_BOXES_ATOL,
    data_atol: float = DATA_ATOL,
) -> dict:
    from det3d.detection.nndet_train import det3d_batch_to_pre_trafo_input

    dm = setup_parity_dm(device="cuda:0", debug=False)
    case_idx = find_case_idx(dm, case_id)
    batch = run_disk_box_pipeline_with_instances(dm, case_idx, seed)

    batch_pre_in = det3d_batch_to_pre_trafo_input(
        batch, forward_patch_size=dm.plan["patch_size"], fg_labels=fg_labels
    )
    batch_post = run_pre_trafo_stepped(batch_pre_in, verbose=False)
    targets_pre = pre_trafo_train_targets(batch_post)
    targets_disk = det3d_batch_to_nndet(
        batch,
        fg_labels=fg_labels,
        use_disk_box_plug=True,
    )

    img_diff = float(
        torch.max(torch.abs(targets_pre["data"] - targets_disk["data"])).item()
    )
    seg_ok = torch.equal(
        targets_pre["target_seg"][0].long(),
        targets_disk["target_seg"][0].long(),
    )

    instances = batch["instances"][0]
    box_result = compare_nndet_target_boxes(
        targets_disk["target_boxes"][0],
        targets_disk["target_classes"][0],
        targets_pre["target_boxes"][0],
        targets_pre["target_classes"][0],
        instances,
        batch_post["present_instances"][0],
        fg_labels,
        boxes_atol=boxes_atol,
    )

    passed = (
        img_diff < data_atol
        and seg_ok
        and _sample_passes_gate(box_result, min_matched_frac=0.90)
    )
    return {
        "case_id": case_id,
        "seed": seed,
        "img_diff": img_diff,
        "seg_ok": seg_ok,
        "box_result": box_result,
        "passed": passed,
    }


def run_disk_box_pipeline_with_instances(dm, case_idx: int, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dici = dict(dm.data[case_idx])
    instances = load_sidecar_instances(dici["case_id"])
    item_keys = [k.strip() for k in dm.keys.split(",") if k.strip()]
    for key in item_keys:
        dici = _apply_item_key(dici, key, dm.transforms_dict[key])
    batch = collate_with_instances([[dici]], instances)
    if dm.transforms_batch is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        batch = dm.transforms_batch(batch)
    return batch


def _case_ids_for_sweep(dm, n: int = 20) -> list[str]:
    return [row["case_id"] for row in dm.data[:n]]


def run_all_gates() -> None:
    fg_labels = _fg_labels_from_manifest()
    print("FG_LABELS", fg_labels)
    print("SPATIAL_BOXES_ATOL", SPATIAL_BOXES_ATOL)

    from det3d.configs.parser import ConfigMakerDet
    from fran.managers.project import Project

    P = Project(DET3D_PROJECT)
    C = ConfigMakerDet(P)
    C.setup(DET3D_PLAN_ID)
    conf = deepcopy(C.configs)
    conf["dataset_params"]["fold"] = 0
    dm_cpu = DataManagerDetLBD(P, conf, batch_size=1, split="train", debug=True)
    dm_cpu.prepare_data()
    dm_cpu.setup()
    case_idx = find_case_idx(dm_cpu, CASE_ID)
    seed = PARITY_SEED
    batch_l2 = run_pipeline_to_checkpoint(dm_cpu, case_idx, seed, "L2", gpu_tail=False)
    _sidecar_boxes, _labels, instances = load_detection_sidecar(
        bbox_sidecar_path(LBD_FOLDER / "bboxes", CASE_ID)
    )
    ref_boxes, ref_classes, present = pre_trafo_oracle_boxes(
        batch_l2, fg_labels, forward_patch_size=dm_cpu.plan["patch_size"]
    )
    res_l2 = compare_instance_aligned(
        batch_l2["bbox"][0],
        batch_l2["label"][0],
        ref_boxes,
        ref_classes,
        instances,
        present,
        fg_labels,
        boxes_atol=SPATIAL_BOXES_ATOL,
    )
    assert res_l2["coord_drift"] == 0 and res_l2["high_iou_bad_corner"] == 0, res_l2
    print("L2 micro gate (unmatched_disk expected when sidecar count > lm)", res_l2)

    dm_gpu = setup_parity_dm(device="cuda:0", debug=False)
    case_ids = _case_ids_for_sweep(dm_gpu, 20)
    seeds = [PARITY_SEED, PARITY_SEED + 1, PARITY_SEED + 2]
    gpu_p1 = {}
    for cp in CHECKPOINTS:
        gpu_p1[cp] = sweep_checkpoint(
            dm_gpu, case_ids, seeds, cp, fg_labels, gpu_tail=True
        )
        print("gpu_tail p=1.0", cp, gpu_p1[cp])

    dm_cpu = setup_cpu_dm(device="cpu", debug=False)
    case_ids = _case_ids_for_sweep(dm_cpu, 20)
    cpu_p1 = {}
    for cp in CHECKPOINTS:
        cpu_p1[cp] = sweep_checkpoint(
            dm_cpu, case_ids, seeds, cp, fg_labels, gpu_tail=False
        )
        print("cpu p=1.0", cp, cpu_p1[cp])

    stress_gpu = stress_sweep(
        dm_gpu, 200, PARITY_SEED + 1000, "full", fg_labels, gpu_tail=True, affine_p=0.5
    )
    print("stress gpu_tail full p=0.5", stress_gpu)

    stress_cpu = stress_sweep(
        dm_cpu, 200, PARITY_SEED + 2000, "full", fg_labels, gpu_tail=False, affine_p=0.5
    )
    print("stress cpu full p=0.5", stress_cpu)

    dual = dual_pipeline_oracle(CASE_ID, PARITY_SEED, fg_labels)
    print("dual_pipeline_oracle (nnDet boundary, same aug batch)", dual)
    assert dual["passed"], dual


if __name__ == "__main__":
    run_all_gates()
