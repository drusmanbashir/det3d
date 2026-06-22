"""#AI Fake SWI tile-offset harness for DetPatchInfererRetinaUNet postprocess path.

Builds a lidc_0001-shaped random volume, GT/pred fg cuboids with recorded voxel/world
coords, runs real `_run_swi` + cascade patch postprocess (`Pack,SqL`) with a
fake `_swi_predictor` that emits two nnDet xyxyzz boxes per overlapping tile (tile-local,
then `_offset_boxes`). Asserts merged/postprocessed boxes recover the recorded global
pred cuboid.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor
from monai.inferers import SlidingWindowInferer
from monai.transforms import Compose

from det3d.inference.patch import DetPatchInfererRetinaUNet

REF_NIFTI = Path("/media/UB/datasets/lidc_all/images/lidc_0001.nii.gz")
PATCH_SIZE = (128, 128, 64)
PATCH_OVERLAP = 0.5
SCORE_THRESH = 0.01


@dataclass
class CuboidRecord:
    voxel_lo: tuple[int, int, int]
    voxel_hi: tuple[int, int, int]
    nndet_xyxyzz: tuple[float, float, float, float, float, float]
    world_lo: tuple[float, float, float]
    world_hi: tuple[float, float, float]


@dataclass
class FakeCaseRecord:
    shape_xyz: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    affine: np.ndarray
    gt: CuboidRecord
    pred: CuboidRecord


class _FakeModel:
    def __init__(self, patch_size, device):
        self.forward_patch_size = tuple(int(v) for v in patch_size)
        self.val_patch_size = self.forward_patch_size
        self.plan = {"score_thresh": SCORE_THRESH}
        self.nndet_plan = {
            "architecture": {
                "topk_candidates": 1000,
                "detections_per_img": 100,
                "remove_small_boxes": 0.0,
            }
        }
        self.net = torch.nn.Linear(1, 1, device=device)

    def eval(self):
        pass


def _load_ref_meta(ref_nifti=REF_NIFTI):
    ref = nib.load(str(ref_nifti))
    shape_xyz = tuple(int(v) for v in ref.shape)
    spacing_xyz = tuple(float(v) for v in ref.header.get_zooms()[:3])
    affine = np.asarray(ref.affine, dtype=np.float64)
    return shape_xyz, spacing_xyz, affine


def _world_corner(affine, i, j, k):
    v = affine @ np.array([i, j, k, 1.0], dtype=np.float64)
    return float(v[0]), float(v[1]), float(v[2])


def _inclusive_cuboid_to_nndet(lo, hi):
    """Inclusive voxel corners lo..hi -> nnDet xyxyzz (instances_to_boxes convention)."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    return (
        float(x0 - 1),
        float(y0 - 1),
        float(x1 + 1),
        float(y1 + 1),
        float(z0 - 1),
        float(z1 + 1),
    )


def _record_cuboid(lo, hi, affine):
    nndet = _inclusive_cuboid_to_nndet(lo, hi)
    world_lo = _world_corner(affine, lo[0], lo[1], lo[2])
    world_hi = _world_corner(affine, hi[0], hi[1], hi[2])
    return CuboidRecord(lo, hi, nndet, world_lo, world_hi)


def make_fake_case(seed=0, ref_nifti=REF_NIFTI):
    rng = np.random.default_rng(seed)
    shape_xyz, spacing_xyz, affine = _load_ref_meta(ref_nifti)
    gt_lo = (210, 190, 44)
    gt_hi = (236, 216, 56)
    pred_lo = (gt_lo[0] + 3, gt_lo[1] + 3, gt_lo[2] + 2)
    pred_hi = (gt_hi[0] - 3, gt_hi[1] - 3, gt_hi[2] - 2)
    image = rng.normal(0.0, 1.0, size=shape_xyz).astype(np.float32)
    gt_mask = np.zeros(shape_xyz, dtype=np.uint8)
    pred_mask = np.zeros(shape_xyz, dtype=np.uint8)
    gt_mask[gt_lo[0] : gt_hi[0] + 1, gt_lo[1] : gt_hi[1] + 1, gt_lo[2] : gt_hi[2] + 1] = 1
    pred_mask[
        pred_lo[0] : pred_hi[0] + 1,
        pred_lo[1] : pred_hi[1] + 1,
        pred_lo[2] : pred_hi[2] + 1,
    ] = 1
    meta = {
        "filename_or_obj": str(ref_nifti),
        "spatial_shape": np.array(shape_xyz, dtype=np.int64),
        "spacing": np.array(spacing_xyz, dtype=np.float64),
        "affine": torch.tensor(affine, dtype=torch.float64),
        "original_affine": torch.tensor(affine, dtype=torch.float64),
        "space": "RAS",
    }
    vol = MetaTensor(
        torch.from_numpy(image[None].copy()),
        meta=meta,
    )
    record = FakeCaseRecord(
        shape_xyz=shape_xyz,
        spacing_xyz=spacing_xyz,
        affine=affine,
        gt=_record_cuboid(gt_lo, gt_hi, affine),
        pred=_record_cuboid(pred_lo, pred_hi, affine),
    )
    return vol, gt_mask, pred_mask, record


def _slice_origin_xyz(unravel_entry):
    spatial = unravel_entry[2:]
    return (
        int(spatial[0].start),
        int(spatial[1].start),
        int(spatial[2].start),
    )


def _global_nndet_for_tile_intersection(record, origin_xyz, tile_size):
    ox, oy, oz = origin_xyz
    tx, ty, tz = tile_size
    pred_lo = record.pred.voxel_lo
    pred_hi = record.pred.voxel_hi
    ix0 = max(pred_lo[0], ox)
    iy0 = max(pred_lo[1], oy)
    iz0 = max(pred_lo[2], oz)
    ix1 = min(pred_hi[0], ox + tx - 1)
    iy1 = min(pred_hi[1], oy + ty - 1)
    iz1 = min(pred_hi[2], oz + tz - 1)
    if ix0 > ix1 or iy0 > iy1 or iz0 > iz1:
        return None
    return torch.tensor(
        _inclusive_cuboid_to_nndet((ix0, iy0, iz0), (ix1, iy1, iz1)),
        dtype=torch.float32,
    )


def _global_to_tile_local_nndet(global_box, origin_xyz):
    ox, oy, oz = origin_xyz
    local = global_box.clone()
    local[0] -= ox
    local[1] -= oy
    local[2] -= ox
    local[3] -= oy
    local[4] -= oz
    local[5] -= oz
    return local


def _two_jitter_boxes(local_box):
    a = local_box.clone()
    b = local_box.clone()
    a[0] += 0.5
    a[1] += 0.5
    a[4] += 0.5
    b[2] -= 0.5
    b[3] -= 0.5
    b[5] -= 0.5
    return torch.stack([a, b], dim=0)


def _nndet_xyxyzz_to_inclusive(a):
    ax0, ay0, ax1, ay1, az0, az1 = (float(v) for v in a)
    return (ax0 + 1, ay0 + 1, az0 + 1, ax1 - 1, ay1 - 1, az1 - 1)


def _box_iou_3d_inclusive(a_lohi, b_lohi):
    ax0, ay0, az0, ax1, ay1, az1 = a_lohi
    bx0, by0, bz0, bx1, by1, bz1 = b_lohi
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    iz0 = max(az0, bz0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iz1 = min(az1, bz1)
    iw = max(0.0, ix1 - ix0 + 1.0)
    ih = max(0.0, iy1 - iy0 + 1.0)
    id_ = max(0.0, iz1 - iz0 + 1.0)
    inter = iw * ih * id_
    vol_a = max(0.0, ax1 - ax0 + 1.0) * max(0.0, ay1 - ay0 + 1.0) * max(0.0, az1 - az0 + 1.0)
    vol_b = max(0.0, bx1 - bx0 + 1.0) * max(0.0, by1 - by0 + 1.0) * max(0.0, bz1 - bz0 + 1.0)
    union = vol_a + vol_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class FakeSWIOffsetHarness(DetPatchInfererRetinaUNet):
    keys_postproc = "Pack,SqL"

    def __init__(self, record, device="cpu"):
        self.record = record
        self.patch_overlap = PATCH_OVERLAP
        self._box_acc = []
        self.model = _FakeModel(PATCH_SIZE, device)
        dev = torch.device(device)
        self.inferer = SlidingWindowInferer(
            roi_size=PATCH_SIZE,
            sw_batch_size=1,
            overlap=PATCH_OVERLAP,
            mode="constant",
            progress=False,
            sw_device=dev,
            device=dev,
        )
        self.inferer.with_coord = True
        self.create_postprocess_transforms(None)
        self.postprocess_tfms_keys = self.keys_postproc
        self.postprocess_transforms = [
            self.postprocess_transforms_dict[k]
            for k in self.postprocess_tfms_keys.replace(" ", "").split(",")
        ]
        self.postprocess_compose = Compose(self.postprocess_transforms)

    def _swi_predictor(self, win_data, unravel_slice):
        tile_size = tuple(int(v) for v in win_data.shape[-3:])
        device = win_data.device
        seg_out = torch.zeros(
            (win_data.shape[0], 1, *tile_size),
            dtype=torch.float32,
            device=device,
        )
        for i in range(win_data.shape[0]):
            origin_xyz = _slice_origin_xyz(unravel_slice[i])
            global_box = _global_nndet_for_tile_intersection(
                self.record, origin_xyz, tile_size
            )
            if global_box is None:
                boxes = torch.zeros((0, 6), dtype=torch.float32, device=device)
                scores = torch.zeros((0,), dtype=torch.float32, device=device)
                labels = torch.zeros((0,), dtype=torch.long, device=device)
            else:
                local_box = _global_to_tile_local_nndet(global_box, origin_xyz)
                boxes = _two_jitter_boxes(local_box).to(device)
                origin = self._tile_origin(unravel_slice[i])
                boxes = self._offset_boxes(boxes, origin)
                scores = torch.tensor([0.95, 0.90], dtype=torch.float32, device=device)
                labels = torch.zeros(2, dtype=torch.long, device=device)
            weights = self._box_tile_weight(boxes, tile_size)
            self._box_acc.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                    "weights": weights,
                }
            )
        return seg_out

    def run(self, vol):
        img = vol.float().to(next(self.model.net.parameters()).device)
        if img.dim() == 4:
            img = img.unsqueeze(0)
        seg, boxes, scores, labels = self._run_swi(img)
        batch = {
            "image": vol,
            "stitched_seg": seg,
            "merged_boxes": boxes,
            "merged_scores": scores,
            "merged_labels": labels,
        }
        batch = self.postprocess_compose(batch)
        return batch


def run_fake_swi_offset_test(seed=0, device="cpu", atol=2.0):
    vol, gt_mask, pred_mask, record = make_fake_case(seed=seed)
    harness = FakeSWIOffsetHarness(record, device=device)
    batch = harness.run(vol)
    merged_nndet = batch["pred_box"].detach().cpu()
    target_nndet = torch.tensor([record.pred.nndet_xyxyzz], dtype=torch.float32)
    target_inclusive = _nndet_xyxyzz_to_inclusive(record.pred.nndet_xyxyzz)
    best_iou = 0.0
    best_idx = -1
    if merged_nndet.shape[0] > 0:
        for i in range(merged_nndet.shape[0]):
            iou = _box_iou_3d_inclusive(
                _nndet_xyxyzz_to_inclusive(merged_nndet[i]),
                target_inclusive,
            )
            if iou > best_iou:
                best_iou = iou
                best_idx = i
    n_tiles = len(harness._box_acc)
    n_tile_hits = sum(1 for item in harness._box_acc if item["boxes"].shape[0] > 0)
    report = {
        "shape_xyz": record.shape_xyz,
        "gt_voxel": record.gt.voxel_lo + record.gt.voxel_hi,
        "pred_voxel": record.pred.voxel_lo + record.pred.voxel_hi,
        "pred_nndet": record.pred.nndet_xyxyzz,
        "pred_world_lo": record.pred.world_lo,
        "pred_world_hi": record.pred.world_hi,
        "n_tiles_seen": n_tiles,
        "n_tiles_with_boxes": n_tile_hits,
        "n_merged_boxes": int(merged_nndet.shape[0]),
        "best_iou": best_iou,
        "best_idx": best_idx,
        "best_nndet": merged_nndet[best_idx].tolist() if best_idx >= 0 else None,
        "target_nndet": record.pred.nndet_xyxyzz,
        "pred_mask_voxels": int(pred_mask.sum()),
        "gt_mask_voxels": int(gt_mask.sum()),
    }
    ok = n_tile_hits >= 2 and merged_nndet.shape[0] >= 1 and best_iou >= 0.5
    max_corner_err = None
    if best_idx >= 0:
        err = (merged_nndet[best_idx] - target_nndet[0]).abs().max().item()
        max_corner_err = err
        ok = ok and err <= atol
        report["max_corner_err"] = err
    report["pass"] = ok
    return report, batch


def print_report(report):
    print("=== fake SWI offset test ===")
    for key in [
        "shape_xyz",
        "gt_voxel",
        "pred_voxel",
        "pred_nndet",
        "n_tiles_seen",
        "n_tiles_with_boxes",
        "n_merged_boxes",
        "best_iou",
        "max_corner_err",
        "target_nndet",
        "best_nndet",
        "target_det3d",
        "best_det3d",
        "pass",
    ]:
        if key in report:
            print(f"{key}: {report[key]}")


if __name__ == "__main__":
    report, _batch = run_fake_swi_offset_test(seed=0, device="cpu", atol=2.0)
    print_report(report)
    if not report["pass"]:
        raise SystemExit(1)
