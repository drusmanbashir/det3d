"""LMG ground-truth roundtrip: full lm → crop → S,O → LMG pred → InvPreBox → Off → full."""

from pathlib import Path

import numpy as np
import torch
from label_analysis.geometry_itk import LabelMapGeometryITK
from label_analysis.geometry_pt import LabelMapGeometryPT
from monai.data.meta_tensor import MetaTensor
from monai.transforms import EnsureChannelFirstd, Orientationd, Spacingd

from det3d.geometry.lmg import voxel_start_size_to_xyzxyz
from det3d.inference.post import InvPreprocessBoxd, Offd
from fran.inference.helpers import load_images_nifti

CASE_ID = "lidc_0001"
LIDC_ROOT = Path("/media/UB/datasets/lidc_all")
LM_FN = LIDC_ROOT / "lms" / f"{CASE_ID}.nii.gz"
IGNORE_LABELS = [1]
SPACING = (0.8, 0.8, 1.5)
MARGIN = 30


def lmg_bbox_xyzxyz(lm, ignore_labels):
    if isinstance(lm, (str, Path)):
        L = LabelMapGeometryITK(li=str(lm), ignore_labels=ignore_labels, compute_feret=False)
    else:
        L = LabelMapGeometryPT(li=lm, ignore_labels=ignore_labels, compute_feret=False)
    if L.nbrhoods.empty:
        raise ValueError("LMG found no components")
    row = L.nbrhoods.iloc[0]
    return np.array(voxel_start_size_to_xyzxyz(row["bbox"]), dtype=np.float64)


def slices_around_box(box_xyzxyz, spatial_shape, margin):
    x0, y0, z0, x1, y1, z1 = [int(v) for v in box_xyzxyz]
    nx, ny, nz = (int(v) for v in spatial_shape)
    return [
        slice(0, 1),
        slice(max(0, x0 - margin), min(nx, x1 + margin)),
        slice(max(0, y0 - margin), min(ny, y1 + margin)),
        slice(max(0, z0 - margin), min(nz, z1 + margin)),
    ]


def crop_volume(vol, bounding_box):
    sl = tuple(bounding_box[1:])
    if vol.ndim == 3:
        return vol[sl]
    return vol[(slice(None),) + sl]


def forward_so(volume, spacing=SPACING, axcodes="RAS"):
    d = {"image": volume}
    d = Spacingd(keys=["image"], pixdim=spacing, mode="nearest")(d)
    d = Orientationd(keys=["image"], axcodes=axcodes)(d)
    return d["image"]


def recover_full_bbox(pred_preproc_xyzxyz, lm_so, bounding_box):
    batch = {
        "pred_box": torch.tensor(pred_preproc_xyzxyz, dtype=torch.float32).unsqueeze(0),
        "image": lm_so,
        "bounding_box": bounding_box,
    }
    batch = InvPreprocessBoxd()(batch)
    batch = Offd(box_keys=["pred_box"])(batch)
    return batch["pred_box"][0].numpy()


def run_lmg_roundtrip():
    gt_full = lmg_bbox_xyzxyz(LM_FN, IGNORE_LABELS)
    print("=== full-volume LMG GT (ignore", IGNORE_LABELS, ") ===")
    print("xyzxyz", gt_full)

    dat = load_images_nifti([LM_FN])[0]
    lm = dat["image"]
    if lm.ndim == 4:
        lm = lm[0]
    spatial = tuple(int(v) for v in lm.shape[-3:])
    bounding_box = slices_around_box(gt_full, spatial, MARGIN)
    print("crop slices", bounding_box[1:])

    lm_crop = crop_volume(lm, bounding_box)
    if lm_crop.ndim == 3:
        lm_crop = lm_crop.unsqueeze(0)
    lm_crop = MetaTensor(lm_crop, meta=lm.meta)

    starts = np.array([bounding_box[1].start, bounding_box[2].start, bounding_box[3].start])
    gt_crop = gt_full.copy()
    gt_crop[[0, 3]] -= starts[0]
    gt_crop[[1, 4]] -= starts[1]
    gt_crop[[2, 5]] -= starts[2]
    gt_crop_lmg = lmg_bbox_xyzxyz(lm_crop, IGNORE_LABELS)
    print("native crop LMG vs slice GT |d|", np.max(np.abs(gt_crop_lmg - gt_crop)))

    lm_so = forward_so(lm_crop)
    print("preproc crop shape", tuple(int(v) for v in lm_so.shape[-3:]))

    from det3d.extra.bbox_geom_roundtrip import forward_preprocess_box

    pred_preproc = lmg_bbox_xyzxyz(lm_so, IGNORE_LABELS)
    math_preproc, _ = forward_preprocess_box(gt_crop, lm_crop)
    print("LMG on preproc crop", pred_preproc)
    print("forward(gt_crop) preproc", math_preproc)

    recovered_lmg = recover_full_bbox(pred_preproc, lm_so, bounding_box)
    recovered_math = recover_full_bbox(math_preproc, lm_so, bounding_box)
    print("recover(LMG pred) full", recovered_lmg)
    print("recover(forward gt) full", recovered_math)
    print("GT full", gt_full)
    print("LMG end-to-end |d|", np.max(np.abs(recovered_lmg - gt_full)))
    print("math end-to-end |d|", np.max(np.abs(recovered_math - gt_full)))
    print("math pass", np.max(np.abs(recovered_math - gt_full)) < 1.0)


if __name__ == "__main__":
    run_lmg_roundtrip()
