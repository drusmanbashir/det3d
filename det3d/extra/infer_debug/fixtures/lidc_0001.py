"""Fixture case: lidc_all lidc_0001 real image + LM (2 label-1 lesions, ignore 6)."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from label_analysis.geometry_pt import LabelMapGeometryPT

from det3d.geometry.lmg import voxel_start_size_to_xyzxyz
from det3d.extra.infer_debug.fixtures._case import InferFixtureCase
from det3d.extra.infer_debug.fixtures._geom import (
    crop_volume,
    slices_around_box,
    union_box_xyzxyz,
)
from fran.inference.helpers import full_meta_from_image, load_images_nifti

CASE_ID = "lidc_0001"
LIDC_ROOT = Path("/media/UB/datasets/lidc_all")
IMAGE_FN = LIDC_ROOT / "images" / f"{CASE_ID}.nii.gz"
LM_FN = LIDC_ROOT / "lms" / f"{CASE_ID}.nii.gz"
IGNORE_LABELS = [6]
MARGIN = 30


def _nbrhood_bboxes(nbrhoods_df):
    out = []
    for _, row in nbrhoods_df.iterrows():
        out.append(np.array(voxel_start_size_to_xyzxyz(row["bbox"]), dtype=np.float64))
    return np.stack(out, axis=0)


def build_lidc_0001() -> InferFixtureCase:
    lm_dat = load_images_nifti([LM_FN])[0]
    lm_full = lm_dat["image"]
    if lm_full.ndim == 4:
        lm_full = lm_full[0]
    lm_full = lm_full.detach().cpu()

    L = LabelMapGeometryPT(li=lm_full, ignore_labels=IGNORE_LABELS, compute_feret=False)
    n = len(L.nbrhoods)
    if n != 2:
        raise ValueError(f"{CASE_ID}: expected 2 lesions (ignore {IGNORE_LABELS}), got {n}")

    lesion_boxes = _nbrhood_bboxes(L.nbrhoods)
    spatial = tuple(int(v) for v in lm_full.shape[-3:])
    bounding_box = slices_around_box(union_box_xyzxyz(lesion_boxes), spatial, MARGIN)

    dat = load_images_nifti([IMAGE_FN])[0]
    image_full = dat["image"]
    full_meta = full_meta_from_image(image_full)

    return InferFixtureCase(
        name="lidc_0001",
        image_full=image_full,
        lm_full=lm_full,
        bounding_box=bounding_box,
        ignore_labels=list(IGNORE_LABELS),
        n_lesions=2,
        source_image=str(IMAGE_FN),
        full_meta=full_meta,
        lesion_boxes_full=lesion_boxes,
    )


def crop_fixture(case: InferFixtureCase):
    img = case.image_full
    img_crop = img[tuple(case.bounding_box[1:])]
    lm_crop = crop_volume(case.lm_full, case.bounding_box)
    return img_crop, lm_crop
