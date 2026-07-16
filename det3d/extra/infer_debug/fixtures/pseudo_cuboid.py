"""Fixture case: in-memory pseudo volume + LM with two cuboid lesions."""

from copy import deepcopy

import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor

from det3d.extra.infer_debug.fixtures._case import InferFixtureCase
from det3d.extra.infer_debug.fixtures._geom import (
    crop_volume,
    slices_around_box,
    union_box_xyzxyz,
)


def _affine_for_spacing(spacing):
    sx, sy, sz = spacing
    aff = np.zeros((4, 4), dtype=np.float64)
    aff[0, 0] = -sx
    aff[1, 1] = -sy
    aff[2, 2] = sz
    aff[3, 3] = 1.0
    return aff


def _meta_for_shape(shape, spacing, filename="pseudo.nii.gz"):
    aff = _affine_for_spacing(spacing)
    spatial = tuple(int(v) for v in shape)
    return {
        "affine": torch.tensor(aff, dtype=torch.float64),
        "original_affine": torch.tensor(aff, dtype=torch.float64),
        "spatial_shape": spatial,
        "space": "RAS",
        "filename_or_obj": filename,
    }


def _fill_cuboid(vol, x0, y0, z0, sx, sy, sz, value):
    vol[x0 : x0 + sx, y0 : y0 + sy, z0 : z0 + sz] = value


def build_pseudo_cuboid(spacing=(0.8, 0.8, 1.5), margin=12) -> InferFixtureCase:
    shape = (128, 128, 64)
    meta = _meta_for_shape(shape, spacing, "pseudo_cuboid.nii.gz")

    image = torch.zeros(shape, dtype=torch.float32)
    image[40:90, 35:85, 20:45] = 0.4
    image = MetaTensor(image, meta=deepcopy(meta))

    lm = torch.zeros(shape, dtype=torch.uint8)
    _fill_cuboid(lm, 50, 45, 25, 12, 14, 8, 1)
    _fill_cuboid(lm, 88, 82, 40, 10, 10, 6, 1)

    boxes = np.array(
        [
            [50, 45, 25, 62, 59, 33],
            [88, 82, 40, 98, 92, 46],
        ],
        dtype=np.float64,
    )
    union = union_box_xyzxyz(boxes)
    bounding_box = slices_around_box(union, shape, margin)

    lm_full = lm
    full_meta = deepcopy(meta)
    full_meta["spatial_shape"] = shape

    return InferFixtureCase(
        name="pseudo_cuboid",
        image_full=image,
        lm_full=lm_full,
        bounding_box=bounding_box,
        ignore_labels=[],
        n_lesions=2,
        source_image="pseudo_cuboid.nii.gz",
        full_meta=full_meta,
        lesion_boxes_full=boxes,
    )


def crop_fixture(case: InferFixtureCase):
    img = case.image_full
    img_crop = img[tuple(case.bounding_box[1:])]
    lm_crop = crop_volume(case.lm_full, case.bounding_box)
    return img_crop, lm_crop
