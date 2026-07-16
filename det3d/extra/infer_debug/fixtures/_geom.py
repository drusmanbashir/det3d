"""Shared geometry helpers for infer_debug fixtures."""

import numpy as np


def union_box_xyzxyz(boxes):
    return np.array(
        [
            boxes[:, 0].min(),
            boxes[:, 1].min(),
            boxes[:, 2].min(),
            boxes[:, 3].max(),
            boxes[:, 4].max(),
            boxes[:, 5].max(),
        ],
        dtype=np.float64,
    )


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


def bbox_to_crop_local(box, bounding_box):
    starts = np.array([bounding_box[1].start, bounding_box[2].start, bounding_box[3].start])
    out = box.copy()
    out[[0, 3]] -= starts[2]
    out[[1, 4]] -= starts[1]
    out[[2, 5]] -= starts[0]
    return out
