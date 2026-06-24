import numpy as np
import torch
from det3d.transforms.detection import GenerateExtendedBoxMask
from monai.transforms.croppad.array import Crop
from monai.transforms.utils import map_binary_to_indices
from monai.utils import fall_back_tuple



def mask_fg_bg_flat_indices(mask):
    """Fg/bg flat indices from detection mask (same pools as RandCropByPosNegLabeld)."""
    assert mask.ndim==4, f"Expected 4D tensor with a preceding channel dim, got {mask.ndim}D tensor"
    fg, bg = map_binary_to_indices(mask, image=None, image_threshold=0.0)
    fg = np.asarray(fg, dtype=np.int64).reshape(-1)
    bg = np.asarray(bg, dtype=np.int64).reshape(-1)
    return fg, bg


def volume_fg_bg_flat_indices(volume, bg_subsample=5):
    """RandCrop pools from a 3D/4D volume; bg kept every Nth index (fran subsample_bg=5)."""
    fg, bg = mask_fg_bg_flat_indices(volume)
    if bg_subsample is not None and int(bg_subsample) > 1:
        bg = bg[:: int(bg_subsample)]
    return fg, bg


def volume_fg_flat_indices(volume):
    fg, _ = mask_fg_bg_flat_indices(volume)
    return fg


def monai_crop_center_to_slices(center, roi_size, spatial_shape):
    """Slice tuple matching MONAI SpatialCrop(roi_center, roi_size)."""
    roi_size = fall_back_tuple(roi_size, spatial_shape)
    slices = Crop.compute_slices(roi_center=center, roi_size=roi_size)
    crop_start = tuple(int(s.start) for s in slices)
    crop_end = tuple(int(s.stop) for s in slices)
    return slices, crop_start, crop_end


def _bbox_overlap_center_boxes(boxes, roi_size, src_dims):
    #AI
    gen = GenerateExtendedBoxMask(
        keys="bbox",
        image_key="image",
        spatial_size=tuple(int(v) for v in roi_size),
        whole_box=False,
    )
    result = gen.generate_fg_center_boxes_np(boxes, src_dims, whole_box=False)
    return result


def _center_inside_int_box(center, box, naxis):
    for axis in range(naxis):
        lo = int(box[axis])
        hi = int(box[axis + naxis])
        if center[axis] < lo or center[axis] >= hi:
            return False
    return True


def _center_inside_any_int_box(center, boxes, naxis):
    for box in boxes:
        if _center_inside_int_box(center, box, naxis):
            return True
    return False


def sample_crop_center_from_bboxes(boxes, roi_size, src_dims, is_fg, rng):
    #AI
    """Pos: center where patch intersects bbox (whole_box=False). Neg: outside those centers."""
    roi_size = tuple(int(v) for v in roi_size)
    src_dims = tuple(int(v) for v in src_dims)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 6)
    naxis = len(src_dims)
    center_boxes = _bbox_overlap_center_boxes(boxes, roi_size, src_dims)

    if is_fg:
        box = center_boxes[int(rng.randint(0, center_boxes.shape[0]))]
        center = []
        for axis in range(naxis):
            lo = int(box[axis])
            hi = int(box[axis + naxis])
            if hi <= lo:
                hi = lo + 1
            center.append(int(rng.randint(lo, hi)))
        result = tuple(center)
        return result

    for _ in range(64):
        center = tuple(
            int(rng.randint(0, int(src_dims[axis])))
            for axis in range(naxis)
        )
        if center_boxes.shape[0] == 0 or not _center_inside_any_int_box(
            center, center_boxes, naxis
        ):
            result = center
            return result
    result = tuple(int(rng.randint(0, int(src_dims[axis]))) for axis in range(naxis))
    return result
