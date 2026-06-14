import numpy as np
import torch
from monai.transforms.croppad.array import Crop
from monai.transforms.utils import map_binary_to_indices
from monai.utils import fall_back_tuple


def _mask_tensor(mask):
    if isinstance(mask, np.ndarray):
        t = torch.as_tensor(mask)
    else:
        t = mask
    if t.ndim == 4:
        t = t[0]
    return t


def mask_fg_bg_flat_indices(mask):
    """Fg/bg flat indices from detection mask (same pools as RandCropByPosNegLabeld)."""
    t = _mask_tensor(mask)
    fg, bg = map_binary_to_indices(t, image=None, image_threshold=0.0)
    fg = np.asarray(fg, dtype=np.int64).reshape(-1)
    bg = np.asarray(bg, dtype=np.int64).reshape(-1)
    return fg, bg


def monai_crop_center_to_slices(center, roi_size, spatial_shape):
    """Slice tuple matching MONAI SpatialCrop(roi_center, roi_size)."""
    roi_size = fall_back_tuple(roi_size, spatial_shape)
    slices = Crop.compute_slices(roi_center=center, roi_size=roi_size)
    crop_start = tuple(int(s.start) for s in slices)
    crop_end = tuple(int(s.stop) for s in slices)
    return slices, crop_start, crop_end
