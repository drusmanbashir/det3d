"""Affine augmentation must pad with zeros, never mirror anatomy into the patch.

Defect (Fable box audit, operator review 2026-11-18): RandAffineBoxSyncd built RandAffined without padding_mode,
so MONAI's default reflection copied lesions across the patch edge into the image and target_seg with no box.
nnDetection pads with a constant. Reference: every foreground voxel after the warp must come from inside the patch.
MUTATION: drop padding_mode="zeros" in gpu_det.RandAffineBoxSyncd (reflection returns).
CONTROL: an object away from the edge keeps all its voxels under the same zoom-out.
"""

import numpy as np
import pytest
import torch
from monai.data import MetaTensor

import det3d.managers.data  # noqa: F401  (import before gpu_det avoids a circular import)
from det3d.transforms.gpu_det import RandAffineBoxSyncd

ZOOM_OUT = [(0.3, 0.3)] * 3


def warp(lm):  #AI
    """Zoom out by 1.3 with the production transform; return output mask and the output->input voxel map."""
    data = {"image": MetaTensor(lm.clone()), "lm": MetaTensor(lm.clone()), "bbox": torch.zeros((0, 6))}
    affine = RandAffineBoxSyncd(["image", "lm"], "bbox", ["bilinear", "nearest"], 1.0, None, ZOOM_OUT)
    affine.rand_affine.set_random_state(0)
    a0 = data["image"].meta["affine"].double()
    out = affine(data)
    to_input = torch.linalg.inv(a0) @ out["image"].meta["affine"].double()
    return out["lm"][0].numpy() > 0.5, to_input.numpy()


@pytest.mark.parametrize("x_slice", [slice(0, 6), slice(29, 35)], ids=["touches_edge", "control_interior"])
def test_affine_pads_without_mirrored_foreground(x_slice):  #AI
    lm = torch.zeros((1, 64, 64, 64))
    lm[0, x_slice, 20:30, 20:30] = 1
    mask, to_input = warp(lm)
    centres = np.argwhere(mask).astype(np.float64)
    source = centres @ to_input[:3, :3].T + to_input[:3, 3]
    assert len(centres) > 0
    assert source.min() >= -0.5 and source.max() <= 63.5
