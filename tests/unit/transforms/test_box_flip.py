"""Flips must move exclusive box edges by N - x (edges), not the voxel-centre rule N - 1 - x.

Defect (Fable box audit, operator review 2026-11-18): flipped boxes landed one voxel low in training
(RandFlipBoxSyncd) and when inference restored LPS/LAS orientation (InvPreprocessBoxd, BoxRd).
Reference is always the extent of the flipped voxels, never product box maths.
MUTATION: drop the 0.5 shift in gpu_det._warp_box_corners, or restore `sh[i] - 1.0 - coords[:, i]` in post.py.
CONTROL: the unflipped RAS cases stay exact under that mutation.
"""

import numpy as np
import pytest
import torch
from monai.data import MetaTensor
from monai.transforms import Compose, Orientationd, Spacingd

import det3d.managers.data  # noqa: F401  (import before gpu_det avoids a circular import)
from det3d.inference.post import BoxRd, InvPreprocessBoxd
from det3d.transforms.gpu_det import RandFlipBoxSyncd

SHAPE = (40, 48, 32)
OBJECT = (slice(10, 16), slice(20, 29), slice(5, 9))
ORIENTATIONS = pytest.mark.parametrize("diag", [(1, 1, 1), (-1, -1, 1), (-1, 1, 1)], ids=["RAS", "LPS", "LAS"])


def voxel_box(mask):  #AI
    """Exclusive xyzxyz extent of the nonzero voxels."""
    idx = np.argwhere(mask > 0.5)
    box = np.concatenate([idx.min(0), idx.max(0) + 1])
    return box


def test_rand_flip_box_matches_flipped_voxels():  #AI
    flipped_axes = set()
    for seed in range(20):
        cube = torch.zeros((1, 64, 64, 64))
        cube[0, 10:20, 10:20, 10:20] = 1
        data = {
            "image": MetaTensor(cube.clone()),
            "lm": MetaTensor(cube.clone()),
            "bbox": torch.tensor([[10.0, 10, 10, 20, 20, 20]]),
        }
        flip = RandFlipBoxSyncd(["image", "lm"], "bbox", prob=1.0)
        flip.rand_flip.set_random_state(seed)
        out = flip(data)
        want = voxel_box(out["lm"][0].numpy())
        flipped_axes.update(np.flatnonzero(want[:3] != 10).tolist())
        np.testing.assert_allclose(out["bbox"][0].numpy(), want, atol=1e-4)
    assert flipped_axes == {0, 1, 2}


@ORIENTATIONS
def test_inv_preprocess_box_restores_native_box(diag):  #AI
    native = np.zeros(SHAPE, np.float32)
    native[OBJECT] = 1
    affine = torch.diag(torch.tensor([*diag, 1.0], dtype=torch.float64))
    preprocess = Compose([Spacingd(["image"], pixdim=(1, 1, 1), mode="nearest"), Orientationd(["image"], axcodes="RAS")])
    pre = preprocess({"image": MetaTensor(torch.as_tensor(native)[None], affine=affine)})["image"]
    pred = torch.as_tensor(voxel_box(pre[0].numpy()), dtype=torch.float32)[None]
    out = InvPreprocessBoxd()({"image": pre[None], "pred_box": pred})["pred_box"][0]
    np.testing.assert_allclose(out.numpy(), voxel_box(native), atol=1e-4)


@ORIENTATIONS
def test_box_rd_restores_original_orientation_box(diag):  #AI
    ras = np.zeros(SHAPE, np.float32)
    ras[OBJECT] = 1
    original = np.flip(ras, axis=tuple(i for i, sign in enumerate(diag) if sign < 0))
    meta = {
        "affine": torch.eye(4, dtype=torch.float64),
        "original_affine": torch.diag(torch.tensor([*diag, 1.0], dtype=torch.float64)),
        "spatial_shape": SHAPE,
    }
    pred = torch.as_tensor(voxel_box(ras), dtype=torch.float32)[None]
    out = BoxRd()({"pred_box": pred, "full_meta": meta})["pred_box"][0]
    np.testing.assert_allclose(out.numpy(), voxel_box(original), atol=1e-4)
