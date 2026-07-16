"""Synthetic bbox roundtrip: native full -> crop -> S,O -> bbo2 -> InvPreBox -> Off."""

import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Orientationd, Spacingd

from det3d.inference.post import (
    InvPreprocessBoxd,
    Offd,
    apply_affine_row_points,
    apply_ornt_points_np,
    box_from_corner_points,
    inverse_preprocess_box,
)
from det3d.geometry.lmg import voxel_start_size_to_xyzxyz

ORIGINAL_AFFINE = np.array(
    [
        [-0.703125, 0.0, 0.0, 0.0],
        [0.0, -0.703125, 0.0, 0.0],
        [0.0, 0.0, 2.5, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
SPACING = (0.8, 0.8, 1.5)
FULL_SHAPE = (512, 512, 133)
BOUNDING_BOX = [
    slice(0, 100, None),
    slice(74, 442, None),
    slice(122, 402, None),
    slice(18, 133, None),
]
BBO2 = np.array(
    [91.8539, 12.6529, 111.8375, 126.3961, 51.4721, 127.3187], dtype=np.float64
)
# derived by inverse roundtrip on lidc_0001 constants
BBOX_FULL = np.array(
    [246.79639648, 292.09288086, 204.39583333, 277.15575195, 326.21131836, 230.19783333],
    dtype=np.float64,
)


def box_corners(box):
    x1, y1, z1, x2, y2, z2 = box
    return np.array(
        [
            [x1, y1, z1],
            [x2, y1, z1],
            [x2, y2, z1],
            [x1, y2, z1],
            [x1, y1, z2],
            [x2, y1, z2],
            [x1, y2, z2],
            [x2, y2, z2],
        ],
        dtype=np.float64,
    )


def bbox_to_crop_local(box, bounding_box):
    starts = np.array([bounding_box[1].start, bounding_box[2].start, bounding_box[3].start])
    out = box.copy()
    out[[0, 3]] -= starts[0]
    out[[1, 4]] -= starts[1]
    out[[2, 5]] -= starts[2]
    return out


def forward_preprocess_box(bbox_crop_local, img_native_crop):
    d = {"image": img_native_crop}
    d = Spacingd(keys=["image"], pixdim=SPACING)(d)
    img_s = d["image"]
    d = Orientationd(keys=["image"], axcodes="RAS")({"image": img_s})
    img_so = d["image"]
    ops = img_so.applied_operations
    import nibabel as nib
    from monai.data.utils import to_affine_nd

    pts = box_corners(bbox_crop_local)
    src = ops[0]["extra_info"]["src_affine"]
    dst = img_s.affine.numpy()
    xform = np.linalg.solve(to_affine_nd(3, src), to_affine_nd(3, dst))
    pts = apply_affine_row_points(pts, xform)
    orig_affine = ops[1]["extra_info"]["original_affine"]
    src_o = nib.io_orientation(to_affine_nd(3, orig_affine))
    dst_o = nib.orientations.axcodes2ornt("RAS")
    fwd_o = nib.orientations.ornt_transform(src_o, dst_o)
    pts = apply_ornt_points_np(pts, fwd_o, ops[1]["orig_size"])
    return box_from_corner_points(pts), img_so


def run_roundtrip():
    bbox_crop = bbox_to_crop_local(BBOX_FULL, BOUNDING_BOX)
    crop_shape = (
        BOUNDING_BOX[1].stop - BOUNDING_BOX[1].start,
        BOUNDING_BOX[2].stop - BOUNDING_BOX[2].start,
        BOUNDING_BOX[3].stop - BOUNDING_BOX[3].start,
    )
    img_crop = MetaTensor(
        torch.zeros(1, *crop_shape),
        affine=torch.tensor(ORIGINAL_AFFINE, dtype=torch.float64),
    )
    fwd, img_so = forward_preprocess_box(bbox_crop, img_crop)
    print("forward bboxt2", fwd)
    print("target bbo2   ", BBO2)
    print("forward max |d|", np.max(np.abs(fwd - BBO2)))

    inv = inverse_preprocess_box(
        BBO2,
        img_so.applied_operations,
        tuple(int(v) for v in img_so.shape[1:]),
        img_so.affine.numpy(),
    )
    print("InvPreBox crop", inv)
    print("expected crop ", bbox_crop)
    print("InvPre max |d|", np.max(np.abs(inv - bbox_crop)))

    batch = {
        "pred_box": torch.tensor(BBO2, dtype=torch.float32).unsqueeze(0),
        "image": img_so,
        "bounding_box": BOUNDING_BOX,
    }
    batch = InvPreprocessBoxd()(batch)
    batch = Offd(box_keys=["pred_box"])(batch)
    print("Off full", batch["pred_box"][0].numpy())
    print("expected full", BBOX_FULL)
    print("Off max |d|", np.max(np.abs(batch["pred_box"][0].numpy() - BBOX_FULL)))


def validate_lmg_gt():
    """GT from dataset_stats row: bbox = [x0,y0,z0,sx,sy,sz] voxels on native full volume."""
    gt_idx_size = [299, 343, 86, 37, 45, 8]
    gt_full = np.array(voxel_start_size_to_xyzxyz(gt_idx_size), dtype=np.float64)
    print("\n=== LMG GT lidc_0001 label_cc=1 ===")
    print("GT xyzxyz full", gt_full)

    bbox_crop = bbox_to_crop_local(gt_full, BOUNDING_BOX)
    crop_shape = (
        BOUNDING_BOX[1].stop - BOUNDING_BOX[1].start,
        BOUNDING_BOX[2].stop - BOUNDING_BOX[2].start,
        BOUNDING_BOX[3].stop - BOUNDING_BOX[3].start,
    )
    img_crop = MetaTensor(
        torch.zeros(1, *crop_shape),
        affine=torch.tensor(ORIGINAL_AFFINE, dtype=torch.float64),
    )
    fwd, img_so = forward_preprocess_box(bbox_crop, img_crop)
    print("GT forward preproc", fwd)
    print("inference bbo2     ", BBO2)

    batch = {
        "pred_box": torch.tensor(fwd, dtype=torch.float32).unsqueeze(0),
        "image": img_so,
        "bounding_box": BOUNDING_BOX,
    }
    batch = InvPreprocessBoxd()(batch)
    batch = Offd(box_keys=["pred_box"])(batch)
    recovered = batch["pred_box"][0].numpy()
    print("roundtrip full", recovered)
    print("GT roundtrip |d|", np.max(np.abs(recovered - gt_full)))
    inside = (
        0 <= recovered[0] < recovered[3] <= FULL_SHAPE[0]
        and 0 <= recovered[1] < recovered[4] <= FULL_SHAPE[1]
        and 0 <= recovered[2] < recovered[5] <= FULL_SHAPE[2]
    )
    print("inside full volume", inside)


if __name__ == "__main__":
    run_roundtrip()
    validate_lmg_gt()
