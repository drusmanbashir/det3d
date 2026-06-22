"""ImageBBoxViewer scratch — luna16_training2 NIfTI infer sidecars (world cccwhd -> voxel)."""
# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from det3d.extra.luna16_view_cases import N_VIEW, PRED_DIR, list_val_cases, print_val_cases
from monai.apps.detection.transforms.dictionary import AffineBoxToImageCoordinated
from monai.data import box_utils
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, Orientationd, ScaleIntensityRanged
from utilz.imageviewers import ImageBBoxViewer

USE_SCORE030 = True
INTENSITY_CLIP_RANGE = [-973.0, 429.0]
GT_BOX_MODE = "cccwhd"
AFFINE_LPS_TO_RAS = False
INDEX = 500
# SHOW: gt | pred | both
# ImageBBoxViewer colors boxes by list index (utilz.imageviewers.BBOX_COLORS):
#   #0 #e41a1c red   #1 #377eb8 blue  #2 #4daf4a green  #3 #984ea3 purple
#   #4 #ff7f00 orange #5 #a65628 brown #6 #f781bf pink  #7 #999999 grey  (cycles)
# gt   — all GT boxes: 1st GT red, 2nd GT blue, 3rd green, ...
# pred — all pred boxes: 1st pred red, 2nd pred blue, ...
# both — GT first then pred: 1st GT red, 2nd GT blue, ...; 1st pred takes next color after last GT
SHOW = "both"


def load_nifti_volume(nifti_path):
    intensity = ScaleIntensityRanged(
        keys=["image"],
        a_min=float(INTENSITY_CLIP_RANGE[0]),
        a_max=float(INTENSITY_CLIP_RANGE[1]),
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )
    tfms = Compose(
        [
            LoadImaged(keys=["image"], image_only=False, meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image"], dtype=torch.float32),
            Orientationd(keys=["image"], axcodes="RAS"),
            intensity,
        ]
    )
    data = tfms({"image": str(nifti_path)})
    return data["image"]


def volume_for_viewer(vol):
    if vol.dim() == 4:
        return vol[0]
    return vol


def world_cccwhd_to_voxel(boxes, vol):
    box_t = torch.as_tensor(boxes, dtype=torch.float32)
    if box_t.numel() == 0:
        return box_t.reshape(0, 6)
    if box_t.ndim == 1:
        box_t = box_t.unsqueeze(0)
    data = {
        "image": vol,
        "box": box_utils.convert_box_mode(box_t, src_mode=GT_BOX_MODE, dst_mode="xyzxyz"),
    }
    data = AffineBoxToImageCoordinated(
        box_keys=["box"],
        box_ref_image_keys="image",
        image_meta_key_postfix="meta_dict",
        affine_lps_to_ras=AFFINE_LPS_TO_RAS,
    )(data)
    return data["box"]


def viewer_boxes(sidecar, vol, show=SHOW):
    gt = world_cccwhd_to_voxel(sidecar["gt_box"], vol)
    pred = world_cccwhd_to_voxel(sidecar["pred_box"], vol)
    if show == "gt":
        return gt
    if show == "pred":
        return pred
    if gt.numel() == 0:
        return pred
    if pred.numel() == 0:
        return gt
    return torch.cat([gt, pred], dim=0)


def view_nifti_case(case_row, index, show=SHOW, orientation="axial"):
    sidecar_fn = case_row["sidecar"]
    if not sidecar_fn.is_file():
        raise FileNotFoundError(f"missing sidecar {sidecar_fn} — run infer_luna16_training2_nifti.py first")
    sidecar = json.loads(sidecar_fn.read_text())
    vol = load_nifti_volume(case_row["nifti"])
    boxes = viewer_boxes(sidecar, vol, show=show)
    print(
        f"view {case_row['case_id']} idx={index} nifti={case_row['nifti'].name} "
        f"gt={sidecar['n_gt']} pred={sidecar['n_pred']} boxes_drawn={boxes.shape[0]}"
    )
    viewer = ImageBBoxViewer(volume_for_viewer(vol), boxes, orientation=orientation)
    if boxes.numel() > 0 and orientation == "axial":
        center_z = (boxes[:, 2] + boxes[:, 5]) / 2.0
        viewer.slider.set_val(int(round(float(center_z.mean()))))
    plt.show(block=True)
    return sidecar


# %%
cases, _dm = list_val_cases(n=N_VIEW, pred_dir=PRED_DIR, use_score030=USE_SCORE030)
print_val_cases(cases)

view_nifti_case(cases[INDEX], index=INDEX, show=SHOW)
# %%
