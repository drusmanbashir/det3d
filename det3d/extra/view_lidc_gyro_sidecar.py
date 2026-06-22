"""#AI LIDCA-GYRO sidecar QA — full-volume JSON boxes vs GT lm / pred seg / live pipeline."""
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from det3d.inference.visualize import (
    crop_volume_and_boxes,
    focal_slice_index,
    load_sidecar_volume,
    sidecar_pred_boxes,
)
from det3d.utils.bbox_sidecar import load_inference_sidecar
from utilz.imageviewers import ImageBBoxViewer, ImageMaskBboxViewer

RUN_P = "LIDCA-GYRO"
CASE_ID = "lidc_0007"
SCORE_MIN = 0.02
SHOW = "sidecar"  # sidecar | pipeline | both
ORIENTATION = "axial"
LIDC_ROOT = Path("/media/UB/datasets/lidc_all")
PRED_DIR = Path("/s/fran_storage/predictions/lidca") / RUN_P

# RetinaUNet cascade bbox handoff (patch -> cascade):
#   predict_inner: nnDet xxyyzz inclusive, spaced crop voxels
#   patch BoxFmt:  det3d exclusive [x0,x1,z0,y0,y1,z1], same space
#   cascade BoxPts: scale det3d exclusive -> MONAI xyzxyz native crop (Spacingd inverse)
#   cascade Off:   MONAI xyzxyz, full volume voxels
#   VoxCopy -> JSON bbox_voxel_full (pre gt_box_mode)
# Cascade patch keys_postproc = "Pack,BoxFmt,SqL" (Int is standalone nifti infer only)


def case_paths(case_id=CASE_ID):
    stem = case_id
    image = LIDC_ROOT / "images" / f"{stem}.nii.gz"
    gt_lm = LIDC_ROOT / "lms" / f"{stem}.nii.gz"
    sidecar = PRED_DIR / f"{stem}.json"
    pred_seg = PRED_DIR / f"{stem}.nii.gz"
    return {"image": image, "gt_lm": gt_lm, "sidecar": sidecar, "pred_seg": pred_seg}


def load_nifti_array(path):
    return np.asarray(nib.load(str(path)).dataobj)


def volume_for_viewer(vol):
    if isinstance(vol, torch.Tensor):
        if vol.dim() == 4:
            return vol[0]
        return vol
    return vol


def box_stats(name, boxes, shape):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        print(f"{name}: empty")
        return
    lo = boxes.min(axis=0)
    hi = boxes.max(axis=0)
    neg = int((boxes < 0).sum())
    dims = np.asarray(shape[-3:] if len(shape) == 4 else shape, dtype=float)
    oob = int(((boxes[:, :3] < 0) | (boxes[:, 3:6] > dims)).any(axis=1).sum())
    print(f"{name}: n={boxes.shape[0]} min={lo} max={hi} neg={neg} oob={oob} vol={tuple(int(v) for v in dims)}")


def mask_fg_ijk(mask):
    fg = np.argwhere(mask > 0)
    if fg.size == 0:
        return None
    return fg.min(axis=0), fg.max(axis=0)


def box_overlaps_fg(boxes, fg_min, fg_max):
    if boxes.size == 0 or fg_min is None:
        return 0
    hits = 0
    for b in boxes:
        x0, y0, z0, x1, y1, z1 = b
        if x0 <= fg_max[0] and x1 >= fg_min[0] and y0 <= fg_max[1] and y1 >= fg_min[1] and z0 <= fg_max[2] and z1 >= fg_min[2]:
            hits += 1
    return hits


def sidecar_boxes_tensor(sidecar, score_min=SCORE_MIN):
    boxes = sidecar_pred_boxes(sidecar, score_min=score_min)
    return torch.as_tensor(boxes, dtype=torch.float32)


def sanity_report(image, gt_lm, sidecar, boxes):
    shape = image.shape
    print(f"case_id={sidecar['case_id']} source={sidecar['source_image']}")
    print(f"volume shape={shape} spacing={sidecar['spacing']}")
    print(f"lbd_bounding_box={sidecar['lbd_bounding_box']}")
    print(f"n_preds={len(sidecar['predictions'])} kept={boxes.shape[0]} score_min={SCORE_MIN}")
    box_stats("bbox_voxel_full", boxes.numpy(), shape)
    fg = mask_fg_ijk(gt_lm)
    if fg is not None:
        print(f"gt_lm fg ijk min={fg[0]} max={fg[1]} voxels={int((gt_lm > 0).sum())}")
        print(f"boxes overlapping gt fg: {box_overlaps_fg(boxes.numpy(), fg[0], fg[1])}")


def view_bbox(image, boxes, orientation=ORIENTATION):
    vol = volume_for_viewer(torch.as_tensor(image))
    viewer = ImageBBoxViewer(vol, boxes, orientation=orientation)
    if boxes.numel() > 0 and orientation == "axial":
        viewer.slider.set_val(focal_slice_index(boxes.numpy()))
    plt.show(block=True)


def view_mask_bbox(image, mask, boxes, orientation=ORIENTATION):
    vol = volume_for_viewer(torch.as_tensor(image))
    mask_t = volume_for_viewer(torch.as_tensor(mask))
    viewer = ImageMaskBboxViewer(vol, mask_t, boxes, orientation=orientation)
    if boxes.numel() > 0 and orientation == "axial":
        viewer.slider.set_val(focal_slice_index(boxes.numpy()))
    plt.show(block=True)


def view_lbd_crop(image, sidecar, boxes):
    crop, crop_boxes = crop_volume_and_boxes(image, boxes.numpy(), sidecar["lbd_bounding_box"])
    print(f"LBD crop shape={crop.shape}")
    box_stats("crop boxes", crop_boxes, crop.shape)
    view_bbox(crop, torch.as_tensor(crop_boxes))


def pipeline_probe(case_id=CASE_ID):
    """Step live cascade tfms; compare coords to on-disk sidecar."""
    from fran.inference import helpers
    from fran.utils.common import COMMON_PATHS
    from utilz.fileio import load_yaml

    from det3d.inference.cascade import DetCascadeInfererRetinaUNet

    paths = case_paths(case_id)
    sidecar_disk = load_inference_sidecar(paths["sidecar"])
    disk_boxes = sidecar_pred_boxes(sidecar_disk, score_min=SCORE_MIN)

    def default_run_w():
        fn = Path(COMMON_PATHS["cold_storage_folder"]) / "conf" / "best_runs.yaml"
        return load_yaml(fn)["totalseg"]["whole"]["runs"][0]

    imgs = [paths["image"]]
    En = DetCascadeInfererRetinaUNet(
        run_w=default_run_w(),
        run_p=RUN_P,
        project_title="lidca",
        devices=[0],
        localiser_labels=[6],
        safe_mode=True,
        patch_overlap=0.5,
        save=False,
        debug=False,
    )
    En.setup()
    En.create_and_set_postprocess_transforms()
    data = helpers.load_images_nifti(imgs)
    En.bboxes = En.extract_fg_bboxes(data, overwrite=True)
    data = En.apply_bboxes(data, En.bboxes)
    full_metas = [dat["full_meta"] for dat in data]
    pred_patches = En.patch_prediction(data)
    decollated = En.decollate_patches(pred_patches, En.bboxes, full_metas)
    item = decollated[0]
    tfms = En.postprocess_transforms_dict
    shape = tuple(int(v) for v in load_nifti_array(paths["image"]).shape)

    print("=== pipeline probe ===")
    print("postprocess keys", En.postprocess_tfms_keys)
    box_stats("patch postproc (det3d exclusive)", item["pred_box"].numpy(), item["image"].shape)

    stages = ["Pre", "SqL", "BoxPts", "Clip", "Off", "VoxCopy"]
    d = dict(item)
    for key in stages:
        if key not in tfms:
            continue
        d = tfms[key](d)
        bk = d["pred_box_voxel"] if key == "VoxCopy" else d["pred_box"]
        box_stats(key, bk.numpy(), shape)

    live_boxes = d["pred_box_voxel"].numpy()
    print("=== vs on-disk sidecar ===")
    box_stats("disk bbox_voxel_full", disk_boxes, shape)
    if live_boxes.size and disk_boxes.size:
        n = min(len(live_boxes), len(disk_boxes))
        diff = np.abs(live_boxes[:n] - disk_boxes[:n])
        print(f"mean abs diff (first {n}): {diff.mean():.3f}")
    return d, sidecar_disk


if __name__ == "__main__":
    # %%
    paths = case_paths()
    print(paths)

    # %%
    image = load_sidecar_volume({"source_image": str(paths["image"])})
    gt_lm = load_nifti_array(paths["gt_lm"]) if paths["gt_lm"].is_file() else None
    pred_seg = load_nifti_array(paths["pred_seg"]) if paths["pred_seg"].is_file() else None
    print("image", image.shape, "gt_lm", None if gt_lm is None else gt_lm.shape, "pred_seg", None if pred_seg is None else pred_seg.shape)

    # %%
    sidecar = load_inference_sidecar(paths["sidecar"])
    boxes = sidecar_boxes_tensor(sidecar, score_min=SCORE_MIN)

    # %%
    sanity_report(image, gt_lm, sidecar, boxes)

    # %%
    if SHOW in ("sidecar", "both"):
        view_bbox(image, boxes)

    # %%
    if gt_lm is not None and SHOW in ("sidecar", "both"):
        view_mask_bbox(image, gt_lm, boxes)

    # %%
    if pred_seg is not None and SHOW in ("sidecar", "both"):
        view_mask_bbox(image, pred_seg, boxes)

    # %%
    view_lbd_crop(image, sidecar, boxes)

    # %%
    if SHOW in ("pipeline", "both"):
        pipeline_probe()
