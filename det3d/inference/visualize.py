from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import torch
from det3d.detection.visualize_image import normalize_image_to_uint8
from det3d.utils.bbox_sidecar import load_inference_sidecar
from fran.inference.helpers import infer_project, load_params
from utilz.stringz import strip_extension


def list_sidecar_files(pred_dir):
    return sorted(Path(pred_dir).glob("*.json"))


def sidecar_file_at_index(pred_dir, index):
    sidecars = list_sidecar_files(pred_dir)
    if index < 0 or index >= len(sidecars):
        raise IndexError(f"index {index} out of range for {len(sidecars)} sidecars in {pred_dir}")
    return sidecars[index]


def resolve_sidecar_path(run_p, case, project=None):
    if project is None:
        params = load_params(run_p)
        project = infer_project(params)
    stem = strip_extension(case)
    return project.predictions_folder / run_p / f"{stem}.json"


def list_prediction_sidecars(run_p, project=None):
    if project is None:
        params = load_params(run_p)
        project = infer_project(params)
    fldr = project.predictions_folder / run_p
    return sorted(fldr.glob("*.json"))


def sidecar_at_index(run_p, index, project=None):
    sidecars = list_prediction_sidecars(run_p, project=project)
    if index < 0 or index >= len(sidecars):
        raise IndexError(f"index {index} out of range for {len(sidecars)} sidecars in {run_p}")
    return sidecars[index]


def sidecar_pred_boxes(sidecar, score_min=0.0, top_k=None):
    preds = sidecar["predictions"]
    kept = [p for p in preds if p["score"] >= score_min]
    if top_k is not None:
        kept = sorted(kept, key=lambda p: p["score"], reverse=True)[:top_k]
    if len(kept) == 0:
        return np.zeros((0, 6), dtype=np.float64)
    return np.asarray([p["bbox_voxel_full"] for p in kept], dtype=np.float64)


def load_sidecar_volume(sidecar):
    return np.asarray(nib.load(sidecar["source_image"]).dataobj)


def lbd_spatial_slices(lbd_bounding_box):
    return tuple(slice(int(pair[0]), int(pair[1])) for pair in lbd_bounding_box[1:])


def crop_volume_and_boxes(image, boxes, lbd_bounding_box):
    slc = lbd_spatial_slices(lbd_bounding_box)
    crop = image[slc]
    offsets = np.asarray([int(pair[0]) for pair in lbd_bounding_box[1:]], dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return crop, boxes
    out = boxes.copy()
    out[:, :3] -= offsets
    out[:, 3:6] -= offsets
    return crop, out


def focal_slice_index(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return 0
    center_z = (boxes[:, 2] + boxes[:, 5]) / 2.0
    return int(round(float(center_z.mean())))


def draw_boxes_on_axial_slice(image, boxes, z_slice):
    slc = np.asarray(image[:, :, z_slice], dtype=np.float32)
    draw_img = normalize_image_to_uint8(slc)
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)
    for bbox in boxes:
        x0, y0, z0, x1, y1, z1 = np.round(bbox).astype(int).tolist()
        if z1 < z_slice or z0 > z_slice:
            continue
        cv2.rectangle(draw_img, (y0, x0), (y1, x1), (0, 0, 255), 1)
    return draw_img


def save_sidecar_png(sidecar, out_png, score_min=0.0, top_k=None, crop_lbd=False):
    boxes = sidecar_pred_boxes(sidecar, score_min=score_min, top_k=top_k)
    image = load_sidecar_volume(sidecar)
    if crop_lbd:
        image, boxes = crop_volume_and_boxes(image, boxes, sidecar["lbd_bounding_box"])
    z_slice = focal_slice_index(boxes)
    png = draw_boxes_on_axial_slice(image, boxes, z_slice)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), png)
    return out_png


def view_inference_sidecar(
    sidecar_fn,
    score_min=0.0,
    top_k=None,
    crop_lbd=False,
    orientation="axial",
):
    import matplotlib.pyplot as plt
    from utilz.imageviewers import ImageBBoxViewer

    sidecar = load_inference_sidecar(sidecar_fn)
    boxes = sidecar_pred_boxes(sidecar, score_min=score_min, top_k=top_k)
    if crop_lbd:
        image = load_sidecar_volume(sidecar)
        image, boxes = crop_volume_and_boxes(image, boxes, sidecar["lbd_bounding_box"])
    else:
        image = sidecar["source_image"]
    bbox = torch.as_tensor(boxes, dtype=torch.float32)
    viewer = ImageBBoxViewer(image, bbox, orientation=orientation)
    if boxes.size > 0 and orientation == "axial":
        viewer.slider.set_val(focal_slice_index(boxes))
    plt.show(block=True)
    return sidecar
