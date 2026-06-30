import json
from pathlib import Path

import cv2
import numpy as np
import torch
from det3d.detection.retinanet_detector2 import RetinaNetDetector2
from det3d.detection.visualize_image import draw_slice_boxes, pick_slice_index
from det3d.utils.tensor import to_numpy
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from utilz.stringz import ast_literal_eval


def val_patch_size_from_plan(plan):
    val_patch_size = plan.get("val_patch_size", [512, 512, 208])
    if isinstance(val_patch_size, str):
        val_patch_size = ast_literal_eval(val_patch_size)
    return [int(v) for v in val_patch_size]


def load_plan_json(plan_json):
    with open(plan_json, encoding="utf-8") as f:
        return json.load(f)


def load_plan_from_project(project_title, plan_id):
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers import Project

    project = Project(project_title)
    config_maker = ConfigMakerDet(project)
    config_maker.setup(int(plan_id))
    return config_maker.configs["plan_train"]


def build_hybrid_detector(plan, model_path, device):
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2 ** level for level in range(len(plan["returned_layers"]) + 1)],
        base_anchor_shapes=plan["base_anchor_shapes"],
    )
    net = torch.jit.load(str(model_path), map_location=device)
    detector = RetinaNetDetector2(
        network=net,
        anchor_generator=anchor_generator,
        debug=False,
    ).to(device)
    detector.set_target_keys(box_key="bbox", label_key="label")
    detector.set_box_selector_parameters(
        score_thresh=float(plan["score_thresh"]),
        topk_candidates_per_level=1000,
        nms_thresh=float(plan["nms_thresh"]),
        detections_per_img=int(plan["detections_per_img"]),
    )
    val_patch_size = val_patch_size_from_plan(plan)
    detector.set_sliding_window_inferer(
        roi_size=val_patch_size,
        overlap=0.25,
        sw_batch_size=1,
        mode="constant",
        device=str(device),
    )
    detector.eval()
    return detector


def load_lbd_pt(path):
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        img = obj
    elif hasattr(obj, "as_tensor"):
        img = obj.as_tensor()
    else:
        img = obj
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return img.float()


def intensity_clip_range(project=None, plan=None):
    if project is not None:
        return project.global_properties["intensity_clip_range"]
    if plan is not None and "intensity_clip_range" in plan:
        return plan["intensity_clip_range"]
    return [-1024.0, 300.0]


def normalize_lbd_image(img, clip_range):
    a_min = float(clip_range[0])
    a_max = float(clip_range[1])
    img = img.clone()
    img = torch.clamp(img, a_min, a_max)
    img = (img - a_min) / (a_max - a_min)
    return img


def infer_lbd_volume(detector, img, plan, device, clip_range):
    img = normalize_lbd_image(img, clip_range)
    val_patch_size = val_patch_size_from_plan(plan)
    spatial_dims = int(plan["spatial_dims"])
    val_input = img.to(device=device, dtype=torch.float32)
    if val_input.dim() == spatial_dims + 1 and int(val_input.shape[0]) == 1:
        val_input = val_input.squeeze(0)
    val_input = val_input.contiguous()
    use_inferer = val_input[0, ...].numel() >= int(np.prod(val_patch_size))
    with torch.no_grad():
        outputs = detector([val_input], use_inferer=use_inferer)
    return outputs[0]


def filter_pred(pred, detector, score_min=0.0):
    box_key = detector.target_box_key
    label_key = detector.target_label_key
    score_key = detector.pred_score_key
    boxes = pred[box_key]
    labels = pred[label_key]
    scores = pred[score_key]
    if scores.numel() == 0:
        return boxes, labels, scores
    keep = scores >= float(score_min)
    return boxes[keep], labels[keep], scores[keep]


def volume_numpy_from_input(img):
    vol = img
    if vol.dim() == 4:
        vol = vol[0]
    if isinstance(vol, torch.Tensor):
        vol = to_numpy(vol)
    return np.asarray(vol, dtype=np.float32)


def save_lbd_pred_png(img, pred, detector, out_png, score_min=0.0, slice_axis=2):
    boxes, labels, _scores = filter_pred(pred, detector, score_min=score_min)
    vol = volume_numpy_from_input(img)
    slice_idx = pick_slice_index(boxes, vol.shape, slice_axis=slice_axis)
    png_bgr = draw_slice_boxes(vol, slice_idx, boxes, labels, slice_axis=slice_axis)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), png_bgr)
    return out_png


def full_volume_bounding_box(img):
    """#AI Identity crop bbox (BBoxFromPTd layout) for pre-cropped LBD volumes."""
    sh = tuple(int(v) for v in img.shape)
    if len(sh) == 4:
        depth, height, width = sh[1], sh[2], sh[3]
    elif len(sh) == 3:
        depth, height, width = sh[0], sh[1], sh[2]
    else:
        raise ValueError(f"Expected 3D or 4D LBD image, got shape {sh}")
    return (
        slice(0, 100, None),
        slice(0, depth),
        slice(0, height),
        slice(0, width),
    )


def load_lbd_pt_patch_data(pt_paths):
    """#AI DetPatchInferer input from LBD .pt (full-volume bbox; skip localiser)."""
    from fran.inference.helpers import apply_bboxes, load_images_pt

    paths = [str(p) for p in pt_paths]
    data = load_images_pt(paths)
    bboxes = [full_volume_bounding_box(dat["image"]) for dat in data]
    return apply_bboxes(data, bboxes)


def collect_lbd_pt_paths(input_path=None, folder=None):
    if (input_path is None) == (folder is None):
        raise ValueError("Pass exactly one of input_path or folder")
    if input_path is not None:
        path = Path(input_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return [path]
    paths = sorted(Path(folder).glob("*.pt"))
    if len(paths) == 0:
        raise FileNotFoundError(f"No .pt files under {folder}")
    return paths
