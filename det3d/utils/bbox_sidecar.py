import json
from pathlib import Path

import torch
from utilz.fileio import save_json


def bbox_sidecar_path(bboxes_dir, image_stem):
    return Path(bboxes_dir) / f"{image_stem}.json"


def _box_to_list(box):
    return [float(x) for x in torch.as_tensor(box).reshape(-1).tolist()]


def _label_to_int(label):
    return int(torch.as_tensor(label).reshape(-1)[0].item())


def normalize_detection_sidecar(sidecar: dict) -> dict:
    payload = dict(sidecar)
    if "bbox" not in payload and "box" in payload:
        payload["bbox"] = payload.pop("box")
    elif "bbox" in payload and "box" in payload:
        payload.pop("box")
    return payload


def migrate_detection_sidecar_file(bbox_fn) -> bool:
    path = Path(bbox_fn)
    sidecar = json.loads(path.read_text())
    normalized = normalize_detection_sidecar(sidecar)
    if normalized == sidecar:
        return False
    save_json(normalized, path)
    return True


def save_detection_sidecar(out_fn, boxes, labels, ignore_labels=None):
    if not isinstance(boxes, list):
        boxes = [boxes]
    if not isinstance(labels, list):
        labels = [labels]
    payload = {
        "bbox": [_box_to_list(box) for box in boxes],
        "label": [_label_to_int(label) for label in labels],
    }
    if ignore_labels is not None:
        payload["ignore_labels"] = [int(x) for x in ignore_labels]
    save_json(payload, out_fn)


def valid_detection_box(box):
    b = torch.as_tensor(box).flatten()
    if b.numel() != 6:
        return False
    if (b[3:] < 1).any():
        return False
    return True


def sidecar_bbox_empty(bbox_fn):
    boxes, _labels = load_detection_sidecar(bbox_fn)
    if len(boxes) == 0:
        return True
    for box in boxes:
        if valid_detection_box(box):
            return False
    return True


def load_detection_sidecar(bbox_fn):
    path = Path(bbox_fn)
    sidecar = json.loads(path.read_text())
    normalized = normalize_detection_sidecar(sidecar)
    if normalized != sidecar:
        save_json(normalized, path)
    boxes = normalized["bbox"]
    labels = normalized["label"]
    valid_boxes = []
    valid_labels = []
    for box, label in zip(boxes, labels):
        valid_boxes.append(torch.tensor(box, dtype=torch.float32))
        valid_labels.append(torch.tensor(int(label), dtype=torch.long))
    return valid_boxes, valid_labels


def _boxes_to_list(boxes):
    box_t = torch.as_tensor(boxes, dtype=torch.float32)
    if box_t.numel() == 0:
        return []
    if box_t.ndim == 1:
        box_t = box_t.unsqueeze(0)
    return [_box_to_list(box_t[i]) for i in range(box_t.shape[0])]


def _labels_to_list(labels):
    label_t = torch.as_tensor(labels)
    if label_t.numel() == 0:
        return []
    return [int(v) for v in label_t.reshape(-1).tolist()]


def _scores_to_list(scores):
    score_t = torch.as_tensor(scores, dtype=torch.float32)
    if score_t.numel() == 0:
        return []
    return [float(v) for v in score_t.reshape(-1).tolist()]


def save_inference_sidecar(
    out_fn,
    source_image,
    case_id,
    lbd_bounding_box,
    localiser_run,
    det_run,
    spacing,
    affine,
    boxes_voxel,
    boxes_world,
    labels,
    scores,
    boxes_pre_tfm=None,
):
    voxel_list = _boxes_to_list(boxes_voxel)
    world_list = _boxes_to_list(boxes_world)
    pre_tfm_list = _boxes_to_list(boxes_pre_tfm) if boxes_pre_tfm is not None else []
    label_list = _labels_to_list(labels)
    score_list = _scores_to_list(scores)
    predictions = []
    n = max(len(voxel_list), len(world_list), len(label_list), len(score_list))
    for idx in range(n):
        predictions.append(
            {
                "bbox_voxel_full": voxel_list[idx] if idx < len(voxel_list) else [],
                "bbox_world": world_list[idx] if idx < len(world_list) else [],
                "bbox_pre_tfm": pre_tfm_list[idx] if idx < len(pre_tfm_list) else [],
                "label": label_list[idx] if idx < len(label_list) else 0,
                "score": score_list[idx] if idx < len(score_list) else 0.0,
            }
        )
    payload = {
        "source_image": str(source_image),
        "case_id": case_id,
        "lbd_bounding_box": lbd_bounding_box,
        "localiser_run": localiser_run,
        "det_run": det_run,
        "spacing": spacing,
        "affine": affine,
        "predictions": predictions,
    }
    save_json(payload, out_fn)


def load_inference_sidecar(sidecar_fn):
    path = Path(sidecar_fn)
    return json.loads(path.read_text())
