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


def save_detection_sidecar(out_fn, boxes, labels, ignore_labels=None):
    if not isinstance(boxes, list):
        boxes = [boxes]
    if not isinstance(labels, list):
        labels = [labels]
    payload = {
        "box": [_box_to_list(box) for box in boxes],
        "label": [_label_to_int(label) for label in labels],
    }
    if ignore_labels is not None:
        payload["ignore_labels"] = [int(x) for x in ignore_labels]
    save_json(payload, out_fn)


def load_detection_sidecar(bbox_fn):
    sidecar = json.loads(Path(bbox_fn).read_text())
    boxes = sidecar["box"]
    labels = sidecar["label"]
    valid_boxes = []
    valid_labels = []
    for box, label in zip(boxes, labels):
        valid_boxes.append(torch.tensor(box, dtype=torch.float32))
        valid_labels.append(torch.tensor(int(label), dtype=torch.long))
    return valid_boxes, valid_labels
