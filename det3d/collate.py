import torch
import torch.nn.functional as F
from monai.transforms.utils import compute_divisible_spatial_size


def _spatial_shape(t):
    return tuple(int(v) for v in t.shape[-3:])


def _pad_to_shape(t, target_shape):
    shape = _spatial_shape(t)
    pad_pairs = []
    for current, target in zip(shape, target_shape):
        deficit = max(target - current, 0)
        pad_pairs.append((0, deficit))
    pad = []
    for left, right in pad_pairs[::-1]:
        pad.extend([left, right])
    if sum(pad) == 0:
        return t
    out = F.pad(t, pad, value=0)
    if hasattr(t, "meta"):
        out.meta = t.meta
    return out


def _as_box_tensor(box):
    if isinstance(box, list):
        if len(box) == 0:
            return torch.zeros((0, 6), dtype=torch.float32)
        if len(box) == 1:
            box = box[0]
        else:
            box = torch.stack([torch.as_tensor(b) for b in box])
    box = torch.as_tensor(box, dtype=torch.float32)
    if box.ndim == 1:
        box = box.unsqueeze(0)
    return box


def _as_label_tensor(label):
    if isinstance(label, list):
        if len(label) == 0:
            return torch.zeros((0,), dtype=torch.long)
        if len(label) == 1:
            label = label[0]
        else:
            label = torch.tensor(label, dtype=torch.long)
    return torch.as_tensor(label, dtype=torch.long).reshape(-1)


def obd_det_collate(batch, size_divisible=None):
    max_shape = [0, 0, 0]
    for item in batch:
        shape = _spatial_shape(item["image"])
        max_shape = [max(a, b) for a, b in zip(max_shape, shape)]
    target_shape = tuple(max_shape)
    if size_divisible is not None:
        target_shape = tuple(
            compute_divisible_spatial_size(list(target_shape), k=size_divisible)
        )
    images = []
    boxes = []
    labels = []
    for item in batch:
        images.append(_pad_to_shape(item["image"], target_shape))
        boxes.append(_as_box_tensor(item["box"]))
        labels.append(_as_label_tensor(item["label"]))
    images_out = torch.stack(images, 0)
    if hasattr(batch[0]["image"], "meta"):
        images_out.meta = batch[0]["image"].meta
    return {
        "image": images_out,
        "box": boxes,
        "label": labels,
        "spatial_size": target_shape,
    }
