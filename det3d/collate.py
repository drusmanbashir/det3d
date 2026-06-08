import torch
import torch.nn.functional as F
from monai.transforms.utils import compute_divisible_spatial_size


def spatial_shape(t):
    return tuple(int(v) for v in t.shape[-3:])


def pad_offsets(current_shape, target_shape):
    """End-padding only: voxel indices unchanged (offsets are zero)."""
    return [0, 0, 0]


def adjust_boxes_for_pad(box, offsets):
    """Shift StandardMode xyzxyz boxes when padding adds voxels before content."""
    box = torch.as_tensor(box, dtype=torch.float32)
    if box.numel() == 0:
        return box.reshape(0, 6)
    if box.ndim == 1:
        box = box.unsqueeze(0)
    off = torch.tensor(offsets, dtype=box.dtype)
    box = box.clone()
    box[:, :3] += off
    box[:, 3:6] += off
    return box


def pad_image_to_shape(image, target_shape):
    shape = spatial_shape(image)
    pad = []
    for current, target in zip(reversed(shape), reversed(target_shape)):
        pad.extend([0, max(target - current, 0)])
    if sum(pad) == 0:
        return torch.as_tensor(image).contiguous()
    out = F.pad(torch.as_tensor(image), pad, value=0)
    return out.contiguous()


def as_box_tensor(box):
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
    return box.contiguous()


def as_label_tensor(label):
    if isinstance(label, list):
        if len(label) == 0:
            return torch.zeros((0,), dtype=torch.long)
        if len(label) == 1:
            label = label[0]
        else:
            label = torch.tensor(label, dtype=torch.long)
    return torch.as_tensor(label, dtype=torch.long).reshape(-1)


def obd_det_collate(batch, size_divisible=None):
    """Pad each item to batch max spatial size, adjust boxes, stack images."""
    max_shape = [0, 0, 0]
    for item in batch:
        shape = spatial_shape(item["image"])
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
        shape = spatial_shape(item["image"])
        offsets = pad_offsets(shape, target_shape)
        image = pad_image_to_shape(item["image"], target_shape)
        box = adjust_boxes_for_pad(item["box"], offsets)
        images.append(image)
        boxes.append(box)
        labels.append(as_label_tensor(item["label"]))
    images_out = torch.stack(images, 0)
    return {
        "image": images_out,
        "box": boxes,
        "label": labels,
        "spatial_size": target_shape,
    }


def det_stack_collate(batch):
    """Stack pre-padded items (fixed-pad pipeline)."""
    images = torch.stack([torch.as_tensor(item["image"]).contiguous() for item in batch], 0)
    boxes = [as_box_tensor(item["box"]) for item in batch]
    labels = [as_label_tensor(item["label"]) for item in batch]
    return {"image": images, "box": boxes, "label": labels}
