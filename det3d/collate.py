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


def attach_targets(batch, box_key="bbox", label_key="label"):
    batch["targets"] = [
        {box_key: batch[box_key][i], label_key: batch[label_key][i]}
        for i in range(len(batch[box_key]))
    ]
    return batch


def det_val_collate(batch, box_key="bbox", label_key="label"):
    """Val batch: stack images, keep per-item bbox/label lists, attach MONAI targets."""
    images = []
    boxes = []
    labels = []
    for item in batch:
        images.append(torch.as_tensor(item["image"]).contiguous())
        if box_key in item:
            box = as_box_tensor(item[box_key])
        else:
            box = torch.zeros((0, 6), dtype=torch.float32)
        boxes.append(box)
        labels.append(as_label_tensor(item[label_key]))
    out = {
        "image": torch.stack(images, 0),
        box_key: boxes,
        label_key: labels,
    }
    return attach_targets(out, box_key, label_key)


def obd_det_collate(
    batch,
    size_divisible=None,
    box_key="bbox",
    point_key="points",
    mask_key="mask",
):
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
    points = []
    masks = []
    for item in batch:
        shape = spatial_shape(item["image"])
        offsets = pad_offsets(shape, target_shape)
        image = pad_image_to_shape(item["image"], target_shape)
        if box_key in item:
            box = adjust_boxes_for_pad(item[box_key], offsets)
        else:
            box = torch.zeros((0, 6), dtype=torch.float32)
        images.append(image)
        boxes.append(box)
        labels.append(as_label_tensor(item["label"]))
        if point_key in item:
            points.append(torch.as_tensor(item[point_key]).contiguous())
        if mask_key in item:
            masks.append(pad_image_to_shape(item[mask_key], target_shape))
    images_out = torch.stack(images, 0)
    out = {
        "image": images_out,
        box_key: boxes,
        "label": labels,
        "spatial_size": target_shape,
    }
    if points:
        out[point_key] = points
    if masks:
        out[mask_key] = masks
    return attach_targets(out, box_key, "label")


def _flatten_dict_samples(batch):
    flat = []
    for item in batch:
        if isinstance(item, list):
            flat.extend(_flatten_dict_samples(item))
        else:
            flat.append(item)
    return flat


def lbd_det_collate(
    batch,
    size_divisible=None,
    box_key="bbox",
    point_key="points",
    mask_key="mask",
):
    """Flatten RandCrop / shard multi-sample lists, then pad/stack for batched training."""
    flat = _flatten_dict_samples(batch)
    return obd_det_collate(
        flat,
        size_divisible=size_divisible,
        box_key=box_key,
        point_key=point_key,
        mask_key=mask_key,
    )


def det_stack_collate(batch, box_key="bbox"):
    """Stack pre-padded items (fixed-pad pipeline)."""
    images = torch.stack([torch.as_tensor(item["image"]).contiguous() for item in batch], 0)
    boxes = [as_box_tensor(item[box_key]) for item in batch]
    labels = [as_label_tensor(item["label"]) for item in batch]
    out = {"image": images, box_key: boxes, "label": labels}
    return attach_targets(out, box_key, "label")
