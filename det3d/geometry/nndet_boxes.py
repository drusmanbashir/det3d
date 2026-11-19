"""nnDetection box layout (xyxyzz, padded minima) to and from det3d's exclusive xyzxyz, and prediction dicts."""

import torch


def to_nndet(boxes: torch.Tensor) -> torch.Tensor:  #AI
    """Convert exclusive xyzxyz boxes to nnDetection xyxyzz with padded minima."""
    native = boxes.reshape(-1, 6)[:, [0, 1, 3, 4, 2, 5]]
    native[:, [0, 1, 4]] -= 1
    return native


def from_nndet(boxes: torch.Tensor) -> torch.Tensor:  #AI
    """Convert xyxyzz to exclusive xyzxyz, preserving a clamped zero minimum.

    Native zero is ambiguous after clamping, so a det3d minimum of 1 round-trips to 0.
    """
    canonical = boxes.reshape(-1, 6)[:, [0, 1, 4, 2, 3, 5]]
    minima = canonical[:, :3]
    canonical[:, :3] = torch.where(minima > 0, minima + 1, 0)
    return canonical


def offset_nndet_xyxyzz_boxes(boxes, origin):
    # AI
    """Shift nnDet xyxyzz tile-local boxes to parent-volume coords."""
    if boxes.numel() == 0:
        return boxes
    shifted = boxes.clone()
    ox, oy, oz = (float(v) for v in origin)
    shifted[:, 0] += ox
    shifted[:, 1] += oy
    shifted[:, 2] += ox
    shifted[:, 3] += oy
    shifted[:, 4] += oz
    shifted[:, 5] += oz
    return shifted


def nndet_pred_to_vis(
    pred, box_key="bbox", label_key="label", score_key="label_scores"
):
    # AI
    boxes = pred["pred_boxes"][0]
    labels = pred["pred_labels"][0]
    scores = pred["pred_scores"][0]
    out = {box_key: boxes, label_key: labels, score_key: scores}
    if "pred_seg" in pred:
        seg = pred["pred_seg"]
        out["pred_seg"] = seg[0] if seg.dim() > 3 else seg
    return out
