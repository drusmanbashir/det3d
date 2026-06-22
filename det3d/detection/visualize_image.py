# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
import torch
from det3d.utils.tensor import to_numpy
from fran.utils.colour_palette import colour_palette
from monai.data import box_utils


def normalize_image_to_uint8(image):
    """
    Normalize image to uint8
    Args:
        image: numpy array
    """
    draw_img = image
    if np.amin(draw_img) < 0:
        draw_img -= np.amin(draw_img)
    if np.amax(draw_img) > 1:
        draw_img /= np.amax(draw_img)
    draw_img = (255 * draw_img).astype(np.uint8)
    return draw_img


def _boxes_numpy(boxes):
    if isinstance(boxes, torch.Tensor):
        boxes = to_numpy(boxes)
    else:
        boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim == 1:
        boxes = boxes[None]
    if boxes.size == 0:
        return boxes.reshape(0, 6)
    return boxes


def _labels_numpy(labels):
    if labels is None:
        return np.zeros((0,), dtype=np.int64)
    if isinstance(labels, torch.Tensor):
        labels = to_numpy(labels)
    else:
        labels = np.asarray(labels, dtype=np.int64)
    return labels.reshape(-1)


def _box_visible(box, slice_axis, slice_idx):
    x0, y0, z0, x1, y1, z1 = box
    if slice_axis == 0:
        return x0 <= slice_idx <= x1
    if slice_axis == 1:
        return y0 <= slice_idx <= y1
    return z0 <= slice_idx <= z1


def _box_span(box, slice_axis):
    x0, y0, z0, x1, y1, z1 = box
    if slice_axis == 0:
        return x0, x1
    if slice_axis == 1:
        return y0, y1
    return z0, z1


def class_color_bgr(label):
    rgb = colour_palette[int(label) + 1]
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def pick_slice_index(boxes, volume_shape, slice_axis=2, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    n_slices = int(volume_shape[slice_axis])
    if n_slices <= 0:
        return 0
    boxes = _boxes_numpy(boxes)
    if boxes.shape[0] == 0:
        return int(rng.integers(0, n_slices))
    box = boxes[int(rng.integers(0, boxes.shape[0]))]
    span_lo, span_hi = _box_span(box, slice_axis)
    lo = max(0, min(int(np.floor(span_lo)), n_slices - 1))
    hi = max(0, min(int(np.ceil(span_hi)), n_slices - 1))
    if lo > hi:
        return lo
    return int(rng.integers(lo, hi + 1))


def pick_slice_index_max_visible(
    pred_boxes, gt_boxes, volume_shape, slice_axis=2, rng=None
):
    """Slice where the most pred boxes (then gt) intersect."""
    rng = np.random.default_rng() if rng is None else rng
    n_slices = int(volume_shape[slice_axis])
    pred_boxes = _boxes_numpy(pred_boxes)
    gt_boxes = _boxes_numpy(gt_boxes)
    if pred_boxes.shape[0] == 0 and gt_boxes.shape[0] == 0:
        return pick_slice_index(np.zeros((0, 6)), volume_shape, slice_axis, rng)
    best_slice = 0
    best_count = -1
    for s in range(n_slices):
        count = sum(_box_visible(b, slice_axis, s) for b in pred_boxes)
        if count > best_count:
            best_count = count
            best_slice = s
    if best_count > 0:
        return best_slice
    return pick_slice_index(gt_boxes, volume_shape, slice_axis, rng)


def count_boxes_on_slice(boxes, slice_idx, slice_axis=2):
    boxes = _boxes_numpy(boxes)
    return sum(_box_visible(b, slice_axis, slice_idx) for b in boxes)


def _volume_slice_2d(volume, slice_idx, slice_axis):
    if slice_axis == 0:
        return volume[slice_idx]
    if slice_axis == 1:
        return volume[:, slice_idx]
    return volume[:, :, slice_idx]


def _edge_points(p0, p1):
    x0, y0 = p0
    x1, y1 = p1
    length = max(abs(x1 - x0), abs(y1 - y0))
    if length == 0:
        return [(x0, y0)]
    pts = []
    for i in range(length + 1):
        t = i / length
        pts.append((int(round(x0 + t * (x1 - x0))), int(round(y0 + t * (y1 - y0)))))
    return pts


def _rectangle_edge_points(pt1, pt2):
    x0, y0 = pt1
    x1, y1 = pt2
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    edges = []
    for i in range(4):
        edges.append(_edge_points(corners[i], corners[(i + 1) % 4]))
    return edges


def _draw_rect_solid(draw_img, pt1, pt2, color, thickness=1):
    cv2.rectangle(draw_img, pt1, pt2, color, thickness)


def _draw_rect_dashed(draw_img, pt1, pt2, color, thickness=1, dash_len=6, gap_len=4):
    for edge in _rectangle_edge_points(pt1, pt2):
        on = True
        count = 0
        for x, y in edge:
            if on:
                cv2.circle(draw_img, (x, y), thickness, color, -1)
            count += 1
            if count >= (dash_len if on else gap_len):
                on = not on
                count = 0


def _draw_rect_dotted_gappy(draw_img, pt1, pt2, color, thickness=1, dot_gap=10):
    for edge in _rectangle_edge_points(pt1, pt2):
        for i, (x, y) in enumerate(edge):
            if i % dot_gap == 0:
                cv2.circle(draw_img, (x, y), thickness, color, -1)


def _scores_numpy(scores):
    if scores is None:
        return np.zeros((0,), dtype=np.float64)
    if isinstance(scores, torch.Tensor):
        scores = to_numpy(scores)
    else:
        scores = np.asarray(scores, dtype=np.float64)
    return scores.reshape(-1)


def filter_detection_pred(pred, score_min, top_k=None):
    """Drop sub-threshold preds; optional top_k by score after threshold."""
    box_key = None
    label_key = None
    score_key = None
    for key, val in pred.items():
        if not isinstance(val, torch.Tensor):
            continue
        if val.ndim == 2 and val.shape[-1] == 6:
            box_key = key
        elif val.ndim == 1 and val.dtype in {torch.long, torch.int, torch.int64}:
            label_key = key
        elif val.ndim == 1 and val.is_floating_point():
            score_key = key
    if "label_scores" in pred:
        score_key = "label_scores"
    box_key = box_key or "bbox"
    label_key = label_key or "label"
    if score_key is None:
        out = dict(pred)
        out[box_key] = torch.zeros((0, 6), dtype=torch.float32)
        out[label_key] = torch.zeros((0,), dtype=torch.long)
        out["label_scores"] = torch.zeros((0,), dtype=torch.float32)
        return out
    keep = pred[score_key] >= float(score_min)
    out = dict(pred)
    out[box_key] = pred[box_key][keep]
    out[label_key] = pred[label_key][keep]
    out[score_key] = pred[score_key][keep]
    if top_k is not None and out[box_key].shape[0] > int(top_k):
        order = out[score_key].argsort(descending=True)[: int(top_k)]
        out[box_key] = out[box_key][order]
        out[label_key] = out[label_key][order]
        out[score_key] = out[score_key][order]
    return out


def score_bucket(score, low_min=0.3, mid_min=0.5, high_min=0.8):
    """Half-open buckets: low [low_min, mid_min), mid [mid_min, high_min), high [high_min, ...)."""
    if score < low_min:
        return None
    if score < mid_min:
        return "low"
    if score < high_min:
        return "mid"
    return "high"


def _slice_box_corners(box, slice_axis):
    return (int(round(box[1])), int(round(box[0]))), (
        int(round(box[4])),
        int(round(box[3])),
    )


def _box_2d_side_px(pt1, pt2):
    w = abs(pt2[0] - pt1[0])
    h = abs(pt2[1] - pt1[1])
    return max(w, h)


def _draw_box_dot(draw_img, pt1, pt2, color, radius=2):
    cx = int(round((pt1[0] + pt2[0]) / 2))
    cy = int(round((pt1[1] + pt2[1]) / 2))
    cv2.circle(draw_img, (cx, cy), radius, color, -1)


def _draw_scored_box_2d(draw_img, pt1, pt2, color, bucket, tiny_side_px=4):
    side = _box_2d_side_px(pt1, pt2)
    if side < tiny_side_px:
        dot_r = {"low": 1, "mid": 2, "high": 3}[bucket]
        _draw_box_dot(draw_img, pt1, pt2, color, radius=dot_r)
        return
    if bucket == "low":
        _draw_rect_dotted_gappy(draw_img, pt1, pt2, color, dot_gap=4)
    elif bucket == "mid":
        _draw_rect_dashed(draw_img, pt1, pt2, color, dash_len=4, gap_len=3)
    else:
        _draw_rect_solid(draw_img, pt1, pt2, color, thickness=1)


def _draw_gt_box_2d(draw_img, pt1, pt2, color, tiny_side_px=4):
    if _box_2d_side_px(pt1, pt2) < tiny_side_px:
        _draw_box_dot(draw_img, pt1, pt2, color, radius=2)
        return
    _draw_rect_solid(draw_img, pt1, pt2, color, thickness=1)


def _box_slice_center_pt(box):
    pt1, pt2 = _slice_box_corners(box, 2)
    cx = int(round((pt1[0] + pt2[0]) / 2))
    cy = int(round((pt1[1] + pt2[1]) / 2))
    return cx, cy


def _draw_off_slice_marker(draw_img, box, color):
    cx, cy = _box_slice_center_pt(box)
    cv2.circle(draw_img, (cx, cy), 2, color, 1)


def top_pred_indices_per_gt(gt_boxes, pred_boxes, pred_scores, per_gt=5):
    """Top per_gt preds per GT by 3D IoU then score."""
    gt_boxes = _boxes_numpy(gt_boxes)
    pred_boxes = _boxes_numpy(pred_boxes)
    scores = _scores_numpy(pred_scores)
    n_gt = gt_boxes.shape[0]
    n_pred = pred_boxes.shape[0]
    if n_gt == 0 or n_pred == 0:
        return np.zeros((0,), dtype=np.int64)
    pred_ok = np.all(pred_boxes[:, 3:6] > pred_boxes[:, 0:3], axis=1)
    if not pred_ok.any():
        return np.zeros((0,), dtype=np.int64)
    orig_idx = np.flatnonzero(pred_ok)
    pred_boxes = pred_boxes[pred_ok]
    scores = scores[pred_ok]
    gt_ok = np.all(gt_boxes[:, 3:6] > gt_boxes[:, 0:3], axis=1)
    if not gt_ok.any():
        return np.zeros((0,), dtype=np.int64)
    gt_boxes = gt_boxes[gt_ok]
    ious = to_numpy(
        box_utils.box_iou(
            torch.as_tensor(pred_boxes, dtype=torch.float32),
            torch.as_tensor(gt_boxes, dtype=torch.float32),
        )
    )
    selected = []
    seen = set()
    for g in range(gt_boxes.shape[0]):
        order = sorted(
            range(pred_boxes.shape[0]),
            key=lambda i: (-float(ious[i, g]), -float(scores[i])),
        )
        n = 0
        for idx in order:
            if ious[idx, g] <= 0:
                break
            pred_i = int(orig_idx[idx])
            if pred_i in seen:
                continue
            seen.add(pred_i)
            selected.append(pred_i)
            n += 1
            if n >= int(per_gt):
                break
    out = np.array(selected, dtype=np.int64)
    return out


def _overlay_slice_boxes_scored(
    draw_img,
    slice_idx,
    boxes,
    labels,
    scores,
    slice_axis=2,
    low_min=0.3,
    mid_min=0.5,
    high_min=0.8,
    tiny_side_px=4,
    show_off_slice_markers=False,
):
    boxes = _boxes_numpy(boxes)
    labels = _labels_numpy(labels)
    scores = _scores_numpy(scores)
    for i in range(boxes.shape[0]):
        box = boxes[i]
        score = float(scores[i]) if i < len(scores) else 0.0
        bucket = score_bucket(score, low_min, mid_min, high_min)
        if bucket is None:
            continue
        label = int(labels[i]) if i < len(labels) else 0
        pt1, pt2 = _slice_box_corners(box, slice_axis)
        color = class_color_bgr(label)
        if not _box_visible(box, slice_axis, slice_idx):
            if show_off_slice_markers:
                _draw_off_slice_marker(draw_img, box, color)
            continue
        _draw_scored_box_2d(draw_img, pt1, pt2, color, bucket, tiny_side_px=tiny_side_px)
    return draw_img


def draw_slice_boxes_scored(
    image_vol,
    slice_idx,
    boxes,
    labels,
    scores,
    slice_axis=2,
    low_min=0.3,
    mid_min=0.5,
    high_min=0.8,
    tiny_side_px=4,
    show_off_slice_markers=False,
):
    volume = np.asarray(image_vol, dtype=np.float32)
    draw_img = normalize_image_to_uint8(_volume_slice_2d(volume, slice_idx, slice_axis))
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)
    draw_img = _overlay_slice_boxes_scored(
        draw_img,
        slice_idx,
        boxes,
        labels,
        scores,
        slice_axis=slice_axis,
        low_min=low_min,
        mid_min=mid_min,
        high_min=high_min,
        tiny_side_px=tiny_side_px,
        show_off_slice_markers=show_off_slice_markers,
    )
    return draw_img


def draw_slice_seg_overlay(image_vol, seg_vol, slice_idx, slice_axis=2, alpha=0.45):
    volume = np.asarray(image_vol, dtype=np.float32)
    if isinstance(seg_vol, torch.Tensor):
        seg = to_numpy(seg_vol)
    else:
        seg = np.asarray(seg_vol)
    if seg.ndim == 4:
        if seg.shape[0] == 1:
            seg = seg[0]
        else:
            seg = np.argmax(seg, axis=0)
    draw_img = normalize_image_to_uint8(_volume_slice_2d(volume, slice_idx, slice_axis))
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)
    seg_slice = _volume_slice_2d(seg, slice_idx, slice_axis) > 0
    color = np.array(colour_palette[1], dtype=np.float32)
    mask = seg_slice.astype(np.float32)[..., None]
    blended = draw_img.astype(np.float32)
    blended = blended * (1.0 - alpha * mask) + color * (alpha * mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _heatmap_color_blend(draw_img, disp, alpha=0.55):
    prob_u8 = np.clip(disp * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(prob_u8, cv2.COLORMAP_TURBO)
    mask = disp[..., None]
    blended = draw_img.astype(np.float32)
    blended = blended * (1.0 - alpha * mask) + heat_bgr.astype(np.float32) * (alpha * mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _add_box_overlap_count(counts, box, slice_axis, slice_idx):
    if not _box_visible(box, slice_axis, slice_idx):
        return
    pt1, pt2 = _slice_box_corners(box, slice_axis)
    x0, x1 = min(pt1[0], pt2[0]), max(pt1[0], pt2[0])
    y0, y1 = min(pt1[1], pt2[1]), max(pt1[1], pt2[1])
    h, w = counts.shape
    xa, xb = max(0, x0), min(w, x1 + 1)
    ya, yb = max(0, y0), min(h, y1 + 1)
    counts[ya:yb, xa:xb] += 1


def draw_slice_box_overlap_heatmap(
    image_vol,
    slice_idx,
    boxes,
    slice_axis=2,
    alpha=0.55,
    count_cap=10,
    overlay_boxes=None,
    overlay_labels=None,
    overlay_scores=None,
    overlay_low_min=0.3,
    overlay_mid_min=0.5,
    overlay_high_min=0.8,
    overlay_tiny_side_px=4,
):
    """Per-pixel bbox overlap count on a 2D slice; optional scored bbox overlay."""
    volume = np.asarray(image_vol, dtype=np.float32)
    draw_img = normalize_image_to_uint8(_volume_slice_2d(volume, slice_idx, slice_axis))
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)
    h, w = draw_img.shape[:2]
    boxes = _boxes_numpy(boxes)
    counts = np.zeros((h, w), dtype=np.int32)
    for i in range(boxes.shape[0]):
        _add_box_overlap_count(counts, boxes[i], slice_axis, slice_idx)
    max_count = int(counts.max())
    disp = np.zeros((h, w), dtype=np.float32)
    if max_count > 0:
        disp = np.clip(counts.astype(np.float32) / float(count_cap), 0, 1)
    panel = _heatmap_color_blend(draw_img, disp, alpha=alpha)
    if overlay_boxes is not None:
        overlay_boxes = _boxes_numpy(overlay_boxes)
        if overlay_boxes.shape[0] > 0:
            panel = _overlay_slice_boxes_scored(
                panel,
                slice_idx,
                overlay_boxes,
                overlay_labels,
                overlay_scores,
                slice_axis=slice_axis,
                low_min=overlay_low_min,
                mid_min=overlay_mid_min,
                high_min=overlay_high_min,
                tiny_side_px=overlay_tiny_side_px,
                show_off_slice_markers=True,
            )
    return panel, max_count


def _overlay_slice_gt_boxes(draw_img, slice_idx, boxes, labels, slice_axis=2, tiny_side_px=4):
    boxes = _boxes_numpy(boxes)
    labels = _labels_numpy(labels)
    for i in range(boxes.shape[0]):
        box = boxes[i]
        if not _box_visible(box, slice_axis, slice_idx):
            continue
        label = int(labels[i]) if i < len(labels) else 0
        pt1, pt2 = _slice_box_corners(box, slice_axis)
        color = class_color_bgr(label)
        _draw_gt_box_2d(draw_img, pt1, pt2, color, tiny_side_px=tiny_side_px)
    return draw_img


def draw_slice_seg_with_gt_boxes(
    image_vol, seg_vol, slice_idx, boxes, labels, slice_axis=2, alpha=0.45, tiny_side_px=4
):
    panel = draw_slice_seg_overlay(
        image_vol, seg_vol, slice_idx, slice_axis=slice_axis, alpha=alpha
    )
    panel = _overlay_slice_gt_boxes(
        panel, slice_idx, boxes, labels, slice_axis=slice_axis, tiny_side_px=tiny_side_px
    )
    return panel


def draw_slice_seg_with_pred_boxes_scored(
    image_vol,
    seg_vol,
    slice_idx,
    boxes,
    labels,
    scores,
    slice_axis=2,
    alpha=0.45,
    low_min=0.3,
    mid_min=0.5,
    high_min=0.8,
    tiny_side_px=4,
    show_off_slice_markers=False,
):
    panel = draw_slice_seg_overlay(
        image_vol, seg_vol, slice_idx, slice_axis=slice_axis, alpha=alpha
    )
    panel = _overlay_slice_boxes_scored(
        panel,
        slice_idx,
        boxes,
        labels,
        scores,
        slice_axis=slice_axis,
        low_min=low_min,
        mid_min=mid_min,
        high_min=high_min,
        tiny_side_px=tiny_side_px,
        show_off_slice_markers=show_off_slice_markers,
    )
    return panel


def draw_slice_boxes(image_vol, slice_idx, boxes, labels, slice_axis=2, tiny_side_px=4):
    volume = np.asarray(image_vol, dtype=np.float32)
    draw_img = normalize_image_to_uint8(_volume_slice_2d(volume, slice_idx, slice_axis))
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)
    draw_img = _overlay_slice_gt_boxes(
        draw_img, slice_idx, boxes, labels, slice_axis=slice_axis, tiny_side_px=tiny_side_px
    )
    return draw_img


def overlay_panel_label(panel_bgr, text):
    out = panel_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    text_w, text_h = cv2.getTextSize(text, font, scale, thickness)[0]
    cv2.rectangle(out, (0, 0), (text_w + 4, text_h + 6), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (2, text_h + 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return out


def annotate_snippet_grid(grid_rgb, case_ids, tile_w, tile_h, n_tiles, padding=2):
    out = grid_rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    for slot, case_id in enumerate(case_ids):
        row = slot // n_tiles
        triplet_col = slot % n_tiles
        x0 = padding + triplet_col * 3 * (tile_w + padding)
        y0 = padding + row * (tile_h + padding)
        x1 = padding + (triplet_col * 3 + 3) * (tile_w + padding) - padding
        y1 = y0 + tile_h
        text = str(case_id)
        text_w, text_h = cv2.getTextSize(text, font, scale, thickness)[0]
        band_h = text_h + 6
        band_y0 = max(y0, y1 - band_h)
        text_x = x0 + max(0, (x1 - x0 - text_w) // 2)
        text_y = band_y0 + text_h + 2
        cv2.rectangle(out, (x0, band_y0), (x1, y1), (0, 0, 0), -1)
        cv2.putText(
            out,
            text,
            (text_x, text_y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return out


def overlay_grid_stage_banner(grid_rgb, stage):
    out = grid_rgb.copy()
    text = str(stage).upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.49
    thickness = 1
    text_w, text_h = cv2.getTextSize(text, font, scale, thickness)[0]
    cv2.rectangle(out, (0, 0), (text_w + 6, text_h + 6), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (3, text_h + 3),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return out


def visualize_one_xy_slice_in_3d_image(gt_boxes, image, pred_boxes, gt_box_index=0):
    """
    Prepare a 2D xy-plane image slice from a 3D image for visualization.
    It draws the (gt_box_index)-th GT box and predicted boxes on the same slice.
    The GT box will be green rect overlayed on the image.
    The predicted boxes will be red boxes overlayed on the image.

    Args:
        gt_boxes: numpy sized (M, 6)
        image: image numpy array, sized (H, W, D)
        pred_boxes: numpy array sized (N, 6)
    """
    draw_box = gt_boxes[gt_box_index, :]
    draw_box_center = [round((draw_box[axis] + draw_box[axis + 3] - 1) / 2.0) for axis in range(3)]
    draw_box = np.round(draw_box).astype(int).tolist()
    draw_box_z = draw_box_center[2]  # the z-slice we will visualize

    # draw image
    draw_img = normalize_image_to_uint8(image[:, :, draw_box_z])
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)

    # draw GT box, notice that cv2 uses Cartesian indexing instead of Matrix indexing.
    # so the xy position needs to be transposed.
    cv2.rectangle(
        draw_img,
        pt1=(draw_box[1], draw_box[0]),
        pt2=(draw_box[4], draw_box[3]),
        color=(0, 255, 0),  # green for GT
        thickness=1,
    )
    # draw predicted boxes
    for bbox in pred_boxes:
        bbox = np.round(bbox).astype(int).tolist()
        if bbox[5] < draw_box[2] or bbox[2] > draw_box[5]:
            continue
        cv2.rectangle(
            draw_img,
            pt1=(bbox[1], bbox[0]),
            pt2=(bbox[4], bbox[3]),
            color=(255, 0, 0),  # red for predicted box
            thickness=1,
        )
    return draw_img
