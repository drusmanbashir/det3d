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
from fran.utils.colour_palette import colour_palette


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
        boxes = boxes.detach().cpu().numpy()
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
        labels = labels.detach().cpu().numpy()
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


def _volume_slice_2d(volume, slice_idx, slice_axis):
    if slice_axis == 0:
        return volume[slice_idx]
    if slice_axis == 1:
        return volume[:, slice_idx]
    return volume[:, :, slice_idx]


def draw_slice_boxes(image_vol, slice_idx, boxes, labels, slice_axis=2):
    volume = np.asarray(image_vol, dtype=np.float32)
    draw_img = normalize_image_to_uint8(_volume_slice_2d(volume, slice_idx, slice_axis))
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)
    boxes = _boxes_numpy(boxes)
    labels = _labels_numpy(labels)
    for i in range(boxes.shape[0]):
        box = boxes[i]
        if not _box_visible(box, slice_axis, slice_idx):
            continue
        label = int(labels[i]) if i < len(labels) else 0
        box_i = np.round(box).astype(int).tolist()
        color = class_color_bgr(label)
        cv2.rectangle(
            draw_img,
            pt1=(box_i[1], box_i[0]),
            pt2=(box_i[4], box_i[3]),
            color=color,
            thickness=1,
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
