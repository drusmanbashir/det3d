from __future__ import annotations

from copy import deepcopy
from functools import partial

import numpy as np
import torch
from fran.managers.data.valid_patch_stream import (
    PatchStreamDataset,
    _is_patch_padded,
    _pad_tensor_to_patch_size,
    _rewrite_manual_tail_padding_lm,
    _rewrite_padded_lm,
)
from monai.data import PatchIterd
from monai.data.box_utils import clip_boxes_to_image

from det3d.managers.data.collate import as_box_tensor, as_label_tensor, attach_targets


def _patch_start_pos(patch_out):
    start = patch_out.get("start_pos")
    if start and len(start) >= 3:
        return tuple(int(v) for v in start[:3])
    coords_arr = np.asarray(patch_out["patch_coords"])
    if coords_arr.ndim == 2 and coords_arr.shape[0] >= 4:
        coords_arr = coords_arr[1:]
    return tuple(int(coords_arr[ax, 0]) for ax in range(3))


def patch_stream_det_collated(batch, box_key="bbox", label_key="label", lm_key="lm"):
    images = []
    lms = []
    boxes = []
    labels = []
    case_ids = []
    patch_coords = []
    start_pos = []
    is_padded = []
    patch_index = []
    patches_in_case = []
    original_spatial_shape = []

    for item in batch:
        images.append(torch.as_tensor(item["image"]).contiguous())
        if lm_key is not None:
            lms.append(torch.as_tensor(item[lm_key]).contiguous())
        boxes.append(as_box_tensor(item[box_key]))
        labels.append(as_label_tensor(item[label_key]))
        case_ids.append(item["case_id"])
        patch_coords.append(item["patch_coords"])
        start_pos.append(item["start_pos"])
        is_padded.append(bool(item["is_padded"]))
        patch_index.append(int(item["patch_index"]))
        patches_in_case.append(int(item["patches_in_case"]))
        original_spatial_shape.append(tuple(int(v) for v in item["original_spatial_shape"]))

    out = {
        "image": torch.stack(images, 0),
        box_key: boxes,
        label_key: labels,
        "case_id": case_ids,
        "patch_coords": patch_coords,
        "start_pos": start_pos,
        "is_padded": is_padded,
        "patch_index": patch_index,
        "patches_in_case": patches_in_case,
        "original_spatial_shape": original_spatial_shape,
        "validation_impl": "patch_stream",
    }
    if lm_key is not None:
        out[lm_key] = lms
    return attach_targets(out, box_key, label_key)


class PatchStreamDatasetDet(PatchStreamDataset):
    """Patch-stream val over full LBD cases; bbox/label clipped per patch; lm optional."""

    def __init__(
        self,
        case_dataset,
        patch_size,
        image_key="image",
        lm_key="lm",
        box_key="bbox",
        label_key="label",
    ):
        self.case_dataset = case_dataset
        self.patch_size = tuple(int(v) for v in patch_size)
        self.image_key = image_key
        self.lm_key = lm_key
        self.box_key = box_key
        self.label_key = label_key
        patch_keys = [image_key]
        if lm_key is not None:
            patch_keys.append(lm_key)
        self.patch_iter = PatchIterd(
            keys=patch_keys,
            patch_size=self.patch_size,
            mode="constant",
            constant_values=0,
        )

    def __iter__(self):
        use_lm = self.lm_key is not None
        for case_ds_idx in range(len(self.case_dataset)):
            case_data = self.case_dataset[case_ds_idx]
            case_id = str(case_data["case_id"])
            case_box = torch.as_tensor(case_data[self.box_key], dtype=torch.float32)
            case_label = torch.as_tensor(case_data[self.label_key], dtype=torch.long)
            if case_box.ndim == 1:
                case_box = case_box.reshape(0, 6)
            case_patches = list(self.patch_iter(case_data))
            patches_in_case = len(case_patches)
            for patch_index, (patch_dict, coords) in enumerate(case_patches):
                patch_out = deepcopy(patch_dict)
                patch_out["case_id"] = case_id
                patch_out["patch_index"] = patch_index
                patch_out["patches_in_case"] = patches_in_case
                patch_out["validation_impl"] = "patch_stream"
                patch_out["is_padded"] = _is_patch_padded(
                    coords=patch_out["patch_coords"],
                    original_spatial_shape=patch_out["original_spatial_shape"],
                )

                vol_key = self.lm_key if use_lm else self.image_key
                original_patch_spatial_shape = tuple(
                    int(v) for v in patch_out[vol_key].shape[-3:]
                )
                image, image_was_padded = _pad_tensor_to_patch_size(
                    tensor=patch_out[self.image_key],
                    patch_size=self.patch_size,
                    pad_value=0,
                )
                patch_out[self.image_key] = image
                lm_was_padded = False
                was_padded = False
                was_padded_manual = False
                if use_lm:
                    lm, lm_was_padded = _pad_tensor_to_patch_size(
                        tensor=patch_out[self.lm_key],
                        patch_size=self.patch_size,
                        pad_value=0,
                    )
                    lm_rewritten, was_padded = _rewrite_padded_lm(
                        lm=lm,
                        coords=patch_out["patch_coords"],
                        original_spatial_shape=patch_out["original_spatial_shape"],
                    )
                    lm_rewritten, was_padded_manual = _rewrite_manual_tail_padding_lm(
                        lm=lm_rewritten,
                        original_patch_spatial_shape=original_patch_spatial_shape,
                    )
                    patch_out[self.lm_key] = lm_rewritten
                patch_out["is_padded"] = bool(
                    was_padded
                    or was_padded_manual
                    or image_was_padded
                    or lm_was_padded
                    or patch_out["is_padded"]
                )

                spatial = tuple(int(v) for v in patch_out[self.image_key].shape[-3:])
                start = _patch_start_pos(patch_out)
                if case_box.numel() == 0:
                    patch_out[self.box_key] = case_box
                    patch_out[self.label_key] = case_label
                else:
                    shifted = case_box.clone()
                    for ax in range(3):
                        shifted[:, ax] -= start[ax]
                        shifted[:, ax + 3] -= start[ax]
                    clipped, keep = clip_boxes_to_image(
                        shifted, spatial, remove_empty=True
                    )
                    patch_out[self.box_key] = clipped
                    patch_out[self.label_key] = case_label[keep]

                yield patch_out


def patch_stream_collate_fn(lm_key):
    return partial(patch_stream_det_collated, lm_key=lm_key)
