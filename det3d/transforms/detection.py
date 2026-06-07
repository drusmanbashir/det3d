# Copyright (c) MONAI Consortium — vendored from Project-MONAI/tutorials/detection

import numpy as np
import torch
from monai.apps.detection.transforms.box_ops import convert_box_to_mask
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated,
    BoxToMaskd,
    ClipBoxToImaged,
    ConvertBoxToStandardModed,
    MaskToBoxd,
    StandardizeEmptyBoxd,
)
from monai.config import KeysCollection
from monai.data.box_utils import clip_boxes_to_image
from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
)
from monai.transforms.spatial.dictionary import ConvertBoxToPointsd, ConvertPointsToBoxesd
from monai.transforms.utility.dictionary import ApplyTransformToPointsd
from monai.utils.type_conversion import convert_data_type


def generate_detection_train_transform(
    image_key,
    box_key,
    label_key,
    gt_box_mode,
    intensity_transform,
    patch_size,
    batch_size,
    point_key="points",
    affine_lps_to_ras=True,
    amp=True,
):
    compute_dtype = torch.float16 if amp else torch.float32
    return Compose(
        [
            LoadImaged(keys=[image_key], image_only=False, meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=torch.long),
            StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
            Orientationd(keys=[image_key], axcodes="RAS"),
            intensity_transform,
            EnsureTyped(keys=[image_key], dtype=torch.float16),
            ConvertBoxToStandardModed(box_keys=[box_key], mode=gt_box_mode),
            ConvertBoxToPointsd(keys=[box_key]),
            AffineBoxToImageCoordinated(
                box_keys=[box_key],
                box_ref_image_keys=image_key,
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            GenerateExtendedBoxMask(
                keys=box_key,
                image_key=image_key,
                spatial_size=patch_size,
                whole_box=True,
            ),
            RandCropByPosNegLabeld(
                keys=[image_key],
                label_key="mask_image",
                spatial_size=patch_size,
                num_samples=batch_size,
                pos=1,
                neg=1,
            ),
            RandZoomd(
                keys=[image_key],
                prob=0.2,
                min_zoom=0.7,
                max_zoom=1.4,
                padding_mode="constant",
                keep_size=True,
            ),
            RandFlipd(keys=[image_key], prob=0.5, spatial_axis=0),
            RandFlipd(keys=[image_key], prob=0.5, spatial_axis=1),
            RandFlipd(keys=[image_key], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=[image_key], prob=0.75, max_k=3, spatial_axes=(0, 1)),
            ApplyTransformToPointsd(
                keys=[point_key], refer_keys=image_key, affine_lps_to_ras=affine_lps_to_ras
            ),
            ConvertPointsToBoxesd(keys=[point_key]),
            ClipBoxToImaged(
                box_keys=box_key,
                label_keys=[label_key],
                box_ref_image_keys=image_key,
                remove_empty=True,
            ),
            BoxToMaskd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                box_ref_image_keys=image_key,
                min_fg_label=0,
                ellipse_mask=True,
            ),
            RandRotated(
                keys=[image_key, "box_mask"],
                mode=["nearest", "nearest"],
                prob=0.2,
                range_x=np.pi / 6,
                range_y=np.pi / 6,
                range_z=np.pi / 6,
                keep_size=True,
                padding_mode="zeros",
            ),
            MaskToBoxd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                min_fg_label=0,
            ),
            DeleteItemsd(keys=["box_mask"]),
            RandGaussianNoised(keys=[image_key], prob=0.1, mean=0, std=0.1),
            RandGaussianSmoothd(
                keys=[image_key],
                prob=0.1,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            RandScaleIntensityd(keys=[image_key], prob=0.15, factors=0.25),
            RandShiftIntensityd(keys=[image_key], prob=0.15, offsets=0.1),
            RandAdjustContrastd(keys=[image_key], prob=0.3, gamma=(0.7, 1.5)),
            EnsureTyped(keys=[image_key], dtype=compute_dtype),
            EnsureTyped(keys=[label_key], dtype=torch.long),
        ]
    )


def generate_detection_val_transform(
    image_key,
    box_key,
    label_key,
    gt_box_mode,
    intensity_transform,
    affine_lps_to_ras=True,
    amp=True,
):
    compute_dtype = torch.float16 if amp else torch.float32
    return Compose(
        [
            LoadImaged(keys=[image_key], image_only=False, meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=torch.long),
            StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
            Orientationd(keys=[image_key], axcodes="RAS"),
            intensity_transform,
            ConvertBoxToStandardModed(box_keys=[box_key], mode=gt_box_mode),
            AffineBoxToImageCoordinated(
                box_keys=[box_key],
                box_ref_image_keys=image_key,
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            EnsureTyped(keys=[image_key, box_key], dtype=compute_dtype),
            EnsureTyped(keys=label_key, dtype=torch.long),
        ]
    )


class GenerateExtendedBoxMask(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        image_key: str,
        spatial_size: tuple[int, int, int],
        whole_box: bool,
        mask_image_key: str = "mask_image",
    ) -> None:
        super().__init__(keys)
        self.image_key = image_key
        self.spatial_size = spatial_size
        self.whole_box = whole_box
        self.mask_image_key = mask_image_key

    def generate_fg_center_boxes_np(self, boxes, image_size, whole_box=True):
        spatial_dims = len(image_size)
        boxes_np, *_ = convert_data_type(boxes, np.ndarray)
        extended_boxes = np.zeros_like(boxes_np, dtype=int)
        boxes_start = np.ceil(boxes_np[:, :spatial_dims]).astype(int)
        boxes_stop = np.floor(boxes_np[:, spatial_dims:]).astype(int)
        for axis in range(spatial_dims):
            if not whole_box:
                extended_boxes[:, axis] = boxes_start[:, axis] - self.spatial_size[axis] // 2 + 1
                extended_boxes[:, axis + spatial_dims] = (
                    boxes_stop[:, axis] + self.spatial_size[axis] // 2 - 1
                )
            else:
                extended_boxes[:, axis] = boxes_stop[:, axis] - self.spatial_size[axis] // 2 - 1
                extended_boxes[:, axis] = np.minimum(extended_boxes[:, axis], boxes_start[:, axis])
                extended_boxes[:, axis + spatial_dims] = (
                    extended_boxes[:, axis] + self.spatial_size[axis] // 2
                )
                extended_boxes[:, axis + spatial_dims] = np.maximum(
                    extended_boxes[:, axis + spatial_dims], boxes_stop[:, axis]
                )
        extended_boxes, _ = clip_boxes_to_image(extended_boxes, image_size, remove_empty=True)
        return extended_boxes

    def generate_mask_img(self, boxes, image_size, whole_box=True):
        extended_boxes_np = self.generate_fg_center_boxes_np(boxes, image_size, whole_box)
        mask_img = convert_box_to_mask(
            extended_boxes_np,
            np.ones(extended_boxes_np.shape[0]),
            image_size,
            bg_label=0,
            ellipse_mask=False,
        )
        mask_img = np.amax(mask_img, axis=0, keepdims=True)[0:1, ...]
        return mask_img

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[self.image_key]
            boxes = d[key]
            d[self.mask_image_key] = self.generate_mask_img(
                boxes, image.shape[1:], whole_box=self.whole_box
            )
        return d
