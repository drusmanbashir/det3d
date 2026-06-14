from det3d.collate import attach_targets
from monai.transforms.transform import MapTransform
import numpy as np
import torch
from monai.apps.detection.transforms.dictionary import ClipBoxToImaged
from monai.data import MetaTensor
from monai.transforms import Compose, DeleteItemsd, EnsureTyped, RandFlipd, RandRotate90d, RandRotated, RandZoomd
from monai.transforms.spatial.dictionary import ConvertPointsToBoxesd
from monai.transforms.utils import to_affine_nd
from monai.utils.type_conversion import convert_to_dst_type


def apply_affine_to_points_gpu(data, affine, dtype=torch.float64):
    data_ = data.to(dtype=torch.float64)
    affine = to_affine_nd(
        data_.shape[-1], affine.to(device=data_.device, dtype=torch.float64)
    )
    ones = torch.ones(
        (data_.shape[0], data_.shape[1], 1),
        device=data_.device,
        dtype=torch.float64,
    )
    homogeneous = torch.cat((data_, ones), dim=2)
    transformed = torch.matmul(homogeneous, affine.T)
    out, *_ = convert_to_dst_type(transformed[:, :, :-1], data, dtype=dtype)
    return out


class DetToDeviced(MapTransform):
    def __init__(self, keys, device):
        super().__init__(keys)
        self.device = device

    def _to_dev(self, t):
        t = t.to(self.device)
        if isinstance(t, MetaTensor):
            meta = {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in dict(t.meta).items()
            }
            t = MetaTensor(t, meta=meta)
        return t

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self._to_dev(d[key])
        return d


class SyncMetaAffined(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            val = d[key]
            if not isinstance(val, MetaTensor):
                continue
            dev = val.device
            meta = {
                k: (v.to(dev) if torch.is_tensor(v) else v)
                for k, v in dict(val.meta).items()
            }
            d[key] = MetaTensor(val, meta=meta)
        return d


class SyncPointsMetaToImaged(MapTransform):
    """After CPU ApplyTransformToPointsd, point coords match image but meta affine can lag."""

    def __init__(self, point_key, image_key):
        super().__init__([point_key])
        self.point_key = point_key
        self.image_key = image_key

    def __call__(self, data):
        d = dict(data)
        points = d[self.point_key]
        image = d[self.image_key]
        meta = dict(points.meta)
        meta["affine"] = image.meta["affine"].to(
            device=points.device, dtype=image.meta["affine"].dtype
        )
        d[self.point_key] = MetaTensor(points, meta=meta)
        return d


class GpuApplyTransformToPointsd(MapTransform):
    def __init__(self, keys, refer_keys, affine_lps_to_ras=False):
        super().__init__(keys)
        self.refer_keys = refer_keys if isinstance(refer_keys, tuple) else (refer_keys,)
        self.affine_lps_to_ras = affine_lps_to_ras

    def __call__(self, data):
        d = dict(data)
        for key, refer_key in zip(self.keys, self.refer_keys):
            coords = d[key]
            refer = d[refer_key]
            affine = refer.meta["affine"]
            applied = coords.meta["affine"]
            affine = affine.to(device=coords.device, dtype=torch.float64)
            applied = applied.to(device=coords.device, dtype=torch.float64)
            final = torch.linalg.inv(affine) @ applied
            d[key] = apply_affine_to_points_gpu(coords, final)
        return d


class BatchItemCompose:
    """Apply item-level transforms per batch index (GPU tail path)."""

    def __init__(
        self,
        tfms,
        image_key="image",
        box_key="bbox",
        label_key="label",
        point_key="points",
        mask_key="mask",
    ):
        self.tfms = tfms
        self.image_key = image_key
        self.box_key = box_key
        self.label_key = label_key
        self.point_key = point_key
        self.mask_key = mask_key

    def __call__(self, batch):
        d = dict(batch)
        n = d[self.image_key].shape[0]
        items = []
        passthrough_keys = (self.point_key, self.label_key, self.mask_key, self.box_key)
        for i in range(n):
            item = {self.image_key: d[self.image_key][i]}
            for key in passthrough_keys:
                if key not in d:
                    continue
                val = d[key]
                if isinstance(val, list):
                    item[key] = val[i]
                else:
                    item[key] = val
            items.append(self.tfms(item))
        d[self.image_key] = torch.stack([it[self.image_key] for it in items], 0)
        d[self.box_key] = [it[self.box_key] for it in items]
        d[self.label_key] = [it[self.label_key] for it in items]
        if self.point_key in d:
            del d[self.point_key]
        if self.mask_key in d:
            del d[self.mask_key]
        return attach_targets(d, self.box_key, self.label_key)


def build_train_gpu_tail_compose(
    *,
    device,
    image_key,
    box_key,
    label_key,
    point_key,
    mask_key,
    affine_lps_to_ras,
    compute_dtype,
    intensity_tfms,
    spatial_prob=1.0,
):
    p = float(spatial_prob)
    return Compose(
        [
            DetToDeviced(keys=[image_key, point_key, label_key], device=device),
            SyncPointsMetaToImaged(point_key=point_key, image_key=image_key),
            RandZoomd(
                keys=[image_key],
                prob=0.2 * p,
                min_zoom=0.7,
                max_zoom=1.4,
                padding_mode="constant",
                keep_size=True,
            ),
            RandFlipd(keys=[image_key], prob=0.5 * p, spatial_axis=0),
            RandFlipd(keys=[image_key], prob=0.5 * p, spatial_axis=1),
            RandFlipd(keys=[image_key], prob=0.5 * p, spatial_axis=2),
            RandRotate90d(
                keys=[image_key], prob=0.75 * p, max_k=3, spatial_axes=(0, 1)
            ),
            RandRotated(
                keys=[image_key],
                mode="nearest",
                prob=0.2 * p,
                range_x=np.pi / 6,
                range_y=np.pi / 6,
                range_z=np.pi / 6,
                keep_size=True,
                padding_mode="zeros",
            ),
            SyncMetaAffined(keys=[image_key, point_key]),
            GpuApplyTransformToPointsd(
                keys=[point_key],
                refer_keys=image_key,
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            ConvertPointsToBoxesd(keys=[point_key], box_key=box_key),
            ClipBoxToImaged(
                box_keys=box_key,
                label_keys=[label_key],
                box_ref_image_keys=image_key,
                remove_empty=True,
            ),
            DeleteItemsd(keys=[mask_key]),
            *intensity_tfms,
            Compose(
                [
                    EnsureTyped(keys=[image_key], dtype=compute_dtype),
                    EnsureTyped(keys=[box_key], dtype=torch.float32),
                    EnsureTyped(keys=[label_key], dtype=torch.long),
                ]
            ),
        ]
    )
