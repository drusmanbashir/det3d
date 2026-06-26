from det3d.managers.data.collate import attach_targets
from monai.transforms.transform import MapTransform
import torch
from monai.apps.detection.transforms.dictionary import ClipBoxToImaged
from monai.data import MetaTensor
from monai.transforms import Compose
from monai.transforms.croppad.array import Crop
from monai.transforms.croppad.dictionary import ResizeWithPadOrCropd
from monai.transforms.spatial.dictionary import ConvertPointsToBoxesd, ConvertBoxToPointsd, RandAffined, RandFlipd
from monai.transforms.utility.array import ApplyTransformToPoints
from monai.transforms.utils import to_affine_nd
from monai.utils import fall_back_tuple
from monai.utils.type_conversion import convert_to_dst_type


def _resize_pad_crop_voxel_shift(pre_spatial, spatial_size):
    """Voxel-index shift for boxes when ResizeWithPadOrCrop center-crops then pads."""
    # AI
    pre = tuple(int(v) for v in pre_spatial)
    target = fall_back_tuple(tuple(int(v) for v in spatial_size), pre)
    roi_center = [i // 2 for i in pre]
    slices = Crop.compute_slices(roi_center=roi_center, roi_size=target)
    crop_start = [int(s.start) for s in slices]
    cropped_shape = [int(s.stop - s.start) for s in slices]
    pad_before = [(int(target[i]) - cropped_shape[i]) // 2 for i in range(3)]
    shift = torch.tensor(
        [pad_before[i] - crop_start[i] for i in range(3)], dtype=torch.float64
    )
    return shift


def _warp_box_corners(corners, a0, aff):
    dev = corners.device
    a0 = a0.to(device=dev, dtype=torch.float64)
    aff = aff.to(device=dev, dtype=torch.float64)
    final_affine = torch.linalg.inv(aff) @ a0
    return apply_affine_to_points_gpu(corners, final_affine, dtype=torch.float32)


class ResizeWithPadOrCropBoxSyncd(MapTransform):
    """ResizeWithPadOrCropd + shift box coords by center-crop / symmetric-pad offset."""

    def __init__(
        self,
        keys,
        box_key,
        label_key,
        spatial_size,
        mode=None,
        lazy=False,
        allow_missing_keys=False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.box_key = box_key
        self.label_key = label_key
        self.spatial_size = tuple(int(v) for v in spatial_size)
        self.resize = ResizeWithPadOrCropd(
            keys=keys,
            spatial_size=self.spatial_size,
            lazy=lazy,
        )

    def __call__(self, data):
        d = dict(data)
        ref = d[self.keys[0]]
        pre = tuple(int(v) for v in ref.shape[-3:])
        d = self.resize(d)
        shift = _resize_pad_crop_voxel_shift(pre, self.spatial_size)
        if float(shift.abs().sum()) == 0.0:
            return d
        box = torch.as_tensor(d[self.box_key], dtype=torch.float32)
        if box.numel() == 0:
            return d
        shift = shift.to(device=box.device, dtype=box.dtype)
        box = box.clone()
        for i in range(3):
            box[:, i] += shift[i]
            box[:, i + 3] += shift[i]
        d[self.box_key] = box
        return d


class RandAffineBoxSyncd(MapTransform):
    """RandAffine on spatial keys + warp patch-voxel boxes via 8-corner affine delta."""

    _corner_key = "__box_corners__"

    def __init__(
        self,
        spatial_keys,
        box_key,
        mode,
        prob,
        rotate_range,
        scale_range,
    ):
        super().__init__(spatial_keys)
        self.box_key = box_key
        self.ref_key = spatial_keys[0]
        self.rand_affine = RandAffined(
            keys=spatial_keys,
            mode=mode,
            prob=prob,
            rotate_range=rotate_range,
            scale_range=scale_range,
        )

    def __call__(self, data):
        d = dict(data)
        a0 = d[self.ref_key].meta["affine"].clone()
        d = self.rand_affine(d)
        d = SyncMetaAffined(keys=self.keys)(d)
        box = d[self.box_key]
        if box.numel() == 0:
            return d
        d = ConvertBoxToPointsd(
            keys=[self.box_key], point_key=self._corner_key
        )(d)
        corners = d[self._corner_key].to(d[self.ref_key].device)
        warped = _warp_box_corners(
            corners, a0, d[self.ref_key].meta["affine"]
        )
        d[self._corner_key] = MetaTensor(warped, meta=corners.meta)
        d = ConvertPointsToBoxesd(
            keys=[self._corner_key], box_key=self.box_key
        )(d)
        del d[self._corner_key]
        return d


class RandFlipBoxSyncd(MapTransform):
    """RandFlip on spatial keys + sync patch-voxel boxes via 8-corner affine delta."""

    _corner_key = "__box_corners__"

    def __init__(self, spatial_keys, box_key, prob, spatial_axis):
        super().__init__(spatial_keys)
        self.box_key = box_key
        self.ref_key = spatial_keys[0]
        self.rand_flip = RandFlipd(
            keys=spatial_keys,
            prob=prob,
            spatial_axis=spatial_axis,
        )

    def __call__(self, data):
        d = dict(data)
        a0 = d[self.ref_key].meta["affine"].clone()
        d = self.rand_flip(d)
        d = SyncMetaAffined(keys=self.keys)(d)
        box = d[self.box_key]
        if box.numel() == 0:
            return d
        d = ConvertBoxToPointsd(
            keys=[self.box_key], point_key=self._corner_key
        )(d)
        corners = d[self._corner_key].to(d[self.ref_key].device)
        warped = _warp_box_corners(
            corners, a0, d[self.ref_key].meta["affine"]
        )
        d[self._corner_key] = MetaTensor(warped, meta=corners.meta)
        d = ConvertPointsToBoxesd(
            keys=[self._corner_key], box_key=self.box_key
        )(d)
        del d[self._corner_key]
        return d


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
        from monai.transforms.utility.array import ApplyTransformToPoints

        self.converter = ApplyTransformToPoints(
            invert_affine=True,
            affine_lps_to_ras=affine_lps_to_ras,
        )

    def __call__(self, data):
        d = dict(data)
        for key, refer_key in zip(self.keys, self.refer_keys):
            coords = d[key]
            refer = d[refer_key]
            affine = refer.meta["affine"]
            d[key] = self.converter(coords, affine)
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
        lm_key=None,
    ):
        self.tfms = tfms
        self.image_key = image_key
        self.box_key = box_key
        self.label_key = label_key
        self.point_key = point_key
        self.mask_key = mask_key
        self.lm_key = lm_key

    def __call__(self, batch):
        d = dict(batch)
        n = d[self.image_key].shape[0]
        items = []
        passthrough_keys = (
            self.point_key,
            self.label_key,
            self.box_key,
        )
        if self.mask_key is not None:
            passthrough_keys = (*passthrough_keys, self.mask_key)
        if self.lm_key is not None:
            passthrough_keys = (*passthrough_keys, self.lm_key)
        if "instances" in d:
            passthrough_keys = (*passthrough_keys, "instances")
        for i in range(n):
            item = {self.image_key: d[self.image_key][i]}
            for key in passthrough_keys:
                if key not in d:
                    continue
                val = d[key]
                if isinstance(val, list):
                    item[key] = val[i]
                elif torch.is_tensor(val) and val.shape[0] == n:
                    item[key] = val[i]
                else:
                    item[key] = val
            ref = item[self.image_key]
            dev = ref.device if torch.is_tensor(ref) else None
            if dev is not None:
                for key in passthrough_keys:
                    t = item.get(key)
                    if torch.is_tensor(t):
                        item[key] = t.to(dev)
            items.append(self.tfms(item))
        d[self.image_key] = torch.stack([it[self.image_key] for it in items], 0)
        d[self.box_key] = [it[self.box_key] for it in items]
        d[self.label_key] = [it[self.label_key] for it in items]
        if self.mask_key is not None and self.mask_key in items[0]:
            d[self.mask_key] = torch.stack([it[self.mask_key] for it in items], 0)
        if self.lm_key is not None and self.lm_key in items[0]:
            d[self.lm_key] = torch.stack([it[self.lm_key] for it in items], 0)
        if "instances" in d:
            inst_in = d["instances"]
            d["instances"] = [
                it["instances"]
                if "instances" in it
                else inst_in[i]
                if isinstance(inst_in, list)
                else inst_in
                for i, it in enumerate(items)
            ]
        if self.point_key in d:
            del d[self.point_key]
        return attach_targets(d, self.box_key, self.label_key)


class PreTrafoBatchItemCompose:
    """GPU tail for pre_trafo: spatial aug on image+lm only (no box path)."""

    def __init__(
        self,
        tfms,
        image_key="image",
        label_key="label",
        lm_key="lm",
        instances_key="instances",
    ):
        self.tfms = tfms
        self.image_key = image_key
        self.label_key = label_key
        self.lm_key = lm_key
        self.instances_key = instances_key

    def __call__(self, batch):
        d = dict(batch)
        n = d[self.image_key].shape[0]
        items = []
        passthrough_keys = (self.label_key, self.lm_key, self.instances_key)
        for i in range(n):
            item = {self.image_key: d[self.image_key][i]}
            for key in passthrough_keys:
                val = d[key]
                if isinstance(val, list):
                    item[key] = val[i]
                elif torch.is_tensor(val) and val.shape[0] == n:
                    item[key] = val[i]
                else:
                    item[key] = val
            items.append(self.tfms(item))
        d[self.image_key] = torch.stack([it[self.image_key] for it in items], 0)
        d[self.label_key] = [it[self.label_key] for it in items]
        d[self.lm_key] = torch.stack([it[self.lm_key] for it in items], 0)
        d[self.instances_key] = [it[self.instances_key] for it in items]
        return d


def build_train_gpu_tail_compose_pre_trafo(
    *,
    image_key,
    lm_key,
    intensity_tfms,
    affine3d,
    patch_size,
    flip_prob,
    spatial_prob=1.0,
):
    p = float(spatial_prob)
    spatial_keys = [image_key, lm_key]
    flip_p = float(flip_prob)
    return Compose(
        [
            RandFlipd(
                keys=spatial_keys,
                prob=flip_p,
                spatial_axis=0,
            ),
            RandFlipd(
                keys=spatial_keys,
                prob=flip_p,
                spatial_axis=1,
            ),
            RandAffined(
                keys=spatial_keys,
                mode=["bilinear", "nearest"],
                prob=float(affine3d["p"]) * p,
                rotate_range=affine3d["rotate_range"],
                scale_range=affine3d["scale_range"],
            ),
            SyncMetaAffined(keys=spatial_keys),
            ResizeWithPadOrCropd(
                keys=spatial_keys,
                spatial_size=tuple(int(v) for v in patch_size),
                lazy=False,
            ),
            *intensity_tfms,
        ]
    )


def build_train_gpu_tail_compose(
    *,
    device,
    image_key,
    box_key,
    label_key,
    point_key,
    mask_key=None,
    lm_key=None,
    affine_lps_to_ras,
    intensity_tfms,
    affine3d,
    patch_size,
    flip_prob,
    spatial_prob=1.0,
):
    p = float(spatial_prob)
    spatial_keys = [image_key]
    affine_mode = ["bilinear"]
    if mask_key is not None:
        spatial_keys.append(mask_key)
        affine_mode.append("nearest")
    if lm_key is not None:
        spatial_keys.append(lm_key)
        affine_mode.append("nearest")
    flip_p = float(flip_prob)
    return Compose(
        [
            RandFlipBoxSyncd(
                spatial_keys=spatial_keys,
                box_key=box_key,
                prob=flip_p,
                spatial_axis=0,
            ),
            RandFlipBoxSyncd(
                spatial_keys=spatial_keys,
                box_key=box_key,
                prob=flip_p,
                spatial_axis=1,
            ),
            RandAffineBoxSyncd(
                spatial_keys=spatial_keys,
                box_key=box_key,
                mode=affine_mode,
                prob=float(affine3d["p"]) * p,
                rotate_range=affine3d["rotate_range"],
                scale_range=affine3d["scale_range"],
            ),
            ResizeWithPadOrCropBoxSyncd(
                keys=spatial_keys,
                box_key=box_key,
                label_key=label_key,
                spatial_size=tuple(int(v) for v in patch_size),
                lazy=False,
            ),
            ClipBoxToImaged(
                box_keys=box_key,
                label_keys=[label_key],
                box_ref_image_keys=image_key,
                remove_empty=True,
            ),
            *intensity_tfms,
        ]
    )
