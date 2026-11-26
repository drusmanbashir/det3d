from det3d.managers.data.collate import attach_targets
from det3d.transforms.box_tracker import CornerTracker, voxel_map
from fran.transforms.spatialtransforms import IndependentRandAffined, RandMirrord, affine_draw_kwargs
from monai.transforms.transform import MapTransform
import torch
from monai.data import MetaTensor
from monai.transforms import Compose
from monai.transforms.croppad.array import Crop
from monai.transforms.croppad.dictionary import ResizeWithPadOrCropd
from monai.utils import fall_back_tuple


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
        tracker=None,
    ):
        super().__init__(keys, allow_missing_keys)
        self.box_key = box_key
        self.label_key = label_key
        self.tracker = tracker if tracker is not None else CornerTracker(box_key, label_key, keys[0])
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
        return self.tracker.translate(d, shift)


class RandAffineBoxSyncd(MapTransform):
    """RandAffine on spatial keys + synchronize box tracking in the new voxel grid."""

    def __init__(
        self,
        spatial_keys,
        box_key,
        mode,
        prob=None,
        rotate_range=None,
        scale_range=None,
        tracker=None,
        *,
        p_rotate=None,
        p_scale=None,
    ):
        super().__init__(spatial_keys)
        self.box_key = box_key
        self.ref_key = spatial_keys[0]
        self.tracker = tracker if tracker is not None else CornerTracker(box_key, image_key=self.ref_key)
        self.rand_affine = IndependentRandAffined(
            keys=spatial_keys,
            mode=mode,
            prob=prob,
            p_rotate=p_rotate,
            p_scale=p_scale,
            rotate_range=rotate_range,
            scale_range=scale_range,
            padding_mode="zeros",
        )

    def __call__(self, data):
        d = dict(data)
        a0 = d[self.ref_key].meta["affine"].clone()
        d = self.rand_affine(d)
        d = SyncMetaAffined(keys=self.keys)(d)
        ref = d[self.ref_key]
        return self.tracker.warp(d, voxel_map(a0, ref.meta["affine"], ref.device))



class RandMirrorBoxSyncd(RandMirrord):
    """Mirror spatial_keys (reference image first) and update box_key in the mirrored voxel grid.

    prob (0 to 1) applies independently to each axis in axes (a subset of 0, 1, 2, or empty).
    tracker defaults to CornerTracker; lazy=True queues the image flip and updates the tracker immediately.
    """

    def __init__(self, spatial_keys, box_key, prob=0.5, axes=(0, 1, 2), tracker=None, lazy=False):
        super().__init__(spatial_keys, prob, axes, lazy=lazy)
        self.ref_key = self.keys[0]
        self.tracker = tracker if tracker is not None else CornerTracker(box_key, image_key=self.ref_key)

    def __call__(self, data, lazy=None):
        lazy = self.lazy if lazy is None else lazy
        d = super().__call__(data, lazy=lazy)
        if not self.selected_axes:
            return d
        before = data[self.ref_key].peek_pending_affine()
        if not lazy:
            d = SyncMetaAffined(keys=self.keys)(d)
        ref = d[self.ref_key]
        return self.tracker.warp(d, voxel_map(before, ref.peek_pending_affine(), ref.device))


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
        passthrough_keys += tuple(k for k in ("hull_xyz", "inst", "inst_labels") if k in d)
        if "df" in d:
            passthrough_keys = (*passthrough_keys, "df")
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
        lm_key = self.lm_key or "lm"
        if lm_key in items[0]:
            d[lm_key] = torch.stack([it[lm_key] for it in items], 0)
        for key in ("hull_xyz", "inst", "inst_labels"):
            d.pop(key, None)
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
        if "df" in items[0]:
            d["df"] = [it["df"] for it in items]
        if self.point_key in d:
            del d[self.point_key]
        return attach_targets(d, self.box_key, self.label_key)


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
    mirror,
    spatial_prob=1.0,
    tracker=None,
    inst_key=None,
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
    if inst_key is not None:
        spatial_keys.append(inst_key)
        affine_mode.append("nearest")
    if tracker is None:
        tracker = CornerTracker(box_key, label_key, image_key)
    return Compose(
        [
            RandAffineBoxSyncd(
                spatial_keys=spatial_keys,
                box_key=box_key,
                mode=affine_mode,
                **affine_draw_kwargs(affine3d, p),
                tracker=tracker,
            ),
            ResizeWithPadOrCropBoxSyncd(
                keys=spatial_keys,
                box_key=box_key,
                label_key=label_key,
                spatial_size=tuple(int(v) for v in patch_size),
                lazy=False,
                tracker=tracker,
            ),
            *intensity_tfms,
            mirror,
            tracker,
        ]
    )
