from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from det3d.geometry.lmg import DetectionLabelMapGeometryPT
from det3d.managers.data.collate import adjust_boxes_for_pad
from det3d.inference.markups import save_inference_markups
from det3d.utils.bbox_sidecar import save_inference_sidecar
from fran.data.dataset import FillBBoxPatchesd
from fran.transforms.inferencetransforms import SqueezeListofListsd
from fran.transforms.spatialtransforms import RestoreOriginalOrientationd, ResizeToMetaSpatialShaped
from monai.config import KeysCollection
from monai.transforms import EnsureChannelFirstd, EnsureTyped, ScaleIntensityRanged
from monai.transforms.spatial.dictionary import ConvertBoxToPointsd, ConvertPointsToBoxesd, Orientationd, Spacingd
from monai.transforms.utility.dictionary import ApplyTransformToPointsd, CastToTyped
from monai.data.meta_tensor import MetaTensor
from monai.transforms.io.dictionary import SaveImaged
from monai.transforms.transform import MapTransform
from utilz.fileio import maybe_makedirs
from utilz.stringz import info_from_filename, strip_extension


def bbox_slice_starts(bounding_box):
    starts = []
    for sl in bounding_box[1:]:
        starts.append(int(sl.start) if isinstance(sl, slice) else int(sl))
    return starts


def encode_bounding_box(bounding_box):
    out = []
    for sl in bounding_box:
        if isinstance(sl, slice):
            out.append([int(sl.start), int(sl.stop)])
        else:
            out.append(int(sl))
    return out


class LoadInferImaged(MapTransform):
    """#AI Load path/.pt or pass through pre-loaded sitk / MetaTensor."""

    def __init__(self, keys=("image",)):
        super().__init__(keys=keys)
        from fran.inference.helpers import SmartImageLoader

        self.smart = SmartImageLoader(keys=list(keys))

    def __call__(self, data):
        import SimpleITK as sitk

        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if isinstance(img, (str, Path)):
                d = self.smart(d)
                continue
            if isinstance(img, (torch.Tensor, MetaTensor, sitk.Image)):
                continue
            d = self.smart(d)
        return d


class ScaleBoxToCropNatived(MapTransform):
    """Map pred boxes from preprocessed crop voxels to native crop voxels."""

    def __init__(
        self,
        box_keys: KeysCollection,
        image_key: str = "image",
        crop_shape_key: str = "crop_spatial_shape",
        box_mode: str = "nndet",
    ):
        super().__init__(box_keys)
        self.image_key = image_key
        self.crop_shape_key = crop_shape_key
        self.box_mode = box_mode

    def __call__(self, data):
        d = dict(data)
        pre = [int(v) for v in d[self.image_key].shape[-3:]]
        native = [int(v) for v in d[self.crop_shape_key]]
        scale_d = native[0] / pre[0]
        scale_h = native[1] / pre[1]
        scale_w = native[2] / pre[2]
        for key in self.key_iterator(d):
            box = torch.as_tensor(d[key], dtype=torch.float32).clone()
            if box.numel() == 0:
                d[key] = box.reshape(0, 6)
                continue
            if box.ndim == 1:
                box = box.unsqueeze(0)
            if self.box_mode == "nndet":
                box[:, 0] *= scale_d
                box[:, 1] *= scale_h
                box[:, 2] *= scale_d
                box[:, 3] *= scale_h
                box[:, 4] *= scale_w
                box[:, 5] *= scale_w
            elif self.box_mode == "xxyyzz":
                box[:, 0] *= scale_w
                box[:, 1] *= scale_w
                box[:, 2] *= scale_h
                box[:, 3] *= scale_h
                box[:, 4] *= scale_d
                box[:, 5] *= scale_d
            else:
                raise ValueError(f"unsupported box_mode {self.box_mode!r}")
            d[key] = box
        return d


def _scale_det3d_exclusive_to_monai_native(box, pre_shape, native_shape):
    """#AI xyzxyz exclusive spaced crop -> xyzxyz exclusive native crop (x,y,z tensor axes)."""
    pre = [int(v) for v in pre_shape]
    native = [int(v) for v in native_shape]
    sx = native[0] / pre[0]
    sy = native[1] / pre[1]
    sz = native[2] / pre[2]
    monai = torch.empty_like(box)
    monai[:, 0] = box[:, 0] * sx
    monai[:, 1] = box[:, 1] * sy
    monai[:, 2] = box[:, 2] * sz
    monai[:, 3] = box[:, 3] * sx
    monai[:, 4] = box[:, 4] * sy
    monai[:, 5] = box[:, 5] * sz
    return monai


class PredBoxToNativeCropViaPointsd(MapTransform):
    """Scale nnDet pred boxes from spaced patch crop to native crop voxel coords."""

    def __init__(
        self,
        box_key: str = "pred_box",
        image_key: str = "image",
        crop_shape_key: str = "crop_spatial_shape",
        box_mode: str = "nndet",
    ):
        super().__init__(keys=[box_key])
        self.box_key = box_key
        self.image_key = image_key
        self.crop_shape_key = crop_shape_key
        self.box_mode = box_mode

    def __call__(self, data):
        d = dict(data)
        box = torch.as_tensor(d[self.box_key], dtype=torch.float32).detach().cpu()
        if box.numel() == 0:
            d[self.box_key] = box.reshape(0, 6)
            return d
        if box.ndim == 1:
            box = box.unsqueeze(0)
        pre = d[self.image_key].shape[-3:]
        native = d[self.crop_shape_key]
        if self.box_mode == "nndet":
            scaled = ScaleBoxToCropNatived(
                box_keys=[self.box_key],
                image_key=self.image_key,
                crop_shape_key=self.crop_shape_key,
                box_mode="nndet",
            )({self.box_key: box, self.image_key: d[self.image_key], self.crop_shape_key: native})
            d[self.box_key] = scaled[self.box_key]
            return d
        d[self.box_key] = _scale_det3d_exclusive_to_monai_native(box, pre, native)
        return d


class RetinaUNetBoxInversed(MapTransform):
    """#AI Inverse preprocess spatial tfms on nnDet pred boxes via corner points."""

    def __init__(
        self,
        box_key: str = "pred_box",
        image_key: str = "image",
        point_key: str = "pred_box_points",
    ):
        super().__init__(keys=[box_key])
        self.box_key = box_key
        self.image_key = image_key
        self.point_key = point_key

    def __call__(self, data):
        from det3d.detection.nndet_train import nndet_batch_to_xyzxyz

        d = dict(data)
        d[self.image_key] = d[self.image_key].detach().cpu()
        boxes = torch.as_tensor(d[self.box_key], dtype=torch.float32).detach().cpu()
        box = nndet_batch_to_xyzxyz(boxes)
        if box.numel() == 0:
            d[self.box_key] = box.reshape(0, 6)
            return d
        if box.ndim == 1:
            box = box.unsqueeze(0)
        monai = torch.empty_like(box)
        monai[:, 0] = box[:, 0]
        monai[:, 1] = box[:, 3]
        monai[:, 2] = box[:, 2]
        monai[:, 3] = box[:, 1]
        monai[:, 4] = box[:, 4]
        monai[:, 5] = box[:, 5]
        d[self.box_key] = monai
        d = ConvertBoxToPointsd(keys=[self.box_key], point_key=self.point_key)(d)
        d = ApplyTransformToPointsd(keys=[self.point_key], refer_keys=self.image_key)(d)
        d = ConvertPointsToBoxesd(keys=[self.point_key], box_key=self.box_key)(d)
        return d


class OffsetBoxByBBoxd(MapTransform):
    """Shift pred boxes from crop space to full-volume voxels."""

    def __init__(
        self,
        box_keys: KeysCollection,
        bbox_key: str = "bounding_box",
        box_mode: str = "xyzxyz",
        allow_missing_keys: bool = False,
    ):
        super().__init__(box_keys, allow_missing_keys)
        self.bbox_key = bbox_key
        self.box_mode = box_mode

    def __call__(self, data):
        from det3d.detection.nndet_train import offset_nndet_xyxyzz_boxes

        d = dict(data)
        if self.box_mode == "nndet":
            origin = bbox_slice_starts(d[self.bbox_key])
            for key in self.key_iterator(d):
                d[key] = offset_nndet_xyxyzz_boxes(
                    torch.as_tensor(d[key], dtype=torch.float32), origin
                )
            return d
        offsets = bbox_slice_starts(d[self.bbox_key])
        for key in self.key_iterator(d):
            d[key] = adjust_boxes_for_pad(d[key], offsets)
        return d


class UseFullMetaForImaged(MapTransform):
    """Point box world conversion at full-volume affine."""

    def __init__(self, keys: KeysCollection, full_meta_key: str = "full_meta"):
        super().__init__(keys)
        self.full_meta_key = full_meta_key

    def __call__(self, data):
        d = dict(data)
        full_meta = deepcopy(d[self.full_meta_key])
        for key in self.key_iterator(d):
            d[key].meta = full_meta
        return d


class CopyBoxKeyd(MapTransform):
    """Keep full-voxel xyzxyz boxes before gt_box_mode conversion."""

    def __init__(self, src_key: str, dst_key: str):
        super().__init__(keys=[src_key])
        self.dst_key = dst_key

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[self.dst_key] = torch.as_tensor(d[key]).clone()
        return d


class PreservePreTfmBoxd(MapTransform):
    """Snapshot model-output boxes before cascade post transforms."""

    def __init__(self, box_key: str = "pred_box", dst_key: str = "pred_box_pre_tfm"):
        super().__init__(keys=[box_key])
        self.dst_key = dst_key

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[self.dst_key] = torch.as_tensor(d[key]).clone()
        return d


class AttachInferenceMetad(MapTransform):
    def __init__(
        self,
        box_keys: KeysCollection,
        run_w: str,
        run_p: str,
        bbox_key: str = "bounding_box",
        full_meta_key: str = "full_meta",
        source_image_key: str = "source_image",
    ):
        super().__init__(box_keys)
        self.run_w = run_w
        self.run_p = run_p
        self.bbox_key = bbox_key
        self.full_meta_key = full_meta_key
        self.source_image_key = source_image_key

    def __call__(self, data):
        d = dict(data)
        source_image = d[self.source_image_key]
        fname = Path(str(source_image))
        case_id = info_from_filename(fname.name, full_caseid=True)["case_id"]
        full_meta = d[self.full_meta_key]
        d["case_id"] = case_id
        d["source_image"] = str(source_image)
        d["localiser_run"] = self.run_w
        d["det_run"] = self.run_p
        d["lbd_bounding_box"] = encode_bounding_box(d[self.bbox_key])
        d["spacing"] = [float(v) for v in full_meta["spacing"]]
        d["affine"] = torch.as_tensor(full_meta["affine"]).cpu().tolist()
        return d


class SaveInferenceSidecard(MapTransform):
    def __init__(
        self,
        box_keys: KeysCollection,
        label_key: str,
        score_key: str,
        output_dir: str | Path,
        voxel_box_key: str = "pred_box_voxel",
        world_box_key: str = "pred_box",
        score_min: float = 0.0,
    ):
        super().__init__(box_keys)
        self.label_key = label_key
        self.score_key = score_key
        self.output_dir = Path(output_dir)
        self.voxel_box_key = voxel_box_key
        self.world_box_key = world_box_key
        self.score_min = float(score_min)

    def __call__(self, data):
        d = dict(data)
        maybe_makedirs(self.output_dir)
        stem = strip_extension(Path(d["source_image"]).name)
        out_fn = self.output_dir / f"{stem}.json"
        boxes_voxel = torch.as_tensor(d[self.voxel_box_key])
        boxes_world = torch.as_tensor(d[self.world_box_key])
        labels = torch.as_tensor(d[self.label_key])
        scores = torch.as_tensor(d[self.score_key])
        if boxes_voxel.ndim == 1:
            boxes_voxel = boxes_voxel.unsqueeze(0)
        if boxes_world.ndim == 1:
            boxes_world = boxes_world.unsqueeze(0)
        if labels.ndim == 0:
            labels = labels.unsqueeze(0)
        if scores.ndim == 0:
            scores = scores.unsqueeze(0)
        keep = scores >= self.score_min
        if keep.ndim == 0:
            keep = keep.unsqueeze(0)
        boxes_voxel = boxes_voxel[keep]
        boxes_world = boxes_world[keep]
        labels = labels[keep]
        scores = scores[keep]
        save_inference_sidecar(
            out_fn,
            source_image=d["source_image"],
            case_id=d["case_id"],
            lbd_bounding_box=d["lbd_bounding_box"],
            localiser_run=d["localiser_run"],
            det_run=d["det_run"],
            spacing=d["spacing"],
            affine=d["affine"],
            boxes_voxel=boxes_voxel,
            boxes_world=boxes_world,
            labels=labels,
            scores=scores,
            boxes_pre_tfm=d["pred_box_pre_tfm"],
        )
        d["sidecar_path"] = str(out_fn)
        return d


class SaveInferenceMarkupsd(MapTransform):
    """#AI Write Slicer ROI Box `.mrk.json` from batch world boxes (RAS mm)."""

    def __init__(
        self,
        label_key: str,
        score_key: str,
        output_dir: str | Path,
        world_box_key: str = "pred_box_world",
        score_min: float = 0.0,
    ):
        super().__init__(keys=[world_box_key])
        self.label_key = label_key
        self.score_key = score_key
        self.output_dir = Path(output_dir)
        self.world_box_key = world_box_key
        self.score_min = float(score_min)

    def __call__(self, data):
        d = dict(data)
        maybe_makedirs(self.output_dir)
        stem = strip_extension(Path(d["source_image"]).name)
        out_fn = self.output_dir / f"{stem}.mrk.json"
        boxes = torch.as_tensor(d[self.world_box_key])
        labels = torch.as_tensor(d[self.label_key])
        scores = torch.as_tensor(d[self.score_key])
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)
        if labels.ndim == 0:
            labels = labels.unsqueeze(0)
        if scores.ndim == 0:
            scores = scores.unsqueeze(0)
        predictions = []
        for idx in range(boxes.shape[0]):
            if float(scores[idx]) < self.score_min:
                continue
            predictions.append(
                {
                    "bbox_world": [float(v) for v in boxes[idx].tolist()],
                    "label": int(labels[idx].item()),
                    "score": float(scores[idx].item()),
                }
            )
        sidecar = {
            "case_id": d["case_id"],
            "predictions": predictions,
        }
        save_inference_markups(out_fn, sidecar, score_min=self.score_min)
        d["markups_path"] = str(out_fn)
        return d


class ScaleSegToCropNatived(MapTransform):
    """Resize pred_seg from preprocessed crop voxels to native crop voxels."""

    def __init__(
        self,
        seg_key: str = "pred_seg",
        image_key: str = "image",
        crop_shape_key: str = "crop_spatial_shape",
    ):
        super().__init__(keys=[seg_key])
        self.seg_key = seg_key
        self.image_key = image_key
        self.crop_shape_key = crop_shape_key

    def __call__(self, data):
        d = dict(data)
        pre = [int(v) for v in d[self.image_key].shape[-3:]]
        native = [int(v) for v in d[self.crop_shape_key]]
        seg = torch.as_tensor(d[self.seg_key])
        if tuple(pre) == tuple(native):
            d[self.seg_key] = seg
            return d
        if seg.ndim == 3:
            seg = seg.unsqueeze(0).unsqueeze(0)
        elif seg.ndim == 4:
            seg = seg.unsqueeze(0)
        seg = F.interpolate(seg.float(), size=native, mode="nearest")
        seg = seg.squeeze(0).squeeze(0).to(torch.uint8)
        d[self.seg_key] = seg
        return d


class ArgmaxSegd(MapTransform):
    """Argmax / threshold seg logits or probs to uint8 label map."""

    def __init__(self, seg_key: str = "pred_seg", prob_thresh: float = 0.5):
        super().__init__(keys=[seg_key])
        self.seg_key = seg_key
        self.prob_thresh = float(prob_thresh)

    def __call__(self, data):
        d = dict(data)
        seg_t = torch.as_tensor(d[self.seg_key])
        if seg_t.dim() == 4:
            if int(seg_t.shape[0]) == 1:
                out = (seg_t[0] > self.prob_thresh).to(torch.uint8)
            else:
                out = seg_t.argmax(dim=0).to(torch.uint8)
        else:
            out = seg_t.to(torch.uint8)
        d[self.seg_key] = out
        return d


def _sitk_to_monai_seg(sitk_img):
    import numpy as np
    import SimpleITK as sitk

    arr = sitk.GetArrayFromImage(sitk_img)
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 1, 0).contiguous()
    return t.unsqueeze(0)


class DustSegd(MapTransform):
    """Remove connected components below dust_mm (major axis, physical mm)."""

    def __init__(
        self,
        seg_key: str = "pred_seg",
        dust_mm: float = 1.0,
        ignore_labels=None,
    ):
        super().__init__(keys=[seg_key])
        self.seg_key = seg_key
        self.dust_mm = float(dust_mm)
        self.ignore_labels = list(ignore_labels) if ignore_labels is not None else [0]

    def __call__(self, data):
        d = dict(data)
        seg = d[self.seg_key]
        meta = deepcopy(seg.meta)
        lm = seg.squeeze(0) if seg.dim() == 4 else seg
        lm = lm.as_tensor()
        L = DetectionLabelMapGeometryPT(
            li=lm,
            ignore_labels=self.ignore_labels,
            compute_feret=False,
        )
        L.dust(self.dust_mm)
        out = _sitk_to_monai_seg(L.li_cc_sitk).to(torch.uint8)
        d[self.seg_key] = MetaTensor(out.contiguous(), meta=meta)
        return d


class WrapPredSegMetad(MapTransform):
    """Attach crop image meta to pred_seg for FillBBox / RestoreOrientation."""

    def __init__(self, seg_key: str = "pred_seg", image_key: str = "image"):
        super().__init__(keys=[seg_key])
        self.seg_key = seg_key
        self.image_key = image_key

    def __call__(self, data):
        d = dict(data)
        seg = torch.as_tensor(d[self.seg_key])
        if seg.ndim == 3:
            seg = seg.unsqueeze(0)
        meta = deepcopy(d[self.image_key].meta)
        d[self.seg_key] = MetaTensor(seg.contiguous(), meta=meta)
        return d


class AttachPredSegPathd(MapTransform):
    """Record saved pred_seg NIfTI path on the batch dict."""

    def __init__(self, seg_key: str = "pred_seg", path_key: str = "pred_seg_nii"):
        super().__init__(keys=[seg_key])
        self.seg_key = seg_key
        self.path_key = path_key

    def __call__(self, data):
        d = dict(data)
        seg = d[self.seg_key]
        d[self.path_key] = str(seg.meta["filename_or_obj"])
        return d


def crop_around_boxes(image, sidecar, margin_mm=10.0):
    """Return cropped subvolumes around each prediction bbox (+/- margin)."""
    spacing = sidecar["spacing"]
    margin_vox = [int(margin_mm / sp) for sp in spacing]
    img = torch.as_tensor(image)
    crops = []
    for pred in sidecar["predictions"]:
        x1, y1, z1, x2, y2, z2 = [int(v) for v in pred["bbox_voxel_full"]]
        slc = (
            slice(max(x1 - margin_vox[0], 0), min(x2 + margin_vox[0], img.shape[-3])),
            slice(max(y1 - margin_vox[1], 0), min(y2 + margin_vox[1], img.shape[-2])),
            slice(max(z1 - margin_vox[2], 0), min(z2 + margin_vox[2], img.shape[-1])),
        )
        crops.append(img[(...,) + slc].contiguous())
    return crops


class NndetBoxToXyzxyzd(MapTransform):
    """Reorder nnDet xyxyzz pred_box to xyzxyz for MONAI world export (no +/-1)."""

    def __init__(self, box_key: str = "pred_box"):
        super().__init__(keys=[box_key])

    def __call__(self, data):
        from det3d.detection.nndet_train import nndet_batch_to_xyzxyz

        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = nndet_batch_to_xyzxyz(d[key])
        return d


class PackRetinaUNetPredsd(MapTransform):
    """Map predict_inner outputs to pred_box/label/score/seg.

    pred_box stays nnDet xyxyzz (same values as inference_step pred_boxes).
    """

    def __init__(self, keys=("image",)):
        super().__init__(keys=keys)

    def _pack_seg(self, seg):
        seg = seg.detach()
        if seg.dim() == 5:
            seg = seg[0]
        if seg.dim() == 4:
            if seg.shape[0] == 1:
                return (seg[0] > 0).to(torch.uint8)
            return seg.argmax(dim=0).to(torch.uint8)
        return (seg > 0).to(torch.uint8)

    def __call__(self, data):
        from det3d.detection.nndet_train import nndet_pred_to_vis

        d = dict(data)
        if "raw_pred" in d:
            vis = nndet_pred_to_vis(d.pop("raw_pred"))
            d["pred_box"] = vis["bbox"].detach()
            d["pred_label"] = vis["label"].detach()
            d["pred_score"] = vis["label_scores"].detach()
            seg = vis["pred_seg"].detach()
            if seg.dim() == 4:
                d["pred_seg_logits"] = seg
            d["pred_seg"] = self._pack_seg(seg)
            return d
        d["pred_box"] = d.pop("merged_boxes").detach()
        d["pred_label"] = d.pop("merged_labels").detach()
        d["pred_score"] = d.pop("merged_scores").detach()
        d["pred_seg"] = self._pack_seg(d.pop("stitched_seg"))
        return d
