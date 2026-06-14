from copy import deepcopy
from pathlib import Path

import torch
from det3d.collate import adjust_boxes_for_pad
from det3d.utils.bbox_sidecar import save_inference_sidecar
from fran.transforms.inferencetransforms import SqueezeListofListsd
from monai.config import KeysCollection
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



class ScaleBoxToCropNatived(MapTransform):
    """Map pred boxes from preprocessed crop voxels to native crop voxels."""

    def __init__(
        self,
        box_keys: KeysCollection,
        image_key: str = "image",
        crop_shape_key: str = "crop_spatial_shape",
    ):
        super().__init__(box_keys)
        self.image_key = image_key
        self.crop_shape_key = crop_shape_key

    def __call__(self, data):
        d = dict(data)
        pre = [int(v) for v in d[self.image_key].shape[-3:]]
        native = [int(v) for v in d[self.crop_shape_key]]
        for key in self.key_iterator(d):
            box = torch.as_tensor(d[key], dtype=torch.float32).clone()
            scale = torch.tensor(
                [native[i] / pre[i] for i in range(3)], dtype=torch.float32, device=box.device
            )
            if box.numel() == 0:
                d[key] = box.reshape(0, 6)
                continue
            if box.ndim == 1:
                box = box.unsqueeze(0)
            box[:, :3] *= scale
            box[:, 3:6] *= scale
            d[key] = box
        return d


class OffsetBoxByBBoxd(MapTransform):
    """Shift xyzxyz boxes from LBD crop space to full-volume voxel space."""

    def __init__(
        self,
        box_keys: KeysCollection,
        bbox_key: str = "bounding_box",
        allow_missing_keys: bool = False,
    ):
        super().__init__(box_keys, allow_missing_keys)
        self.bbox_key = bbox_key

    def __call__(self, data):
        d = dict(data)
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
        if "pixdim" in full_meta:
            d["spacing"] = [float(v) for v in full_meta["pixdim"][1:4]]
        elif "spacing" in full_meta:
            d["spacing"] = [float(v) for v in full_meta["spacing"][:3]]
        else:
            d["spacing"] = [1.0, 1.0, 1.0]
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
    ):
        super().__init__(box_keys)
        self.label_key = label_key
        self.score_key = score_key
        self.output_dir = Path(output_dir)
        self.voxel_box_key = voxel_box_key
        self.world_box_key = world_box_key

    def __call__(self, data):
        d = dict(data)
        maybe_makedirs(self.output_dir)
        stem = strip_extension(Path(d["source_image"]).name)
        out_fn = self.output_dir / f"{stem}.json"
        save_inference_sidecar(
            out_fn,
            source_image=d["source_image"],
            case_id=d["case_id"],
            lbd_bounding_box=d["lbd_bounding_box"],
            localiser_run=d["localiser_run"],
            det_run=d["det_run"],
            spacing=d["spacing"],
            affine=d["affine"],
            boxes_voxel=d[self.voxel_box_key],
            boxes_world=d[self.world_box_key],
            labels=d[self.label_key],
            scores=d[self.score_key],
            boxes_pre_tfm=d.get("pred_box_pre_tfm"),
        )
        d["sidecar_path"] = str(out_fn)
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


def build_det_postprocess_transforms_dict(
    inferer,
    image_key: str = "image",
    pred_box_key: str = "pred_box",
    pred_label_key: str = "pred_label",
    pred_score_key: str = "pred_score",
    gt_box_mode: str = "cccwhd",
    affine_lps_to_ras: bool = True,
):
    from monai.apps.detection.transforms.dictionary import (
        AffineBoxToWorldCoordinated,
        ClipBoxToImaged,
        ConvertBoxModed,
    )

    Pre = PreservePreTfmBoxd(box_key=pred_box_key, dst_key="pred_box_pre_tfm")
    SqL = SqueezeListofListsd(keys=["bounding_box"])
    Clip = ClipBoxToImaged(
        box_keys=[pred_box_key],
        label_keys=[pred_label_key, pred_score_key],
        box_ref_image_keys=image_key,
        remove_empty=True,
    )
    Scale = ScaleBoxToCropNatived(box_keys=[pred_box_key], image_key=image_key)
    Off = OffsetBoxByBBoxd(box_keys=[pred_box_key])
    VoxCopy = CopyBoxKeyd(src_key=pred_box_key, dst_key="pred_box_voxel")
    FullMeta = UseFullMetaForImaged(keys=[image_key])
    World = AffineBoxToWorldCoordinated(
        box_keys=[pred_box_key],
        box_ref_image_keys=image_key,
        affine_lps_to_ras=affine_lps_to_ras,
    )
    Mode = ConvertBoxModed(
        box_keys=[pred_box_key],
        src_mode="xyzxyz",
        dst_mode=gt_box_mode,
    )
    Meta = AttachInferenceMetad(
        box_keys=[pred_box_key],
        run_w=inferer.run_w,
        run_p=inferer.run_p,
    )
    WorldCopy = CopyBoxKeyd(src_key=pred_box_key, dst_key="pred_box_world")
    Sav = SaveInferenceSidecard(
        box_keys=[pred_box_key],
        label_key=pred_label_key,
        score_key=pred_score_key,
        output_dir=inferer.output_folder,
        world_box_key="pred_box_world",
    )
    return {
        "Pre": Pre,
        "SqL": SqL,
        "Clip": Clip,
        "Scale": Scale,
        "Off": Off,
        "VoxCopy": VoxCopy,
        "FullMeta": FullMeta,
        "World": World,
        "WorldCopy": WorldCopy,
        "Mode": Mode,
        "Meta": Meta,
        "Sav": Sav,
    }
