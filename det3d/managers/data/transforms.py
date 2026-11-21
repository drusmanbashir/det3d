"""MONAI dictionary transforms for detection training: HDF5 shard/crop loading, box-centred crops and padding."""

import json
from ast import literal_eval
from pathlib import Path
import numpy as np
import torch
from det3d.transforms.crop_indices import monai_crop_center_to_slices
from det3d.transforms.box_tracker import CornerTracker
from fran.transforms.detection import patch_size_manifest_key
from fran.managers.data.valid_patch_stream import _pad_tensor_to_patch_size
from fran.utils.bbox_sidecar import extended_bbox_df_column
from fran.managers.data.main import RandCropExtendedBBoxd  # noqa: F401  det Rtr; oversample_fg slots as fran's D2
from fran.preprocessing.helpers import import_h5py
from monai.data import MetaTensor
from monai.transforms import MapTransform


class LoadHDF5DetShardExtendedBBoxd(MapTransform):
    """Load HDF5 shard paths, bbox, src_dims, and precomputed extended center boxes."""

    def __init__(
        self,
        keys,
        manifest_fn: str,
        data_folder: str,
        patch_size,
        box_key: str = "bbox",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.manifest_fn = Path(manifest_fn)
        self.data_folder = Path(data_folder)
        self.patch_size_key = patch_size_manifest_key(patch_size)
        self.box_key = box_key
        self._manifest_cache = {}

    def _cached_manifest(self):
        manifest_key = str(self.manifest_fn)
        cached = self._manifest_cache.get(manifest_key)
        if cached is not None:
            return cached

        with open(self.manifest_fn, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        case_to_shard: dict[str, str] = {}
        for shard_info in manifest["shards"]:
            shard_name = shard_info["shard"]
            shard_path = Path(shard_name)
            if not shard_path.is_absolute():
                shard_path = self.manifest_fn.parent / shard_path
            for case_id in shard_info["case_ids"]:
                case_to_shard[str(case_id)] = str(shard_path)

        cached = {"case_to_shard": case_to_shard}
        self._manifest_cache[manifest_key] = cached
        return cached

    def __call__(self, data):
        d = dict(data)
        case_id = str(d["case_id"])
        manifest = self._cached_manifest()
        shard_path = manifest["case_to_shard"][case_id]
        h5py = import_h5py()
        case_path = f"/cases/{case_id}"
        with h5py.File(shard_path, "r") as h5f:
            case_grp = h5f[case_path]
            src_dims = tuple(int(v) for v in case_grp["image"].shape)
        col = extended_bbox_df_column(self.patch_size_key)
        raw = d["df"][col]
        center_boxes = [
            literal_eval(v) if isinstance(v, str) else v for v in raw
        ]
        d["hdf5_shard_path"] = str(shard_path)
        d["hdf5_case_path"] = case_path
        # Volume bounds for RandCrop (Rtr runs before L2 loads pixels). Per-case HDF5 dataset shape, not plan or manifest tag.
        d["src_dims"] = src_dims
        d["extended_center_boxes"] = np.asarray(center_boxes, dtype=np.int64)
        return d


class CropDetPatchd(MapTransform):
    """Crop in-memory det patch tensors using RandCropByFlatIndicesd metadata."""

    def __init__(
        self,
        keys,
        box_key="bbox",
        crop_slices_key="crop_slices",
        crop_start_key="crop_start",
        allow_missing_keys=False,
        tracker=None,
    ):
        super().__init__(keys, allow_missing_keys)
        self.box_key = box_key
        self.tracker = tracker if tracker is not None else CornerTracker(box_key)
        self.crop_slices_key = crop_slices_key
        self.crop_start_key = crop_start_key

    def __call__(self, data):
        d = dict(data)
        crop_slices = tuple(d[self.crop_slices_key])
        crop_start = tuple(int(v) for v in d[self.crop_start_key])
        meta_updates = {
            "crop_start": crop_start,
            "crop_end": d.get("crop_end"),
            "sampled_flat_index": d.get("sampled_flat_index"),
            "sample_is_fg": d.get("sample_is_fg"),
        }
        for key in self.key_iterator(d):
            val = d[key]
            meta = dict(val.meta) if isinstance(val, MetaTensor) else {}
            meta.update(meta_updates)
            if val.ndim == 4:
                cropped = val[(slice(None), *crop_slices)]
            elif val.ndim == 3:
                cropped = val[crop_slices]
            else:
                raise ValueError(f"expected 3D or 4D tensor for {key}, got {val.shape}")
            if isinstance(val, MetaTensor):
                d[key] = MetaTensor(cropped.contiguous(), meta=meta)
            else:
                d[key] = cropped.contiguous()
        d = self.tracker.translate(d, [-v for v in crop_start])
        return d


class BboxCenterCropSlicesd(MapTransform):
    """Deterministic val crop: center on first bbox, MONAI allow_smaller slice math."""

    def __init__(
        self,
        box_key="bbox",
        image_key="image",
        roi_size=(128, 128, 64),
        allow_missing_keys=False,
    ):
        super().__init__([box_key], allow_missing_keys)
        self.box_key = box_key
        self.image_key = image_key
        self.roi_size = tuple(int(v) for v in roi_size)

    def __call__(self, data):
        d = dict(data)
        img = d[self.image_key]
        spatial_shape = tuple(int(v) for v in img.shape[-3:])
        box = torch.as_tensor(d[self.box_key], dtype=torch.float32)
        if box.ndim == 1:
            box = box.reshape(1, 6)
        b0 = box[0]
        center = tuple(
            int(round((b0[i].item() + b0[i + 3].item()) / 2)) for i in range(3)
        )
        slices, crop_start, crop_end = monai_crop_center_to_slices(
            center, self.roi_size, spatial_shape
        )
        d["crop_center"] = center
        d["crop_slices"] = slices
        d["crop_start"] = crop_start
        d["crop_end"] = crop_end
        d["validation_impl"] = "bbox_anchor"
        return d


class PadDetPatchd(MapTransform):
    """End-pad cropped patch tensors to fixed patch_size (boxes unchanged)."""

    def __init__(
        self,
        keys,
        patch_size,
        pad_value=0,
        allow_missing_keys=False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.pad_value = pad_value

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key], _ = _pad_tensor_to_patch_size(
                tensor=d[key],
                patch_size=self.patch_size,
                pad_value=self.pad_value,
            )
        return d


class LoadHDF5DetCropd(MapTransform):
    def __init__(
        self,
        keys,
        box_key="bbox",
        shard_path_key="hdf5_shard_path",
        case_path_key="hdf5_case_path",
        crop_slices_key="crop_slices",
        crop_start_key="crop_start",
        allow_missing_keys=False,
        tracker=None,
    ):
        super().__init__(keys, allow_missing_keys)
        self.box_key = box_key
        self.tracker = tracker if tracker is not None else CornerTracker(box_key)
        self.shard_path_key = shard_path_key
        self.case_path_key = case_path_key
        self.crop_slices_key = crop_slices_key
        self.crop_start_key = crop_start_key

    def __call__(self, data):
        d = dict(data)
        shard_path = Path(d[self.shard_path_key])
        case_path = str(d[self.case_path_key])
        crop_slices = tuple(d[self.crop_slices_key])
        crop_start = tuple(int(v) for v in d[self.crop_start_key])
        h5py = import_h5py()
        h5_keys = tuple(dict.fromkeys(("image", *self.keys)))
        with h5py.File(shard_path, "r") as h5f:
            case_grp = h5f[case_path]
            loaded = {
                key: np.asarray(case_grp[key][crop_slices])
                for key in h5_keys
                if key in case_grp
            }
            if "inst" in self.keys:
                d["inst_labels"] = _read_instance_labels(case_grp)
            bbox = np.asarray(case_grp[self.box_key][:], dtype=np.float32)
        meta = {
            "filename_or_obj": f"{shard_path}:{case_path}",
            "case_id": d["case_id"],
            "crop_start": crop_start,
            "original_channel_dim": 0,
        }
        for key, arr in loaded.items():
            if arr.ndim == 3:
                arr = arr[np.newaxis, ...]
            d[key] = MetaTensor(torch.as_tensor(arr), meta=dict(meta))
        d[self.box_key] = torch.as_tensor(bbox, dtype=torch.float32)
        d = self.tracker.translate(d, [-v for v in crop_start])
        del (
            d[self.shard_path_key],
            d[self.case_path_key],
            d[self.crop_slices_key],
            d[self.crop_start_key],
        )
        return d


class LoadHDF5DetCaseFulld(MapTransform):
    def __init__(
        self,
        keys,
        shard_path_key="hdf5_shard_path",
        case_path_key="hdf5_case_path",
        allow_missing_keys=False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.shard_path_key = shard_path_key
        self.case_path_key = case_path_key

    def __call__(self, data):
        d = dict(data)
        shard_path = Path(d[self.shard_path_key])
        case_path = str(d[self.case_path_key])
        h5py = import_h5py()
        with h5py.File(shard_path, "r") as h5f:
            image = np.asarray(h5f[case_path]["image"][:])
        if image.ndim == 3:
            image = image[np.newaxis, ...]
        meta = {
            "filename_or_obj": f"{shard_path}:{case_path}",
            "case_id": d.get("case_id"),
            "original_channel_dim": 0,
        }
        d["image"] = MetaTensor(torch.as_tensor(image), meta=dict(meta))
        return d


class LoadHDF5DetInstancesd(MapTransform):
    """Read validation instance IDs while retaining the PT image metadata."""

    def __init__(self):  #AI
        """Use the loaded validation image as the spatial reference."""
        super().__init__(["image"])

    def __call__(self, data):  #AI
        """Load the full instance map and its sidecar row labels."""
        d = dict(data)
        path = d["hdf5_shard_path"]
        case = d["hdf5_case_path"]
        with import_h5py().File(path, "r") as h5f:
            group = h5f[case]
            d["inst_labels"] = _read_instance_labels(group)
            inst = torch.as_tensor(group["inst"][:])[None]
        meta = dict(d["image"].meta, original_channel_dim=0)
        d["inst"] = MetaTensor(inst.contiguous(), meta=meta)
        return d


def _read_instance_labels(group):  #AI
    labels = torch.as_tensor(group["label"][:])
    return labels
