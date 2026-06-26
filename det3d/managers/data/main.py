import json
import resource
from functools import partial
from pathlib import Path
from utilz.imageviewers import ImageBBoxViewer, ImageMaskBboxViewer, ImageMaskViewer
from typing import Optional

import numpy as np
import pandas as pd
import torch
from det3d.managers.data.collate import det_val_collate, lbd_det_collate, lbd_det_collate_train, lbd_det_collate_val
from det3d.managers.data.valid_patch_stream import (
    PatchStreamDatasetDet,
    patch_stream_collate_fn,
)
from det3d.preprocessing.hdf5_shards_det import ensure_hdf5_shards_for_plan
from det3d.transforms.crop_indices import (
    monai_crop_center_to_slices,
    sample_crop_center_from_extended_boxes,
)
from det3d.transforms.gpu_det import RandAffineBoxSyncd, RandFlipBoxSyncd, ResizeWithPadOrCropBoxSyncd
from det3d.transforms.detection import patch_size_manifest_key
from fran.managers.data.valid_patch_stream import _pad_tensor_to_patch_size
from det3d.utils.bbox_sidecar import bbox_sidecar_path, load_detection_sidecar, valid_detection_box
from fran.configs.helpers import is_excel_None
from fran.managers.data.main import (
    DataManager,
    DataManagerDual,
    DataManagerLBD,
    DataManagerMulti,
    DataManagerPatch,
    DataManagerRBD,
    DataManagerShort,
    DataManagerSource,
    DataManagerWhole,
    RandCropByFlatIndicesd,
)
from fran.preprocessing.helpers import import_h5py
from fran.run.preproc.archive_preprocessed import ensure_rapid_data_folder
from fran.transforms.imageio import TorchReader
from monai.apps.detection.transforms.dictionary import ClipBoxToImaged
from monai.data import DataLoader, Dataset, MetaTensor
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    ScaleIntensityRanged,
)
from monai.transforms.croppad.dictionary import ResizeWithPadOrCropd
from fran.utils.folder_names import FolderNames
from utilz.fileio import load_json
from utilz.stringz import info_from_filename


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

        # Shard-level build tag; symlink dir name may differ. Not copied to sample dict — RandCrop uses per-case image.shape.
        case_to_shard: dict[str, str] = {}
        for shard_info in manifest["shards"]:
            shard_name = shard_info["shard"]
            shard_path = Path(shard_name)
            if not shard_path.is_absolute():
                shard_path = self.manifest_fn.parent / shard_path
            for case_id in shard_info["case_ids"]:
                case_to_shard[str(case_id)] = str(shard_path)

        src_dims = tuple(int(v) for v in manifest["src_dims"])
        cached = {
            "case_to_shard": case_to_shard,
            "src_dims": src_dims,
        }
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
        ext_fn = self.data_folder / "extended_bboxes" / f"{case_id}.json"
        ext_all = json.loads(ext_fn.read_text(encoding="utf-8"))
        center_boxes = ext_all[self.patch_size_key]
        d["hdf5_shard_path"] = str(shard_path)
        d["hdf5_case_path"] = case_path
        # Volume bounds for RandCrop (Rtr runs before L2 loads pixels). Per-case HDF5 dataset shape, not plan or manifest tag.
        d["src_dims"] = src_dims
        d["extended_center_boxes"] = np.asarray(center_boxes, dtype=np.int64)
        return d


class RandCropByFgIndicesd(RandCropByFlatIndicesd):
    """RandCropByFlatIndicesd with uniform-random neg centers (no stored bg pool)."""

    def __call__(self, data):
        d = dict(data)
        src_dims = tuple(int(v) for v in d[self.src_dims_key])
        fg = np.asarray(d[self.fg_indices_key], dtype=np.int64).reshape(-1)
        n_voxels = int(np.prod(src_dims))
        out = []
        for _ in range(self.num_samples):
            sample = dict(d)
            choose_fg = self.R.rand() < self.pos / (self.pos + self.neg)
            if choose_fg and fg.size > 0:
                sample_is_fg = True
                sampled_flat_index = int(fg[self.R.randint(0, fg.size)])
            else:
                sample_is_fg = False
                sampled_flat_index = int(self.R.randint(0, n_voxels))
            center = tuple(
                int(v) for v in np.unravel_index(sampled_flat_index, src_dims)
            )
            crop_slices, crop_start, crop_end = self._compute_crop(center, src_dims)
            sample["crop_center"] = center
            sample["crop_slices"] = crop_slices
            sample["crop_start"] = crop_start
            sample["crop_end"] = crop_end
            sample["sample_is_fg"] = bool(sample_is_fg)
            sample["sampled_flat_index"] = sampled_flat_index
            out.append(sample)
        return out


class RandCropExtendedBBoxd(RandCropByFgIndicesd):
    """RandCrop from precomputed extended center boxes per patch size."""

    def __init__(
        self,
        keys,
        roi_size,
        pos=1.0,
        neg=1.0,
        num_samples=1,
        extended_boxes_key="extended_center_boxes",
        src_dims_key="src_dims",
        allow_missing_keys=False,
    ):
        RandCropByFgIndicesd.__init__(
            self,
            keys=keys,
            roi_size=roi_size,
            pos=pos,
            neg=neg,
            num_samples=num_samples,
            fg_indices_key="fg_indices",
            src_dims_key=src_dims_key,
            allow_missing_keys=allow_missing_keys,
        )
        self.extended_boxes_key = extended_boxes_key

    def __call__(self, data):
        d = dict(data)
        src_dims = tuple(int(v) for v in d[self.src_dims_key])
        center_boxes = np.asarray(d[self.extended_boxes_key], dtype=np.int64).reshape(-1, 6)
        out = []
        for _ in range(self.num_samples):
            sample = dict(d)
            choose_fg = self.R.rand() < self.pos / (self.pos + self.neg)
            sample_is_fg = bool(choose_fg and center_boxes.shape[0] > 0)
            center = sample_crop_center_from_extended_boxes(
                center_boxes, src_dims, sample_is_fg, self.R
            )
            crop_slices, crop_start, _crop_end = self._compute_crop(center, src_dims)
            sample["crop_slices"] = crop_slices
            sample["crop_start"] = crop_start
            del sample[self.src_dims_key], sample[self.extended_boxes_key]
            out.append(sample)
        return out


class CropDetPatchd(MapTransform):
    """Crop in-memory det patch tensors using RandCropByFlatIndicesd metadata."""

    def __init__(
        self,
        keys,
        box_key="bbox",
        crop_slices_key="crop_slices",
        crop_start_key="crop_start",
        allow_missing_keys=False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.box_key = box_key
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
        box = torch.as_tensor(d[self.box_key], dtype=torch.float32)
        if box.numel() > 0:
            start = torch.tensor(crop_start, dtype=box.dtype)
            box = box.clone()
            box[:, :3] -= start
            box[:, 3:] -= start
            d[self.box_key] = box
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
    ):
        super().__init__(keys, allow_missing_keys)
        self.box_key = box_key
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
        box = torch.as_tensor(bbox, dtype=torch.float32)
        if box.numel() > 0:
            start = torch.tensor(crop_start, dtype=box.dtype)
            box = box.clone()
            box[:, :3] -= start
            box[:, 3:] -= start
            d[self.box_key] = box
        else:
            d[self.box_key] = box
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


class DataManagerDet(DataManager):
    data_keys = ("image",)
    spatial_aug_keys = ("image",)
    image_key = "image"
    box_key = "bbox"
    label_key = "label"
    point_key = "points"
    mask_key = "mask"
    lm_key = "lm"
    keys_tr_batch = None
    keys_val_batch = None

    def __init__(
        self,
        project,
        configs: dict,
        batch_size=64,
        cache_rate=0.0,
        split="train",
        device="cuda:0",
        ds_type=None,
        save_hyperparameters=False,
        keys=None,
        collate_fn=None,
        data_folder: Optional[str | Path] = None,
        val_sampling=1.0,
        debug=False,
    ):
        super().__init__(
            project=project,
            configs=configs,
            batch_size=batch_size,
            cache_rate=cache_rate,
            split=split,
            device=device,
            ds_type=ds_type,
            save_hyperparameters=save_hyperparameters,
            keys=keys,
            collate_fn=collate_fn,
            data_folder=data_folder,
            val_sampling=val_sampling,
            debug=debug,
        )
        self.amp = True
        self.affine_lps_to_ras = False

    def __repr__(self):
        n = len(self.data) if self.data else 0
        return f"{self.__class__.__name__}(split={self.split}, n={n})"

    def __str__(self):
        return self.__repr__()

    # def create_dataset(self):
    #     self.ds = self._create_modal_ds()

    def _create_modal_ds(self):
        if is_excel_None(self.ds_type):
            ds = Dataset(data=self.data, transform=self.transforms)
        else:
            ds = super()._create_modal_ds()
        return ds

    def maybe_fix_remapping_dtype(self):
        pass

    def set_preprocessing_params(self):
        from utilz.fileio import load_dict

        global_properties = load_dict(self.project.global_properties_filename)
        self.dataset_params = self.configs["dataset_params"]
        self.dataset_params["intensity_clip_range"] = global_properties[
            "intensity_clip_range"
        ]
        self.dataset_params["mean_fg"] = global_properties["mean_fg"]
        self.dataset_params["std_fg"] = global_properties["std_fg"]
        transform_factors = self.configs.get("transform_factors")
        if transform_factors:
            self._assimilate_tfm_factors(transform_factors)
        self.dataset_params.setdefault("valid_impl", "bbox_anchor")

    def _valid_impl(self):
        return self.dataset_params["valid_impl"]

    def set_effective_batch_size(self):
        self.effective_batch_size = self.batch_size

    def _size_divisible(self):
        from det3d.architectures.create_detector import size_divisible_from_conf

        return size_divisible_from_conf(self.configs)

    def _patch_size(self):
        return tuple(int(v) for v in self.plan["patch_size"])

    def uses_lm_seg(self):
        from det3d.architectures.create_detector import arch_from_conf

        return arch_from_conf(self.configs) in ("retinaunet", "retinaunet_v3")

    def _rand_crop_patch_size(self):
        if self.uses_lm_seg():
            from det3d.detection.nndet_train import forward_patch_size_from_configs

            ps = forward_patch_size_from_configs(self.configs)
            if ps is not None:
                return tuple(int(v) for v in ps)
        return self._patch_size()

    def _set_collate_fn(self):
        if self.is_eval_split():
            if self.uses_lm_seg():
                self.collate_fn = lbd_det_collate_val
                return
            collate_kwargs = {
                "box_key": self.box_key,
                "label_key": self.label_key,
            }
            self.collate_fn = partial(det_val_collate, **collate_kwargs)
            return

        collate_kwargs = {
            "size_divisible": self._size_divisible(),
            "box_key": self.box_key,
        }
        if self.uses_lm_seg():
            self.collate_fn = lbd_det_collate_train
            return
        self.collate_fn = partial(lbd_det_collate, **collate_kwargs)

    def override_batch_size_valid_split(self, split="valid"):
        pass

    def _compute_dtype(self):
        return torch.float32

    def _num_workers(self):
        if self.debug:
            return 0, False
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_limit < 1024:
            return 0, False
        if self.is_train_split():
            num_workers = min(4, max(2, self.effective_batch_size // 8))
        else:
            num_workers = min(4, max(2, int(self.configs["dataset_params"].get("num_workers_val", 2))))
        return num_workers, False

    def create_dataloader(self):
        num_workers, persistent_workers = self._num_workers()
        if self.is_train_split():
            batch_size = self.effective_batch_size
            collate_fn = self.collate_fn
        else:
            batch_size = 1
            collate_fn = self.collate_fn
        pin_memory = torch.cuda.is_available() and not self.debug
        if num_workers > 0:
            persistent_workers = True
        dl_kwargs = dict(
            batch_size=batch_size,
            shuffle=self.is_train_split(),
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
        )
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = 2
        self.dl = DataLoader(self.ds, **dl_kwargs)

    def derive_data_folder(self, plan):
        data_folder = Path(FolderNames(self.project, plan).folders["data_folder_lbd"])
        data_folder = ensure_rapid_data_folder(data_folder)
        if not data_folder.exists():
            raise FileNotFoundError(f"Data folder {data_folder} does not exist")
        images_dir = data_folder / "images"
        bboxes_dir = data_folder / "bboxes"
        if not images_dir.is_dir() or not bboxes_dir.is_dir():
            raise FileNotFoundError(
                f"Expected images/ and bboxes/ under {data_folder}"
            )
        if len(list(images_dir.glob("*.pt"))) == 0:
            raise FileNotFoundError(f"No label-bounded cases under {images_dir}")
        if len(list(bboxes_dir.glob("*.pt"))) > 0:
            raise FileNotFoundError(
                f"Legacy bbox .pt sidecars under {bboxes_dir}; re-preproc to JSON"
            )
        if len(list(bboxes_dir.glob("*.json"))) == 0:
            raise FileNotFoundError(f"No bbox JSON sidecars under {bboxes_dir}")
        csv_fn = data_folder / "dataset_details.csv"
        if not csv_fn.is_file():
            raise FileNotFoundError(f"Missing dataset_details.csv under {data_folder}")
        return data_folder

    def cases_from_project_split(self):
        ds_tokens = [x.strip() for x in self.plan["datasources"].split(",") if x.strip()]
        nnz_allowed = self.plan.get("nnz_allowed", False)
        train_cases, valid_cases = self.project.get_train_val_case_ids(
            self.dataset_params["fold"],
            ds_tokens,
            nnz_allowed=nnz_allowed,
        )
        self.cases = train_cases if self.is_train_split() else valid_cases
        case_ids_on_disk = set()
        for img_fn in (self.data_folder / "images").glob("*.pt"):
            case_ids_on_disk.add(
                info_from_filename(img_fn.name, full_caseid=True)["case_id"]
            )
        self.cases = [case_id for case_id in self.cases if case_id in case_ids_on_disk]
        self.cases = self._filter_cases_by_stats_nnz(self.cases, nnz_allowed)
        assert len(self.cases) > 0, "There are no cases, aborting!"

    def _filter_cases_by_stats_nnz(self, cases, nnz_allowed):
        if nnz_allowed:
            return cases
        df = pd.read_csv(self.data_folder / "dataset_details.csv")
        fg_case_ids = self._fg_case_ids_from_stats()
        bbox_ok = set(df.loc[~df["bbox_empty"], "case_id"].astype(str))
        return [
            case_id
            for case_id in cases
            if str(case_id) in fg_case_ids and str(case_id) in bbox_ok
        ]

    def _load_bbox_sidecar(self, bbox_fn):
        boxes, labels, _instances = load_detection_sidecar(bbox_fn)
        valid_boxes = []
        valid_labels = []
        for box, label in zip(boxes, labels):
            if valid_detection_box(box):
                valid_boxes.append(box.reshape(-1))
                valid_labels.append(label.reshape(-1))
        if len(valid_boxes) == 0:
            box_t = torch.zeros((0, 6), dtype=torch.float32)
            label_t = torch.zeros((0,), dtype=torch.long)
        else:
            box_t = torch.stack(valid_boxes)
            label_t = torch.stack(valid_labels).reshape(-1)
        return box_t, label_t, _instances

    @property
    def hdf5_manifest_fn(self):
        src_tag = "_".join(str(int(v)) for v in self.plan["src_dims"])
        return self.data_folder / "hdf5_shards" / f"src_{src_tag}" / "manifest.json"

    def _case_ids_on_disk(self):
        manifest = json.loads(self.hdf5_manifest_fn.read_text())
        case_ids = set()
        for shard_info in manifest["shards"]:
            case_ids.update(str(case_id) for case_id in shard_info["case_ids"])
        return case_ids

    def _load_case_dicts_from_shards(self, case_ids):
        case_ids = set(str(case_id) for case_id in case_ids)
        data = []
        bboxes_dir = self.data_folder / "bboxes"
        skipped = 0
        manifest = json.loads(self.hdf5_manifest_fn.read_text())
        manifest_parent = self.hdf5_manifest_fn.parent
        for shard_info in manifest["shards"]:
            shard_path = Path(shard_info["shard"])
            if not shard_path.is_absolute():
                shard_path = manifest_parent / shard_path
            for case_id in shard_info["case_ids"]:
                case_id = str(case_id)
                if case_id not in case_ids:
                    continue
                bbox_fn = bbox_sidecar_path(bboxes_dir, case_id)
                if not bbox_fn.is_file():
                    skipped += 1
                    continue
                box_t, label_t, instances = self._load_bbox_sidecar(bbox_fn)
                row = {
                    "case_id": case_id,
                    "data_folder": str(self.data_folder),
                    "hdf5_shard_path": str(shard_path),
                    "hdf5_case_path": f"/cases/{case_id}",
                    self.box_key: box_t,
                    self.label_key: label_t,
                    "instances": instances,
                }
                data.append(row)
        return data, skipped

    def _require_shard_manifest(self, data_folder):
        data_folder = Path(data_folder)
        ensure_hdf5_shards_for_plan(data_folder, self.plan["src_dims"])
        manifest_fn = data_folder / "hdf5_shards" / (
            f"src_{'_'.join(str(int(v)) for v in self.plan['src_dims'])}"
        ) / "manifest.json"
        if not manifest_fn.is_file():
            raise FileNotFoundError(f"Missing HDF5 shard manifest {manifest_fn}")

    def _shard_cases_from_project_split(self):
        ds_tokens = [x.strip() for x in self.plan["datasources"].split(",") if x.strip()]
        nnz_allowed = self.plan.get("nnz_allowed", False)
        train_cases, valid_cases = self.project.get_train_val_case_ids(
            self.dataset_params["fold"],
            ds_tokens,
            nnz_allowed=nnz_allowed,
        )
        self.cases = train_cases if self.is_train_split() else valid_cases
        self.cases = [
            case_id for case_id in self.cases if case_id in self._case_ids_on_disk()
        ]
        self.cases = self._filter_cases_by_stats_nnz(self.cases, nnz_allowed)
        assert len(self.cases) > 0, "There are no cases, aborting!"


class DataManagerDetSource(DataManagerDet, DataManagerSource):
    keys_tr = (
        "Ld,Rtr,L2,E,Norm,F1,F2,Affine,ResizePC,BoxClip,IntensityTfms"
    )

    def __init__(self, project, configs: dict, batch_size=8, cache_rate=0.0, **kwargs):
        provided_keys = kwargs["keys"] if "keys" in kwargs else None
        super().__init__(project, configs, batch_size, cache_rate, **kwargs)
        if provided_keys is None and self.uses_train_keys():
            self.keys = self.keys_tr

    def derive_data_folder(self, plan):
        data_folder = super(DataManagerDet, self).derive_data_folder(plan)
        self._require_shard_manifest(data_folder)
        manifest_fn = data_folder / "manifest.json"
        if not manifest_fn.is_file():
            raise FileNotFoundError(f"Missing manifest.json under {data_folder}")
        return data_folder

    def set_effective_batch_size(self):
        spf = int(self.plan["samples_per_file"])
        assert self.batch_size % spf == 0, (
            f"batch_size {self.batch_size} must be divisible by "
            f"samples_per_file {spf}"
        )
        self.effective_batch_size = self.batch_size // spf

    def cases_from_project_split(self):
        self._shard_cases_from_project_split()

    def create_data_dicts(self, case_ids):
        data, skipped = self._load_case_dicts_from_shards(case_ids)
        if skipped:
            print(
                f"DataManagerDetSource: skipped {skipped} cases "
                "(missing sidecar)"
            )
        return data

    def create_transforms(self):
        super().create_transforms()
        ik, bk, lk, lmk = (
            self.image_key,
            self.box_key,
            self.label_key,
            self.lm_key,
        )
        load_keys = [ik]
        use_lm = self.uses_lm_seg()
        if use_lm:
            load_keys.append(lmk)
        spatial_aug_keys = [ik]
        if use_lm:
            spatial_aug_keys.append(lmk)
        affine_modes = ["bilinear" if k == ik else "nearest" for k in spatial_aug_keys]
        plan = self.plan
        patch_size = self._patch_size()
        scale = float(self.dataset_params["prezoom_scale"])
        patch_size_prezoom = tuple(int(v * scale) for v in patch_size)
        data_manifest = load_json(self.data_folder / "manifest.json")
        manifest_patch_sizes = [
            tuple(int(v) for v in ps)
            for ps in data_manifest["extended_bboxes_patch_sizes"]
        ]
        if patch_size_prezoom not in manifest_patch_sizes:
            raise ValueError(
                f"patch_size_prezoom {patch_size_prezoom} not in manifest "
                f"extended_bboxes_patch_sizes {manifest_patch_sizes}"
            )
        clip = self.dataset_params["intensity_clip_range"]
        affine3d = self.configs["affine3d"]

        self.flip["prob"]= 0.8 #HACK: temporary for debugging
        flip_prob = float(self.flip["prob"])

        self.Ld = LoadHDF5DetShardExtendedBBoxd(
            keys=["case_id"],
            manifest_fn=str(self.hdf5_manifest_fn),
            data_folder=str(self.data_folder),
            patch_size=patch_size_prezoom,
            box_key=bk,
        )
        self.Rtr = RandCropExtendedBBoxd(
            keys=[ik],
            roi_size=patch_size_prezoom,
            num_samples=int(plan["samples_per_file"]),
            pos=self.dataset_params["fgbg_ratio"],
            neg=1,
            src_dims_key="src_dims",
        )
        self.L2 = LoadHDF5DetCropd(keys=load_keys, box_key=bk)
        self.E = EnsureChannelFirstd(keys=load_keys)
        self.F1 = RandFlipBoxSyncd(
            spatial_keys=spatial_aug_keys,
            box_key=bk,
            prob=flip_prob,
            spatial_axis=0,
        )
        self.F2 = RandFlipBoxSyncd(
            spatial_keys=spatial_aug_keys,
            box_key=bk,
            prob=flip_prob,
            spatial_axis=1,
        )
        self.Affine = RandAffineBoxSyncd(
            spatial_keys=spatial_aug_keys,
            box_key=bk,
            mode=affine_modes,
            prob=affine3d["p"],
            rotate_range=affine3d["rotate_range"],
            scale_range=affine3d["scale_range"],
        )
        self.ResizePC = ResizeWithPadOrCropBoxSyncd(
            keys=spatial_aug_keys,
            box_key=bk,
            label_key=lk,
            spatial_size=patch_size,
            lazy=False,
        )
        self.Norm = ScaleIntensityRanged(
            keys=[ik],
            a_min=float(clip[0]),
            a_max=float(clip[1]),
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
        self.BoxClip = ClipBoxToImaged(
            box_keys=bk,
            label_keys=[lk],
            box_ref_image_keys=ik,
            remove_empty=True,
        )
        self.transforms_dict["Ld"] = self.Ld
        self.transforms_dict["Rtr"] = self.Rtr
        self.transforms_dict["L2"] = self.L2
        self.transforms_dict["E"] = self.E
        self.transforms_dict["Norm"] = self.Norm
        self.transforms_dict["F1"] = self.F1
        self.transforms_dict["F2"] = self.F2
        self.transforms_dict["Affine"] = self.Affine
        self.transforms_dict["ResizePC"] = self.ResizePC
        self.transforms_dict["BoxClip"] = self.BoxClip
        if type(self.Ld) is not LoadHDF5DetShardExtendedBBoxd:
            raise RuntimeError(
                f"det shard train must use LoadHDF5DetShardExtendedBBoxd, got {type(self.Ld)}"
            )


class DataManagerDetWhole(DataManagerDet, DataManagerWhole):
    pass


class DataManagerDetLBD(DataManagerDetSource, DataManagerLBD):
    keys_val = "L,E,Norm,DtypeVal"
    keys_val_seg = "L,E,BoxClip,Norm,DtypeVal"
    keys_val_bbox = "L,E,Norm,BboxCrop,CropPatch,PadPatch,BoxClip,DtypeVal"
    keys_val_batch = None

    def __repr__(self):
        n = len(self.data) if self.data else 0
        return f"{self.__class__.__name__}(split={self.split}, n={n})"

    def __str__(self):
        return (
            f"{self.__class__.__name__} split={self.split} n="
            f"{len(self.data) if self.data else 0} folder={self.data_folder}"
        )

    def __init__(self, project, configs: dict, batch_size=8, cache_rate=0.0, **kwargs):
        provided_keys = kwargs["keys"] if "keys" in kwargs else None
        super().__init__(project, configs, batch_size, cache_rate, **kwargs)
        if provided_keys is None and self.is_eval_split():
            if self._valid_impl() == "patch_stream":
                self.keys = self.keys_val_seg if self.uses_lm_seg() else self.keys_val
            else:
                self.keys = self.keys_val_bbox
        self.override_batch_size_valid_split(split=self.split)

    def override_batch_size_valid_split(self, split="valid"):
        if (
            split == "valid"
            and not self.is_train_all_split()
            and self._valid_impl() == "patch_stream"
        ):
            self.batch_size = 1
            self.effective_batch_size = 1
            lm_key = self.lm_key if self.uses_lm_seg() else None
            self.collate_fn = patch_stream_collate_fn(lm_key)

    def create_dataset(self):
        if self.is_train_all_split():
            return DataManagerDetSource.create_dataset(self)
        if self.is_eval_split():
            if not hasattr(self, "data") or len(self.data) == 0:
                print("No data. DS is not being created at this point.")
                return 0
            if self._valid_impl() == "patch_stream":
                case_ds = self._create_modal_ds()
                patch_size = self._rand_crop_patch_size()
                lm_key = self.lm_key if self.uses_lm_seg() else None
                self.ds = PatchStreamDatasetDet(
                    case_dataset=case_ds,
                    patch_size=patch_size,
                    lm_key=lm_key,
                )
                return
            self.ds = self._create_modal_ds()
            return
        DataManagerDet.create_dataset(self)

    def create_dataloader(self):
        if not self.is_eval_split():
            return DataManagerDet.create_dataloader(self)
        if isinstance(self.ds, PatchStreamDatasetDet):
            self.dl = DataLoader(
                self.ds,
                batch_size=1,
                num_workers=0,
                collate_fn=self.collate_fn,
                persistent_workers=False,
                pin_memory=not self.debug,
                shuffle=False,
            )
            return
        num_workers, persistent_workers = self._num_workers()
        pin_memory = torch.cuda.is_available() and not self.debug
        if num_workers > 0:
            persistent_workers = True
        dl_kwargs = dict(
            batch_size=self.effective_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
        )
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = 2
        self.dl = DataLoader(self.ds, **dl_kwargs)

    def create_valid_dataloader(self):
        self.create_dataloader()

    def cases_from_project_split(self):
        if self.is_eval_split():
            DataManagerDet.cases_from_project_split(self)
            return
        self._shard_cases_from_project_split()

    def create_data_dicts(self, case_ids):
        if self.is_train_all_split():
            return DataManagerDetSource.create_data_dicts(self, case_ids)
        case_ids = set(str(case_id) for case_id in case_ids)
        skipped = 0
        bboxes_dir = self.data_folder / "bboxes"
        data = []
        images_dir = self.data_folder / "images"
        for img_fn in sorted(images_dir.glob("*.pt")):
            case_id = info_from_filename(img_fn.name, full_caseid=True)["case_id"]
            if case_id not in case_ids:
                continue
            bbox_fn = bbox_sidecar_path(bboxes_dir, img_fn.stem)
            if not bbox_fn.is_file():
                skipped += 1
                continue
            box_t, label_t, instances = self._load_bbox_sidecar(bbox_fn)
            row = {
                "case_id": case_id,
                "data_folder": str(self.data_folder),
                "image": str(img_fn),
                self.box_key: box_t,
                self.label_key: label_t,
                "instances": instances,
            }
            if self.uses_lm_seg():
                lm_fn = self.data_folder / "lms" / img_fn.name
                if not lm_fn.is_file():
                    skipped += 1
                    continue
                row[self.lm_key] = str(lm_fn)
            data.append(row)
        if skipped:
            print(
                f"DataManagerDetLBD: skipped {skipped} cases "
                "(missing sidecar)"
            )
        return data

    def create_transforms(self):
        if self.is_train_all_split():
            return DataManagerDetSource.create_transforms(self)
        ik, bk, lk, mk, lmk = (
            self.image_key,
            self.box_key,
            self.label_key,
            self.mask_key,
            self.lm_key,
        )
        use_lm = self.uses_lm_seg()
        load_keys = [ik]
        if use_lm:
            load_keys.append(lmk)
        compute_dtype = self._compute_dtype()
        clip = self.dataset_params["intensity_clip_range"]

        L = LoadImaged(
            keys=load_keys,
            image_only=False,
            ensure_channel_first=False,
            simple_keys=True,
        )
        L.register(TorchReader())

        dtype_val = [
            EnsureTyped(keys=[ik], dtype=compute_dtype),
            EnsureTyped(keys=[bk], dtype=torch.float32),
            EnsureTyped(keys=[lk], dtype=torch.long),
        ]
        if use_lm:
            dtype_val.insert(1, EnsureTyped(keys=[lmk], dtype=torch.long))

        self.transforms_dict = {
            "L": L,
            "E": EnsureChannelFirstd(keys=load_keys),
            "Norm": ScaleIntensityRanged(
                keys=[ik],
                a_min=float(clip[0]),
                a_max=float(clip[1]),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            "DtypeVal": Compose(dtype_val),
        }
        box_clip = ClipBoxToImaged(
            box_keys=bk,
            label_keys=[lk],
            box_ref_image_keys=ik,
            remove_empty=True,
        )
        if self.is_eval_split() and self._valid_impl() == "bbox_anchor":
            patch_size = self._rand_crop_patch_size()
            crop_keys = [ik]
            if use_lm:
                crop_keys.append(lmk)
            self.transforms_dict["BboxCrop"] = BboxCenterCropSlicesd(
                box_key=bk,
                image_key=ik,
                roi_size=patch_size,
            )
            self.transforms_dict["CropPatch"] = CropDetPatchd(keys=crop_keys, box_key=bk)
            self.transforms_dict["PadPatch"] = PadDetPatchd(
                keys=crop_keys,
                patch_size=patch_size,
            )
            self.transforms_dict["BoxClip"] = box_clip
        elif use_lm:
            self.transforms_dict["BoxClip"] = box_clip


class DataManagerDetRBD(DataManagerDetLBD, DataManagerRBD):
    pass


class DataManagerDetShort(DataManagerDet, DataManagerShort):
    pass


class DataManagerDetPatch(DataManagerDet, DataManagerPatch):
    pass


class DataManagerDualDet(DataManagerDual):
    _DET_MANAGER_CLASSES = {
        "source": DataManagerDetSource,
        "whole": DataManagerDetWhole,
        "pbd": DataManagerDetPatch,
        "sourcepbd": DataManagerDetPatch,
        "lbd": DataManagerDetLBD,
        "rbd": DataManagerDetRBD,
    }

    def infer_manager_classes(self, configs):
        train_mode = configs["plan_train"]["mode"]
        valid_mode = configs["plan_valid"]["mode"]
        mode_to_class = dict(self._DET_MANAGER_CLASSES)
        for mode in (train_mode, valid_mode):
            if mode not in mode_to_class:
                raise ValueError(
                    f"Unrecognized mode: {mode}. Must be one of {list(mode_to_class.keys())}"
                )
        return mode_to_class[train_mode], mode_to_class[valid_mode]

    def _assert_det_manager_class(self, cls):
        if not cls.__name__.startswith("DataManagerDet"):
            raise RuntimeError(
                f"Expected a det3d DataManagerDet* class, got {cls.__name__}. "
                "TrainerDet must build DataManagerDualDet* — not fran DataManagerDual."
            )

    def _build_managers(self):
        cls_tr, cls_val = self.infer_manager_classes(self.configs)
        cls_tr = self.manager_class_train or cls_tr
        cls_val = self.manager_class_valid or cls_val
        self._assert_det_manager_class(cls_tr)
        self._assert_det_manager_class(cls_val)
        lbd_folder = FolderNames(self.project, self.configs["plan_train"]).lbd_folder
        from utilz.cprint import cprint

        cprint(f"train manager class: {cls_tr.__name__}", color="cyan")
        cprint(f"valid manager class: {cls_val.__name__}", color="cyan")
        self.train_manager = cls_tr(
            project=self.project,
            configs=self.configs,
            batch_size=self.batch_size,
            cache_rate=self.cache_rate,
            split="train",
            device=self.device,
            ds_type=self.ds_type,
            keys=self.keys_tr,
            data_folder=lbd_folder,
            debug=self.debug,
        )
        self.valid_manager = cls_val(
            project=self.project,
            configs=self.configs,
            batch_size=self.batch_size,
            cache_rate=self.cache_rate,
            split="valid",
            device=self.device,
            ds_type=self.ds_type,
            keys=self.keys_val,
            data_folder=lbd_folder,
            val_sampling=self.val_sampling,
            debug=self.debug,
        )

    def __repr__(self):
        return (
            f"DataManagerDualDet("
            f"train={self.train_manager!r}, valid={self.valid_manager!r})"
        )

    def __str__(self):
        return self.__repr__()


class DataManagerMultiDet(DataManagerMulti):
    pass


# %%
if __name__ == "__main__":
#SECTION:--- setup ---
    from fran.managers import Project

    from det3d.configs.parser import ConfigMakerDet
    from det3d.managers.data.batch_tfms import DataManagerDualDetBTfms

    project_title = "lidca"
    plan_id = 4
    conf_fold = 0

    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = C.configs
    conf["dataset_params"]["fold"] = conf_fold
    conf["plan_train"]["patch_size"]=[128,128,64]
    conf["model_params"]["arch"] = "retinanet"
    conf["model_params"]["arch"] = "retinaunet"
    conf["affine3d"]["p"]=1.0
    conf["affine3d"]["translate_factor"]=0.3
    conf["affine3d"]["shear"]=0.5
    

#SECTION:--- dualdet datamanager ---
# %%
    batch_size = 2
    batch_tfms = False
    debug_ = True
    train_indices = 40
    val_indices = 10
    val_sampling = 1.0
    device = 0

    for key in ("plan_train", "plan_valid", "plan_test"):
        plan = conf[key]
        if plan["mode"] in {"det", "lbd"}:
            plan["mode"] = "lbd"

    plan["patch_size"]=[128,128,64]
    DmCls = DataManagerDualDetBTfms if batch_tfms else DataManagerDualDet
    D = DmCls(
        project_title=project_title,
        configs=conf,
        batch_size=batch_size,
        cache_rate=conf["dataset_params"]["cache_rate"],
        device=device,
        ds_type=conf["dataset_params"]["ds_type"],
        train_indices=train_indices,
        val_indices=val_indices,
        val_sampling=val_sampling,
        debug=debug_,
        batch_tfms=batch_tfms,
    )
# %%
    D.prepare_data()

    D.setup(stage="fit")
    tmt = D.train_manager
    tmv = D.valid_manager

# %%
    dat = tmt.ds[0]
    dat[0].keys()
    img = dat[0]['image']
    lm  = dat[0]['lm']
    bbox = dat[0]['bbox']
# %%
#SECTION:-------------------- ts--------------------------------------------------------------------------------------
#SECTION:--- train dataloader ---
# %%
# %%
    tfms = tmt.transforms
 
    tfms = tmt.transforms_dict
#%%
    dici = dat[0]
    L = tfms["Ld"]
    dici = L(dici)

    R = tfms["Rtr"]
    dici = R(dici)
    dici2=dici[0]

    L2 = tfms["L2"]
    dici2 = L2(dici2)
    img = dici2['image']
    box = dici2['bbox']
    img.meta['affine']

    E = tfms["E"]
    dici2 = E(dici2)
# %%
    F1= tfms["F1"]
    dici2 = F1(dici2)
    print(dici2['image'].meta['affine'])
    print(dici2['bbox'][0])
    F2 = tfms["F2"]
# %%
    dici2=F2(dici2)
    print(dici2['image'].meta['affine'])
    img = dici2['image']
    box = dici2['bbox']
    print(dici2['bbox'][0])
# %%
    N = tfms["Norm"]
    dici2 = N(dici2)
    dici2['image'].min()
    print(dici2['bbox'][0])

    ImageBBoxViewer(img, box)
    A = tfms["Affine"]
# %%
    dici3 = A(dici2)
    img = dici3['image']
    box = dici3['bbox']
    print(img.meta['affine'])
    print(dici3['bbox'][0])

    ImageBBoxViewer(img,box)
# %%

    Re = tfms["ResizePC"]
    dici4 = Re(dici3)

    B = tfms["BoxClip"]
    dici4 = B(dici4)

    I = tfms["IntensityTfms"]
    dici4 = I(dici4)
    img = dici4['image']
    box = dici4['bbox']
    
    ImageBBoxViewer(img,box)
# %%


# %%
    print(dici.keys())
    tmt.setup()
    train_dl = tmt.dl
    print(f"train: {tmt}")
    print(f"train keys: {tmt.keys}")
    train_batch = next(iter(train_dl))
    train_batch.keys()
    train_batch["image"].shape

#SECTION:--- valid dataloader ---
# %%
    tmv.setup()
    val_dl = tmv.dl
    print(f"valid: {tmv}")
    print(f"valid keys: {tmv.keys}")
    print(f"valid_impl: {tmv.dataset_params['valid_impl']}")
    iteri=iter(val_dl)
# %%
    val_batch = next(iteri)
    val_batch.keys()
    val_batch['image'].meta


    img = val_batch["image"]
    bbox = val_batch[tmv.box_key]
# %%
    ImageBBoxViewer(img[1], bbox)
    val_batch["image"].shape
    val_batch.get("validation_impl")
    val_batch["case_id"]

# %%
