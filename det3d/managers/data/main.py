import json
import resource
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from det3d.collate import det_val_collate, lbd_det_collate
from det3d.utils.bbox_sidecar import bbox_sidecar_path, load_detection_sidecar, valid_detection_box
from det3d.utils.folder_names import lbd_det_folder_from_plan
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
from fran.transforms.intensitytransforms import RandRandGaussianNoised
from monai.apps.detection.transforms.dictionary import AffineBoxToWorldCoordinated, ClipBoxToImaged
from monai.data import DataLoader, Dataset, MetaTensor
from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    RandAdjustContrastd,
    RandFlipd,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
    ScaleIntensityRanged,
)
from monai.transforms.spatial.dictionary import ConvertBoxToPointsd, ConvertPointsToBoxesd
from monai.transforms.utility.dictionary import ApplyTransformToPointsd
from utilz.stringz import info_from_filename


class LoadHDF5DetShardIndexd(MapTransform):
    """Resolve case -> HDF5 shard; load mask-derived fg/bg flat indices."""

    def __init__(
        self,
        keys,
        manifest_fn: str,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.manifest_fn = Path(manifest_fn)
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

        src_dims = tuple(int(v) for v in manifest["src_dims"])
        cached = {
            "case_to_shard": case_to_shard,
            "src_dims": src_dims,
        }
        self._manifest_cache[manifest_key] = cached
        return cached

    @staticmethod
    def _read_flat_indices(case_grp):
        if "fg_indices" in case_grp:
            fg_key, bg_key = "fg_indices", "bg_indices"
        else:
            fg_key, bg_key = "lm_fg_indices", "lm_bg_indices"
        fg = np.asarray(case_grp[fg_key][:], dtype=np.int64).reshape(-1)
        bg = np.asarray(case_grp[bg_key][:], dtype=np.int64).reshape(-1)
        return fg, bg

    def __call__(self, data):
        d = dict(data)
        case_id = str(d["case_id"])
        manifest = self._cached_manifest()
        shard_path = manifest["case_to_shard"][case_id]
        h5py = import_h5py()
        case_path = f"/cases/{case_id}"
        with h5py.File(shard_path, "r") as h5f:
            case_grp = h5f[case_path]
            fg, bg = self._read_flat_indices(case_grp)
            src_dims = tuple(int(v) for v in case_grp["mask"].shape)
        d["hdf5_shard_path"] = str(shard_path)
        d["hdf5_case_path"] = case_path
        d["src_dims"] = src_dims
        d["fg_indices"] = fg
        d["bg_indices"] = bg
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
        with h5py.File(shard_path, "r") as h5f:
            case_grp = h5f[case_path]
            image = np.asarray(case_grp["image"][crop_slices])
            mask = np.asarray(case_grp["mask"][crop_slices])
        if image.ndim == 3:
            image = image[np.newaxis, ...]
        if mask.ndim == 3:
            mask = mask[np.newaxis, ...]
        meta = {
            "filename_or_obj": f"{shard_path}:{case_path}",
            "case_id": d.get("case_id"),
            "crop_start": crop_start,
            "crop_end": d.get("crop_end"),
            "sampled_flat_index": d.get("sampled_flat_index"),
            "sample_is_fg": d.get("sample_is_fg"),
            "original_channel_dim": 0,
        }
        d["image"] = MetaTensor(torch.as_tensor(image), meta=dict(meta))
        d["mask"] = MetaTensor(torch.as_tensor(mask), meta=dict(meta))
        box = torch.as_tensor(d[self.box_key], dtype=torch.float32)
        if box.numel() > 0:
            start = torch.tensor(crop_start, dtype=box.dtype)
            box = box.clone()
            box[:, :3] -= start
            box[:, 3:] -= start
            d[self.box_key] = box
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

    def create_dataset(self):
        self.ds = self._create_modal_ds()

    def _create_modal_ds(self):
        if is_excel_None(self.ds_type):
            ds = Dataset(data=self.data, transform=self.transforms)
        else:
            ds = super()._create_modal_ds()
        return ds

    def maybe_fix_remapping_dtype(self):
        pass

    def set_preprocessing_params(self):
        self.dataset_params = self.configs["dataset_params"]
        transform_factors = self.configs.get("transform_factors")
        if transform_factors:
            self._assimilate_tfm_factors(transform_factors)

    def set_effective_batch_size(self):
        self.effective_batch_size = self.batch_size

    def _size_divisible(self):
        plan = self.plan
        return [
            step * 2 * 2 ** max(plan["returned_layers"])
            for step in plan["conv1_t_stride"]
        ]

    def _patch_size(self):
        return tuple(int(v) for v in self.plan["patch_size"])

    def _set_collate_fn(self):
        if self.is_eval_split():
            self.collate_fn = partial(
                det_val_collate,
                box_key=self.box_key,
                label_key=self.label_key,
            )
            return
        self.collate_fn = partial(
            lbd_det_collate,
            size_divisible=self._size_divisible(),
            box_key=self.box_key,
        )

    def _compute_dtype(self):
        return torch.float16 if self.amp else torch.float32

    def _num_workers(self):
        if self.debug:
            return 0, False
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_limit < 1024:
            return 0, False
        if self.is_train_split():
            num_workers = min(4, max(2, self.effective_batch_size // 8))
        else:
            num_workers = min(4, max(2, int(self.plan.get("num_workers_val", 2))))
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
        data_folder = lbd_det_folder_from_plan(self.project, plan)
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
        boxes, labels = load_detection_sidecar(bbox_fn)
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
        return box_t, label_t

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
                box_t, label_t = self._load_bbox_sidecar(bbox_fn)
                data.append(
                    {
                        "case_id": case_id,
                        "data_folder": str(self.data_folder),
                        "hdf5_shard_path": str(shard_path),
                        "hdf5_case_path": f"/cases/{case_id}",
                        self.box_key: box_t,
                        self.label_key: label_t,
                    }
                )
        return data, skipped

    def _require_shard_manifest(self, data_folder):
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
    data_keys = ("image", "mask")

    keys_tr = (
        "Ld,Rtr,L2,E,Norm,BoxToWorld,ToPoints,Zoom,Flip0,Flip1,Flip2,Rand90,Rot,"
        "AffinePts,ToBoxes,BoxClip,DelMask,IntensityTfms,Dtype"
    )

    def __init__(self, project, configs: dict, batch_size=8, cache_rate=0.0, **kwargs):
        provided_keys = kwargs["keys"] if "keys" in kwargs else None
        super().__init__(project, configs, batch_size, cache_rate, **kwargs)
        if provided_keys is None and self.uses_train_keys():
            self.keys = self.keys_tr

    def derive_data_folder(self, plan):
        data_folder = super(DataManagerDet, self).derive_data_folder(plan)
        self._require_shard_manifest(data_folder)
        masks_dir = data_folder / "masks"
        if not masks_dir.is_dir():
            raise FileNotFoundError(f"Expected masks/ under {data_folder}")
        if len(list(masks_dir.glob("*.pt"))) == 0:
            raise FileNotFoundError(f"No label-bounded masks under {masks_dir}")
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
        ik, bk, lk, mk, pk = (
            self.image_key,
            self.box_key,
            self.label_key,
            self.mask_key,
            self.point_key,
        )
        load_keys = list(self.data_keys)
        patch_size = self._patch_size()
        plan = self.plan
        compute_dtype = self._compute_dtype()
        affine_lps_to_ras = self.affine_lps_to_ras

        IntensityTfms = [
            RandScaleIntensityd(
                keys=[ik], factors=self.scale["value"], prob=self.scale["prob"]
            ),
            RandRandGaussianNoised(
                keys=[ik], std_limits=self.noise["value"], prob=self.noise["prob"]
            ),
            RandShiftIntensityd(
                keys=[ik], offsets=self.shift["value"], prob=self.shift["prob"]
            ),
            RandAdjustContrastd(
                keys=[ik], gamma=self.contrast["value"], prob=self.contrast["prob"]
            ),
        ]

        self.transforms_dict = {
            "Ld": LoadHDF5DetShardIndexd(
                keys=["case_id"],
                manifest_fn=str(self.hdf5_manifest_fn),
            ),
            "Rtr": RandCropByFlatIndicesd(
                keys=[ik],
                roi_size=patch_size,
                num_samples=int(plan["samples_per_file"]),
                pos=1,
                neg=1,
                fg_indices_key="fg_indices",
                bg_indices_key="bg_indices",
            ),
            "L2": LoadHDF5DetCropd(keys=load_keys, box_key=bk),
            "E": EnsureChannelFirstd(keys=load_keys),
            "Norm": ScaleIntensityRanged(
                keys=[ik],
                a_min=float(plan["intensity_a_min"]),
                a_max=float(plan["intensity_a_max"]),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            "BoxToWorld": AffineBoxToWorldCoordinated(
                box_keys=[bk],
                box_ref_image_keys=ik,
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            "ToPoints": ConvertBoxToPointsd(keys=[bk]),
            "Zoom": RandZoomd(
                keys=[ik],
                prob=0.2,
                min_zoom=0.7,
                max_zoom=1.4,
                padding_mode="constant",
                keep_size=True,
            ),
            "Flip0": RandFlipd(keys=[ik], prob=0.5, spatial_axis=0),
            "Flip1": RandFlipd(keys=[ik], prob=0.5, spatial_axis=1),
            "Flip2": RandFlipd(keys=[ik], prob=0.5, spatial_axis=2),
            "Rand90": RandRotate90d(keys=[ik], prob=0.75, max_k=3, spatial_axes=(0, 1)),
            "Rot": RandRotated(
                keys=[ik],
                mode="nearest",
                prob=0.2,
                range_x=np.pi / 6,
                range_y=np.pi / 6,
                range_z=np.pi / 6,
                keep_size=True,
                padding_mode="zeros",
            ),
            "AffinePts": ApplyTransformToPointsd(
                keys=[pk], refer_keys=ik, affine_lps_to_ras=affine_lps_to_ras
            ),
            "ToBoxes": ConvertPointsToBoxesd(keys=[pk], box_key=bk),
            "BoxClip": ClipBoxToImaged(
                box_keys=bk,
                label_keys=[lk],
                box_ref_image_keys=ik,
                remove_empty=True,
            ),
            "DelMask": DeleteItemsd(keys=[mk]),
            "IntensityTfms": IntensityTfms,
            "Dtype": Compose(
                [
                    EnsureTyped(keys=[ik], dtype=compute_dtype),
                    EnsureTyped(keys=[bk], dtype=torch.float32),
                    EnsureTyped(keys=[lk], dtype=torch.long),
                ]
            ),
        }
        ld = self.transforms_dict["Ld"]
        if type(ld) is not LoadHDF5DetShardIndexd:
            raise RuntimeError(
                f"det shard train must use LoadHDF5DetShardIndexd, got {type(ld)}"
            )


class DataManagerDetWhole(DataManagerDet, DataManagerWhole):
    pass


class DataManagerDetLBD(DataManagerDetSource, DataManagerLBD):
    keys_val = "L,E,Norm,DtypeVal"
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
            self.keys = self.keys_val

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
            box_t, label_t = self._load_bbox_sidecar(bbox_fn)
            data.append(
                {
                    "case_id": case_id,
                    "data_folder": str(self.data_folder),
                    "image": str(img_fn),
                    self.box_key: box_t,
                    self.label_key: label_t,
                }
            )
        if skipped:
            print(
                f"DataManagerDetLBD: skipped {skipped} cases "
                "(missing sidecar)"
            )
        return data

    def create_transforms(self):
        if self.is_train_all_split():
            return DataManagerDetSource.create_transforms(self)
        ik, bk, lk = self.image_key, self.box_key, self.label_key
        load_keys = [self.image_key]
        compute_dtype = self._compute_dtype()
        plan = self.plan

        L = LoadImaged(
            keys=load_keys,
            image_only=False,
            ensure_channel_first=False,
            simple_keys=True,
        )
        L.register(TorchReader())

        self.transforms_dict = {
            "L": L,
            "E": EnsureChannelFirstd(keys=load_keys),
            "Norm": ScaleIntensityRanged(
                keys=[ik],
                a_min=float(plan["intensity_a_min"]),
                a_max=float(plan["intensity_a_max"]),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            "DtypeVal": Compose(
                [
                    EnsureTyped(keys=[ik], dtype=compute_dtype),
                    EnsureTyped(keys=[bk], dtype=torch.float32),
                    EnsureTyped(keys=[lk], dtype=torch.long),
                ]
            ),
        }


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
        lbd_folder = lbd_det_folder_from_plan(self.project, self.configs["plan_train"])
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
