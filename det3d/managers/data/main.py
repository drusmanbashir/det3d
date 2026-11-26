"""Detection data managers (source, whole, patch, LBD, RBD) on FRAN's DataManager."""

import json
import os
import resource
from ast import literal_eval
from functools import partial
from pathlib import Path
from typing import Optional
import pandas as pd
import torch
from det3d.managers.data.collate import (
    det_val_collate,
    lbd_det_collate,
    lbd_det_collate_train,
    lbd_det_collate_val,
)
from det3d.managers.data.valid_patch_stream import PatchStreamDatasetDet, patch_stream_collate_fn
from fran.preprocessing.hdf5_shards import find_hdf5_shard_folder
from det3d.transforms.box_tracker import BOX_TRACKERS
from fran.transforms.detection import patch_size_manifest_key
from det3d.transforms.gpu_det import RandAffineBoxSyncd, RandMirrorBoxSyncd, ResizeWithPadOrCropBoxSyncd
from det3d.utils.bbox_sidecar import bbox_sidecar_path, load_detection_sidecar, valid_detection_box
from fran.utils.bbox_sidecar import extended_bbox_df_column, nbrhood_df_to_detection_records
from fran.configs.parser import ConfigMaker
from fran.configs.helpers import is_excel_None
from fran.managers.data.main import (
    DATA_MANAGER_BATCH_SIZE,
    DataManager,
    ForcedFgBatchSampler,
    ForcedFgDataset,
    DataManagerLBD,
    DataManagerPatch,
    DataManagerRBD,
    DataManagerShort,
    DataManagerSource,
    DataManagerWhole,
)
from fran.run.preproc.archive_preprocessed import ensure_rapid_data_folder
from fran.transforms.imageio import TorchReader
from monai.data import DataLoader, Dataset
from torch.utils.data import RandomSampler
from fran.transforms.intensitytransforms import NormaliseClipd
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
)
from fran.utils.folder_names import FolderNames
from utilz.stringz import info_from_filename
from utilz.fileio import load_json
from det3d.managers.data.transforms import (
    LoadHDF5DetShardExtendedBBoxd,
    RandCropExtendedBBoxd,
    CropDetPatchd,
    BboxCenterCropSlicesd,
    PadDetPatchd,
    LoadHDF5DetCropd,
    LoadHDF5DetInstancesd,
)


_FIXED = ConfigMaker.fixed()


class DetectionCropDataset(ForcedFgDataset):
    """Keep random crops on the drawn scan and draw a positive scan for its forced crops if needed."""

    def __init__(self, ds, crop, box_key):
        super().__init__(ds)
        self.crop = crop
        self.positive = [i for i, row in enumerate(ds.data) if len(row[box_key])]
        self.box_key = box_key

    def __getitem__(self, key):
        if self.crop.oversample_fg is None:
            index = key
            p = self.crop.pos / (self.crop.pos + self.crop.neg)
            n_forced = int((torch.rand(self.crop.num_samples) < p).sum())
        else:
            index, n_forced = key
        n_random = self.crop.num_samples - n_forced
        row = self.ds.data[index]
        if not n_forced or len(row[self.box_key]):
            return self.ds.transform(dict(row, crop_counts=(n_random, n_forced)))
        if not self.positive:
            raise ValueError("A foreground crop was requested but no training case has a detection box")
        draw = torch.randint(len(self.positive), ()).item()
        positive = self.ds.data[self.positive[draw]]
        patches = self.ds.transform(dict(row, crop_counts=(n_random, 0))) if n_random else []
        return patches + self.ds.transform(dict(positive, crop_counts=(0, n_forced)))


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
    AUGMENT_VALUES = ("monai",)  # future: nnDetection's base_more preset

    def __init__(
        self,
        project,
        configs: dict,
        batch_size=_FIXED["DET_BASE_DATA_MANAGER_BATCH_SIZE"],
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

        self.dataset_params = self.configs["dataset_params"]
        transform_factors = self.configs.get("transform_factors")
        if transform_factors:
            self._assimilate_tfm_factors(transform_factors)
        self.dataset_params.setdefault("valid_impl", "bbox_anchor")
        name = self.dataset_params.get("box_tracker", "corner")
        name = "corner" if is_excel_None(name) else name
        if name not in BOX_TRACKERS:
            raise ValueError(f"Unknown box_tracker {name!r}. Choose from {tuple(BOX_TRACKERS)}")
        self.box_tracker_name = name
        self.box_tracker = BOX_TRACKERS[name](self.box_key, self.label_key, self.image_key)

    def augment(self):
        """det3d's own `augment` row (det workbook data_params, merged over FRAN's dataset_params).
        `monai` = det3d's MONAI flip, affine and intensity transforms (`keys_tr`); FRAN's nnU-Net path stays off."""
        augment = self.dataset_params["augment"]
        if augment not in self.AUGMENT_VALUES:
            raise ValueError(
                f"det3d dataset_params.augment = {augment!r} is not allowed; allowed: {self.AUGMENT_VALUES}. "
                "Fix: set augment to monai in configurations/experiment_configs_det.xlsx, sheet data_params"
            )
        return augment

    def _training_spatial_keys(self):
        keys = [self.image_key]
        if self.uses_tracker_lm():
            keys.append(self.lm_key)
        if self.box_tracker_name == "remeasure":
            keys.append("inst")
        return keys

    def _create_mirror(self):
        return RandMirrorBoxSyncd(
            self._training_spatial_keys(), self.box_key, prob=self.mirror["prob"],
            axes=self.mirror_axes, tracker=self.box_tracker,
        )

    def uses_tracker_lm(self):  #AI
        """Load the semantic map for RetinaUNet or hull filtering."""
        return self.uses_lm_seg() or self.box_tracker_name == "hull"

    def _intensity_transform(self, keys):  #AI
        """Build detection normalization from the intensity properties for this split."""
        transform = NormaliseClipd.from_plan(
            keys=keys,
            plan=self.plan,
            stored_plan=self.plan,
            detection_params=self.dataset_params,
        )
        return transform

    def _valid_impl(self):
        return self.dataset_params["valid_impl"]

    def _size_divisible(self):
        from det3d.architectures.create_detector import size_divisible_from_conf

        return size_divisible_from_conf(self.configs)

    def _patch_size(self):
        return tuple(int(v) for v in self.plan["patch_size"])

    def uses_lm_seg(self):
        from det3d.architectures.create_detector import arch_from_conf

        return arch_from_conf(self.configs) == "retinaunet"

    def _rand_crop_patch_size(self): # garbage AI slop
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
        if soft_limit < _FIXED["DET_MIN_OPEN_FILES_FOR_WORKERS"]:
            return 0, False
        if self.is_train_all_split():
            return super()._num_workers()
        available_cpus = len(os.sched_getaffinity(0)) - _FIXED["DET_VALID_RESERVED_CPUS"]
        num_workers = min(_FIXED["DET_VALID_MAX_WORKERS"], max(1, available_cpus))
        return num_workers, False

    def train_batching(self):  #AI
        """Batch the resolved case draws and force the configured share of foreground slots."""
        sampler = self.epoch_sampler("train")
        ds = self.ds
        batching = dict(batch_size=self.effective_batch_size, shuffle=sampler is None, sampler=sampler)
        if self.oversample_fg() is not None and "Rtr" in self.keys:
            ds = ForcedFgDataset(self.ds)
            batch_sampler = ForcedFgBatchSampler(
                sampler or RandomSampler(self.ds),
                self.effective_batch_size,
                self.plan["samples_per_file"],
                self.oversample_fg(),
            )
            batching = dict(batch_sampler=batch_sampler)
        if self.dataset_params["nnz_allowed"] and "Rtr" in self.keys:
            ds = DetectionCropDataset(self.ds, self.Rtr, self.box_key)
        return ds, batching

    def create_dataloader(self):
        num_workers, persistent_workers = self._num_workers()
        if self.is_train_all_split():
            ds, batching = self.train_batching()
        else:
            sampler = None
            if "epochs" in self.configs["plan_train"]:
                sampler = self.epoch_sampler("valid", batch_size=1)
            ds, batching = self.ds, dict(batch_size=1, shuffle=False, sampler=sampler)
        pin_memory = torch.cuda.is_available() and not self.debug
        if num_workers > 0:
            persistent_workers = True
        dl_kwargs = dict(
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            **batching,
        )
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = _FIXED["DET_LOADER_PREFETCH_FACTOR"]
        self.dl = DataLoader(ds, **dl_kwargs)

    def derive_data_folder(self, plan):
        data_folder = Path(FolderNames(self.project, plan).lbd_folder)
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
        if (
            len(list(bboxes_dir.glob("*.csv"))) == 0
            and len(list(bboxes_dir.glob("*.json"))) == 0
        ):
            raise FileNotFoundError(f"No bbox sidecars under {bboxes_dir}")
        csv_fn = data_folder / "dataset_details.csv"
        if not csv_fn.is_file():
            raise FileNotFoundError(f"Missing dataset_details.csv under {data_folder}")
        return data_folder

    def cases_from_project_split(self):
        ds_tokens = [x.strip() for x in self.plan["datasources"].split(",") if x.strip()]
        full_volume = self.is_eval_split() and self._valid_impl() == "full_volume"
        nnz_allowed = full_volume or self.dataset_params["nnz_allowed"]
        if self.split == "all":
            self.cases = list(
                self.project.get_train_val_case_ids(
                    fold=None,
                    ds=ds_tokens,
                    nnz_allowed=nnz_allowed,
                )
            )
        else:
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
        include_empty = full_volume or (self.is_train_all_split() and nnz_allowed)
        if include_empty:
            missing = set(self.cases) - case_ids_on_disk
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} {self.split} volumes: {sorted(missing)[:5]}")
        self.cases = [case_id for case_id in self.cases if case_id in case_ids_on_disk]
        self.cases = self._filter_cases_by_stats_nnz(self.cases, include_empty)
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

    def _bbox_sidecar_ext_col(self):
        patch_size = self._patch_size()
        scale = float(self.dataset_params["prezoom_scale"])
        patch_size_prezoom = tuple(int(v * scale) for v in patch_size)
        return extended_bbox_df_column(patch_size_manifest_key(patch_size_prezoom))

    def _load_bbox_sidecar(self, bbox_fn):
        bbox_fn = Path(bbox_fn)
        if bbox_fn.suffix == ".csv":
            ext_col = self._bbox_sidecar_ext_col()
            columns = ["label_org", "bbox_xyzxyz", "cent", ext_col]
            if self.box_tracker_name == "hull":
                columns.append("hull_xyz")
                df = pd.read_csv(bbox_fn, usecols=lambda key: key in columns)
                if "hull_xyz" not in df or df["hull_xyz"].isna().any():
                    raise ValueError(f"Missing hull_xyz in {bbox_fn}; rerun LBD with --overwrite")
                df = df[columns]
            else:
                df = pd.read_csv(bbox_fn, usecols=columns)
            boxes, labels, _instances = nbrhood_df_to_detection_records(df)
        else:
            if self.box_tracker_name == "hull":
                raise ValueError(f"Missing hull_xyz in {bbox_fn}; rerun LBD with --overwrite")
            df = None
            boxes, labels, _instances = load_detection_sidecar(bbox_fn)
        valid_boxes = []
        valid_labels = []
        keep_idx = []
        for i, (box, label) in enumerate(zip(boxes, labels)):
            if valid_detection_box(box):
                valid_boxes.append(box.reshape(-1))
                valid_labels.append(label.reshape(-1))
                keep_idx.append(i)
        if self.box_tracker_name != "corner" and len(keep_idx) != len(boxes):
            raise ValueError(f"Invalid detection rows in {bbox_fn}; rerun LBD with --overwrite")
        if df is not None and len(keep_idx) != len(df):
            df = df.iloc[keep_idx].reset_index(drop=True)
        if len(valid_boxes) == 0:
            box_t = torch.zeros((0, 6), dtype=torch.float32)
            label_t = torch.zeros((0,), dtype=torch.long)
        else:
            box_t = torch.stack(valid_boxes)
            label_t = torch.stack(valid_labels).reshape(-1)
        labels_int = [int(v) for v in label_t.reshape(-1).tolist()]
        instances = {str(x): x for x in set(labels_int)}
        return box_t, label_t, instances, df

    @property
    def hdf5_manifest_fn(self):
        return find_hdf5_shard_folder(self.data_folder / "hdf5_shards") / "manifest.json"

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
                    if self.is_train_all_split() and self.dataset_params["nnz_allowed"]:
                        raise FileNotFoundError(bbox_fn)
                    skipped += 1
                    continue
                box_t, label_t, instances, df = self._load_bbox_sidecar(bbox_fn)
                row = {
                    "case_id": case_id,
                    "data_folder": str(self.data_folder),
                    "hdf5_shard_path": str(shard_path),
                    "hdf5_case_path": f"/cases/{case_id}",
                    self.box_key: box_t,
                    self.label_key: label_t,
                    "instances": instances,
                    "df": df,
                }
                if self.box_tracker_name == "hull":
                    row["hull_xyz"] = df["hull_xyz"].map(literal_eval).tolist()
                data.append(row)
        return data, skipped

    def _require_shard_manifest(self, data_folder):
        data_folder = Path(data_folder)
        manifest_fn = find_hdf5_shard_folder(data_folder / "hdf5_shards") / "manifest.json"
        if not manifest_fn.is_file():
            raise FileNotFoundError(f"Missing HDF5 shard manifest {manifest_fn}")

    def _shard_cases_from_project_split(self):
        ds_tokens = [x.strip() for x in self.plan["datasources"].split(",") if x.strip()]
        nnz_allowed = self.dataset_params["nnz_allowed"]
        train_cases, valid_cases = self.project.get_train_val_case_ids(
            self.dataset_params["fold"],
            ds_tokens,
            nnz_allowed=nnz_allowed,
        )
        self.cases = train_cases if self.is_train_split() else valid_cases
        on_disk = self._case_ids_on_disk()
        include_empty = self.is_train_all_split() and nnz_allowed
        if include_empty and (missing := set(self.cases) - on_disk):
            raise FileNotFoundError(f"Missing {len(missing)} training cases in {self.hdf5_manifest_fn}: {sorted(missing)[:5]}")
        self.cases = [case_id for case_id in self.cases if case_id in on_disk]
        self.cases = self._filter_cases_by_stats_nnz(self.cases, include_empty)
        assert len(self.cases) > 0, "There are no cases, aborting!"


class DataManagerDetSource(DataManagerDet, DataManagerSource):
    keys_tr = (
        "Ld,Rtr,L2,E,Norm,Mir,Affine,ResizePC,BoxClip,IntensityTfms"
    )

    def __init__(self, project, configs: dict, batch_size=DATA_MANAGER_BATCH_SIZE, cache_rate=0.0, **kwargs):
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
        ik, bk, lk = self.image_key, self.box_key, self.label_key
        spatial_aug_keys = self._training_spatial_keys()
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
        affine3d = self.configs["affine3d"]

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
            oversample_fg=self.oversample_fg(),
            src_dims_key="src_dims",
        )
        self.L2 = LoadHDF5DetCropd(keys=spatial_aug_keys, box_key=bk, tracker=self.box_tracker)
        self.E = EnsureChannelFirstd(keys=spatial_aug_keys)
        self.Affine = RandAffineBoxSyncd(
            spatial_keys=spatial_aug_keys,
            box_key=bk,
            mode=affine_modes,
            prob=affine3d["p"],
            rotate_range=affine3d["rotate_range"],
            scale_range=affine3d["scale_range"],
            tracker=self.box_tracker,
        )
        self.ResizePC = ResizeWithPadOrCropBoxSyncd(
            keys=spatial_aug_keys,
            box_key=bk,
            label_key=lk,
            spatial_size=patch_size,
            lazy=False,
            tracker=self.box_tracker,
        )
        self.Norm = self.transforms_dict["N"]
        self.BoxClip = self.box_tracker
        self.transforms_dict["Ld"] = self.Ld
        self.transforms_dict["Rtr"] = self.Rtr
        self.transforms_dict["L2"] = self.L2
        self.transforms_dict["E"] = self.E
        self.transforms_dict["Norm"] = self.Norm
        # LBD crops are padded on disk only to PATCH_MIN, which the patch planned after LBD can exceed.
        # End-pad short patches before collate and pass full-size patches through unchanged.
        self.transforms_dict["PadPatch"] = PadDetPatchd(keys=spatial_aug_keys, patch_size=patch_size_prezoom)
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

    def __init__(self, project, configs: dict, batch_size=DATA_MANAGER_BATCH_SIZE, cache_rate=0.0, **kwargs):
        provided_keys = kwargs["keys"] if "keys" in kwargs else None
        super().__init__(project, configs, batch_size, cache_rate, **kwargs)
        if provided_keys is None and self.is_eval_split():
            self.keys = self.validation_keys()
        self.override_batch_size_valid_split(split=self.split)

    def validation_keys(self):
        mode = self._valid_impl()
        if mode == "bbox_anchor":
            return self.keys_val_bbox
        if mode == "patch_stream" and self.uses_lm_seg() and self.box_tracker_name == "corner":
            return self.keys_val_seg
        return self.keys_val

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
                lm_key = self.lm_key if self.uses_tracker_lm() else None
                self.ds = PatchStreamDatasetDet(
                    case_dataset=case_ds,
                    patch_size=patch_size,
                    lm_key=lm_key,
                    tracker=self.box_tracker if self.box_tracker_name != "corner" else None,
                    instance_key="inst" if self.box_tracker_name == "remeasure" else None,
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
        full_volume = self._valid_impl() == "full_volume"
        num_workers, persistent_workers = (int(self.dataset_params["num_workers_val"]), False) if full_volume else self._num_workers()
        pin_memory = torch.cuda.is_available() and not self.debug
        if num_workers > 0:
            persistent_workers = True
        batch_size = 1 if full_volume else self.effective_batch_size
        sample_cases = not full_volume or "epochs" in self.configs["plan_train"]
        dl_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            sampler=self.epoch_sampler("valid", batch_size) if sample_cases else None,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
        )
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = _FIXED["DET_LOADER_PREFETCH_FACTOR"]
        self.dl = DataLoader(self.ds, **dl_kwargs)

    def create_valid_dataloader(self):
        self.create_dataloader()

    def cases_from_project_split(self):
        if self.is_eval_split():
            DataManagerDet.cases_from_project_split(self)
            return
        if self.split == "all":
            DataManagerDet.cases_from_project_split(self)
            return
        self._shard_cases_from_project_split()

    def create_data_dicts(self, case_ids):
        if self.is_train_all_split():
            return DataManagerDetSource.create_data_dicts(self, case_ids)
        case_ids = set(str(case_id) for case_id in case_ids)
        remeasure = self.box_tracker_name == "remeasure"
        if remeasure:
            shard_rows, _ = self._load_case_dicts_from_shards(case_ids)
            shard_rows = {row["case_id"]: row for row in shard_rows}
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
            if remeasure:
                if case_id not in shard_rows:
                    raise ValueError(f"Missing instance shard for {case_id}; rerun LBD with --overwrite")
                row = shard_rows[case_id]
            else:
                box_t, label_t, instances, df = self._load_bbox_sidecar(bbox_fn)
                row = {
                    "case_id": case_id,
                    "data_folder": str(self.data_folder),
                    self.box_key: box_t,
                    self.label_key: label_t,
                    "instances": instances,
                    "df": df,
                }
            row[self.image_key] = str(img_fn)
            if self.box_tracker_name == "hull":
                row["hull_xyz"] = df["hull_xyz"].map(literal_eval).tolist()
            if self.uses_tracker_lm():
                lm_fn = self.data_folder / "lms" / img_fn.name
                if not lm_fn.is_file():
                    skipped += 1
                    continue
                row[self.lm_key] = str(lm_fn)
            data.append(row)
        if self._valid_impl() == "full_volume":
            missing = case_ids.difference(row["case_id"] for row in data)
            if missing:
                raise FileNotFoundError(f"Missing masks or box sidecars for validation: {sorted(missing)[:5]}")
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
        use_lm = self.uses_tracker_lm()
        load_keys = [ik]
        if use_lm:
            load_keys.append(lmk)
        remeasure = self.box_tracker_name == "remeasure"
        compute_dtype = self._compute_dtype()

        L = LoadImaged(
            keys=load_keys,
            image_only=False,
            ensure_channel_first=False,
            simple_keys=True,
        )
        L.register(TorchReader())
        if remeasure:
            L = Compose([L, LoadHDF5DetInstancesd()])

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
            "Norm": self._intensity_transform(keys=[ik]),
            "DtypeVal": Compose(dtype_val),
        }
        box_clip = self.box_tracker
        if self.is_eval_split() and self._valid_impl() == "bbox_anchor":
            patch_size = self._rand_crop_patch_size()
            crop_keys = [*load_keys, "inst"] if remeasure else load_keys
            self.transforms_dict["BboxCrop"] = BboxCenterCropSlicesd(
                box_key=bk,
                image_key=ik,
                roi_size=patch_size,
            )
            self.transforms_dict["CropPatch"] = CropDetPatchd(
                keys=crop_keys, box_key=bk, tracker=self.box_tracker
            )
            self.transforms_dict["PadPatch"] = PadDetPatchd(
                keys=crop_keys,
                patch_size=patch_size,
            )
            self.transforms_dict["BoxClip"] = box_clip
        elif use_lm or remeasure:
            self.transforms_dict["BoxClip"] = box_clip


class DataManagerDetRBD(DataManagerDetLBD, DataManagerRBD):
    pass


class DataManagerDetShort(DataManagerDet, DataManagerShort):
    pass


class DataManagerDetPatch(DataManagerDet, DataManagerPatch):
    pass
