import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from det.collate import obd_det_collate
from functools import partial
from det.utils.bbox_sidecar import bbox_sidecar_path, load_detection_sidecar
from det.utils.folder_names import lbd_det_folder_from_plan, obd_folder_from_plan
from fran.managers.data.main import (
    DataManager,
    DataManagerDual,
    LoadHDF5ShardIndexd,
)
from fran.run.preproc.archive_preprocessed import ensure_rapid_data_folder
from fran.transforms.imageio import TorchReader
from monai.apps.detection.transforms.dictionary import ClipBoxToImaged
from monai.data import DataLoader, MetaTensor
from fran.transforms.intensitytransforms import RandRandGaussianNoised
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    RandAdjustContrastd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
)
from monai.transforms.spatial.dictionary import ConvertBoxToPointsd, ConvertPointsToBoxesd
from monai.transforms.utility.dictionary import ApplyTransformToPointsd
from fran.preprocessing.helpers import import_h5py
from utilz.stringz import info_from_filename


PATCH_FNAME_RE = re.compile(r"^(?P<case_id>.+)_bbox(?P<idx>\d+)\.pt$")


def _valid_detection_box(box):
    b = torch.as_tensor(box).flatten()
    if b.numel() != 6:
        return False
    if (b[3:] < 1).any():
        return False
    return True


class LoadHDF5CaseFulld(MapTransform):
    """Load a full case image tensor from an HDF5 shard (fran LBD shard layout)."""

    def __init__(
        self,
        keys,
        shard_path_key: str = "hdf5_shard_path",
        case_path_key: str = "hdf5_case_path",
        allow_missing_keys: bool = False,
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
            case_grp = h5f[case_path]
            image = np.asarray(case_grp["image"][:])
        if image.ndim == 3:
            image = image[np.newaxis, ...]
        filename_or_obj = f"{shard_path}:{case_path}"
        meta = {
            "filename_or_obj": filename_or_obj,
            "case_id": d.get("case_id"),
            "original_channel_dim": 0,
        }
        d["image"] = MetaTensor(torch.as_tensor(image), meta=dict(meta))
        return d


class _DetManagerBase(DataManager):
    data_keys = ("image",)
    spatial_aug_keys = ("image",)
    image_key = "image"
    box_key = "box"
    label_key = "label"
    point_key = "points"

    def __init__(
        self,
        project,
        configs: dict,
        batch_size=4,
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
        if keys is None:
            keys = self.keys_tr if split in ("train", "all") else self.keys_val
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
        self.affine_lps_to_ras = bool(self.plan.get("affine_lps_to_ras", True))

    def maybe_fix_remapping_dtype(self):
        pass

    def set_preprocessing_params(self):
        self.dataset_params = self.configs["dataset_params"]
        transform_factors = self.configs.get("transform_factors")
        if transform_factors:
            self._assimilate_tfm_factors(transform_factors)

    def set_effective_batch_size(self):
        self.effective_batch_size = self.batch_size

    def _compute_size_divisible(self):
        plan = self.plan
        return [
            step * 2 * 2 ** max(plan["returned_layers"])
            for step in plan["conv1_t_stride"]
        ]

    def _set_collate_fn(self):
        self.collate_fn = partial(
            obd_det_collate, size_divisible=self._compute_size_divisible()
        )

    def _compute_dtype(self):
        return torch.float16 if self.amp else torch.float32

    def _intensity_tfms(self, ik):
        scale = getattr(self, "scale", {"value": 0.1, "prob": 0.15})
        noise = getattr(self, "noise", {"value": (0.0, 0.1), "prob": 0.15})
        shift = getattr(self, "shift", {"value": 0.1, "prob": 0.15})
        contrast = getattr(self, "contrast", {"value": (0.7, 1.3), "prob": 0.3})
        return [
            RandScaleIntensityd(keys=[ik], factors=scale["value"], prob=scale["prob"]),
            RandRandGaussianNoised(
                keys=[ik], std_limits=noise["value"], prob=noise["prob"]
            ),
            RandShiftIntensityd(keys=[ik], offsets=shift["value"], prob=shift["prob"]),
            RandAdjustContrastd(
                keys=[ik], gamma=contrast["value"], prob=contrast["prob"]
            ),
        ]

    def _box_aug_transforms(self):
        ik, bk, lk, pk = self.image_key, self.box_key, self.label_key, self.point_key
        return {
            "BoxToPts": ConvertBoxToPointsd(keys=[bk], point_key=pk),
            "BoxPtAug": ApplyTransformToPointsd(
                keys=[pk], refer_keys=ik, affine_lps_to_ras=self.affine_lps_to_ras
            ),
            "BoxConv": ConvertPointsToBoxesd(keys=[pk], box_key=bk),
            "BoxClip": ClipBoxToImaged(
                box_keys=bk,
                label_keys=[lk],
                box_ref_image_keys=ik,
                remove_empty=False,
            ),
        }

    def create_dataloader(self):
        if self.is_train_split():
            num_workers = int(self.plan.get("num_workers_train", 7))
        else:
            num_workers = int(self.plan.get("num_workers_val", 2))
        if self.debug:
            num_workers = 0
        persistent_workers = num_workers > 0
        batch_size = 1 if self.is_eval_split() else self.effective_batch_size
        self.dl = DataLoader(
            self.ds,
            batch_size=batch_size,
            shuffle=self.is_train_split(),
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=persistent_workers,
            pin_memory=not self.debug and torch.cuda.is_available(),
        )


class DataManagerDetOBD(_DetManagerBase):
    keys_tr = (
        "L,E,BoxToPts,F1,F2,F3,Rand90,DetZoom,BoxPtAug,BoxConv,BoxClip,IntensityTfms,Dtype"
    )
    keys_val = "L,E,Dtype"

    def derive_data_folder(self, plan):
        data_folder = obd_folder_from_plan(self.project, plan)
        data_folder = ensure_rapid_data_folder(data_folder)
        if not data_folder.exists():
            raise FileNotFoundError(f"Data folder {data_folder} does not exist")
        images_dir = data_folder / "images"
        lms_dir = data_folder / "lms"
        bboxes_dir = data_folder / "bboxes"
        if not images_dir.is_dir() or not lms_dir.is_dir() or not bboxes_dir.is_dir():
            raise FileNotFoundError(
                f"Expected images/, lms/, and bboxes/ under {data_folder}"
            )
        if len(list(images_dir.glob("*_bbox*.pt"))) == 0:
            raise FileNotFoundError(f"No object-bounded patches under {images_dir}")
        if len(list(bboxes_dir.glob("*.pt"))) > 0:
            raise FileNotFoundError(
                f"Legacy bbox .pt sidecars under {bboxes_dir}; re-preproc to JSON"
            )
        if len(list(bboxes_dir.glob("*.json"))) == 0:
            raise FileNotFoundError(f"No bbox JSON sidecars under {bboxes_dir}")
        return data_folder

    def cases_from_project_split(self):
        ds_tokens = [x.strip() for x in self.plan["datasources"].split(",") if x.strip()]
        train_cases, valid_cases = self.project.get_train_val_case_ids(
            self.dataset_params["fold"],
            ds_tokens,
            nnz_allowed=self.plan.get("nnz_allowed", False),
        )
        self.cases = train_cases if self.is_train_split() else valid_cases
        patch_case_ids = set()
        for img_fn in (self.data_folder / "images").glob("*_bbox*.pt"):
            match = PATCH_FNAME_RE.match(img_fn.name)
            patch_case_ids.add(match.group("case_id"))
        self.cases = [case_id for case_id in self.cases if case_id in patch_case_ids]
        assert len(self.cases) > 0, "There are no cases, aborting!"

    def create_data_dicts(self, case_ids):
        case_ids = set(case_ids)
        data = []
        images_dir = self.data_folder / "images"
        bboxes_dir = self.data_folder / "bboxes"
        skipped = 0
        for img_fn in sorted(images_dir.glob("*_bbox*.pt")):
            match = PATCH_FNAME_RE.match(img_fn.name)
            case_id = match.group("case_id")
            if case_id not in case_ids:
                continue
            lm_fn = self.data_folder / "lms" / img_fn.name
            bbox_fn = bbox_sidecar_path(bboxes_dir, img_fn.stem)
            if not lm_fn.is_file() or not bbox_fn.is_file():
                skipped += 1
                continue
            boxes, labels = load_detection_sidecar(bbox_fn)
            box = boxes[0]
            label = labels[0]
            if not _valid_detection_box(box):
                skipped += 1
                continue
            data.append(
                {
                    "case_id": case_id,
                    "data_folder": str(self.data_folder),
                    "image": str(img_fn),
                    self.box_key: box.reshape(1, -1),
                    self.label_key: label.reshape(-1),
                }
            )
        if skipped:
            print(
                f"DataManagerDetOBD: skipped {skipped} patches "
                "(missing sidecar or invalid box)"
            )
        return data

    def create_transforms(self):
        ik, bk, lk, pk = self.image_key, self.box_key, self.label_key, self.point_key
        compute_dtype = self._compute_dtype()
        data_keys = list(self.data_keys)
        spatial_keys = list(self.spatial_aug_keys)

        L = LoadImaged(
            keys=data_keys,
            image_only=False,
            ensure_channel_first=False,
            simple_keys=True,
        )
        L.register(TorchReader())

        self.transforms_dict = {
            "L": L,
            "E": EnsureChannelFirstd(keys=data_keys),
            "F1": RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=0),
            "F2": RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=1),
            "F3": RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=2),
            "Rand90": RandRotate90d(
                keys=[ik], prob=0.75, max_k=3, spatial_axes=(0, 1)
            ),
            "DetZoom": RandZoomd(
                keys=[ik],
                prob=0.2,
                min_zoom=0.7,
                max_zoom=1.4,
                padding_mode="constant",
                keep_size=True,
            ),
            "IntensityTfms": self._intensity_tfms(ik),
            "Dtype": Compose(
                [
                    EnsureTyped(keys=[ik], dtype=compute_dtype),
                    EnsureTyped(keys=[bk], dtype=compute_dtype),
                    EnsureTyped(keys=[lk], dtype=torch.long),
                ]
            ),
        }
        self.transforms_dict.update(self._box_aug_transforms())

    def __repr__(self):
        n = len(self.data) if hasattr(self, "data") and self.data else 0
        return f"DataManagerDetOBD(split={self.split}, n={n})"


class DataManagerDetLBD(_DetManagerBase):
    keys_tr = (
        "L,E,BoxToPts,F1,F2,F3,Rand90,DetZoom,BoxPtAug,BoxConv,BoxClip,IntensityTfms,Dtype"
    )
    keys_val = "L,E,DtypeVal"
    keys_val_shard = "Ld,Lfull,E,DtypeVal"

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
        return data_folder

    @property
    def hdf5_manifest_fn(self):
        src_tag = "_".join(str(int(v)) for v in self.plan["src_dims"])
        return self.data_folder / "hdf5_shards" / f"src_{src_tag}" / "manifest.json"

    def uses_hdf5_shards(self):
        return self.hdf5_manifest_fn.is_file()

    def cases_from_project_split(self):
        ds_tokens = [x.strip() for x in self.plan["datasources"].split(",") if x.strip()]
        train_cases, valid_cases = self.project.get_train_val_case_ids(
            self.dataset_params["fold"],
            ds_tokens,
            nnz_allowed=self.plan.get("nnz_allowed", False),
        )
        self.cases = train_cases if self.is_train_split() else valid_cases
        if self.uses_hdf5_shards():
            manifest = json.loads(self.hdf5_manifest_fn.read_text())
            case_ids_on_disk = set()
            for shard_info in manifest["shards"]:
                case_ids_on_disk.update(str(case_id) for case_id in shard_info["case_ids"])
        else:
            case_ids_on_disk = set()
            for img_fn in (self.data_folder / "images").glob("*.pt"):
                case_ids_on_disk.add(
                    info_from_filename(img_fn.name, full_caseid=True)["case_id"]
                )
        self.cases = [case_id for case_id in self.cases if case_id in case_ids_on_disk]
        assert len(self.cases) > 0, "There are no cases, aborting!"

    def _load_bbox_sidecar(self, bbox_fn):
        boxes, labels = load_detection_sidecar(bbox_fn)
        valid_boxes = []
        valid_labels = []
        for box, label in zip(boxes, labels):
            if _valid_detection_box(box):
                valid_boxes.append(box.reshape(-1))
                valid_labels.append(label.reshape(-1))
        box_t = torch.stack(valid_boxes)
        label_t = torch.stack(valid_labels).reshape(-1)
        return box_t, label_t

    def create_data_dicts(self, case_ids):
        case_ids = set(str(case_id) for case_id in case_ids)
        data = []
        bboxes_dir = self.data_folder / "bboxes"
        skipped = 0

        if self.uses_hdf5_shards():
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
                    if box_t.shape[0] == 0:
                        skipped += 1
                        continue
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
            if skipped:
                print(
                    f"DataManagerDetLBD: skipped {skipped} shard cases "
                    "(missing sidecar or invalid boxes)"
                )
            return data

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
            if box_t.shape[0] == 0:
                skipped += 1
                continue
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
                "(missing sidecar or invalid boxes)"
            )
        return data

    def create_transforms(self):
        ik, bk, lk = self.image_key, self.box_key, self.label_key
        compute_dtype = self._compute_dtype()
        data_keys = list(self.data_keys)
        spatial_keys = list(self.spatial_aug_keys)

        L = LoadImaged(
            keys=data_keys,
            image_only=False,
            ensure_channel_first=False,
            simple_keys=True,
        )
        L.register(TorchReader())

        self.transforms_dict = {
            "L": L,
            "Lfull": LoadHDF5CaseFulld(keys=data_keys),
            "Ld": LoadHDF5ShardIndexd(
                keys=["case_id"],
                manifest_fn=str(self.hdf5_manifest_fn),
            ),
            "E": EnsureChannelFirstd(keys=data_keys),
            "F1": RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=0),
            "F2": RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=1),
            "F3": RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=2),
            "Rand90": RandRotate90d(
                keys=[ik], prob=0.75, max_k=3, spatial_axes=(0, 1)
            ),
            "DetZoom": RandZoomd(
                keys=[ik],
                prob=0.2,
                min_zoom=0.7,
                max_zoom=1.4,
                padding_mode="constant",
                keep_size=True,
            ),
            "IntensityTfms": self._intensity_tfms(ik),
            "Dtype": Compose(
                [
                    EnsureTyped(keys=[ik], dtype=compute_dtype),
                    EnsureTyped(keys=[bk], dtype=compute_dtype),
                    EnsureTyped(keys=[lk], dtype=torch.long),
                ]
            ),
            "DtypeVal": EnsureTyped(keys=[ik], dtype=compute_dtype),
        }
        self.transforms_dict.update(self._box_aug_transforms())

    def set_transforms(self, keys):
        if self.is_eval_split() and self.uses_hdf5_shards():
            keys = self.keys_val_shard
        self.keys = keys
        super().set_transforms(keys)

    def __repr__(self):
        n = len(self.data) if hasattr(self, "data") and self.data else 0
        return f"DataManagerDetLBD(split={self.split}, n={n}, shards={self.uses_hdf5_shards()})"


class DataManagerDualDet(DataManagerDual):
    def _build_managers(self):
        self.train_manager = DataManagerDetOBD(
            project=self.project,
            configs=self.configs,
            batch_size=self.batch_size,
            cache_rate=self.cache_rate,
            split="train",
            device=self.device,
            ds_type=self.ds_type,
            keys=DataManagerDetOBD.keys_tr,
            data_folder=self.data_folder,
            debug=self.debug,
        )
        lbd_folder = lbd_det_folder_from_plan(self.project, self.configs["plan_train"])
        self.valid_manager = DataManagerDetLBD(
            project=self.project,
            configs=self.configs,
            batch_size=self.batch_size,
            cache_rate=self.cache_rate,
            split="valid",
            device=self.device,
            ds_type=self.ds_type,
            keys=DataManagerDetLBD.keys_val,
            data_folder=lbd_folder,
            val_sampling=self.val_sampling,
            debug=self.debug,
        )

    def __repr__(self):
        return (
            f"DataManagerDualDet(train={self.train_manager}, valid={self.valid_manager})"
        )


# Backward-compatible alias
DataManagerDet = DataManagerDetOBD
