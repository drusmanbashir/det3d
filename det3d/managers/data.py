from utilz.imageviewers import ImageBBoxViewer, ImageMaskBboxViewer
import json
import resource
from functools import partial
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from det3d.collate import lbd_det_collate
from det3d.utils.bbox_sidecar import bbox_sidecar_path, load_detection_sidecar
from det3d.utils.folder_names import lbd_det_folder_from_plan
from monai.apps.detection.transforms.dictionary import AffineBoxToWorldCoordinated, ClipBoxToImaged
from monai.data import DataLoader, MetaTensor
from monai.transforms import Compose, DeleteItemsd, EnsureChannelFirstd, EnsureTyped, LoadImaged, MapTransform, RandAdjustContrastd, RandCropByPosNegLabeld, RandFlipd, RandRotate90d, RandRotated, RandScaleIntensityd, RandShiftIntensityd, RandZoomd, ScaleIntensityRanged
from monai.transforms.spatial.dictionary import ConvertBoxToPointsd, ConvertPointsToBoxesd
from monai.transforms.utility.dictionary import ApplyTransformToPointsd
import torch
from fran.managers.data.main import DataManager, DataManagerDual, LoadHDF5ShardIndexd
from fran.run.preproc.archive_preprocessed import ensure_rapid_data_folder
from fran.transforms.imageio import TorchReader
from fran.transforms.intensitytransforms import RandRandGaussianNoised
from fran.preprocessing.helpers import import_h5py
from utilz.stringz import info_from_filename


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
    box_key = "bbox"
    label_key = "label"
    point_key = "points"
    mask_key = "mask"

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
        # TorchReader PT affines are already in stored image space; not ITK LPS.
        self.affine_lps_to_ras = False

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
            self.collate_fn = None
            return
        self.collate_fn = partial(
            lbd_det_collate,
            size_divisible=self._size_divisible(),
            box_key=self.box_key,
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

    def _norm_transform(self, ik):
        plan = self.plan
        return ScaleIntensityRanged(
            keys=[ik],
            a_min=float(plan["intensity_a_min"]),
            a_max=float(plan["intensity_a_max"]),
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )

    def _train_spatial_transforms(self):
        ik, bk, lk, pk, mk = (
            self.image_key,
            self.box_key,
            self.label_key,
            self.point_key,
            self.mask_key,
        )
        patch_size = self._patch_size()
        affine_lps_to_ras = self.affine_lps_to_ras
        return {
            "BoxToWorld": AffineBoxToWorldCoordinated(
                box_keys=[bk],
                box_ref_image_keys=ik,
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            "ToPoints": ConvertBoxToPointsd(keys=[bk]),
            "RandCrop": RandCropByPosNegLabeld(
                keys=[ik],
                label_key=mk,
                spatial_size=patch_size,
                num_samples=int(self.plan["samples_per_file"]),
                pos=1,
                neg=1,
            ),
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
        }

    def _num_workers(self):
        if self.debug:
            return 0, False
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_limit < 1024:
            return 0, False
        if self.is_train_split():
            num_workers = min(4, max(2, self.effective_batch_size // 8))
        else:
            num_workers = 0
        return num_workers, False

    def create_dataloader(self):
        num_workers, persistent_workers = self._num_workers()
        if self.is_train_split():
            batch_size = self.effective_batch_size
            collate_fn = self.collate_fn
        else:
            batch_size = 1
            collate_fn = None
        self.dl = DataLoader(
            self.ds,
            batch_size=batch_size,
            shuffle=self.is_train_split(),
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
            pin_memory=False,
        )


class _LBDDetMixin:
    def derive_data_folder(self, plan, require_masks=False):
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
        if require_masks:
            masks_dir = data_folder / "masks"
            if not masks_dir.is_dir():
                raise FileNotFoundError(f"Expected masks/ under {data_folder}")
            if len(list(masks_dir.glob("*.pt"))) == 0:
                raise FileNotFoundError(f"No label-bounded masks under {masks_dir}")
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
        if self.is_train_split():
            return False
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
        return data, skipped

    def set_transforms(self, keys):
        if self.is_eval_split() and self.uses_hdf5_shards():
            keys = self.keys_val_shard
        self.keys = keys
        super().set_transforms(keys)


class DataManagerTrainDet(_LBDDetMixin, _DetManagerBase):
    data_keys = ("image", "mask")

    keys_tr = (
        "L,E,Norm,BoxToWorld,ToPoints,RandCrop,Zoom,Flip0,Flip1,Flip2,Rand90,Rot,"
        "AffinePts,ToBoxes,BoxClip,DelMask,IntensityTfms,Dtype"
    )

    def derive_data_folder(self, plan):
        return super().derive_data_folder(plan, require_masks=True)

    def set_effective_batch_size(self):
        spf = int(self.plan["samples_per_file"])
        assert self.batch_size % spf == 0, (
            f"batch_size {self.batch_size} must be divisible by "
            f"samples_per_file {spf}"
        )
        self.effective_batch_size = self.batch_size // spf

    def create_data_dicts(self, case_ids):
        case_ids = set(str(case_id) for case_id in case_ids)
        data = []
        bboxes_dir = self.data_folder / "bboxes"
        masks_dir = self.data_folder / "masks"
        skipped = 0
        images_dir = self.data_folder / "images"
        for img_fn in sorted(images_dir.glob("*.pt")):
            case_id = info_from_filename(img_fn.name, full_caseid=True)["case_id"]
            if case_id not in case_ids:
                continue
            bbox_fn = bbox_sidecar_path(bboxes_dir, img_fn.stem)
            mask_fn = masks_dir / img_fn.name
            if not bbox_fn.is_file() or not mask_fn.is_file():
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
                    self.mask_key: str(mask_fn),
                    self.box_key: box_t,
                    self.label_key: label_t,
                }
            )
        if skipped:
            print(
                f"DataManagerTrainDet: skipped {skipped} cases "
                "(missing sidecar/mask or invalid boxes)"
            )
        return data

    def create_transforms(self):
        ik, bk, lk, mk = self.image_key, self.box_key, self.label_key, self.mask_key
        compute_dtype = self._compute_dtype()
        load_keys = list(self.data_keys)

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
            "Norm": self._norm_transform(ik),
            "IntensityTfms": self._intensity_tfms(ik),
            "Dtype": Compose(
                [
                    EnsureTyped(keys=[ik], dtype=compute_dtype),
                    EnsureTyped(keys=[bk], dtype=torch.float32),
                    EnsureTyped(keys=[lk], dtype=torch.long),
                ]
            ),
        }
        self.transforms_dict.update(self._train_spatial_transforms())

    def __repr__(self):
        n = len(self.data) if hasattr(self, "data") and self.data else 0
        return f"DataManagerTrainDet(split={self.split}, n={n})"


class DataManagerDetLBD(_LBDDetMixin, _DetManagerBase):
    keys_val = "L,E,Norm,DtypeVal"
    keys_val_shard = "Ld,Lfull,E,Norm,DtypeVal"

    def uses_hdf5_shards(self):
        return False

    def create_data_dicts(self, case_ids):
        case_ids = set(str(case_id) for case_id in case_ids)
        bboxes_dir = self.data_folder / "bboxes"
        skipped = 0

        if self.uses_hdf5_shards():
            data, skipped = self._load_case_dicts_from_shards(case_ids)
            if skipped:
                print(
                    f"DataManagerDetLBD: skipped {skipped} shard cases "
                    "(missing sidecar or invalid boxes)"
                )
            return data

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
        load_keys = list(self.data_keys)

        L = LoadImaged(
            keys=load_keys,
            image_only=False,
            ensure_channel_first=False,
            simple_keys=True,
        )
        L.register(TorchReader())

        self.transforms_dict = {
            "L": L,
            "Lfull": LoadHDF5CaseFulld(keys=load_keys),
            "Ld": LoadHDF5ShardIndexd(
                keys=["case_id"],
                manifest_fn=str(self.hdf5_manifest_fn),
            ),
            "E": EnsureChannelFirstd(keys=load_keys),
            "Norm": self._norm_transform(ik),
            "DtypeVal": Compose(
                [
                    EnsureTyped(keys=[ik], dtype=compute_dtype),
                    EnsureTyped(keys=[bk], dtype=torch.float32),
                    EnsureTyped(keys=[lk], dtype=torch.long),
                ]
            ),
        }

    def __repr__(self):
        n = len(self.data) if hasattr(self, "data") and self.data else 0
        return f"DataManagerDetLBD(split={self.split}, n={n})"


class DataManagerDualDet(DataManagerDual):
    def _build_managers(self):
        lbd_folder = lbd_det_folder_from_plan(self.project, self.configs["plan_train"])
        common = dict(
            project=self.project,
            configs=self.configs,
            batch_size=self.batch_size,
            cache_rate=self.cache_rate,
            device=self.device,
            ds_type=self.ds_type,
            data_folder=lbd_folder,
            debug=self.debug,
        )
        self.train_manager = DataManagerTrainDet(
            **common,
            split="train",
            keys=DataManagerTrainDet.keys_tr,
        )
        self.valid_manager = DataManagerDetLBD(
            **common,
            split="valid",
            keys=DataManagerDetLBD.keys_val,
            val_sampling=self.val_sampling,
        )

    def __repr__(self):
        return (
            f"DataManagerDualDet(train={self.train_manager}, valid={self.valid_manager})"
        )


DataManagerDet = DataManagerTrainDet
# %%
#SECTION:-------------------- setup--------------------------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers.project import Project
    from utilz.imageviewers import ImageBBoxViewer, ImageMaskViewer
    import warnings
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers.project import Project
    from utilz.imageviewers import ImageBBoxViewer, ImageMaskViewer

    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")
    torch.set_float32_matmul_precision("medium")

    def bbv(dici):
        ImageBBoxViewer(dici["image"], dici[bk])

# SECTION:-------------------- LIDC det plan1 (LBD train + val) --------------------

    # plan 1 → .../lbd/spc_070_070_125_lbl1_ex000 (labelbounded.py preproc)
    batch_size = 4
    ds_type = None

    proj_lidc = Project(project_title="lidc")
    CL = ConfigMakerDet(proj_lidc)
    CL.setup(1)
    config_det = CL.configs
    config_det["dataset_params"]["cache_rate"] = 0.0

# %%
    DM = DataManagerDualDet(
        project_title=proj_lidc.project_title,
        configs=config_det,
        batch_size=batch_size,
        ds_type=ds_type,
        debug=True,
    )

    # bbox_fns = list(G.data_folder.glob("bboxes/*.json"))
    # fn =bbox_fns[0]
    # dici = load_dict(fn)
# %%
    plan =config_det['plan_train']
    data_folder = lbd_det_folder_from_plan(DM.project, plan)
    DM.prepare_data()
    DM.setup("fit")

# %%
    tmt = DM.train_manager
    tmv = DM.valid_manager
    tmt.data_folder
    tmt.cases
    tmv.cases
    dat = tmt.data[2]
    # ImageMaskViewer([im, im], "im")

# %%
    DataManagerTrainDet.keys_tr
    td = tmt.transforms_dict
    td.keys()
    bk = tmt.box_key

# %%
    dici0 = dat
    dici = td["L"](dici0)
    dici["image"].shape
    dici["image"].meta
    img = dici["image"]
    lm = dici["mask"]
    ImageMaskBboxViewer(img, lm, dici[bk])

# %%

    dici = td["E"](dici)
    dici = td["Norm"](dici)
    print(dici['bbox'])
    
# %%
    dici = td["BoxToWorld"](dici)
    print(dici['bbox'])
    dici = td["ToPoints"](dici)
    dici.keys()
    dici["points"]
    dici = td["RandCrop"](dici)

# %%
    n = 0
    dici2 = dici[n]
# %%


    dici2 = td["Zoom"](dici2)
    dici2 = td["Flip0"](dici2)
    dici2 = td["Flip1"](dici2)
    dici2 = td["Flip2"](dici2)
    dici2 = td["Rand90"](dici2)
    dici2 = td["Rot"](dici2)
    print(dici2.keys())
    dici2 = td["AffinePts"](dici2)
    print(dici2.keys())
    dici2 = td["ToBoxes"](dici2)
    print(dici2['bbox'])

# %%
    bbv(dici2)
# %%
    tmt.point_key
# %%
    dici = td["BoxClip"](dici2)
    dici[bk]
    dici = td["DelMask"](dici)
    dici = td["IntensityTfms"](dici)
    dici = td["Dtype"](dici)

# %%
    b = DM.train_ds[0]
    b["image"].shape
    b[bk].shape
    b["label"].shape

# %%
    tmv.setup()
    tmv.prepare_data()
    dl = DM.val_dataloader()
    batch = next(iter(dl))
    batch["image"].shape
    batch['bbox'].shape
    bbox = batch['bbox'][0]
    img = batch["image"][0]
    ImageBBoxViewer(img, bbox)
    bbv(batch)
    batch['image'].shape
    img = batch["image"][0]
    n= 1
    ImageBBoxViewer(batch["image"][n], batch[bk][n])
    img.shape
   
    batch['bbox'][3]
    batch[bk][0].shape

    ImageBBoxViewer(batch["image"][0], batch[bk][0])

# %%
    bv = DM.valid_ds[0]
    bv["image"].shape
    bv[bk].shape




