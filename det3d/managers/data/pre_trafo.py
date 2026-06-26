from pathlib import Path

import numpy as np
import torch
from det3d.managers.data.batch_tfms import DataManagerDualBTfms
from det3d.managers.data.collate import (
    lbd_det_collate_train_pre_trafo,
    lbd_det_collate_val_pre_trafo,
)
from det3d.managers.data.main import (
    CropDetPatchd,
    DataManagerDet,
    DataManagerDetLBD,
    DataManagerDetPatch,
    DataManagerDetRBD,
    DataManagerDetSource,
    DataManagerDetWhole,
    DataManagerDualDet,
    DataManagerMultiDet,
    LoadHDF5DetCropd,
)
from det3d.transforms.gpu_det import (
    PreTrafoBatchItemCompose,
    build_train_gpu_tail_compose_pre_trafo,
)
from fran.managers.data.main import DataManagerRBD
from fran.preprocessing.helpers import import_h5py
from monai.data import MetaTensor
from monai.transforms import EnsureChannelFirstd, EnsureTyped, ScaleIntensityRanged
from utilz.fileio import load_json
from utilz.stringz import info_from_filename


class LoadHDF5DetCropLmOnlyd(LoadHDF5DetCropd):
    """Load HDF5 crop for image+lm only (pre_trafo derives boxes from lm)."""

    def __call__(self, data):
        d = dict(data)
        shard_path = Path(d[self.shard_path_key])
        case_path = str(d[self.case_path_key])
        crop_slices = tuple(d[self.crop_slices_key])
        crop_start = tuple(int(v) for v in d[self.crop_start_key])
        h5py = import_h5py()
        h5_keys = tuple(dict.fromkeys(self.keys))
        with h5py.File(shard_path, "r") as h5f:
            case_grp = h5f[case_path]
            loaded = {
                key: np.asarray(case_grp[key][crop_slices])
                for key in h5_keys
                if key in case_grp
            }
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
        del (
            d[self.shard_path_key],
            d[self.case_path_key],
            d[self.crop_slices_key],
            d[self.crop_start_key],
        )
        return d


class CropDetPatchLmOnlyd(CropDetPatchd):
    """Crop image+lm patch tensors without box adjustment."""

    def __call__(self, data):
        d = dict(data)
        crop_slices = tuple(d[self.crop_slices_key])
        crop_start = tuple(int(v) for v in d[self.crop_start_key])
        meta_updates = {
            "crop_start": crop_start,
            "crop_end": d["crop_end"],
            "sampled_flat_index": d["sampled_flat_index"],
            "sample_is_fg": d["sample_is_fg"],
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
        return d


class DataManagerDetPreTrafoMixin:
    pre_trafo = True

    def uses_pre_trafo(self):
        return True

    def _set_collate_fn(self):
        if self.is_eval_split():
            self.collate_fn = lbd_det_collate_val_pre_trafo
            return
        self.collate_fn = lbd_det_collate_train_pre_trafo


class DataManagerDetPreTrafo(DataManagerDetPreTrafoMixin, DataManagerDet):
    pass


class DataManagerDetSourcePreTrafo(DataManagerDetPreTrafoMixin, DataManagerDetSource):
    keys_tr = "Ld,Rtr,L2,E,Norm,Affine,ResizePC,IntensityTfms"

    def create_data_dicts(self, case_ids):
        case_ids = set(str(case_id) for case_id in case_ids)
        data = []
        bboxes_dir = self.data_folder / "bboxes"
        skipped = 0
        manifest = load_json(self.hdf5_manifest_fn)
        manifest_parent = self.hdf5_manifest_fn.parent
        for shard_info in manifest["shards"]:
            shard_path = Path(shard_info["shard"])
            if not shard_path.is_absolute():
                shard_path = manifest_parent / shard_path
            for case_id in shard_info["case_ids"]:
                case_id = str(case_id)
                if case_id not in case_ids:
                    continue
                from det3d.utils.bbox_sidecar import bbox_sidecar_path

                bbox_fn = bbox_sidecar_path(bboxes_dir, case_id)
                if not bbox_fn.is_file():
                    skipped += 1
                    continue
                _box_t, label_t, instances = self._load_bbox_sidecar(bbox_fn)
                row = {
                    "case_id": case_id,
                    "data_folder": str(self.data_folder),
                    "hdf5_shard_path": str(shard_path),
                    "hdf5_case_path": f"/cases/{case_id}",
                    self.label_key: label_t,
                    "instances": instances,
                }
                data.append(row)
        if skipped:
            print(
                f"DataManagerDetSourcePreTrafo: skipped {skipped} cases "
                "(missing sidecar)"
            )
        return data, skipped

    def create_transforms(self):
        super().create_transforms()
        ik, bk, lk, pk, lmk = (
            self.image_key,
            self.box_key,
            self.label_key,
            self.point_key,
            self.lm_key,
        )
        load_keys = [ik, lmk]
        spatial_aug_keys = [ik, lmk]
        affine_modes = ["bilinear", "nearest"]
        plan = self.plan
        patch_size = self._patch_size()
        scale = float(self.dataset_params["prezoom_scale"])
        patch_size_prezoom = tuple(int(v) for v in patch_size)
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
        for key in ("BoxToWorld", "ToPoints", "AffinePts", "ToBoxes", "BoxClip"):
            self.transforms_dict.pop(key, None)
        self.L2 = LoadHDF5DetCropLmOnlyd(keys=load_keys)
        self.transforms_dict["L2"] = self.L2
        self.Affine = self.transforms_dict["Affine"]
        self.Affine.keys = spatial_aug_keys
        self.Affine.mode = affine_modes
        self.Affine.prob = affine3d["p"]
        self.Affine.rotate_range = affine3d["rotate_range"]
        self.Affine.scale_range = affine3d["scale_range"]
        self.ResizePC = self.transforms_dict["ResizePC"]
        self.ResizePC.keys = spatial_aug_keys


class DataManagerDetSourcePreTrafoBTfms(DataManagerDetSourcePreTrafo):
    keys_tr = "Ld,Rtr,L2,E,Norm"
    keys_tr_batch = "GpuTailPreTrafo"
    keys_val = "L,E,Norm,DtypeVal"
    keys_val_batch = None

    def __init__(self, project, configs: dict, batch_size=8, cache_rate=0.0, **kwargs):
        provided_keys = kwargs["keys"] if "keys" in kwargs else None
        super().__init__(project, configs, batch_size, cache_rate, **kwargs)
        if provided_keys is None:
            if self.uses_train_keys():
                self.keys = self.keys_tr
            elif self.is_eval_split():
                self.keys = self.keys_val

    def install_gpu_tail_pre_trafo(self):
        ik, lk, lmk = self.image_key, self.label_key, self.lm_key
        self.GpuTailPreTrafo = PreTrafoBatchItemCompose(
            build_train_gpu_tail_compose_pre_trafo(
                image_key=ik,
                lm_key=lmk,
                intensity_tfms=self.transforms_dict["IntensityTfms"],
                affine3d=self.configs["affine3d"],
                patch_size=self.plan["patch_size"],
            ),
            image_key=ik,
            label_key=lk,
            lm_key=lmk,
            instances_key="instances",
        )
        self.transforms_dict["GpuTailPreTrafo"] = self.GpuTailPreTrafo

    def create_transforms(self):
        DataManagerDetSourcePreTrafo.create_transforms(self)
        if self.uses_train_keys():
            self.install_gpu_tail_pre_trafo()
            self.keys = self.keys_tr
            for key in ("Affine", "ResizePC", "IntensityTfms"):
                self.transforms_dict.pop(key, None)


class DataManagerDetWholePreTrafo(DataManagerDetPreTrafoMixin, DataManagerDetWhole):
    pass


class DataManagerDetLBDPreTrafo(DataManagerDetLBD, DataManagerDetSourcePreTrafo):
    keys_val_seg = "L,E,Norm,DtypeVal"
    keys_val_bbox = "L,E,Norm,BboxCrop,CropPatch,PadPatch,DtypeVal"

    def __init__(self, project, configs: dict, batch_size=8, cache_rate=0.0, **kwargs):
        provided_keys = kwargs["keys"] if "keys" in kwargs else None
        super().__init__(project, configs, batch_size, cache_rate, **kwargs)
        if provided_keys is None and self.is_eval_split():
            if self._valid_impl() == "patch_stream":
                self.keys = self.keys_val_seg
            else:
                self.keys = self.keys_val_bbox

    def create_data_dicts(self, case_ids):
        if self.is_train_all_split():
            return DataManagerDetSourcePreTrafo.create_data_dicts(self, case_ids)
        case_ids = set(str(case_id) for case_id in case_ids)
        skipped = 0
        bboxes_dir = self.data_folder / "bboxes"
        data = []
        images_dir = self.data_folder / "images"
        for img_fn in sorted(images_dir.glob("*.pt")):
            case_id = info_from_filename(img_fn.name, full_caseid=True)["case_id"]
            if case_id not in case_ids:
                continue
            from det3d.utils.bbox_sidecar import bbox_sidecar_path

            bbox_fn = bbox_sidecar_path(bboxes_dir, img_fn.stem)
            if not bbox_fn.is_file():
                skipped += 1
                continue
            box_t, label_t, instances = self._load_bbox_sidecar(bbox_fn)
            lm_fn = self.data_folder / "lms" / img_fn.name
            if not lm_fn.is_file():
                skipped += 1
                continue
            row = {
                "case_id": case_id,
                "data_folder": str(self.data_folder),
                "image": str(img_fn),
                self.box_key: box_t,
                self.label_key: label_t,
                "instances": instances,
                self.lm_key: str(lm_fn),
            }
            data.append(row)
        if skipped:
            print(
                f"DataManagerDetLBDPreTrafo: skipped {skipped} cases "
                "(missing sidecar)"
            )
        return data

    def create_transforms(self):
        if self.is_train_all_split():
            return DataManagerDetSourcePreTrafo.create_transforms(self)
        ik, lk, lmk = self.image_key, self.label_key, self.lm_key
        load_keys = [ik, lmk]
        compute_dtype = self._compute_dtype()
        clip = self.dataset_params["intensity_clip_range"]
        from det3d.managers.data.main import BboxCenterCropSlicesd, PadDetPatchd
        from fran.transforms.imageio import TorchReader
        from monai.transforms import Compose, LoadImaged

        L = LoadImaged(
            keys=load_keys,
            image_only=False,
            ensure_channel_first=False,
            simple_keys=True,
        )
        L.register(TorchReader())
        dtype_val = [
            EnsureTyped(keys=[ik], dtype=compute_dtype),
            EnsureTyped(keys=[lmk], dtype=torch.long),
            EnsureTyped(keys=[lk], dtype=torch.long),
        ]
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
        if self.is_eval_split() and self._valid_impl() == "bbox_anchor":
            patch_size = self._rand_crop_patch_size()
            crop_keys = [ik, lmk]
            self.transforms_dict["BboxCrop"] = BboxCenterCropSlicesd(
                box_key=self.box_key,
                image_key=ik,
                roi_size=patch_size,
            )
            self.transforms_dict["CropPatch"] = CropDetPatchLmOnlyd(keys=crop_keys)
            self.transforms_dict["PadPatch"] = PadDetPatchd(
                keys=crop_keys,
                patch_size=patch_size,
            )


class DataManagerDetLBDPreTrafoBTfms(DataManagerDetLBDPreTrafo, DataManagerDetSourcePreTrafoBTfms):
    def create_transforms(self):
        if self.is_train_all_split():
            DataManagerDetSourcePreTrafo.create_transforms(self)
            if self.uses_train_keys():
                self.install_gpu_tail_pre_trafo()
                self.keys = self.keys_tr
            return
        DataManagerDetLBDPreTrafo.create_transforms(self)


class DataManagerDetRBDPreTrafo(DataManagerDetLBDPreTrafo, DataManagerRBD):
    pass


class DataManagerDetRBDPreTrafoBTfms(DataManagerDetLBDPreTrafoBTfms, DataManagerDetRBD):
    pass


class DataManagerDetPatchPreTrafo(DataManagerDetPreTrafoMixin, DataManagerDetPatch):
    pass


class DataManagerDualDetPreTrafo(DataManagerDualDet):
    _DET_MANAGER_CLASSES = {
        "source": DataManagerDetSourcePreTrafo,
        "whole": DataManagerDetWholePreTrafo,
        "pbd": DataManagerDetPatchPreTrafo,
        "sourcepbd": DataManagerDetPatchPreTrafo,
        "lbd": DataManagerDetLBDPreTrafo,
        "rbd": DataManagerDetRBDPreTrafo,
    }


class DataManagerDualDetPreTrafoBTfms(DataManagerDualDetPreTrafo, DataManagerDualBTfms):
    _DET_MANAGER_CLASSES = {
        "source": DataManagerDetSourcePreTrafoBTfms,
        "whole": DataManagerDetWholePreTrafo,
        "pbd": DataManagerDetPatchPreTrafo,
        "sourcepbd": DataManagerDetPatchPreTrafo,
        "lbd": DataManagerDetLBDPreTrafoBTfms,
        "rbd": DataManagerDetRBDPreTrafoBTfms,
    }


class DataManagerMultiDetPreTrafo(DataManagerMultiDet):
    def infer_manager_classes(self, configs):
        train_mode = configs["plan_train"]["mode"]
        valid_mode = configs["plan_valid"]["mode"]
        mode_to_class = dict(DataManagerDualDetPreTrafo._DET_MANAGER_CLASSES)
        for mode in (train_mode, valid_mode):
            if mode not in mode_to_class:
                raise ValueError(
                    f"Unrecognized mode: {mode}. Must be one of {list(mode_to_class.keys())}"
                )
        return mode_to_class[train_mode], mode_to_class[valid_mode]
