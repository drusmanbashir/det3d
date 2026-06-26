"""Transform-debug DataManagers — train and valid share the same cases and tfm keys.

Used when ``TrainerDet.setup(debug=True, debug_tfm_keys=...)``.
"""
from __future__ import annotations

from typing import Optional

from det3d.managers.data.batch_tfms import (
    DataManagerDetBTfms,
    DataManagerDetLBDBTfms,
    DataManagerDetPatchBTfms,
    DataManagerDetRBDBTfms,
    DataManagerDetSourceBTfms,
    DataManagerDetWholeBTfms,
    DataManagerDualDetBTfms,
)
from det3d.managers.data.main import (
    DataManagerDetLBD,
    DataManagerDetPatch,
    DataManagerDetRBD,
    DataManagerDetSource,
    DataManagerDetWhole,
    DataManagerDualDet,
)
from det3d.transforms.gpu_det import BatchItemCompose, ResizeWithPadOrCropBoxSyncd
from fran.utils.folder_names import FolderNames
from monai.apps.detection.transforms.dictionary import ClipBoxToImaged
from monai.transforms import Compose
from utilz.cprint import cprint

KEYS_ITEM_NO_SPATIAL = "Ld,Rtr,L2,E,Norm"
KEYS_CPU_NO_SPATIAL = "Ld,Rtr,L2,E,Norm,ResizePC,BoxClip,IntensityTfms"


def build_train_gpu_tail_compose_no_spatial(
    *,
    image_key,
    box_key,
    label_key,
    lm_key,
    intensity_tfms,
    patch_size,
):
    spatial_keys = [image_key]
    if lm_key is not None:
        spatial_keys.append(lm_key)
    return Compose(
        [
            ResizeWithPadOrCropBoxSyncd(
                keys=spatial_keys,
                box_key=box_key,
                label_key=label_key,
                spatial_size=tuple(int(v) for v in patch_size),
                lazy=False,
            ),
            ClipBoxToImaged(
                box_keys=box_key,
                label_keys=[label_key],
                box_ref_image_keys=image_key,
                remove_empty=True,
            ),
            *intensity_tfms,
        ]
    )


class DataManagerDetBTfmsTfmDebug(DataManagerDetBTfms):
    def install_gpu_tail(self):
        self.install_gpu_tail_no_spatial()

    def install_gpu_tail_no_spatial(self):
        ik, bk, lk, pk = (
            self.image_key,
            self.box_key,
            self.label_key,
            self.point_key,
        )
        lm_key = self.lm_key if self.uses_lm_seg() else None
        self.GpuTail = BatchItemCompose(
            build_train_gpu_tail_compose_no_spatial(
                image_key=ik,
                box_key=bk,
                label_key=lk,
                lm_key=lm_key,
                intensity_tfms=self.transforms_dict["IntensityTfms"],
                patch_size=self.plan["patch_size"],
            ),
            image_key=ik,
            box_key=bk,
            label_key=lk,
            point_key=pk,
            mask_key=None,
            lm_key=lm_key,
        )
        self.transforms_dict["GpuTail"] = self.GpuTail


class DataManagerDetSourceBTfmsTfmDebug(DataManagerDetSourceBTfms, DataManagerDetBTfmsTfmDebug):
    def __init__(self, project, configs: dict, batch_size=8, cache_rate=0.0, **kwargs):
        tfm_keys = kwargs.pop("tfm_keys", None)
        if tfm_keys is not None:
            kwargs["keys"] = tfm_keys
        super().__init__(project, configs, batch_size, cache_rate, **kwargs)
        if tfm_keys is not None:
            self.keys = tfm_keys


class DataManagerDetLBDBTfmsTfmDebug(DataManagerDetLBD, DataManagerDetSourceBTfmsTfmDebug):
    def __init__(self, project, configs: dict, batch_size=8, cache_rate=0.0, **kwargs):
        tfm_keys = kwargs.pop("tfm_keys", None)
        if tfm_keys is not None:
            kwargs["keys"] = tfm_keys
        super().__init__(project, configs, batch_size, cache_rate, **kwargs)
        self.keys = tfm_keys if tfm_keys is not None else self.keys


class DataManagerDetWholeBTfmsTfmDebug(DataManagerDetWhole, DataManagerDetBTfmsTfmDebug):
    pass


class DataManagerDetRBDBTfmsTfmDebug(DataManagerDetLBDBTfmsTfmDebug, DataManagerDetRBD):
    pass


class DataManagerDetPatchBTfmsTfmDebug(DataManagerDetPatch, DataManagerDetBTfmsTfmDebug):
    pass


class DataManagerDetSourceTfmDebug(DataManagerDetSource):
    def __init__(self, project, configs: dict, batch_size=8, cache_rate=0.0, **kwargs):
        tfm_keys = kwargs.pop("tfm_keys", None)
        if tfm_keys is not None:
            kwargs["keys"] = tfm_keys
        super().__init__(project, configs, batch_size, cache_rate, **kwargs)
        if tfm_keys is not None:
            self.keys = tfm_keys


class DataManagerDetLBDTfmDebug(DataManagerDetLBD, DataManagerDetSourceTfmDebug):
    def __init__(self, project, configs: dict, batch_size=8, cache_rate=0.0, **kwargs):
        tfm_keys = kwargs.pop("tfm_keys", None)
        if tfm_keys is not None:
            kwargs["keys"] = tfm_keys
        super().__init__(project, configs, batch_size, cache_rate, **kwargs)
        self.keys = tfm_keys if tfm_keys is not None else self.keys


class DataManagerDetWholeTfmDebug(DataManagerDetWhole, DataManagerDetSourceTfmDebug):
    pass


class DataManagerDetRBDTfmDebug(DataManagerDetLBDTfmDebug, DataManagerDetRBD):
    pass


class DataManagerDetPatchTfmDebug(DataManagerDetPatch, DataManagerDetSourceTfmDebug):
    pass


class DataManagerDualDetTfmDebug(DataManagerDualDet):
    _DET_MANAGER_CLASSES = {
        "source": DataManagerDetSourceTfmDebug,
        "whole": DataManagerDetWholeTfmDebug,
        "pbd": DataManagerDetPatchTfmDebug,
        "sourcepbd": DataManagerDetPatchTfmDebug,
        "lbd": DataManagerDetLBDTfmDebug,
        "rbd": DataManagerDetRBDTfmDebug,
    }

    def __init__(
        self,
        project_title,
        configs: dict,
        batch_size: int,
        cache_rate=0.0,
        device="cuda",
        ds_type=None,
        save_hyperparameters=True,
        data_folder=None,
        manager_class_train: Optional[type] = None,
        manager_class_valid: Optional[type] = None,
        train_indices=None,
        val_indices=None,
        val_sampling=1.0,
        debug=False,
        batch_tfms: bool = True,
        debug_tfm_keys: str | None = None,
        debug_n_cases: int | None = None,
    ):
        self.debug_tfm_keys = debug_tfm_keys
        self.debug_n_cases = debug_n_cases
        super().__init__(
            project_title=project_title,
            configs=configs,
            batch_size=batch_size,
            cache_rate=cache_rate,
            device=device,
            ds_type=ds_type,
            save_hyperparameters=save_hyperparameters,
            data_folder=data_folder,
            manager_class_train=manager_class_train,
            manager_class_valid=manager_class_valid,
            train_indices=train_indices,
            val_indices=val_indices,
            val_sampling=val_sampling,
            debug=debug,
            batch_tfms=batch_tfms,
        )

    def _manager_kwargs(self) -> dict:
        return dict(tfm_keys=self.debug_tfm_keys)

    def _build_managers(self):
        cls_tr, cls_val = self.infer_manager_classes(self.configs)
        cls_tr = self.manager_class_train or cls_tr
        cls_val = self.manager_class_valid or cls_val
        self._assert_det_manager_class(cls_tr)
        self._assert_det_manager_class(cls_val)
        extra = self._manager_kwargs()
        lbd_folder = FolderNames(self.project, self.configs["plan_train"]).lbd_folder
        cprint(f"train manager class: {cls_tr.__name__}", color="cyan")
        cprint(f"valid manager class: {cls_val.__name__}", color="cyan")
        cprint(f"tfm_debug keys: {self.debug_tfm_keys}", color="yellow")
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
            **extra,
        )
        self.valid_manager = cls_val(
            project=self.project,
            configs=self.configs,
            batch_size=self.batch_size,
            cache_rate=self.cache_rate,
            split="train",
            device=self.device,
            ds_type=self.ds_type,
            keys=self.keys_tr,
            data_folder=lbd_folder,
            val_sampling=self.val_sampling,
            debug=self.debug,
            **extra,
        )

    def prepare_data(self):
        self._build_managers()
        self._call_prepare_data()
        n = self.debug_n_cases
        if n is None and isinstance(self.train_indices, int):
            n = self.train_indices
        n = int(n)
        self.train_manager.select_cases_from_inds(n)
        self.train_manager.data = self.train_manager.create_data_dicts(
            self.train_manager.cases
        )
        self.valid_manager.cases = list(self.train_manager.cases)
        self.valid_manager.data = self.valid_manager.create_data_dicts(
            self.valid_manager.cases
        )
        cprint(
            f"TfmDebug: train+valid share {len(self.train_manager.cases)} cases, "
            f"keys={self.debug_tfm_keys}",
            color="yellow",
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        if (
            not self.trainer.training
            and isinstance(batch, dict)
            and "validation_impl" not in batch
        ):
            batch["validation_impl"] = "bbox_anchor"
        return batch


class DataManagerDualDetBTfmsTfmDebug(DataManagerDualDetTfmDebug, DataManagerDualDetBTfms):
    _DET_MANAGER_CLASSES = {
        "source": DataManagerDetSourceBTfmsTfmDebug,
        "whole": DataManagerDetWholeBTfmsTfmDebug,
        "pbd": DataManagerDetPatchBTfmsTfmDebug,
        "sourcepbd": DataManagerDetPatchBTfmsTfmDebug,
        "lbd": DataManagerDetLBDBTfmsTfmDebug,
        "rbd": DataManagerDetRBDBTfmsTfmDebug,
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
