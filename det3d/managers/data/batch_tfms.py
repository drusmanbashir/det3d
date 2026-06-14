from typing import Optional

from det3d.managers.data.main import (
    DataManagerDet,
    DataManagerDetLBD,
    DataManagerDetPatch,
    DataManagerDetRBD,
    DataManagerDetSource,
    DataManagerDetWhole,
    DataManagerDualDet,
    DataManagerMultiDet,
)
from det3d.transforms.gpu_det import BatchItemCompose, build_train_gpu_tail_compose
from fran.managers.data.batch_tfms import DataManagerDualBTfms
from utilz.cprint import cprint


class DataManagerDetBTfms(DataManagerDet):
    def print_transform_summary(self):
        item_keys = self.keys or ""
        batch_keys = self.active_batch_keys() or ""
        cprint("Transforms are set up", color="green")
        cprint(f"Item Transforms: {item_keys}", color="yellow")
        cprint(f"Batch Transforms: {batch_keys}", color="yellow")

    def install_gpu_tail(self):
        ik, bk, lk, pk, mk = (
            self.image_key,
            self.box_key,
            self.label_key,
            self.point_key,
            self.mask_key,
        )
        compute_dtype = self._compute_dtype()
        self.transforms_dict["GpuTail"] = BatchItemCompose(
            build_train_gpu_tail_compose(
                device=self.device,
                image_key=ik,
                box_key=bk,
                label_key=lk,
                point_key=pk,
                mask_key=mk,
                affine_lps_to_ras=self.affine_lps_to_ras,
                compute_dtype=compute_dtype,
                intensity_tfms=self.transforms_dict["IntensityTfms"],
            ),
            image_key=ik,
            box_key=bk,
            label_key=lk,
            point_key=pk,
            mask_key=mk,
        )

    def create_transforms(self):
        super().create_transforms()
        if not self.uses_train_keys():
            return
        self.install_gpu_tail()


class DataManagerDetSourceBTfms(DataManagerDetSource, DataManagerDetBTfms):
    keys_tr = "Ld,Rtr,L2,E,Norm,BoxToWorld,ToPoints,AffinePts"
    keys_tr_batch = "GpuTail"
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

    def create_transforms(self):
        DataManagerDetSource.create_transforms(self)
        if self.uses_train_keys():
            self.install_gpu_tail()


class DataManagerDetWholeBTfms(DataManagerDetWhole, DataManagerDetBTfms):
    pass


class DataManagerDetLBDBTfms(DataManagerDetLBD, DataManagerDetSourceBTfms):
    def __repr__(self):
        n = len(self.data) if self.data else 0
        return f"{self.__class__.__name__}(split={self.split}, n={n})"

    def __str__(self):
        return self.__repr__()

    def create_transforms(self):
        if self.is_train_all_split():
            DataManagerDetSource.create_transforms(self)
            self.install_gpu_tail()
            return
        DataManagerDetLBD.create_transforms(self)


class DataManagerDetRBDBTfms(DataManagerDetLBDBTfms, DataManagerDetRBD):
    pass


class DataManagerDetPatchBTfms(DataManagerDetPatch, DataManagerDetBTfms):
    pass


class DataManagerDualDetBTfms(DataManagerDualDet, DataManagerDualBTfms):
    def infer_manager_classes(self, configs):
        train_mode = configs["plan_train"]["mode"]
        valid_mode = configs["plan_valid"]["mode"]
        mode_to_class = {
            "source": DataManagerDetSourceBTfms,
            "whole": DataManagerDetWholeBTfms,
            "pbd": DataManagerDetPatchBTfms,
            "sourcepbd": DataManagerDetPatchBTfms,
            "lbd": DataManagerDetLBDBTfms,
            "rbd": DataManagerDetRBDBTfms,
        }
        for mode in (train_mode, valid_mode):
            if mode not in mode_to_class:
                raise ValueError(
                    f"Unrecognized mode: {mode}. Must be one of {list(mode_to_class.keys())}"
                )
        return mode_to_class[train_mode], mode_to_class[valid_mode]


class DataManagerMultiDetBTfms(DataManagerMultiDet):
    def infer_manager_classes(self, configs):
        train_mode = configs["plan_train"]["mode"]
        valid_mode = configs["plan_valid"]["mode"]
        mode_to_class = {
            "source": DataManagerDetSourceBTfms,
            "whole": DataManagerDetWholeBTfms,
            "pbd": DataManagerDetPatchBTfms,
            "sourcepbd": DataManagerDetPatchBTfms,
            "lbd": DataManagerDetLBDBTfms,
            "rbd": DataManagerDetRBDBTfms,
        }
        for mode in (train_mode, valid_mode):
            if mode not in mode_to_class:
                raise ValueError(
                    f"Unrecognized mode: {mode}. Must be one of {list(mode_to_class.keys())}"
                )
        return mode_to_class[train_mode], mode_to_class[valid_mode]
