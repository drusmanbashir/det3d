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
        lm_key = self.lm_key if self.uses_lm_seg() else None
        self.transforms_dict["GpuTail"] = BatchItemCompose(
            build_train_gpu_tail_compose(
                device=self.device,
                image_key=ik,
                box_key=bk,
                label_key=lk,
                point_key=pk,
                mask_key=mk,
                lm_key=lm_key,
                affine_lps_to_ras=self.affine_lps_to_ras,
                compute_dtype=compute_dtype,
                intensity_tfms=self.transforms_dict["IntensityTfms"],
            ),
            image_key=ik,
            box_key=bk,
            label_key=lk,
            point_key=pk,
            mask_key=mk,
            lm_key=lm_key,
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
    def __init__(self, project, configs: dict, batch_size=8, cache_rate=0.0, **kwargs):
        provided_keys = kwargs["keys"] if "keys" in kwargs else None
        super().__init__(project, configs, batch_size, cache_rate, **kwargs)
        if provided_keys is None and self.is_eval_split():
            if self._valid_impl() == "patch_stream":
                if self.uses_lm_seg():
                    self.keys = self.keys_val_seg
                else:
                    self.keys = self.keys_val
            else:
                self.keys = self.keys_val_bbox

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


# %%
if __name__ == "__main__":
#SECTION:--- setup ---
    from fran.managers import Project
    from utilz.helpers import pp

    from det3d.configs.parser import ConfigMakerDet

    project_title = "lidca"
    plan_id = 1
    conf_fold = 0

    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = C.configs
    conf["dataset_params"]["fold"] = conf_fold
    pp(conf["plan_train"])

#SECTION:--- dualdet datamanager ---
# %%
    batch_size = 1
    batch_tfms = True
    debug_ = False
    train_indices = None
    val_indices = None
    val_sampling = 1.0
    device = 1

    for key in ("plan_train", "plan_valid", "plan_test"):
        plan = conf[key]
        if plan["mode"] in {"det", "lbd"}:
            plan["mode"] = "lbd"

    D = DataManagerDualDetBTfms(
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
    D.prepare_data()
    D.setup(stage="fit")
    tmt = D.train_manager
    tmv = D.valid_manager
    tmt.setup()
    tmv.setup()
    train_dl = tmt.dl
    val_dl = tmv.dl
    print(f"train: {tmt}")
    print(f"valid: {tmv}")

#SECTION:--- tmt ---
# %%
    batch = next(iter(train_dl))

#SECTION:--- tmv ---
# %%
    batch2 = next(iter(val_dl))
