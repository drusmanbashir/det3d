"""DataManagerDualDet: paired train/valid detection data managers for TrainerDet."""

from utilz.imageviewers import ImageBBoxViewer
from fran.managers.data.main import DataManagerDual
from fran.utils.folder_names import FolderNames
from det3d.managers.data.main import (
    DataManagerDet,
    DataManagerDetSource,
    DataManagerDetWhole,
    DataManagerDetLBD,
    DataManagerDetRBD,
    DataManagerDetPatch,
)


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
