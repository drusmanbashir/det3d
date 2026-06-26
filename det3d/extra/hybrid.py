"""Det3d fast nnDetection path — LBD HDF5 + disk sidecar boxes, no materialize.

Baseline: ``DataManagerDualDet`` — full train item tfms on CPU
(``Ld,Rtr,L2,E,Norm,F1,F2,Affine,ResizePC,BoxClip,IntensityTfms``).

Optional ``use_gpu_tail=True``: ``DataManagerDualDetBTfms`` — load/crop/norm on CPU,
``GpuTail`` batch tfms (GPU RandAffine + intensity) via ``on_after_batch_transfer``.

→ ``det3d_batch_to_nndet`` → ``RetinaUNetV001`` → ``fit_nndet_module``.

Val (``valid_impl``): ``patch_stream`` → ``L,E,BoxClip,Norm,DtypeVal``;
``bbox_anchor`` → ``L,E,Norm,BboxCrop,CropPatch,PadPatch,BoxClip,DtypeVal``.
"""

from __future__ import annotations

import os
import types
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Sequence

from nndet.ptmodule.retinaunet import RetinaUNetV001

from fran.managers.project import Project
from omegaconf import OmegaConf

from det3d.configs.parser import ConfigMakerDet
from det3d.detection.nndet_train import (
    build_nndet_retinaunet_module,
    det3d_batch_to_nndet,
    forward_patch_size_from_configs,
    maybe_store_batch_grid_preds,
)
from det3d.extra.nndet_native_lbd import setup_nndet_env
from det3d.managers.data import DataManagerDualDet, DataManagerDualDetBTfms
from det3d.managers.data.batch_tfms import DataManagerDetLBDBTfms
from det3d.managers.data.main import DataManagerDetLBD
from det3d.preprocessing.run_build import build_from_plan
from utilz.imageviewers import ImageBBoxViewer, ImageMaskBboxViewer

FAST_DM_KEYS_TR = DataManagerDetLBDBTfms.keys_tr
FAST_DM_KEYS_TR_BATCH = DataManagerDetLBDBTfms.keys_tr_batch
FAST_DM_KEYS_VAL_SEG = DataManagerDetLBD.keys_val_seg

DEFAULT_FAST_DET_MODELS = Path("/s/agent_rw/nndet_models_benchmark")
DEFAULT_FAST_PROJECT = "lidca"
DEFAULT_FAST_PLAN_ID = 4
_DET_PIPELINE_MODES = frozenset({"det", "lbd"})


def _normalize_plan_modes_lbd(conf: dict) -> None:
    for key in ("plan_train", "plan_valid", "plan_test"):
        plan = conf[key]
        if plan["mode"] in _DET_PIPELINE_MODES:
            plan["mode"] = "lbd"


def resolve_lbd_fg_case_ids(
    project,
    conf: dict,
    *,
    split: str = "train",
    n_cases: int | None = None,
) -> List[str]:
    import pandas as pd
    from fran.utils.folder_names import FolderNames

    fold = int(conf["dataset_params"]["fold"])
    ds = [x.strip() for x in conf["plan_train"]["datasources"].split(",") if x.strip()]
    train_ids, val_ids = project.get_train_val_case_ids(fold, ds, nnz_allowed=False)
    pool = set(train_ids if split == "train" else val_ids)
    lbd_folder = Path(FolderNames(project, conf["plan_train"]).lbd_folder)
    df = pd.read_csv(lbd_folder / "dataset_details.csv")
    rows = df[(df["has_fg"]) & (~df["bbox_empty"])].sort_values("case_id")
    out = [str(c) for c in rows["case_id"] if str(c) in pool]
    if n_cases is not None:
        out = out[: int(n_cases)]
    return out


def setup_det3d_fast_dm(
    project_title: str = DEFAULT_FAST_PROJECT,
    plan_id: int = DEFAULT_FAST_PLAN_ID,
    *,
    train_case_ids: List[str] | None = None,
    val_case_ids: List[str] | None = None,
    case_ids: List[str] | None = None,
    batch_size: int = 1,
    fold: int = 0,
    debug: bool = False,
    use_gpu_tail: bool = False,
):
    # AI
    """TrainerDet-style DualDet datamodule for hybrid fast LBD (GpuTail when use_gpu_tail)."""
    from det3d.managers.data.labels import infer_det_labels_from_data_folder

    if train_case_ids is None:
        train_case_ids = case_ids

    C = ConfigMakerDet(Project(project_title))
    C.setup(plan_id)
    conf = deepcopy(C.configs)
    conf["dataset_params"]["fold"] = int(fold)
    conf["dataset_params"]["batch_size"] = int(batch_size)
    _, conf = build_from_plan(project_title, plan_id, configs=conf)
    _normalize_plan_modes_lbd(conf)

    dm_cls = DataManagerDualDetBTfms if use_gpu_tail else DataManagerDualDet
    dm = dm_cls(
        project_title=project_title,
        configs=conf,
        batch_size=int(batch_size),
        cache_rate=conf["dataset_params"]["cache_rate"],
        device="cuda",
        ds_type=conf["dataset_params"]["ds_type"],
        train_indices=train_case_ids,
        val_indices=val_case_ids,
        val_sampling=1.0,
        debug=debug,
        batch_tfms=use_gpu_tail,
    )
    dm.prepare_data()
    infer_det_labels_from_data_folder(dm=dm, configs=dm.configs)
    dm.setup(stage="fit")

    train_m = dm.train_manager
    collate_name = getattr(train_m.collate_fn, "__name__", repr(train_m.collate_fn))
    print(
        f"fast LBD train: {type(train_m).__name__} n={len(train_m.cases)} keys={train_m.keys}"
    )
    if train_m.transforms_batch is not None:
        print(f"fast LBD train batch tfms: {FAST_DM_KEYS_TR_BATCH}")
    print(f"fast LBD train collate_fn: {collate_name}")
    val_m = dm.valid_manager
    val_collate = getattr(val_m.collate_fn, "__name__", repr(val_m.collate_fn))
    print(
        f"fast LBD val: {type(val_m).__name__} n={len(val_m.cases)} keys={val_m.keys}"
    )
    print(f"fast LBD val collate_fn: {val_collate}")
    return dm


def _log_nndet_losses(pl_module, losses, prefix):
    """RetinaUNetManager._log_losses parity: train0_* / val0_* per step + epoch."""
    cls_seg = losses["cls"] + losses["seg_ce"] + losses["seg_dice"]
    total = sum(losses.values())
    pl_module.log(
        f"{prefix}_cls_seg_loss",
        cls_seg,
        prog_bar=(prefix == "train0"),
        on_step=True,
        on_epoch=True,
        sync_dist=True,
    )
    pl_module.log(
        f"{prefix}_loss",
        total,
        prog_bar=False,
        on_step=True,
        on_epoch=True,
        sync_dist=True,
    )
    for key, val in losses.items():
        pl_module.log(
            f"{prefix}_{key}",
            val,
            on_step=(prefix == "train0"),
            on_epoch=True,
            sync_dist=True,
        )
    return total


def _fast_nndet_batch_to_device(nb, device):
    nb["data"] = nb["data"].to(device)
    nb["target_boxes"] = [b.to(device) for b in nb["target_boxes"]]
    nb["target_classes"] = [c.to(device) for c in nb["target_classes"]]
    nb["target_seg"] = nb["target_seg"].to(device)
    return nb


def patch_module_for_det3d_fast_batch(
    module, *, fg_labels: list[int], forward_patch_size
):
    fps = forward_patch_size
    fg = list(fg_labels)

    def training_step(self, batch, batch_idx):
        nb = det3d_batch_to_nndet(batch, forward_patch_size=fps, fg_labels=fg)
        nb = _fast_nndet_batch_to_device(nb, batch["image"].device)
        losses, _ = self.model.train_step(
            images=nb["data"],
            targets={
                "target_boxes": nb["target_boxes"],
                "target_classes": nb["target_classes"],
                "target_seg": nb["target_seg"],
            },
            evaluation=False,
            batch_num=batch_idx,
        )
        loss = _log_nndet_losses(self, losses, "train0")
        return loss

    def validation_step(self, batch, batch_idx):
        nb = det3d_batch_to_nndet(batch, forward_patch_size=fps, fg_labels=fg)
        nb = _fast_nndet_batch_to_device(nb, batch["image"].device)
        targets = {
            "target_boxes": nb["target_boxes"],
            "target_classes": nb["target_classes"],
            "target_seg": nb["target_seg"],
        }
        losses, prediction = self.model.train_step(
            images=nb["data"],
            targets=targets,
            evaluation=True,
            batch_num=batch_idx,
        )
        maybe_store_batch_grid_preds(self, batch, prediction)
        self.evaluation_step(prediction=prediction, targets=targets)
        loss = sum(losses.values())
        return {
            "loss": float(loss.detach().item()),
            **{key: float(val.detach().item()) for key, val in losses.items()},
        }

    module.training_step = types.MethodType(training_step, module)
    module.validation_step = types.MethodType(validation_step, module)
    return module


def run_det3d_fast_training_loop(
    *,
    case_ids: List[str] | None,
    epochs: int,
    batches_per_epoch: int | None,
    batch_size: int = 1,
    project_title: str = DEFAULT_FAST_PROJECT,
    plan_id: int = DEFAULT_FAST_PLAN_ID,
    exp_id: str = "Det3dFastRetinaUNet_lbd",
    train_mode: str = "overwrite",
    det_models: Path = DEFAULT_FAST_DET_MODELS,
    wandb: bool = True,
    run_name: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    notes: str = "",
    extra_callbacks=None,
    wandb_grid_epoch_freq: int = 10,
    val_every_n_epochs: int = 20,
    permanent_checkpoint_every_n_epochs: int = 100,
    val_batches_per_epoch: int | None = None,
    val_case_ids: List[str] | None = None,
    val_enabled: bool = True,
    debug: bool = False,
    use_gpu_tail: bool = False,
):
    from scripts.train import init_train_dir

    from det3d.detection.nndet_wandb import (
        build_nndet_retinaunet_wandb_grid_callback,
        fit_nndet_module,
    )

    setup_nndet_env(det_models=det_models)
    datamodule = setup_det3d_fast_dm(
        project_title,
        plan_id,
        train_case_ids=case_ids,
        val_case_ids=val_case_ids if val_enabled else None,
        batch_size=batch_size,
        debug=debug,
        use_gpu_tail=use_gpu_tail,
    )

    if batches_per_epoch is None:
        batches_per_epoch = len(datamodule.train_dataloader())
    if val_enabled and val_batches_per_epoch is None:
        val_batches_per_epoch = len(datamodule.val_dataloader())

    configs = datamodule.configs
    fg_labels = configs["plan_train"]["fg_labels"]
    fps = forward_patch_size_from_configs(configs)
    module, plan = build_nndet_retinaunet_module(
        configs, num_train_batches=int(batches_per_epoch)
    )

    def _reapply_fast_patch(mod):
        patch_module_for_det3d_fast_batch(
            mod, fg_labels=fg_labels, forward_patch_size=fps
        )

    trainer_cfg = deepcopy(module.trainer_cfg)
    trainer_cfg["num_train_batches_per_epoch"] = int(batches_per_epoch)
    trainer_cfg["num_val_batches_per_epoch"] = (
        int(val_batches_per_epoch) if val_enabled and val_batches_per_epoch else 0
    )
    trainer_cfg["max_num_epochs"] = int(epochs)
    trainer_cfg["swa_epochs"] = 0
    trainer_cfg["monitor_key"] = "val0_metric"
    trainer_cfg["monitor_mode"] = "max"
    module.trainer_cfg = trainer_cfg

    cfg = OmegaConf.create(
        {
            "task": "det3d_fast_lbd",
            "host": {"parent_results": os.environ["det_models"]},
            "exp": {"id": exp_id, "fold": 0},
            "train": {"mode": train_mode},
            "trainer_cfg": trainer_cfg,
        }
    )
    train_dir = init_train_dir(cfg)

    callbacks = list(extra_callbacks) if extra_callbacks else []
    if wandb and val_enabled:
        grid_cb = build_nndet_retinaunet_wandb_grid_callback(
            configs,
            datamodule.project.log_folder,
            wandb_grid_epoch_freq=int(wandb_grid_epoch_freq),
        )
        callbacks.append(grid_cb)
        print(
            f"fast LBD wandb grid callback epoch_freq={wandb_grid_epoch_freq} "
            f"local={Path(datamodule.project.log_folder) / 'wandb_grid'}",
            flush=True,
        )

    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader() if val_enabled else None

    res = fit_nndet_module(
        module,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        train_dir=train_dir,
        trainer_cfg=trainer_cfg,
        task="det3d_fast_lbd",
        exp_id=exp_id,
        project_title=project_title,
        run_name=run_name or exp_id,
        train_mode=train_mode,
        max_epochs=int(epochs),
        wandb=wandb,
        tags=list(tags) if tags else ["det3d_fast_lbd", "disk_boxes"],
        notes=notes,
        extra_callbacks=callbacks,
        val_enabled=val_enabled,
        permanent_checkpoint_every_n_epochs=permanent_checkpoint_every_n_epochs,
        patch_pl2=True,
        log_train_det_loss=False,
        limit_train_batches=int(batches_per_epoch),
        limit_val_batches=int(val_batches_per_epoch)
        if val_enabled and val_batches_per_epoch
        else None,
        check_val_every_n_epoch=int(val_every_n_epochs),
        after_pl2_patch=_reapply_fast_patch,
    )

    return res


# %%
if __name__ == "__main__":
# SECTION:--- setup ---
    import torch

    #RetinaUNetV001
    project_title = DEFAULT_FAST_PROJECT
    plan_id = DEFAULT_FAST_PLAN_ID
    det_models = DEFAULT_FAST_DET_MODELS

    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = C.configs
    conf["dataset_params"]["fold"] = 0

# %%
# SECTION:--- repl knobs ---
    repl_n_cases = 100
    repl_val_cases = 30
    batch_size = 1
    device_id = 0
    full_run = False
    use_gpu_tail = False

# %%
# SECTION:--- build dm + module (mirror run_det3d_fast_training_loop setup) ---
    setup_nndet_env(det_models=det_models)
    repl_train_ids = resolve_lbd_fg_case_ids(
        P, conf, split="train", n_cases=repl_n_cases
    )
    repl_val_ids = resolve_lbd_fg_case_ids(
        P, conf, split="valid", n_cases=repl_val_cases
    )
    print(f"repl train={len(repl_train_ids)} val={len(repl_val_ids)}")

    datamodule = setup_det3d_fast_dm(
        project_title,
        plan_id,
        train_case_ids=repl_train_ids,
        val_case_ids=repl_val_ids,
        batch_size=batch_size,
        use_gpu_tail=use_gpu_tail,
    )
    train_m = datamodule.train_manager
    configs = datamodule.configs
    fg_labels = configs["plan_train"]["fg_labels"]
    patch_size = forward_patch_size_from_configs(configs)
    batches_per_epoch = len(datamodule.train_dataloader())
    R, nndet_plan = build_nndet_retinaunet_module(
        configs, num_train_batches=batches_per_epoch
    )
    patch_module_for_det3d_fast_batch(R, fg_labels=fg_labels, forward_patch_size=patch_size)

    iteri = iter(datamodule.train_dataloader())
# %%
# SECTION:--- inspect batch ---
    train_batch = next(iteri)
    print(train_batch.keys())
    print(train_batch["instances"])
    print(train_batch['lm'].unique())
    print(train_batch['bbox'][0].shape)
# %%

    if train_m.transforms_batch is not None:
        train_batch = train_m.transforms_batch(train_batch)
    train_batch.keys()
    print(
        train_batch["image"].min(),
        train_batch["image"].max(),
        train_batch["image"].dtype,
    )
    print(
        train_batch["image"].shape,
        train_batch["bbox"][0].shape,
        train_batch["lm"].shape,
    )

# %%
# SECTION:--- single train step ---
    device = torch.device(f"cuda:{device_id}")
    R = R.to(device)
    batch_dev = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in train_batch.items()
    }
# %%
    batch = batch_dev
    batch_idx = 0
    print(batch.keys())
    box = batch["bbox"][0]
    img = batch["image"]
    lm = batch["lm"][0]
    lm.unique()
# %%
    ImageBBoxViewer(img, box)
    ImageMaskBboxViewer(img,lm,box)


# %%
    train_out = R.training_step(batch_dev, 0)
    train_out

# %%
# SECTION:--- single val step ---
    val_batch = next(iter(datamodule.val_dataloader()))
    val_batch_dev = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in val_batch.items()
    }
    val_out = R.validation_step(val_batch_dev, 0)
    val_out
    val_batch_dev["pred"][0].keys()

# %%
# SECTION:--- full run knobs ---
    epochs = 500 if full_run else 100
    run_name = "DET3D-FAST-LBD-E500-FULL"
    wandb = True
    wandb_grid_epoch_freq = 10
    val_every_n_epochs = 20
    permanent_checkpoint_every_n_epochs = 100
    train_mode = "overwrite"
    tags = ["det3d_fast_lbd", "disk_boxes", "nndet_v001"]
    notes = "det3d fast LBD nnDet path"

# %%
# SECTION:--- fit ---
    if full_run:
        train_ids = resolve_lbd_fg_case_ids(P, conf, split="train")
        val_ids = resolve_lbd_fg_case_ids(P, conf, split="valid")
        print(f"full train={len(train_ids)} val={len(val_ids)}")
        fit_out = run_det3d_fast_training_loop(
            case_ids=train_ids,
            val_case_ids=val_ids,
            epochs=epochs,
            batches_per_epoch=None,
            batch_size=batch_size,
            project_title=project_title,
            plan_id=plan_id,
            exp_id=run_name,
            train_mode=train_mode,
            det_models=det_models,
            wandb=wandb,
            run_name=run_name,
            tags=tags,
            notes=notes,
            wandb_grid_epoch_freq=wandb_grid_epoch_freq,
            val_every_n_epochs=val_every_n_epochs,
            val_batches_per_epoch=None,
            permanent_checkpoint_every_n_epochs=permanent_checkpoint_every_n_epochs,
            use_gpu_tail=use_gpu_tail,
        )
    else:
        fit_out = run_det3d_fast_training_loop(
            case_ids=repl_train_ids,
            val_case_ids=repl_val_ids,
            epochs=epochs,
            batches_per_epoch=4,
            batch_size=batch_size,
            project_title=project_title,
            plan_id=plan_id,
            exp_id="Det3dFastRetinaUNet_lbd_repl",
            train_mode=train_mode,
            det_models=det_models,
            wandb=wandb,
            run_name=None,
            tags=tags,
            notes="repl smoke",
            wandb_grid_epoch_freq=wandb_grid_epoch_freq,
            val_every_n_epochs=val_every_n_epochs,
            val_batches_per_epoch=None,
            permanent_checkpoint_every_n_epochs=permanent_checkpoint_every_n_epochs,
            use_gpu_tail=use_gpu_tail,
        )
    fit_out["train_dir"]

    
# %%  # T:block_start|RetinaUNetV001.training_step
#SECTION:--------------------  TS--------------------------------------------------------------------------------------
# /home/ub/code/nnDetection/nndet/ptmodule/retinaunet/base.py  # T:block_donor|/home/ub/code/nnDetection/nndet/ptmodule/retinaunet/base.py
    batch = batch_dev
    print(batch.keys())
#SECTION:-------------------- training_step --------------------------------------------------------------------------------------  # T:block_meta|RetinaUNetV001.training_step
    # requires R = RetinaUNetV001(...) in __main__  # T:requires_alias|R = RetinaUNetV001(...)
    """
    Computes a single training step
    See :class:`BaseRetinaNet` for more information
    """
    with torch.no_grad():
        batch = R.pre_trafo(**batch)  # T:self_ref|    batch = self.pre_trafo(**batch)
    losses, _ = R.model.train_step(  # T:self_ref|losses, _ = self.model.train_step(
        images=batch["data"],
        targets={
            "target_boxes": batch["boxes"],
            "target_classes": batch["classes"],
            "target_seg": batch["target"][:, 0],  # Remove channel dimension
        },
        evaluation=False,
        batch_num=batch_idx,
    )
    loss = sum(losses.values())
    training_step_result = {"loss": loss, **{key: l.detach().item() for key, l in losses.items()}}  # T:return|return {"loss": loss, **{key: l.detach().item() for key, l in losses.items()}}
#SECTION:-------------------- training_step end --------------------------------------------------------------------------------------  # T:block_meta_end|RetinaUNetV001.training_step
    # end PythonMethodScratch  # T:block_end|RetinaUNetV001.training_step

# %%


