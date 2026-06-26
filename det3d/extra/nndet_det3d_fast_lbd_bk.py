"""Det3d fast nnDetection path — LBD HDF5 + disk sidecar boxes, no materialize.

Uses ``DataManagerDetLBDBTfms``: item load/crop/norm/box-points on CPU, then
``GpuTail`` batch tfms (GPU RandAffine + intensity) via ``on_after_batch_transfer``
→ ``det3d_batch_to_nndet`` → ``RetinaUNetV001`` → ``fit_nndet_module``.

Train item ``keys`` (``DataManagerDetSourceBTfms.keys_tr``):

``Ld,Rtr,L2,E,Norm,BoxToWorld,ToPoints``

Train batch ``keys`` (``keys_tr_batch``): ``GpuTail`` — spatial + intensity on GPU.

Val (``valid_impl``): ``patch_stream`` → ``L,E,BoxClip,Norm,DtypeVal``;
``bbox_anchor`` → ``L,E,Norm,BboxCrop,CropPatch,PadPatch,BoxClip,DtypeVal``.
"""

from __future__ import annotations

import os
import types
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Sequence

from det3d.extra.trainer_nndet import (
    apply_det3d_plan_to_nndet_model_cfg,
    load_nndet_train_cfgs,
    plan_from_det3d,
)
from fran.managers.project import Project
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule

from det3d.configs.parser import ConfigMakerDet
from det3d.detection.nndet_train import (
    build_nndet_retinaunet_module,
    det3d_batch_to_nndet,
    ensure_nndet_importable,
    forward_patch_size_from_configs,
    maybe_store_batch_grid_preds,
)
from det3d.extra.nndet_native_lbd import setup_nndet_env
from det3d.managers.data.batch_tfms import DataManagerDetLBDBTfms
from det3d.managers.data.main import DataManagerDetLBD, DataManagerDetSource

FAST_DM_KEYS_TR = DataManagerDetLBDBTfms.keys_tr
FAST_DM_KEYS_TR_BATCH = DataManagerDetLBDBTfms.keys_tr_batch
FAST_DM_KEYS_VAL_SEG = DataManagerDetLBD.keys_val_seg
from det3d.preprocessing.run_build import build_from_plan

DEFAULT_FAST_DET_MODELS = Path("/s/agent_rw/nndet_models_benchmark")
DEFAULT_FAST_PROJECT = "lidca"
DEFAULT_FAST_PLAN_ID = 4


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
    case_ids: List[str] | None = None,
    batch_size: int = 1,
    debug: bool = False,
    use_gpu_tail: bool = True,
) -> DataManagerDetLBD:
    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = deepcopy(C.configs)
    conf["dataset_params"]["fold"] = 0
    conf["dataset_params"]["batch_size"] = int(batch_size)
    _, conf = build_from_plan(project_title, plan_id, configs=conf)

    dm_cls = DataManagerDetLBDBTfms if use_gpu_tail else DataManagerDetLBD
    dm = dm_cls(
        P,
        conf,
        batch_size=int(batch_size),
        split="train",
        device="cuda",
        debug=debug,
    )
    from det3d.managers.data.labels import infer_det_labels_from_data_folder

    dm.prepare_data()
    infer_det_labels_from_data_folder(dm, dm.configs)
    if case_ids is not None:
        dm.select_cases_from_inds(list(case_ids))
        dm.data = dm.create_data_dicts(dm.cases)
    dm.setup()
    collate_name = getattr(dm.collate_fn, "__name__", repr(dm.collate_fn))
    print(f"fast LBD train keys: {dm.keys}")
    if dm.transforms_batch is not None:
        print(f"fast LBD train batch tfms: {FAST_DM_KEYS_TR_BATCH}")
    print(f"fast LBD train collate_fn: {collate_name}")
    return dm


class Det3dFastLbdDataModule(LightningDataModule):
    """Lightning datamodule: ``DataManagerDetLBDBTfms`` + GpuTail on batch transfer."""

    def __init__(
        self,
        train_dm: DataManagerDetLBD,
        val_dm: DataManagerDetLBD | None = None,
    ):
        super().__init__()
        self.train_dm = train_dm
        self.val_dm = val_dm

    def setup(self, stage: str | None = None) -> None:
        if self.train_dm.ds is None:
            self.train_dm.create_dataset()
        if self.train_dm.dl is None:
            self.train_dm.create_dataloader()
        if self.val_dm is not None:
            if self.val_dm.ds is None:
                self.val_dm.create_dataset()
            if self.val_dm.dl is None:
                self.val_dm.create_dataloader()

    def _apply_batch_tfms(self, batch, dataloader_idx):
        if not isinstance(batch, dict) or "image" not in batch:
            return batch
        if self.trainer.training:
            manager = self.train_dm
        elif self.trainer.validating or self.trainer.sanity_checking:
            manager = self.val_dm
        else:
            return batch
        if manager is None or manager.transforms_batch is None:
            return batch
        return manager.transforms_batch(batch)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return self._apply_batch_tfms(batch, dataloader_idx)

    def train_dataloader(self):
        self.setup()
        return self.train_dm.dl

    def val_dataloader(self):
        if self.val_dm is None:
            return []
        self.setup()
        return self.val_dm.dl


def setup_det3d_fast_val_dm(
    train_dm: DataManagerDetLBD,
    case_ids: List[str] | None = None,
    *,
    batch_size: int = 1,
    split: str = "valid",
) -> DataManagerDetLBD:
    if case_ids is None:
        case_ids = resolve_lbd_fg_case_ids(
            train_dm.project, train_dm.configs, split=split
        )
    val_dm = DataManagerDetLBD(
        train_dm.project,
        train_dm.configs,
        batch_size=int(batch_size),
        split=split,
        debug=False,
    )
    val_dm.prepare_data()
    val_dm.select_cases_from_inds(list(case_ids))
    val_dm.data = val_dm.create_data_dicts(val_dm.cases)
    val_dm.setup()
    collate_name = getattr(val_dm.collate_fn, "__name__", repr(val_dm.collate_fn))
    print(f"fast LBD val keys: {val_dm.keys}")
    print(f"fast LBD val collate_fn: {collate_name}")
    return val_dm


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
    wandb_grid_epoch_freq: int = 20,
    val_every_n_epochs: int = 20,
    permanent_checkpoint_every_n_epochs: int = 100,
    val_batches_per_epoch: int | None = None,
    val_case_ids: List[str] | None = None,
    val_enabled: bool = True,
    val_split: str = "valid",
    debug: bool = False,
    use_gpu_tail: bool = True,
):
    from scripts.train import init_train_dir

    from det3d.detection.nndet_wandb import (
        build_nndet_retinaunet_wandb_grid_callback,
        fit_nndet_module,
    )

    setup_nndet_env(det_models=det_models)
    dm = setup_det3d_fast_dm(
        project_title,
        plan_id,
        case_ids=case_ids,
        batch_size=batch_size,
        debug=debug,
        use_gpu_tail=use_gpu_tail,
    )
    val_dm = None
    if val_enabled:
        val_dm = setup_det3d_fast_val_dm(
            dm, val_case_ids, batch_size=batch_size, split=val_split
        )
    datamodule = Det3dFastLbdDataModule(dm, val_dm=val_dm)
    datamodule.setup()

    if batches_per_epoch is None:
        batches_per_epoch = len(datamodule.train_dataloader())
    if val_enabled and val_batches_per_epoch is None and val_dm is not None:
        val_batches_per_epoch = len(datamodule.val_dataloader())

    configs = dm.configs
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
            dm.project.log_folder,
            wandb_grid_epoch_freq=int(wandb_grid_epoch_freq),
        )
        callbacks.append(grid_cb)
        print(
            f"fast LBD wandb grid callback epoch_freq={wandb_grid_epoch_freq} "
            f"local={Path(dm.project.log_folder) / 'wandb_grid'}",
            flush=True,
        )

    res =  fit_nndet_module(
        module,
        datamodule,
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

def build_nndet_retinaunet_module(configs, num_train_batches):
    ensure_nndet_importable()
    from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001

    from det3d.configs.parser import resolve_nndet_plan_path

    plan_train = configs["plan_train"]
    plan_path = configs["model_params"].get("nndet_plan_path")
    if plan_path is None:
        plan_path = resolve_nndet_plan_path(
            configs["mnemonic"], Path(configs["configurations_dir"])
        )
    else:
        plan_path = Path(plan_path)
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    model_cfg, trainer_cfg = load_nndet_train_cfgs()
    model_cfg = apply_det3d_plan_to_nndet_model_cfg(model_cfg, plan_train)
    trainer_cfg["num_train_batches_per_epoch"] = int(num_train_batches)
    trainer_cfg["max_num_epochs"] = int(configs["model_params"].get("max_epochs", 600))
    plan = plan_from_det3d(plan_train, plan_path=str(plan_path))
    module = RetinaUNetV001(
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        plan=plan,
    )
    return module, plan


# %%
if __name__ == "__main__":
# SECTION:--- setup ---
    import torch

    project_title = "lidca"
    plan_id = 4
    det_models =Path("/s/agent_rw/nndet_models_benchmark")

    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = C.configs
    conf["dataset_params"]["fold"] = 0

# %%
# SECTION:--- repl knobs ---
    repl_n_cases = 8
    batch_size = 1
    device_id = 1
    full_run = False

# %%
# SECTION:--- build dm + module (mirror run_det3d_fast_training_loop setup) ---
    setup_nndet_env(det_models=
    os.environ["det_data"] = str(det_data)
    os.environ["det_models"] = str(det_models)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("det_num_threads", "2")
    os.environ.setdefault("det_verbose", "1")
    _nndet_import_shim()
    _lightning_import_shim()
    if str(NNDET_ROOT) not in sys.path:
        sys.path.insert(0, str(NNDET_ROOT))
    import nndet.compat  # noqa: F401


    repl_train_ids = resolve_lbd_fg_case_ids(
        P, conf, split="train", n_cases=repl_n_cases
    )
    repl_val_ids = resolve_lbd_fg_case_ids(
        P, conf, split="valid", n_cases=min(4, repl_n_cases)
    )
    print(f"repl train={len(repl_train_ids)} val={len(repl_val_ids)}")

    train_dm = setup_det3d_fast_dm(
        project_title,
        plan_id,
        case_ids=repl_train_ids,
        batch_size=batch_size,
    )
    val_dm = setup_det3d_fast_val_dm(
        train_dm, repl_val_ids, batch_size=batch_size
    )
    datamodule = Det3dFastLbdDataModule(train_dm, val_dm)
    datamodule.setup()

    configs = train_dm.configs
    fg_labels = configs["plan_train"]["fg_labels"]
    fps = forward_patch_size_from_configs(configs)
    batches_per_epoch = len(datamodule.train_dataloader())
    R, nndet_plan = build_nndet_retinaunet_module(
        configs, num_train_batches=batches_per_epoch
    )
    patch_module_for_det3d_fast_batch(
        R, fg_labels=fg_labels, forward_patch_size=fps
    )

# %%
# SECTION:--- inspect batch ---
    train_batch = next(iter(datamodule.train_dataloader()))
    if train_dm.transforms_batch is not None:
        train_batch = train_dm.transforms_batch(train_batch)
    train_batch.keys()
    print(train_batch["image"].min(), train_batch["image"].max(), train_batch["image"].dtype)
    print(
        train_batch["image"].shape,
        train_batch["bbox"].shape,
        train_batch["lm"].shape,
    )

# %%
# SECTION:--- single train step ---
    device = torch.device(f"cuda:{device_id}")
    R = R.to(device)
    batch_dev = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in train_batch.items()
    }
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
    epochs = 500 if full_run else 2
    run_name = "DET3D-FAST-LBD-E500-FULL"
    wandb = True
    wandb_grid_epoch_freq = 20
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
            wandb_grid_epoch_freq=1,
            val_every_n_epochs=1,
            val_batches_per_epoch=2,
            permanent_checkpoint_every_n_epochs=epochs + 1,
        )
    fit_out["train_dir"]

