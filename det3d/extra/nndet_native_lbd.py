"""Native nnDetection on det3d LBD — shared helpers and APIs.

Interactive run-through: ``nndet_native_lbd_runthrough.py``
Parity CP-0..4: ``nndet_parity_cp0_4.py``
W&B + checkpoints: ``det3d/detection/nndet_wandb.py`` (``run_native_training_loop(..., wandb=True)``).
"""
from __future__ import annotations

import json
import os
import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import yaml

from det3d.extra.lbd_nndet_materialize import (
    DUSTING_THRESHOLD,
    instance_seg_to_nndet_boxes_pkl,
    lbd_case_to_nndet_arrays,
    lbd_lm_to_sidecar_tensors,
    lm_pt_to_instance_seg,
    materialize_lbd_nndet_task as _materialize_lbd_nndet_task,
    nndet_boxes_pkl_to_seg_mask,
    select_fg_case_ids,
    verify_lbd_format_roundtrip,
)

NNDET_ROOT = Path("/home/ub/code/nnDetection")
DEFAULT_LBD_FOLDER = Path(
    "/r/datasets/preprocessed/lidca/lbd/spc_080_080_150_rlb40c36831_rlb40c36831_ex000"
)
SCRATCH_DET_DATA = Path("/s/agent_rw/nndet_native_lbd")
DEFAULT_DET_MODELS = Path("/s/agent_rw/nndet_models")
NATIVE_PLAN_SRC = Path("/r/datasets/nndet_data/Task012_LIDC/preprocessed/D3V001_3d.pkl")
NATIVE_DATASET_JSON_SRC = Path("/r/datasets/nndet_data/Task012_LIDC/dataset.json")
TASK = "Task012_LIDC_lbd"
PLAN_ID = "D3V001_3d"
FOLD = 0
SCRATCH_BATCH_SIZE = 1
SCRATCH_MAX_EPOCHS = 2
SCRATCH_N_CASES = 16
SCRATCH_CASE_IDS: Optional[List[str]] = None
SCRATCH_TRAIN_MODE = "overwrite"
SCRATCH_EXP_ID = "RetinaUNetV001_lbd"
SCRATCH_WANDB = True
SCRATCH_RUN_NAME: Optional[str] = None  # default EXP_ID
SCRATCH_WANDB_PROJECT = "lidca"
PARITY_CASE_ID = "lidc_0067"


# Legacy inline stages: nndet_native_lbd_runthrough.py


def _nndet_import_shim():
    if "torch._six" not in sys.modules:
        torch_six = types.ModuleType("torch._six")
        torch_six.string_classes = (str,)
        sys.modules["torch._six"] = torch_six


def _lightning_import_shim():
    mem_key = "pytorch_lightning.core.memory"
    existing = sys.modules.get(mem_key)
    if existing is None or not hasattr(existing, "ModelSummary"):
        try:
            from pytorch_lightning.core.memory import ModelSummary  # noqa: F401
        except ImportError:
            from pytorch_lightning.utilities.model_summary import ModelSummary

            import pytorch_lightning as pl

            core = pl.core
            memory = types.ModuleType(mem_key)
            memory.__spec__ = None
            memory.ModelSummary = ModelSummary
            sys.modules[mem_key] = memory
            core.memory = memory

    opt_key = "pytorch_lightning.trainer.optimizers"
    if opt_key not in sys.modules or not hasattr(sys.modules[opt_key], "_get_default_scheduler_config"):
        optimizers = types.ModuleType(opt_key)
        optimizers.__spec__ = None

        def _get_default_scheduler_config():
            return {
                "scheduler": None,
                "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": False,
                "monitor": "val_loss",
                "strict": True,
                "name": None,
            }

        optimizers._get_default_scheduler_config = _get_default_scheduler_config
        sys.modules[opt_key] = optimizers


def setup_nndet_env(
    det_data: Path = SCRATCH_DET_DATA,
    det_models: Path = DEFAULT_DET_MODELS,
) -> None:
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


def load_nndet_train_cfgs(cfg_path=None):
    cfg_path = cfg_path or NNDET_ROOT / "nndet/conf/train/v001.yaml"
    with open(cfg_path) as handle:
        train_cfg = yaml.safe_load(handle)
    return deepcopy(train_cfg["model_cfg"]), deepcopy(train_cfg["trainer_cfg"])


def clear_cuda_scratch() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def inspect_nndet_batch(batch) -> dict:
    out = {"keys": sorted(batch.keys())}
    out["data"] = tuple(batch["data"].shape)
    out["data_dtype"] = str(batch["data"].dtype)
    out["data_device"] = str(batch["data"].device)
    if "target" in batch:
        out["target"] = tuple(batch["target"].shape)
        out["target_dtype"] = str(batch["target"].dtype)
    if "boxes" in batch:
        out["boxes"] = [tuple(b.shape) for b in batch["boxes"]]
    if "classes" in batch:
        out["classes"] = [tuple(c.shape) for c in batch["classes"]]
    return out


def nndet_batch_to_device(batch, device):
    out = dict(batch)
    out["data"] = out["data"].float().to(device)
    out["target"] = out["target"].float().to(device)
    return out


def select_scratch_case_ids(
    lbd_folder: Path,
    n_cases: int = SCRATCH_N_CASES,
    case_ids: Optional[Sequence[str]] = None,
) -> list[str]:
    return select_fg_case_ids(lbd_folder, n_cases=n_cases, case_ids=case_ids)


def materialize_lbd_nndet_task(
    lbd_folder: Path = DEFAULT_LBD_FOLDER,
    scratch_det_data: Path = SCRATCH_DET_DATA,
    task_name: str = TASK,
    plan_id: str = PLAN_ID,
    case_ids: Optional[Sequence[str]] = None,
    n_cases: int = SCRATCH_N_CASES,
    plan_src: Path = NATIVE_PLAN_SRC,
    dataset_json_src: Path = NATIVE_DATASET_JSON_SRC,
    dusting_threshold: float = DUSTING_THRESHOLD,
    train_as_val: bool = True,
) -> dict:
    return _materialize_lbd_nndet_task(
        lbd_folder=lbd_folder,
        scratch_det_data=scratch_det_data,
        task_name=task_name,
        plan_id=plan_id,
        case_ids=case_ids,
        n_cases=n_cases,
        plan_src=plan_src,
        dataset_json_src=dataset_json_src,
        dusting_threshold=dusting_threshold,
        train_as_val=train_as_val,
    )


def compose_native_lbd_cfg(
    *,
    det_data: Path = SCRATCH_DET_DATA,
    fold: int = FOLD,
    max_epochs: int,
    batch_size: int,
    num_train_batches_per_epoch: int,
    num_val_batches_per_epoch: int = 0,
    precision: int | str = 16,
):
    #AI
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from nndet.utils.config import compose

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_module(config_module="nndet.conf", version_base="1.1")
    cfg = compose(
        TASK,
        "config.yaml",
        overrides=[
            f"host.parent_data={det_data}",
            f"exp.fold={fold}",
            f"+augment_cfg.batch_size={int(batch_size)}",
            "augment_cfg.multiprocessing=False",
            f"augment_cfg.num_train_batches_per_epoch={int(num_train_batches_per_epoch)}",
            f"augment_cfg.num_val_batches_per_epoch={int(num_val_batches_per_epoch)}",
            f"trainer_cfg.num_train_batches_per_epoch={int(num_train_batches_per_epoch)}",
            f"trainer_cfg.num_val_batches_per_epoch={int(num_val_batches_per_epoch)}",
            f"trainer_cfg.precision={precision}",
            f"trainer_cfg.max_num_epochs={int(max_epochs)}",
        ],
    )
    return cfg


def scratch_compose_cfg(fold=FOLD, max_epochs=SCRATCH_MAX_EPOCHS):
    #AI
    from hydra import initialize_config_module
    from nndet.utils.config import compose

    initialize_config_module(config_module="nndet.conf", version_base="1.1")
    cfg = compose(
        TASK,
        "config.yaml",
        overrides=[
            f"host.parent_data={SCRATCH_DET_DATA}",
            f"exp.fold={fold}",
            f"+augment_cfg.batch_size={SCRATCH_BATCH_SIZE}",
            "augment_cfg.multiprocessing=False",
            "augment_cfg.num_train_batches_per_epoch=4",
            "augment_cfg.num_val_batches_per_epoch=2",
            "trainer_cfg.num_train_batches_per_epoch=4",
            "trainer_cfg.num_val_batches_per_epoch=2",
            "trainer_cfg.precision=32",
            f"trainer_cfg.max_num_epochs={int(max_epochs)}",
        ],
    )
    return cfg


def run_native_training_loop(
    module,
    datamodule,
    trainer_cfg,
    plan,
    fold=FOLD,
    exp_id=SCRATCH_EXP_ID,
    train_mode=SCRATCH_TRAIN_MODE,
    max_epochs=SCRATCH_MAX_EPOCHS,
    *,
    wandb: bool = SCRATCH_WANDB,
    run_name: Optional[str] = None,
    project_title: str = SCRATCH_WANDB_PROJECT,
    tags: Optional[Sequence[str]] = None,
    notes: str = "",
    extra_callbacks=None,
    val_enabled: bool | None = None,
    permanent_checkpoint_every_n_epochs: int = 100,
    check_val_every_n_epoch: int = 20,
    wandb_grid_epoch_freq: int = 20,
    det3d_configs: dict | None = None,
):
    #AI
    """pl.Trainer.fit with native Datamodule — W&B + checkpoint layout like TrainerDet."""
    from omegaconf import OmegaConf
    from scripts.train import init_train_dir

    from det3d.detection.nndet_train import patch_module_for_native_wandb_grid
    from det3d.detection.nndet_wandb import (
        build_nndet_retinaunet_wandb_grid_callback,
        fit_nndet_module,
    )
    from fran.managers.project import Project

    trainer_cfg = deepcopy(trainer_cfg)
    trainer_cfg["num_train_batches_per_epoch"] = int(
        datamodule.augment_cfg["num_train_batches_per_epoch"]
    )
    module.trainer_cfg["num_train_batches_per_epoch"] = trainer_cfg["num_train_batches_per_epoch"]
    trainer_cfg["max_num_epochs"] = int(max_epochs)
    module.trainer_cfg["max_num_epochs"] = int(max_epochs)

    cfg = OmegaConf.create(
        {
            "task": TASK,
            "host": {"parent_results": os.environ["det_models"]},
            "exp": {"id": exp_id, "fold": int(fold)},
            "train": {"mode": train_mode},
            "trainer_cfg": trainer_cfg,
        }
    )
    train_dir = init_train_dir(cfg)

    callbacks = list(extra_callbacks) if extra_callbacks else []
    log_folder = Project(project_title).log_folder
    grid_patch = None
    if wandb and val_enabled and det3d_configs is not None:
        grid_cb = build_nndet_retinaunet_wandb_grid_callback(
            det3d_configs,
            log_folder,
            wandb_grid_epoch_freq=int(wandb_grid_epoch_freq),
        )
        callbacks.append(grid_cb)
        grid_patch = patch_module_for_native_wandb_grid
        print(
            f"native LBD wandb grid callback epoch_freq={wandb_grid_epoch_freq} "
            f"local={Path(log_folder) / 'wandb_grid'}",
            flush=True,
        )

    return fit_nndet_module(
        module,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
        train_dir=train_dir,
        trainer_cfg=trainer_cfg,
        task=TASK,
        exp_id=exp_id,
        project_title=project_title,
        run_name=run_name or SCRATCH_RUN_NAME or exp_id,
        train_mode=train_mode,
        max_epochs=int(max_epochs),
        wandb=wandb,
        tags=tags or ["native_lbd", "nndet", TASK],
        notes=notes,
        extra_callbacks=callbacks,
        val_enabled=val_enabled,
        permanent_checkpoint_every_n_epochs=permanent_checkpoint_every_n_epochs,
        patch_pl2=True,
        log_train_det_loss=False,
        check_val_every_n_epoch=int(check_val_every_n_epoch),
        after_pl2_patch=grid_patch,
    )


def append_metrics_log(train_dir: Path, tag: str, trainer) -> None:
    #AI
    log_path = SCRATCH_DET_DATA / "metrics_log.jsonl"
    row = {
        "tag": tag,
        "train_dir": str(train_dir),
        "epoch": int(trainer.current_epoch),
    }
    if trainer.callback_metrics:
        for key, val in trainer.callback_metrics.items():
            if hasattr(val, "item"):
                row[str(key)] = float(val.item())
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
