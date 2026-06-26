"""W&B logging + checkpoint wiring for native nnDetection (RetinaUNetV001) training."""

from __future__ import annotations

import os
import types
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from fran.callback.wandb.wandb_bk import WandbLogBestCkpt
from fran.configs.helpers import normalize_logging_payload
from fran.managers.project import Project
from fran.managers.wandb.wandb import (
    WandbManager,
    derive_wandb_project_name,
    wandb_run_exists,
)
from fran.trainers.trainer import _flatten_dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from utilz.stringz import ast_literal_eval, headline


BENCHMARK_TRAIN_DET_LOSS = "train_det_loss"
BENCHMARK_PERF_KEY = "val0_metric"
NNDET_TRAIN_LOSS_KEYS = ("cls", "reg", "seg_ce", "seg_dice", "loss")


def _mean_step_vals(vals: dict, key: str) -> float | None:
    key_vals = vals.get(key)
    if not key_vals:
        return None
    return float(np.mean(key_vals))


def _aggregate_nndet_train_metrics(vals: dict, *, prefix: str = "train0") -> dict:
    metrics: dict[str, float] = {}
    cls_m = _mean_step_vals(vals, "cls")
    reg_m = _mean_step_vals(vals, "reg")
    seg_ce_m = _mean_step_vals(vals, "seg_ce")
    seg_dice_m = _mean_step_vals(vals, "seg_dice")
    total_m = _mean_step_vals(vals, "loss")
    if cls_m is not None:
        metrics[f"{prefix}_cls"] = cls_m
    if reg_m is not None:
        metrics[f"{prefix}_reg"] = reg_m
    if seg_ce_m is not None:
        metrics[f"{prefix}_seg_ce"] = seg_ce_m
    if seg_dice_m is not None:
        metrics[f"{prefix}_seg_dice"] = seg_dice_m
    if cls_m is not None and reg_m is not None:
        metrics[BENCHMARK_TRAIN_DET_LOSS] = cls_m + reg_m
    if cls_m is not None and seg_ce_m is not None and seg_dice_m is not None:
        metrics[f"{prefix}_cls_seg_loss"] = cls_m + seg_ce_m + seg_dice_m
    if total_m is not None:
        metrics[f"{prefix}_loss"] = total_m
    return metrics


def install_nndet_pl2_val_hooks(module) -> None:
    """PL2 val hooks: val0_* losses + box/seg metrics (TrainerDet / RetinaUNetManager parity)."""
    if getattr(module, "_nndet_pl2_val_hooks", False):
        return

    val_step = module.validation_step

    def on_validation_epoch_start(self):
        self._nndet_val_epoch_outputs = []
        self.box_evaluator.reset()
        self.seg_evaluator.reset()

    def validation_step(self, batch, batch_idx):
        out = val_step(batch, batch_idx)
        self._nndet_val_epoch_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        outputs = self._nndet_val_epoch_outputs
        self._nndet_val_epoch_outputs = []
        vals = defaultdict(list)
        for step_out in outputs:
            for key, val in step_out.items():
                vals[key].append(
                    float(val.item()) if hasattr(val, "item") else float(val)
                )
        metrics = _aggregate_nndet_train_metrics(vals, prefix="val0")
        if "val0_cls" in metrics and "val0_seg_ce" in metrics and "val0_seg_dice" in metrics:
            metrics["val0_cls_seg_loss"] = (
                metrics["val0_cls"] + metrics["val0_seg_ce"] + metrics["val0_seg_dice"]
            )
        for log_key, mean_val in metrics.items():
            prog = log_key in ("val0_loss", "val0_cls_seg_loss")
            self.log(log_key, mean_val, prog_bar=prog, sync_dist=True)

        metric_scores, _ = self.box_evaluator.finish_online_evaluation()
        self.box_evaluator.reset()
        for key, value in metric_scores.items():
            log_key = f"val0_{key}"
            metrics[log_key] = float(value)
            self.log(log_key, float(value), sync_dist=True)

        val_metric = float(metric_scores[self.eval_score_key])
        metrics[BENCHMARK_PERF_KEY] = val_metric
        self.log(BENCHMARK_PERF_KEY, val_metric, prog_bar=True, sync_dist=True)

        seg_scores, _ = self.seg_evaluator.finish_online_evaluation()
        self.seg_evaluator.reset()
        for key, value in seg_scores.items():
            log_key = f"val0_{key}"
            metrics[log_key] = float(value)
            self.log(log_key, float(value), sync_dist=True)

        self._nndet_last_val_metrics = metrics

    module.on_validation_epoch_start = types.MethodType(on_validation_epoch_start, module)
    module.validation_step = types.MethodType(validation_step, module)
    module.on_validation_epoch_end = types.MethodType(on_validation_epoch_end, module)
    module._nndet_pl2_val_hooks = True


def patch_nndet_module_pl2(module, *, log_train_det_loss: bool = False) -> None:
    """nnDetection targets PL 1.x; dl env uses Lightning 2.x epoch-end hooks."""
    from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001

    pl_major = int(pl.__version__.split(".")[0])
    if pl_major < 2:
        return

    module.training_epoch_end = None
    module.validation_epoch_end = None
    module.configure_callbacks = lambda: []

    def training_step(self, batch, batch_idx):
        out = RetinaUNetV001.training_step(self, batch, batch_idx)
        self._nndet_epoch_outputs.append(out)
        return out

    def on_train_epoch_end(self):
        outputs = self._nndet_epoch_outputs
        self._nndet_epoch_outputs = []
        vals = defaultdict(list)
        for step_out in outputs:
            for key, val in step_out.items():
                if key == "loss":
                    vals[key].append(
                        val.detach().item() if hasattr(val, "detach") else float(val)
                    )
                else:
                    vals[key].append(val)
        if log_train_det_loss:
            metrics = _aggregate_nndet_train_metrics(vals, prefix="train0")
            for log_key, mean_val in metrics.items():
                self.log(log_key, mean_val, sync_dist=True)
        else:
            metrics = {}
            for key, key_vals in vals.items():
                mean_val = float(np.mean(key_vals))
                log_key = "train_loss" if key == "loss" else f"train_{key}"
                self.log(log_key, mean_val, sync_dist=True)
                metrics[log_key] = mean_val
        self._nndet_last_epoch_metrics = metrics

    module._nndet_epoch_outputs = []
    module._nndet_last_epoch_metrics = {}
    module.training_step = types.MethodType(training_step, module)
    module.on_train_epoch_end = types.MethodType(on_train_epoch_end, module)


def nndet_checkpoint_monitor(
    trainer_cfg: dict,
    *,
    val_enabled: bool,
    log_train_det_loss: bool = False,
) -> tuple[str, str]:
    if val_enabled:
        if log_train_det_loss:
            return BENCHMARK_PERF_KEY, "max"
        return str(trainer_cfg["monitor_key"]), str(trainer_cfg["monitor_mode"])
    if log_train_det_loss:
        return BENCHMARK_TRAIN_DET_LOSS, "min"
    return "train_loss", "min"


def build_nndet_model_checkpoints(
    train_dir: Path,
    monitor_key: str,
    monitor_mode: str,
    *,
    permanent_checkpoint_every_n_epochs: int | None = 100,
) -> list:
    train_dir = Path(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    best = ModelCheckpoint(
        dirpath=str(train_dir),
        filename="model_best",
        save_last=True,
        save_top_k=1,
        monitor=monitor_key,
        mode=monitor_mode,
    )
    best.CHECKPOINT_NAME_LAST = "model_last"
    callbacks = [best, LearningRateMonitor(logging_interval="epoch")]
    if permanent_checkpoint_every_n_epochs is not None:
        callbacks.insert(
            1,
            ModelCheckpoint(
                dirpath=str(train_dir),
                filename="epoch{epoch:04d}-snapshot",
                save_top_k=-1,
                save_last=False,
                every_n_epochs=int(permanent_checkpoint_every_n_epochs),
                enable_version_counter=False,
                auto_insert_metric_name=False,
            ),
        )
    return callbacks


def build_nndet_wandb_logger(
    project_title: str,
    run_name: str,
    *,
    tags: Sequence[str] | None = None,
    notes: str = "",
    log_model_checkpoints: bool = False,
    wb_mode: str | None = None,
) -> WandbManager:
    project = Project(project_title=project_title)
    mode = wb_mode or os.environ.get("WANDB_MODE", "online")
    return WandbManager(
        project=project,
        run_id=run_name,
        wandb_project_name=derive_wandb_project_name(project),
        log_model_checkpoints=log_model_checkpoints,
        tags=list(tags) if tags else [],
        notes=notes,
        wb_mode=mode,
    )


def log_nndet_run_config(logger: WandbManager, *, trainer_cfg: dict, task: str, exp_id: str) -> None:
    payload = {
        "nndet/task": task,
        "nndet/exp_id": exp_id,
        "nndet/trainer_cfg": normalize_logging_payload(deepcopy(trainer_cfg)),
    }
    logger.experiment.config.update(_flatten_dict(payload, base="configs"), allow_val_change=True)


def ensure_local_ckpt_on_wandb_resume(
    train_dir: Path,
    run_name: str,
    project_title: str,
    logger: WandbManager | None,
    *,
    wandb_run_is_new: bool | None,
) -> Path | None:
    last_local = Path(train_dir) / "model_last.ckpt"
    if last_local.is_file():
        return last_local
    if logger is None or not run_name or wandb_run_is_new:
        return None
    wb_ckpt = logger.model_checkpoint
    if wb_ckpt and Path(wb_ckpt).is_file():
        headline(f"W&B resume: using checkpoint from summary {wb_ckpt}")
        return Path(wb_ckpt)
    try:
        logger.download_checkpoints()
        wb_ckpt = logger.model_checkpoint
        if wb_ckpt and Path(wb_ckpt).is_file():
            headline(f"W&B resume: downloaded checkpoint {wb_ckpt}")
            return Path(wb_ckpt)
    except Exception as exc:
        headline(f"W&B resume: checkpoint sync attempt failed: {exc}")
    return None


def resolve_wandb_run_is_new(project_title: str, run_name: str | None, wandb: bool) -> bool | None:
    if not wandb or not run_name:
        return None
    project = Project(project_title=project_title)
    wb_project = derive_wandb_project_name(project)
    mode = os.environ.get("WANDB_MODE", "online")
    if mode in {"offline", "disabled", "dryrun"}:
        return False
    return not wandb_run_exists(wb_project, run_name, wb_mode=mode)


def build_nndet_retinaunet_wandb_grid_callback(
    configs: dict,
    log_folder: Path | str,
    *,
    wandb_grid_epoch_freq: int = 5,
):
    #AI
    """WandbRetinaUNetImageGridCallback for nnDet train_step preds (xyxyzz → xyzxyz in grid)."""
    from det3d.callback.wandb_det_grid import WandbRetinaUNetImageGridCallback

    val_patch_size = configs["model_params"]["val_patch_size"]
    if val_patch_size is None:
        val_patch_size = configs["plan_train"]["patch_size"]
    if isinstance(val_patch_size, str):
        val_patch_size = ast_literal_eval(val_patch_size)
    plan = configs["plan_train"]
    callback = WandbRetinaUNetImageGridCallback(
        patch_size=[int(v) for v in val_patch_size],
        epoch_freq=max(1, int(wandb_grid_epoch_freq)),
        local_folder=str(Path(log_folder) / "wandb_grid"),
        score_min=float(plan.get("wandb_grid_score_min", 0.3)),
        score_mid_min=float(plan.get("wandb_grid_score_mid_min", 0.5)),
        score_high_min=float(plan.get("wandb_grid_score_high_min", 0.8)),
        tiny_side_px=int(plan.get("wandb_grid_tiny_side_px", 4)),
        pred_top_k=plan.get("wandb_grid_pred_top_k", 5),
        show_fg_heatmap=bool(plan.get("wandb_grid_show_fg_heatmap", True)),
        show_pred_seg=bool(plan.get("wandb_grid_show_pred_seg", True)),
        show_gt_seg=bool(plan.get("wandb_grid_show_gt_seg", True)),
        gt_seg_key="lm",
        adapt_nndet_boxes=True,
    )
    return callback


def build_nndet_trainer_callbacks(
    train_dir: Path,
    trainer_cfg: dict,
    *,
    val_enabled: bool,
    wandb: bool,
    permanent_checkpoint_every_n_epochs: int | None = 100,
    extra_callbacks: Sequence | None = None,
    log_train_det_loss: bool = False,
) -> list:
    monitor_key, monitor_mode = nndet_checkpoint_monitor(
        trainer_cfg,
        val_enabled=val_enabled,
        log_train_det_loss=log_train_det_loss,
    )
    callbacks = build_nndet_model_checkpoints(
        train_dir,
        monitor_key,
        monitor_mode,
        permanent_checkpoint_every_n_epochs=permanent_checkpoint_every_n_epochs,
    )
    if wandb:
        callbacks.append(WandbLogBestCkpt())
    if extra_callbacks:
        callbacks = callbacks + list(extra_callbacks)
    return callbacks


def build_nndet_pl_trainer_kwargs(
    trainer_cfg: dict,
    *,
    max_epochs: int,
    callbacks: list,
    logger: WandbManager | bool | None,
    val_enabled: bool,
    ckpt_path: Path | None = None,
    check_val_every_n_epoch: int = 1,
) -> dict:
    pl_major = int(pl.__version__.split(".")[0])
    trainer_kwargs: dict[str, Any] = {
        "max_epochs": int(max_epochs),
        "callbacks": callbacks,
        "logger": logger if logger is not None else True,
        "deterministic": trainer_cfg["deterministic"],
        "benchmark": trainer_cfg["benchmark"],
        "num_sanity_val_steps": 0,
    }
    if not val_enabled:
        trainer_kwargs["limit_val_batches"] = 0
        trainer_kwargs["check_val_every_n_epoch"] = 0
    else:
        trainer_kwargs["check_val_every_n_epoch"] = int(check_val_every_n_epoch)
        n_val = int(trainer_cfg.get("num_val_batches_per_epoch", 0))
        if n_val > 0:
            trainer_kwargs["limit_val_batches"] = n_val
    if ckpt_path is not None and pl_major < 2:
        trainer_kwargs["resume_from_checkpoint"] = str(ckpt_path)

    num_gpus = int(trainer_cfg["gpus"])
    if pl_major >= 2:
        devices = list(range(num_gpus)) if num_gpus > 1 else num_gpus
        trainer_kwargs["accelerator"] = trainer_cfg["accelerator"] or "gpu"
        trainer_kwargs["devices"] = devices
        trainer_kwargs["precision"] = trainer_cfg["precision"]
        trainer_kwargs["reload_dataloaders_every_n_epochs"] = 0
    else:
        trainer_kwargs["gpus"] = list(range(num_gpus)) if num_gpus > 1 else num_gpus
        trainer_kwargs["accelerator"] = trainer_cfg["accelerator"]
        trainer_kwargs["precision"] = trainer_cfg["precision"]
        trainer_kwargs["amp_backend"] = trainer_cfg["amp_backend"]
        trainer_kwargs["amp_level"] = trainer_cfg["amp_level"]
        trainer_kwargs["reload_dataloaders_every_epoch"] = False
    return trainer_kwargs


def fit_nndet_module(
    module,
    datamodule=None,
    *,
    train_dataloaders=None,
    val_dataloaders=None,
    train_dir: Path,
    trainer_cfg: dict,
    task: str,
    exp_id: str,
    project_title: str = "lidca",
    run_name: str | None = None,
    train_mode: str = "overwrite",
    max_epochs: int | None = None,
    wandb: bool = True,
    tags: Sequence[str] | None = None,
    notes: str = "",
    extra_callbacks: Sequence | None = None,
    val_enabled: bool | None = None,
    permanent_checkpoint_every_n_epochs: int | None = 100,
    patch_pl2: bool = True,
    log_train_det_loss: bool = False,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
    check_val_every_n_epoch: int = 1,
    after_pl2_patch=None,
):
    """Lightning fit with local + W&B checkpoint parity (TrainerDet-style)."""
    from nndet.io.load import save_pickle

    if datamodule is None and train_dataloaders is None:
        raise ValueError("fit_nndet_module needs datamodule or train_dataloaders")

    trainer_cfg = deepcopy(trainer_cfg)
    if max_epochs is not None:
        trainer_cfg["max_num_epochs"] = int(max_epochs)
        module.trainer_cfg["max_num_epochs"] = int(max_epochs)

    train_dir = Path(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(module.plan, train_dir / "plan.pkl")

    if patch_pl2:
        patch_nndet_module_pl2(module, log_train_det_loss=log_train_det_loss)

    if after_pl2_patch is not None:
        after_pl2_patch(module)

    if val_enabled is None:
        val_enabled = int(trainer_cfg.get("num_val_batches_per_epoch", 0)) > 0

    if val_enabled and patch_pl2:
        install_nndet_pl2_val_hooks(module)

    run_name = run_name or exp_id
    wandb_run_is_new = resolve_wandb_run_is_new(project_title, run_name, wandb)
    logger = None
    if wandb:
        logger = build_nndet_wandb_logger(
            project_title,
            run_name,
            tags=tags,
            notes=notes,
        )
        log_nndet_run_config(logger, trainer_cfg=trainer_cfg, task=task, exp_id=exp_id)

    ckpt_path = None
    if str(train_mode).lower() == "resume":
        ckpt_path = ensure_local_ckpt_on_wandb_resume(
            train_dir,
            run_name,
            project_title,
            logger,
            wandb_run_is_new=wandb_run_is_new,
        )
        if ckpt_path is None and (train_dir / "model_last.ckpt").is_file():
            ckpt_path = train_dir / "model_last.ckpt"

    callbacks = build_nndet_trainer_callbacks(
        train_dir,
        trainer_cfg,
        val_enabled=val_enabled,
        wandb=wandb,
        permanent_checkpoint_every_n_epochs=permanent_checkpoint_every_n_epochs,
        extra_callbacks=extra_callbacks,
        log_train_det_loss=log_train_det_loss,
    )
    trainer_kwargs = build_nndet_pl_trainer_kwargs(
        trainer_cfg,
        max_epochs=int(trainer_cfg["max_num_epochs"]),
        callbacks=callbacks,
        logger=logger if wandb else True,
        val_enabled=val_enabled,
        ckpt_path=ckpt_path,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )
    trainer = pl.Trainer(**trainer_kwargs)
    if limit_train_batches is not None:
        trainer.limit_train_batches = int(limit_train_batches)
    if limit_val_batches is not None:
        trainer.limit_val_batches = int(limit_val_batches)
    fit_kwargs = {}
    if ckpt_path is not None and int(pl.__version__.split(".")[0]) >= 2:
        fit_kwargs["ckpt_path"] = str(ckpt_path)
    if datamodule is not None:
        trainer.fit(module, datamodule=datamodule, **fit_kwargs)
    else:
        trainer.fit(
            module,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            **fit_kwargs,
        )
    return {
        "trainer": trainer,
        "train_dir": train_dir,
        "module": module,
        "logger": logger,
        "run_name": run_name,
    }
