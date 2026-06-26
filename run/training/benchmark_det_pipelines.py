#!/usr/bin/env python3
"""Benchmark det / nnDetection training pipelines on a fixed 16-case subset.

Baseline: pure nnDetection (native Task012 preprocessed subset). Compare against
LBD-materialized native loop and (later) det3d-integrated paths.

Metric: log all train losses (train0_*); pass/fail uses detection performance
(`val0_metric` / box mAP), not CE or total loss — cls CE drops trivially.

Pass rule from wandb `LIDCA-HOSS` (retinanet, workable) vs `LIDCA-IMPS` (broken).

Examples:
  python run/training/benchmark_det_pipelines.py run --pipelines native_baseline,native_lbd,retinanet
  python run/training/benchmark_det_pipelines.py sweep --pipelines retinanet --gpu 1
  python run/training/benchmark_det_pipelines.py report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
import types
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

NNDET_ROOT = Path("/home/ub/code/nnDetection")
BENCHMARK_ROOT = Path("/s/agent_rw/nndet_benchmark")
BENCHMARK_LOGS = BENCHMARK_ROOT / "logs"
SCRATCH_DET_DATA = BENCHMARK_ROOT / "det_data"
DET_MODELS = Path("/s/agent_rw/nndet_models_benchmark")
DEFAULT_LBD_FOLDER = Path(
    "/r/datasets/preprocessed/lidca/lbd/spc_080_080_150_rlb40c36831_rlb40c36831_ex000"
)
NATIVE_IMAGES_TR = Path(
    "/r/datasets/nndet_data/Task012_LIDC/preprocessed/D3V001_3d/imagesTr"
)
NATIVE_PLAN = Path("/r/datasets/nndet_data/Task012_LIDC/preprocessed/D3V001_3d.pkl")
NATIVE_DATASET_JSON = Path("/r/datasets/nndet_data/Task012_LIDC/dataset.json")
PLAN_ID = "D3V001_3d"
FOLD = 0
DEFAULT_N_CASES = 16
DEFAULT_EPOCHS = 10
DEFAULT_BATCHES_PER_EPOCH = 4
DEFAULT_VAL_BATCHES_PER_EPOCH = 4
DEFAULT_BATCH_SIZE = 1
DEFAULT_DET3D_PROJECT = "lidca"
DEFAULT_DET3D_PLAN = 4
BENCHMARK_LOSS_KEY = "train_det_loss"
BENCHMARK_PERF_KEY = "val0_metric"
HANDOFF_MIN_METRIC_END = 0.03
HANDOFF_MIN_METRIC_RISE = 0.02
DET_LOSS_COMPONENT_GROUPS = (
    ("train_cls", "train_reg"),
    ("train0_cls", "train0_reg"),
    ("train0_cls_loss", "train0_box_reg_loss"),
    ("train0_classification", "train0_box_regression"),
)
DET3D_DET_METRIC_KEYS = {
    "retinanet": ("train0_cls_loss", "train0_box_reg_loss"),
    "retinaunet": ("train0_cls", "train0_reg"),
    "retinaunet_v3": ("train0_classification", "train0_box_regression"),
}
WANDB_ENTITY = "drubashir"
WANDB_PROJECT = "lidc"
REFERENCE_GOOD = "LIDCA-HOSS"
REFERENCE_BAD = "LIDCA-IMPS"
REFERENCE_RUNS = (REFERENCE_GOOD, REFERENCE_BAD)
WANDB_DET_SPECS = {
    REFERENCE_GOOD: ("train0_cls_loss", "train0_box_reg_loss"),
    REFERENCE_BAD: ("train0_cls", "train0_reg"),
}
REFERENCE_CSV_PATHS = {
    REFERENCE_GOOD: BENCHMARK_ROOT / "results" / "retinanet_hoss_workflow_n32_e20.csv",
    REFERENCE_BAD: BENCHMARK_ROOT / "results" / "retinaunet_imps_workflow_n32_e20.csv",
}
GOLDEN_NATIVE_LBD_CSV = BENCHMARK_ROOT / "results" / "native_lbd_native_lbd_ext.csv"
HANDOFF_MIN_DROP = 0.3
SWEEP_PIPELINES = (
    "native_baseline",
    "native_lbd",
    "retinanet",
    "retinaunet",
    "retinaunet_v3",
)
# det3d paths that must match HOSS/IMPS scratch (item tfms, not GpuTail batch path)
DET3D_ITEM_TFMS_ARCHES = frozenset({"retinanet", "retinaunet", "retinaunet_v3"})

PIPELINE_TASK = {
    "native_baseline": "Task012_LIDC_subset16",
    "native_lbd": "Task012_LIDC_lbd",
}


@dataclass
class PipelineSpec:
    name: str
    tag: str
    description: str
    kind: str = "native"
    task: str = ""
    arch: str = ""


PIPELINES: dict[str, PipelineSpec] = {
    "native_baseline": PipelineSpec(
        name="native_baseline",
        task=PIPELINE_TASK["native_baseline"],
        tag="native_baseline",
        description="nnDetection RetinaUNetV001 on native Task012 subset (no val)",
    ),
    "native_lbd": PipelineSpec(
        name="native_lbd",
        task=PIPELINE_TASK["native_lbd"],
        tag="native_lbd",
        description="nnDetection on LBD-materialized subset (no val)",
    ),
    "retinanet": PipelineSpec(
        name="retinanet",
        tag="retinanet_hoss",
        kind="det3d",
        arch="retinanet",
        description="RetinaNetManager — LIDCA-HOSS control (det-only; not retinaunet-like)",
    ),
    "retinaunet": PipelineSpec(
        name="retinaunet",
        tag="retinaunet_imps",
        kind="det3d",
        arch="retinaunet",
        description="RetinaUNetManager → nnDet RetinaUNetV001 — LIDCA-IMPS broken link",
    ),
    "retinaunet_v3": PipelineSpec(
        name="retinaunet_v3",
        tag="retinaunet_v3",
        kind="det3d",
        arch="retinaunet_v3",
        description="RetinaUNetManagerV3 on LBD via TrainerDet (det loss = cls+reg only)",
    ),
    "det3d_fast_lbd": PipelineSpec(
        name="det3d_fast_lbd",
        tag="det3d_fast_lbd",
        kind="det3d_fast",
        description="LBD HDF5 + disk boxes + semantic seg → RetinaUNetV001 via fit_nndet_module",
    ),
}


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


def setup_nndet_env(det_data: Path = SCRATCH_DET_DATA, det_models: Path = DET_MODELS) -> None:
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


def select_case_ids(lbd_folder: Path, n_cases: int, case_ids: list[str] | None) -> list[str]:
    import pandas as pd
    from det3d.configs.parser import ConfigMakerDet
    from det3d.extra.nndet_native_lbd import select_scratch_case_ids
    from fran.managers import Project

    if case_ids is not None:
        return [str(c) for c in case_ids]

    P = Project(DEFAULT_DET3D_PROJECT)
    C = ConfigMakerDet(P)
    C.setup(DEFAULT_DET3D_PLAN)
    conf = C.configs
    fold = int(conf["dataset_params"]["fold"])
    ds = [x.strip() for x in conf["plan_train"]["datasources"].split(",") if x.strip()]
    train_ids, _ = P.get_train_val_case_ids(fold, ds, nnz_allowed=False)
    train_set = set(train_ids)

    df = pd.read_csv(lbd_folder / "dataset_details.csv")
    rows = df[(df["has_fg"]) & (~df["bbox_empty"])].sort_values("case_id")
    pool = [str(c) for c in rows["case_id"] if str(c) in train_set]
    if len(pool) >= int(n_cases):
        return pool[: int(n_cases)]
    return select_scratch_case_ids(lbd_folder, n_cases=n_cases, case_ids=None)


def _symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    os.symlink(src, dst)


def filter_case_ids_native_available(case_ids: list[str]) -> list[str]:
    #AI
    out = []
    for case_id in case_ids:
        if (NATIVE_IMAGES_TR / f"{case_id}.npy").is_file():
            out.append(case_id)
    return out


def sweep_task_name(pipeline: str, n_cases: int) -> str:
    #AI
    if pipeline == "native_baseline":
        return f"Task012_LIDC_subset{n_cases}"
    if pipeline == "native_lbd":
        return f"Task012_LIDC_lbd{n_cases}"
    raise ValueError(f"unknown pipeline {pipeline}")


def materialize_native_baseline_subset(
    case_ids: list[str],
    scratch_det_data: Path = SCRATCH_DET_DATA,
    task_name: str | None = None,
) -> dict:
    #AI
    """Symlink native Task012 D3V001_3d cases into benchmark task tree."""
    from nndet.io.load import save_pickle

    task_name = task_name or PIPELINE_TASK["native_baseline"]
    task_dir = scratch_det_data / task_name
    preprocessed = task_dir / "preprocessed"
    images_tr = preprocessed / PLAN_ID / "imagesTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    _symlink_or_copy(NATIVE_PLAN, preprocessed / f"{PLAN_ID}.pkl")
    dataset_meta = json.loads(NATIVE_DATASET_JSON.read_text())
    dataset_meta["task"] = task_name
    (task_dir / "dataset.json").write_text(json.dumps(dataset_meta, indent=2))
    for case_id in case_ids:
        for name in (
            f"{case_id}.npy",
            f"{case_id}_seg.npy",
            f"{case_id}.pkl",
            f"{case_id}_boxes.pkl",
        ):
            _symlink_or_copy(NATIVE_IMAGES_TR / name, images_tr / name)
    split = {"train": list(case_ids), "val": list(case_ids)}
    save_pickle([split], preprocessed / "splits_final.pkl")
    return {"task_dir": task_dir, "images_tr": images_tr, "case_ids": case_ids}


def materialize_pipeline_data(
    pipeline: str,
    case_ids: list[str],
    lbd_folder: Path,
    task_name: str | None = None,
) -> dict:
    #AI
    if pipeline == "native_baseline":
        return materialize_native_baseline_subset(
            case_ids, task_name=task_name or PIPELINE_TASK["native_baseline"]
        )
    if pipeline == "native_lbd":
        from det3d.extra.nndet_native_lbd import materialize_lbd_nndet_task

        return materialize_lbd_nndet_task(
            lbd_folder=lbd_folder,
            scratch_det_data=SCRATCH_DET_DATA,
            task_name=task_name or PIPELINE_TASK["native_lbd"],
            case_ids=case_ids,
            n_cases=len(case_ids),
            train_as_val=True,
        )
    raise ValueError(f"unknown pipeline {pipeline}")


def load_nndet_train_cfgs():
    cfg_path = NNDET_ROOT / "nndet/conf/train/v001.yaml"
    with open(cfg_path) as handle:
        train_cfg = yaml.safe_load(handle)
    return deepcopy(train_cfg["model_cfg"]), deepcopy(train_cfg["trainer_cfg"])


def compose_benchmark_cfg(
    task: str,
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
):
    #AI
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    from nndet.utils.config import compose

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_module(config_module="nndet.conf", version_base="1.1")
    cfg = compose(
        task,
        "config.yaml",
        overrides=[
            f"host.parent_data={SCRATCH_DET_DATA}",
            f"exp.fold={FOLD}",
            f"+augment_cfg.batch_size={batch_size}",
            "augment_cfg.multiprocessing=False",
            f"augment_cfg.num_train_batches_per_epoch={batches_per_epoch}",
            "augment_cfg.num_val_batches_per_epoch=0",
            f"trainer_cfg.num_train_batches_per_epoch={batches_per_epoch}",
            "trainer_cfg.num_val_batches_per_epoch=0",
            "trainer_cfg.precision=32",
            f"trainer_cfg.max_num_epochs={int(epochs)}",
        ],
    )
    return cfg


def patch_nndet_module_pl2(module) -> None:
    #AI
    from det3d.detection.nndet_wandb import patch_nndet_module_pl2 as _patch

    _patch(module, log_train_det_loss=True)


def make_epoch_train_loss_csv(csv_path: Path):
    #AI
    import pytorch_lightning as pl

    class EpochTrainLossCSV(pl.Callback):
        def __init__(self):
            super().__init__()
            self.csv_path = Path(csv_path)
            self.rows: list[dict] = []
            self._epoch_t0: float | None = None
            self._run_t0 = time.perf_counter()
            self._pending_row: dict | None = None

        def on_train_epoch_start(self, trainer, pl_module) -> None:
            self._epoch_t0 = time.perf_counter()

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            epoch_sec = None
            if self._epoch_t0 is not None:
                epoch_sec = time.perf_counter() - self._epoch_t0
            self._pending_row = {
                "epoch": int(trainer.current_epoch),
                "epoch_sec": round(epoch_sec, 3) if epoch_sec is not None else "",
                "elapsed_sec": round(time.perf_counter() - self._run_t0, 3),
            }
            if int(getattr(trainer, "limit_val_batches", 0) or 0) == 0:
                if hasattr(pl_module, "_nndet_last_epoch_metrics") and pl_module._nndet_last_epoch_metrics:
                    self._pending_row.update(pl_module._nndet_last_epoch_metrics)
                self._fill_train_metrics_from_callback(trainer, self._pending_row)
                self._commit_row()

        def _fill_train_metrics_from_callback(self, trainer, row: dict) -> None:
            if BENCHMARK_LOSS_KEY in row or any(
                str(k).startswith(("train0_", "train_")) for k in row
            ):
                return
            for key, val in trainer.callback_metrics.items():
                if not str(key).startswith(("train0_", "train_")):
                    continue
                row[str(key)] = float(val.item()) if hasattr(val, "item") else float(val)
            if BENCHMARK_LOSS_KEY not in row:
                cls_raw = row.get("train0_cls", row.get("train_cls"))
                reg_raw = row.get("train0_reg", row.get("train_reg"))
                if cls_raw not in ("", None) and reg_raw not in ("", None):
                    row[BENCHMARK_LOSS_KEY] = float(cls_raw) + float(reg_raw)

        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            row = self._pending_row or {
                "epoch": int(trainer.current_epoch),
                "epoch_sec": "",
                "elapsed_sec": round(time.perf_counter() - self._run_t0, 3),
            }
            if hasattr(pl_module, "_nndet_last_epoch_metrics") and pl_module._nndet_last_epoch_metrics:
                row.update(pl_module._nndet_last_epoch_metrics)
            self._fill_train_metrics_from_callback(trainer, row)
            if hasattr(pl_module, "_nndet_last_val_metrics") and pl_module._nndet_last_val_metrics:
                row.update(pl_module._nndet_last_val_metrics)
            else:
                for key, val in trainer.callback_metrics.items():
                    if not str(key).startswith("val0_"):
                        continue
                    row[str(key)] = float(val.item()) if hasattr(val, "item") else float(val)
            self._pending_row = None
            self.rows.append(row)
            self._flush()

        def _commit_row(self) -> None:
            row = self._pending_row
            self._pending_row = None
            if row is None:
                return
            has_loss = BENCHMARK_LOSS_KEY in row or any(
                str(k).startswith(("train0_", "train_")) for k in row
            )
            has_perf = BENCHMARK_PERF_KEY in row
            if not has_loss and not has_perf:
                return
            self.rows.append(row)
            self._flush()

        def _flush(self) -> None:
            if not self.rows:
                return
            fieldnames = sorted({k for row in self.rows for k in row})
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)

    return EpochTrainLossCSV()


def det_loss_from_row(row: dict) -> float | None:
    #AI
    raw = row.get(BENCHMARK_LOSS_KEY, "")
    if raw not in ("", None):
        return float(raw)
    for cls_key, reg_key in DET_LOSS_COMPONENT_GROUPS:
        cls_raw = row.get(cls_key, "")
        reg_raw = row.get(reg_key, "")
        if cls_raw not in ("", None) and reg_raw not in ("", None):
            return float(cls_raw) + float(reg_raw)
    return None


def make_det3d_epoch_csv(csv_path: Path, arch: str):
    #AI
    import pytorch_lightning as pl

    cls_key, reg_key = DET3D_DET_METRIC_KEYS[arch]

    class Det3dEpochCSV(pl.Callback):
        def __init__(self):
            super().__init__()
            self.csv_path = Path(csv_path)
            self.rows: list[dict] = []

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            metrics = trainer.callback_metrics
            if cls_key not in metrics or reg_key not in metrics:
                return
            cls_v = float(metrics[cls_key].item())
            reg_v = float(metrics[reg_key].item())
            row = {
                "epoch": int(trainer.current_epoch),
                cls_key: cls_v,
                reg_key: reg_v,
                BENCHMARK_LOSS_KEY: cls_v + reg_v,
            }
            self.rows.append(row)
            self._flush()

        def _flush(self) -> None:
            if not self.rows:
                return
            fieldnames = sorted({k for row in self.rows for k in row})
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)

    return Det3dEpochCSV()


def run_det3d_pipeline(
    pipeline: str,
    case_ids: list[str],
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
    run_id: str,
    project: str = DEFAULT_DET3D_PROJECT,
    plan_id: int = DEFAULT_DET3D_PLAN,
) -> dict:
    #AI
    from det3d.configs.parser import ConfigMakerDet
    from det3d.preprocessing.run_build import build_from_plan
    from det3d.trainers.trainerdet import TrainerDet
    from fran.managers import Project

    os.environ["WANDB_MODE"] = "disabled"
    spec = PIPELINES[pipeline]
    arch = spec.arch
    P = Project(project)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = C.configs
    conf["dataset_params"]["fold"] = 0

    _, conf = build_from_plan(project, plan_id, configs=conf)
    conf["model_params"]["arch"] = arch
    if arch == "retinaunet_v3":
        conf["loss_params"]["lambda_dice"] = 0.5
        conf["loss_params"]["lambda_ce"] = 0.5

    results_dir = BENCHMARK_ROOT / "results"
    csv_path = results_dir / f"{spec.tag}_{run_id}.csv"
    metrics_cb = make_det3d_epoch_csv(csv_path, arch)
    batch_tfms = arch not in DET3D_ITEM_TFMS_ARCHES

    T = TrainerDet(P.project_title, conf, run_name=None, ckpt_path=None)
    T.setup(
        train_indices=list(case_ids),
        val_indices=None,
        val_every_n_epochs=epochs + 1,
        epochs=int(epochs),
        batch_size=int(batch_size),
        wandb=False,
        debug=False,
        devices=[0],
        cbs=[metrics_cb],
        early_stopping=False,
        batch_tfms=batch_tfms,
        permanent_checkpoint_every_n_epochs=epochs + 1,
    )
    T.trainer.limit_train_batches = int(batches_per_epoch)
    T.fit()

    meta = {
        "pipeline": spec.name,
        "tag": spec.tag,
        "kind": "det3d",
        "arch": arch,
        "project": project,
        "plan_id": int(plan_id),
        "run_id": run_id,
        "csv_path": str(csv_path),
        "case_ids": case_ids,
        "n_cases": len(case_ids),
        "epochs": int(epochs),
        "batches_per_epoch": int(batches_per_epoch),
        "batch_size": int(batch_size),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = results_dir / f"{spec.tag}_{run_id}.json"
    meta = finalize_benchmark_run(meta, epochs=int(epochs))
    summary_path = BENCHMARK_ROOT / "manifest.jsonl"
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(meta) + "\n")
    print_run_verdict(meta)
    return meta


def run_det3d_fast_pipeline(
    case_ids: list[str],
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
    run_id: str,
    *,
    wandb: bool = False,
    run_name: str | None = None,
    project_title: str = DEFAULT_DET3D_PROJECT,
    plan_id: int = DEFAULT_DET3D_PLAN,
) -> dict:
    #AI
    from det3d.detection.nndet_wandb import BENCHMARK_TRAIN_DET_LOSS
from det3d.extra.hybrid import run_det3d_fast_training_loop

    spec = PIPELINES["det3d_fast_lbd"]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if wandb:
        os.environ.pop("WANDB_MODE", None)

    results_dir = BENCHMARK_ROOT / "results"
    csv_path = results_dir / f"{spec.tag}_{run_id}.csv"
    metrics_cb = make_epoch_train_loss_csv(csv_path)

    exp_id = f"bench_{spec.tag}_{run_id}"
    wb_run_name = run_name or f"{spec.tag.upper()}-{run_id}"

    fit_out = run_det3d_fast_training_loop(
        case_ids=list(case_ids),
        epochs=int(epochs),
        batches_per_epoch=int(batches_per_epoch),
        batch_size=int(batch_size),
        project_title=project_title,
        plan_id=int(plan_id),
        exp_id=exp_id,
        det_models=DET_MODELS,
        wandb=wandb,
        run_name=wb_run_name,
        tags=[spec.tag, "benchmark", "disk_boxes"],
        notes=(
            f"benchmark {spec.name} n={len(case_ids)} ep={epochs} "
            f"bpe={batches_per_epoch} bs={batch_size}"
        ),
        extra_callbacks=[metrics_cb],
        permanent_checkpoint_every_n_epochs=int(epochs) + 1,
        val_batches_per_epoch=DEFAULT_VAL_BATCHES_PER_EPOCH,
        val_enabled=True,
    )

    meta = {
        "pipeline": spec.name,
        "tag": spec.tag,
        "kind": "det3d_fast",
        "run_id": run_id,
        "run_name": wb_run_name,
        "wandb": wandb,
        "train_dir": str(fit_out["train_dir"]),
        "csv_path": str(csv_path),
        "case_ids": case_ids,
        "n_cases": len(case_ids),
        "epochs": int(epochs),
        "batches_per_epoch": int(batches_per_epoch),
        "batch_size": int(batch_size),
        "train_det_loss_key": BENCHMARK_TRAIN_DET_LOSS,
        "benchmark_perf_key": BENCHMARK_PERF_KEY,
        "val_batches_per_epoch": DEFAULT_VAL_BATCHES_PER_EPOCH,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = results_dir / f"{spec.tag}_{run_id}.json"
    meta = finalize_benchmark_run(meta, epochs=int(epochs))
    summary_path = BENCHMARK_ROOT / "manifest.jsonl"
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(meta) + "\n")
    print_run_verdict(meta)
    return meta


def run_pipeline(
    pipeline: str,
    case_ids: list[str],
    lbd_folder: Path,
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
    run_id: str,
    task_name: str | None = None,
    *,
    wandb: bool = False,
    run_name: str | None = None,
    project_title: str = DEFAULT_DET3D_PROJECT,
) -> dict:
    #AI
    spec = PIPELINES[pipeline]
    if spec.kind == "det3d":
        return run_det3d_pipeline(
            pipeline=pipeline,
            case_ids=case_ids,
            epochs=epochs,
            batches_per_epoch=batches_per_epoch,
            batch_size=batch_size,
            run_id=run_id,
        )
    if spec.kind == "det3d_fast":
        return run_det3d_fast_pipeline(
            case_ids=case_ids,
            epochs=epochs,
            batches_per_epoch=batches_per_epoch,
            batch_size=batch_size,
            run_id=run_id,
            wandb=wandb,
            run_name=run_name,
            project_title=project_title,
        )
    return run_nndet_pipeline(
        pipeline=pipeline,
        case_ids=case_ids,
        lbd_folder=lbd_folder,
        epochs=epochs,
        batches_per_epoch=batches_per_epoch,
        batch_size=batch_size,
        run_id=run_id,
        task_name=task_name,
        wandb=wandb,
        run_name=run_name,
        project_title=project_title,
    )


def run_nndet_pipeline(
    pipeline: str,
    case_ids: list[str],
    lbd_folder: Path,
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
    run_id: str | None = None,
    task_name: str | None = None,
    *,
    wandb: bool = False,
    run_name: str | None = None,
    project_title: str = DEFAULT_DET3D_PROJECT,
) -> dict:
    #AI
    from det3d.detection.nndet_wandb import BENCHMARK_TRAIN_DET_LOSS, fit_nndet_module
    from nndet.io.datamodule.bg_module import Datamodule
    from nndet.io.load import load_pickle
    from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001
    from omegaconf import OmegaConf
    from scripts.train import init_train_dir

    spec = PIPELINES[pipeline]
    task = task_name or spec.task
    setup_nndet_env()
    materialize_pipeline_data(pipeline, case_ids, lbd_folder, task_name=task)

    cfg = compose_benchmark_cfg(task, epochs, batches_per_epoch, batch_size)
    plan = load_pickle(cfg.host.plan_path)
    data_dir = Path(cfg.host.preprocessed_output_dir) / plan["data_identifier"] / "imagesTr"

    augment_cfg = OmegaConf.to_container(cfg.augment_cfg, resolve=True)
    datamodule = Datamodule(
        augment_cfg=augment_cfg,
        plan=plan,
        data_dir=data_dir,
        fold=FOLD,
    )
    datamodule.setup()

    model_cfg, trainer_cfg = load_nndet_train_cfgs()
    trainer_cfg["num_train_batches_per_epoch"] = int(batches_per_epoch)
    trainer_cfg["num_val_batches_per_epoch"] = int(DEFAULT_VAL_BATCHES_PER_EPOCH)
    trainer_cfg["max_num_epochs"] = int(epochs)
    trainer_cfg["swa_epochs"] = 0
    trainer_cfg["monitor_key"] = BENCHMARK_PERF_KEY
    trainer_cfg["monitor_mode"] = "max"

    module = RetinaUNetV001(model_cfg=model_cfg, trainer_cfg=trainer_cfg, plan=plan)

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    exp_id = f"bench_{spec.tag}_{run_id}"
    train_cfg_omega = OmegaConf.create(
        {
            "task": task,
            "host": {"parent_results": os.environ["det_models"]},
            "exp": {"id": exp_id, "fold": FOLD},
            "train": {"mode": "overwrite"},
            "trainer_cfg": trainer_cfg,
        }
    )
    train_dir = init_train_dir(train_cfg_omega)

    results_dir = BENCHMARK_ROOT / "results"
    csv_path = results_dir / f"{spec.tag}_{run_id}.csv"
    metrics_cb = make_epoch_train_loss_csv(csv_path)

    wb_run_name = run_name or f"{spec.tag.upper()}-{run_id}"
    if wandb:
        os.environ.pop("WANDB_MODE", None)

    fit_nndet_module(
        module,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
        train_dir=train_dir,
        trainer_cfg=trainer_cfg,
        task=task,
        exp_id=exp_id,
        project_title=project_title,
        run_name=wb_run_name,
        train_mode="overwrite",
        max_epochs=int(epochs),
        wandb=wandb,
        tags=[spec.tag, "benchmark", task],
        notes=f"benchmark {spec.name} n={len(case_ids)} ep={epochs} bpe={batches_per_epoch}",
        extra_callbacks=[metrics_cb],
        val_enabled=True,
        permanent_checkpoint_every_n_epochs=int(epochs) + 1,
        patch_pl2=True,
        log_train_det_loss=True,
        limit_val_batches=int(DEFAULT_VAL_BATCHES_PER_EPOCH),
    )

    meta = {
        "pipeline": pipeline,
        "tag": spec.tag,
        "task": task,
        "run_id": run_id,
        "run_name": wb_run_name,
        "wandb": wandb,
        "train_dir": str(train_dir),
        "csv_path": str(csv_path),
        "case_ids": case_ids,
        "n_cases": len(case_ids),
        "epochs": int(epochs),
        "batches_per_epoch": int(batches_per_epoch),
        "batch_size": int(batch_size),
        "train_det_loss_key": BENCHMARK_TRAIN_DET_LOSS,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = results_dir / f"{spec.tag}_{run_id}.json"
    meta = finalize_benchmark_run(meta, epochs=int(epochs))
    summary_path = BENCHMARK_ROOT / "manifest.jsonl"
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(meta) + "\n")
    print_run_verdict(meta)
    return meta


def parse_train_log_losses(train_log: Path) -> list[float]:
    #AI
    text = train_log.read_text()
    losses = []
    for line in text.splitlines():
        match = re.search(r"Train loss reached: ([0-9.]+)", line)
        if match:
            losses.append(float(match.group(1)))
    return losses


def load_epoch_series_from_csv(csv_path: Path, loss_key: str = BENCHMARK_LOSS_KEY) -> list[tuple[int, float]]:
    #AI
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    series = []
    for row in rows:
        epoch = int(float(row["epoch"]))
        if loss_key != BENCHMARK_LOSS_KEY:
            raw = row.get(loss_key, "")
            if raw in ("", None):
                continue
            value = float(raw)
        else:
            value = det_loss_from_row(row)
            if value is None:
                continue
        series.append((epoch, value))
    series.sort(key=lambda item: item[0])
    return series


def summarize_perf_series(series: list[tuple[int, float]]) -> dict:
    #AI
    if len(series) < 2:
        end = series[0][1] if series else None
        return {
            "n_epochs": len(series),
            "metric_start": end,
            "metric_end": end,
            "metric_rise": 0.0,
            "metric_rise_per_epoch": 0.0,
            "monotonic_up_fraction": 0.0,
        }
    values = [v for _, v in series]
    ups = sum(1 for a, b in zip(values, values[1:]) if b > a)
    rise = values[-1] - values[0]
    return {
        "n_epochs": len(values),
        "metric_start": values[0],
        "metric_end": values[-1],
        "metric_rise": rise,
        "metric_rise_per_epoch": rise / max(len(values) - 1, 1),
        "monotonic_up_fraction": ups / max(len(values) - 1, 1),
    }


def load_perf_series_from_csv(
    csv_path: Path,
    perf_key: str = BENCHMARK_PERF_KEY,
) -> list[tuple[int, float]]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    series = []
    for row in rows:
        raw = row.get(perf_key, "")
        if raw in ("", None):
            continue
        series.append((int(float(row["epoch"])), float(raw)))
    series.sort(key=lambda item: item[0])
    return series


def summarize_loss_series(series: list[tuple[int, float]]) -> dict:
    #AI
    if len(series) < 2:
        return {
            "n_epochs": len(series),
            "loss_start": series[0][1] if series else None,
            "loss_end": series[-1][1] if series else None,
            "loss_drop": 0.0,
            "loss_drop_per_epoch": 0.0,
            "monotonic_down_fraction": 0.0,
        }
    values = [v for _, v in series]
    downs = sum(1 for a, b in zip(values, values[1:]) if b < a)
    drop = values[0] - values[-1]
    return {
        "n_epochs": len(values),
        "loss_start": values[0],
        "loss_end": values[-1],
        "loss_drop": drop,
        "loss_drop_per_epoch": drop / max(len(values) - 1, 1),
        "monotonic_down_fraction": downs / max(len(values) - 1, 1),
    }


def summarize_timing_from_csv(csv_path: Path) -> dict:
    rows = list(csv.DictReader(Path(csv_path).open(encoding="utf-8")))
    epoch_secs = []
    for row in rows:
        raw = row.get("epoch_sec", "")
        if raw in ("", None):
            continue
        epoch_secs.append(float(raw))
    if not epoch_secs:
        return {
            "n_timed_epochs": 0,
            "epoch_sec_mean": None,
            "epoch_sec_min": None,
            "epoch_sec_max": None,
            "total_elapsed_sec": None,
        }
    total = float(rows[-1]["elapsed_sec"]) if rows[-1].get("elapsed_sec") not in ("", None) else sum(epoch_secs)
    return {
        "n_timed_epochs": len(epoch_secs),
        "epoch_sec_mean": sum(epoch_secs) / len(epoch_secs),
        "epoch_sec_min": min(epoch_secs),
        "epoch_sec_max": max(epoch_secs),
        "total_elapsed_sec": total,
    }


def reference_summaries_from_csv(n_epochs: int) -> dict:
    out = {}
    for run_name, csv_path in REFERENCE_CSV_PATHS.items():
        if not csv_path.is_file():
            raise FileNotFoundError(f"missing reference CSV {csv_path}")
        series = load_epoch_series_from_csv(csv_path)
        window = series[: min(n_epochs, len(series))]
        out[run_name] = summarize_loss_series(window)
    return out


def reference_summaries_for_window(n_epochs: int) -> dict:
    #AI
    try:
        out = {}
        for run_name in REFERENCE_RUNS:
            series = fetch_wandb_reference_series(run_name)
            window = series[: min(n_epochs, len(series))]
            out[run_name] = summarize_loss_series(window)
        return out
    except Exception:
        return reference_summaries_from_csv(n_epochs)


def golden_native_summary(n_epochs: int) -> dict | None:
    if not GOLDEN_NATIVE_LBD_CSV.is_file():
        return None
    series = load_epoch_series_from_csv(GOLDEN_NATIVE_LBD_CSV)
    window = series[: min(n_epochs, len(series))]
    return summarize_loss_series(window)


def pass_fail_label(verdict: str, loss_drop: float, golden: dict | None) -> str:
    if verdict == "spurious_or_flat" or loss_drop < HANDOFF_MIN_DROP:
        return "fail"
    if verdict == "real_signal":
        if golden is not None and loss_drop < 0.5 * golden["loss_drop"]:
            return "marginal"
        return "pass"
    return "marginal"


def fetch_wandb_reference_perf_series(
    run_name: str,
    perf_key: str = BENCHMARK_PERF_KEY,
) -> list[tuple[int, float]]:
    import wandb

    api = wandb.Api()
    runs = list(
        api.runs(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}",
            filters={"display_name": run_name},
            per_page=1,
        )
    )
    if not runs:
        raise RuntimeError(f"wandb run not found: {run_name}")
    hist = runs[0].history(samples=10000)
    if "epoch" not in hist.columns:
        raise RuntimeError(f"missing epoch in wandb history for {run_name}")
    if perf_key not in hist.columns:
        raise RuntimeError(f"missing {perf_key}/epoch in wandb history for {run_name}")
    ep = hist.groupby("epoch", as_index=False)[perf_key].mean().dropna()
    return [(int(row["epoch"]), float(row[perf_key])) for _, row in ep.iterrows()]


def reference_perf_summaries_for_window(n_epochs: int) -> dict:
    out = {}
    for run_name in REFERENCE_RUNS:
        series = fetch_wandb_reference_perf_series(run_name)
        window = series[: min(n_epochs, len(series))]
        out[run_name] = summarize_perf_series(window)
    return out


def classify_perf_signal(pipeline_summary: dict, references: dict, n_epochs: int) -> dict:
    hoss = references[REFERENCE_GOOD]
    imps = references[REFERENCE_BAD]
    end = pipeline_summary["metric_end"]
    rise = pipeline_summary["metric_rise"]
    imps_end = imps["metric_end"]
    imps_rise = imps["metric_rise"]
    min_pass_end = max(float(imps_end) + 0.015, HANDOFF_MIN_METRIC_END)
    if end <= float(imps_end) + 0.005 and rise <= max(float(imps_rise) + 0.005, 0.01):
        verdict = "spurious_or_flat"
    elif end >= min_pass_end or rise >= HANDOFF_MIN_METRIC_RISE:
        verdict = "real_signal"
    elif rise > float(imps_rise) + 0.005:
        verdict = "weak_signal"
    else:
        verdict = "weak_signal"
    return {
        "verdict": verdict,
        "hoss_ref_metric_end": hoss["metric_end"],
        "hoss_ref_metric_rise": hoss["metric_rise"],
        "imps_ref_metric_end": imps_end,
        "imps_ref_metric_rise": imps_rise,
        "min_pass_metric_end": min_pass_end,
        "min_pass_metric_rise": HANDOFF_MIN_METRIC_RISE,
    }


def pass_fail_label_perf(verdict: str, metric_end: float, metric_rise: float, imps_ref: dict) -> str:
    if verdict == "spurious_or_flat":
        return "fail"
    if metric_end >= HANDOFF_MIN_METRIC_END and metric_rise >= HANDOFF_MIN_METRIC_RISE:
        return "pass"
    if verdict == "real_signal" and metric_end > float(imps_ref["metric_end"]) + 0.01:
        return "pass"
    if metric_end > float(imps_ref["metric_end"]) + 0.005:
        return "marginal"
    return "fail"


def fetch_wandb_reference_series(run_name: str, loss_key: str = BENCHMARK_LOSS_KEY) -> list[tuple[int, float]]:
    #AI
    import wandb

    api = wandb.Api()
    runs = list(
        api.runs(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}",
            filters={"display_name": run_name},
            per_page=1,
        )
    )
    if not runs:
        raise RuntimeError(f"wandb run not found: {run_name}")
    hist = runs[0].history(samples=10000)
    if "epoch" not in hist.columns:
        raise RuntimeError(f"missing epoch in wandb history for {run_name}")
    if loss_key == BENCHMARK_LOSS_KEY and run_name in WANDB_DET_SPECS:
        c1, c2 = WANDB_DET_SPECS[run_name]
        ep = hist.groupby("epoch", as_index=False).agg({c1: "mean", c2: "mean"}).dropna()
        return [
            (int(row["epoch"]), float(row[c1] + row[c2]))
            for _, row in ep.iterrows()
        ]
    if loss_key not in hist.columns:
        raise RuntimeError(f"missing {loss_key}/epoch in wandb history for {run_name}")
    ep = hist.groupby("epoch", as_index=False)[loss_key].mean().dropna()
    return [(int(row["epoch"]), float(row[loss_key])) for _, row in ep.iterrows()]


def classify_signal(pipeline_summary: dict, references: dict, n_epochs: int) -> dict:
    #AI
    hoss = references[REFERENCE_GOOD]
    imps = references[REFERENCE_BAD]
    drop = pipeline_summary["loss_drop"]
    per_ep = pipeline_summary["loss_drop_per_epoch"]
    mono = pipeline_summary["monotonic_down_fraction"]
    delta = hoss["loss_drop"] - imps["loss_drop"]
    min_pass_drop = imps["loss_drop"] + max(0.5 * delta, 0.01)
    min_pass_per_ep = imps["loss_drop_per_epoch"] + max(
        0.5 * (hoss["loss_drop_per_epoch"] - imps["loss_drop_per_epoch"]),
        0.001,
    )
    if drop <= imps["loss_drop"]:
        verdict = "spurious_or_flat"
    elif drop < min_pass_drop or per_ep < min_pass_per_ep:
        verdict = "weak_signal"
    elif mono >= 0.5 and drop >= 0.5 * hoss["loss_drop"]:
        verdict = "real_signal"
    else:
        verdict = "weak_signal"
    return {
        "verdict": verdict,
        "hoss_ref_drop": hoss["loss_drop"],
        "imps_ref_drop": imps["loss_drop"],
        "hoss_imps_delta": delta,
        "min_pass_drop": min_pass_drop,
        "min_pass_per_epoch": min_pass_per_ep,
    }


def finalize_benchmark_run(meta: dict, *, epochs: int | None = None) -> dict:
    """Append train-loss + val-det perf summary, HOSS/IMPS verdict, pass/fail to manifest."""
    csv_path = Path(meta["csv_path"])
    n_epochs = int(epochs if epochs is not None else meta["epochs"])
    manifest_path = Path(csv_path).with_suffix(".json")
    if manifest_path.name != f"{meta['tag']}_{meta['run_id']}.json":
        manifest_path = csv_path.parent / f"{meta['tag']}_{meta['run_id']}.json"

    out = dict(meta)
    out["evaluated_utc"] = datetime.now(timezone.utc).isoformat()

    if not csv_path.is_file():
        out["verdict"] = "no_csv"
        out["pass_fail"] = "unknown"
        out["eval_error"] = f"missing csv {csv_path}"
        manifest_path.write_text(json.dumps(out, indent=2))
        return out

    loss_series = load_epoch_series_from_csv(csv_path)[:n_epochs]
    loss_summary = summarize_loss_series(loss_series)
    perf_series = load_perf_series_from_csv(csv_path)[:n_epochs]
    perf_summary = summarize_perf_series(perf_series) if perf_series else None
    timing = summarize_timing_from_csv(csv_path)
    out.update(loss_summary)

    score_mode = "train_det_loss"
    signal: dict = {}
    ref_source = "none"
    if perf_summary and perf_summary["metric_end"] is not None:
        try:
            perf_refs = reference_perf_summaries_for_window(n_epochs)
            ref_source = "wandb"
            signal = classify_perf_signal(perf_summary, perf_refs, n_epochs)
            score_mode = BENCHMARK_PERF_KEY
            out.update(perf_summary)
            out.update(signal)
            out["pass_fail"] = pass_fail_label_perf(
                signal["verdict"],
                float(perf_summary["metric_end"]),
                float(perf_summary["metric_rise"]),
                perf_refs[REFERENCE_BAD],
            )
        except Exception as exc:
            out["perf_eval_error"] = str(exc)

    if score_mode != BENCHMARK_PERF_KEY:
        try:
            references = {}
            for run_name in REFERENCE_RUNS:
                ref_series = fetch_wandb_reference_series(run_name)
                window = ref_series[: min(n_epochs, len(ref_series))]
                references[run_name] = summarize_loss_series(window)
            ref_source = "wandb"
        except Exception:
            references = reference_summaries_from_csv(n_epochs)
            ref_source = "csv"
        signal = classify_signal(loss_summary, references, n_epochs)
        golden = golden_native_summary(n_epochs)
        out.update(signal)
        out["pass_fail"] = pass_fail_label(signal["verdict"], loss_summary["loss_drop"], golden)
        if golden is not None:
            out["golden_ref_run"] = "native_lbd_ext"
            out["golden_ref_drop"] = golden["loss_drop"]
            out["golden_ref_end"] = golden["loss_end"]
            out["golden_drop_ratio"] = loss_summary["loss_drop"] / max(golden["loss_drop"], 1e-6)

    out["score_mode"] = score_mode
    out["reference_source"] = ref_source
    out.update({f"timing_{k}": v for k, v in timing.items()})

    manifest_path.write_text(json.dumps(out, indent=2))

    BENCHMARK_LOGS.mkdir(parents=True, exist_ok=True)
    log_path = BENCHMARK_LOGS / f"{meta['tag']}_{meta['run_id']}.log"
    lines = [
        f"pipeline={meta.get('pipeline')} run_id={meta.get('run_id')} pass_fail={out['pass_fail']}",
        f"score_mode={score_mode} verdict={out.get('verdict')}",
    ]
    if score_mode == BENCHMARK_PERF_KEY:
        lines.append(
            f"{BENCHMARK_PERF_KEY} {perf_summary.get('metric_start'):.4f} -> "
            f"{perf_summary.get('metric_end'):.4f} rise={perf_summary.get('metric_rise'):.4f} "
            f"(min_pass_end={signal.get('min_pass_metric_end'):.4f})"
        )
    lines.append(
        f"train_det_loss {loss_summary.get('loss_start'):.4f} -> {loss_summary.get('loss_end'):.4f} "
        f"drop={loss_summary.get('loss_drop'):.4f}"
    )
    if out.get("golden_drop_ratio") is not None:
        lines.append(
            f"golden native_lbd_ext drop={out['golden_ref_drop']:.4f} ratio={out['golden_drop_ratio']:.2f}"
        )
    if timing["epoch_sec_mean"] is not None:
        lines.append(
            f"timing mean_epoch={timing['epoch_sec_mean']:.2f}s "
            f"min={timing['epoch_sec_min']:.2f}s max={timing['epoch_sec_max']:.2f}s "
            f"total={timing['total_elapsed_sec']:.1f}s"
        )
    lines.append("--- per-epoch ---")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    for row in rows[:n_epochs]:
        ep = row.get("epoch", "?")
        loss = det_loss_from_row(row)
        loss_s = f"{loss:.4f}" if loss is not None else "?"
        perf_raw = row.get(BENCHMARK_PERF_KEY, "")
        perf_s = f"{float(perf_raw):.4f}" if perf_raw not in ("", None) else "?"
        t_s = row.get("epoch_sec", "")
        lines.append(
            f"epoch {ep}: train_det_loss={loss_s} {BENCHMARK_PERF_KEY}={perf_s} epoch_sec={t_s}"
        )
    log_path.write_text("\n".join(lines) + "\n")
    out["log_path"] = str(log_path)
    return out


def print_run_verdict(meta: dict) -> None:
    pf = meta.get("pass_fail", "?")
    verdict = meta.get("verdict", "?")
    score_mode = meta.get("score_mode", BENCHMARK_LOSS_KEY)
    if score_mode == BENCHMARK_PERF_KEY:
        end = meta.get("metric_end")
        rise = meta.get("metric_rise")
        end_s = f"{end:.4f}" if end is not None else "?"
        rise_s = f"{rise:.4f}" if rise is not None else "?"
        line = f"  pass_fail={pf} verdict={verdict} {BENCHMARK_PERF_KEY}={end_s} rise={rise_s}"
        if meta.get("min_pass_metric_end") is not None:
            line += f" min_pass_end={meta['min_pass_metric_end']:.4f}"
    else:
        drop = meta.get("loss_drop")
        drop_s = f"{drop:.4f}" if drop is not None else "?"
        line = f"  pass_fail={pf} verdict={verdict} loss_drop={drop_s}"
        if meta.get("min_pass_drop") is not None:
            line += f" min_pass_drop={meta['min_pass_drop']:.4f}"
    if meta.get("timing_epoch_sec_mean") is not None:
        line += f" mean_epoch_sec={meta['timing_epoch_sec_mean']:.2f}"
    if meta.get("golden_drop_ratio") is not None:
        line += f" golden_ratio={meta['golden_drop_ratio']:.2f}"
    print(line)
    if meta.get("log_path"):
        print(f"  log={meta['log_path']}")


def report_benchmarks(results_dir: Path, n_epochs: int | None = None) -> dict:
    #AI
    results_dir = Path(results_dir)
    manifests = sorted(
        p
        for p in results_dir.glob("*.json")
        if p.name != "report.json" and not p.name.startswith("sweep_")
    )
    references = None
    rows = []
    for manifest_path in manifests:
        meta = json.loads(manifest_path.read_text())
        csv_path = Path(meta["csv_path"])
        if not csv_path.is_file():
            continue
        series = load_epoch_series_from_csv(csv_path)
        epochs = n_epochs or int(meta["epochs"])
        series = series[:epochs]
        summary = summarize_loss_series(series)
        if references is None:
            references = reference_summaries_for_window(epochs)
        signal = classify_signal(summary, references, epochs)
        row = {
            "pipeline": meta["pipeline"],
            "tag": meta["tag"],
            "run_id": meta["run_id"],
            **summary,
            **signal,
            "csv_path": str(csv_path),
        }
        rows.append(row)
    report = {
        "references": references,
        "pipelines": rows,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path = results_dir / "report.json"
    out_path.write_text(json.dumps(report, indent=2))
    return report


def print_report(report: dict) -> None:
    refs = report["references"]
    print("=== reference (first N epochs, train det loss = cls+reg) ===")
    hoss = refs[REFERENCE_GOOD]
    imps = refs[REFERENCE_BAD]
    print(
        f"{REFERENCE_GOOD} (retinanet, workable): drop={hoss['loss_drop']:.4f} "
        f"per_epoch={hoss['loss_drop_per_epoch']:.4f}"
    )
    print(
        f"{REFERENCE_BAD} (retinaunet, broken): drop={imps['loss_drop']:.4f} "
        f"per_epoch={imps['loss_drop_per_epoch']:.4f} "
        f"delta={hoss['loss_drop'] - imps['loss_drop']:.4f}"
    )
    try:
        n_ep = DEFAULT_EPOCHS
        if report["pipelines"]:
            row0 = report["pipelines"][0]
            n_ep = int(row0.get("epochs", row0.get("n_epochs", DEFAULT_EPOCHS)))
        perf_refs = reference_perf_summaries_for_window(n_ep)
        print(f"=== reference {BENCHMARK_PERF_KEY} (HOSS vs IMPS, same epoch window) ===")
        print(
            f"{REFERENCE_GOOD}: end={perf_refs[REFERENCE_GOOD]['metric_end']:.4f} "
            f"rise={perf_refs[REFERENCE_GOOD]['metric_rise']:.4f}"
        )
        print(
            f"{REFERENCE_BAD}: end={perf_refs[REFERENCE_BAD]['metric_end']:.4f} "
            f"rise={perf_refs[REFERENCE_BAD]['metric_rise']:.4f}"
        )
    except Exception:
        pass
    print("=== benchmark runs ===")
    for row in report["pipelines"]:
        print(
            f"{row['tag']} ({row['run_id']}): start={row['loss_start']:.4f} "
            f"end={row['loss_end']:.4f} drop={row['loss_drop']:.4f} "
            f"per_epoch={row['loss_drop_per_epoch']:.4f} "
            f"verdict={row['verdict']} min_pass_drop={row['min_pass_drop']:.4f}"
        )


def compare_pipeline_pair(
    baseline_summary: dict,
    lbd_summary: dict,
    min_abs_drop_diff: float = 0.05,
    winner_rel_margin: float = 0.15,
    flat_rel_margin: float = 0.10,
) -> dict:
    #AI
    drop_b = baseline_summary["loss_drop"]
    drop_l = lbd_summary["loss_drop"]
    diff = drop_l - drop_b
    denom = max(min(abs(drop_b), abs(drop_l)), 0.05)
    rel = diff / denom
    if abs(diff) < min_abs_drop_diff or abs(rel) < flat_rel_margin:
        outcome = "no_clear_difference"
    elif diff > 0 and rel >= winner_rel_margin:
        outcome = "native_lbd"
    elif diff < 0 and abs(rel) >= winner_rel_margin:
        outcome = "native_baseline"
    else:
        outcome = "inconclusive"
    row = {
        "baseline_drop": drop_b,
        "lbd_drop": drop_l,
        "drop_diff_lbd_minus_baseline": diff,
        "relative_diff": rel,
        "outcome": outcome,
    }
    return row


def batches_per_epoch_for_n(n_cases: int, base: int = DEFAULT_BATCHES_PER_EPOCH) -> int:
    #AI
    scaled = max(base, n_cases // 16)
    return min(scaled, 16)


def run_sample_sweep(
    lbd_folder: Path,
    sizes: list[int],
    epochs: int,
    sweep_id: str,
    gpu: int | None = None,
    pipelines: tuple[str, ...] = SWEEP_PIPELINES,
    min_abs_drop_diff: float = 0.05,
    winner_rel_margin: float = 0.15,
    flat_rel_margin: float = 0.10,
) -> dict:
    #AI
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    BENCHMARK_LOGS.mkdir(parents=True, exist_ok=True)
    SCRATCH_DET_DATA.mkdir(parents=True, exist_ok=True)
    DET_MODELS.mkdir(parents=True, exist_ok=True)

    rows = []
    stop_reason = None
    for n_target in sizes:
        case_ids = select_case_ids(lbd_folder, n_target, None)
        case_ids = filter_case_ids_native_available(case_ids)
        n_cases = len(case_ids)
        if n_cases < 8:
            stop_reason = "too_few_cases"
            break

        batches = batches_per_epoch_for_n(n_cases)
        run_id = f"{sweep_id}_n{n_cases}"
        pair = {"n_cases": n_cases, "n_target": n_target, "batches_per_epoch": batches, "run_id": run_id}
        metas = {}
        for pipeline in pipelines:
            task = sweep_task_name(pipeline, n_cases) if PIPELINES[pipeline].kind == "native" else None
            print(f"=== sweep n={n_cases} {pipeline} epochs={epochs} batches={batches} ===")
            metas[pipeline] = run_pipeline(
                pipeline=pipeline,
                case_ids=case_ids,
                lbd_folder=lbd_folder,
                epochs=epochs,
                batches_per_epoch=batches,
                batch_size=DEFAULT_BATCH_SIZE,
                run_id=run_id,
                task_name=task,
            )

        summaries = {
            p: summarize_loss_series(load_epoch_series_from_csv(Path(metas[p]["csv_path"]))[:epochs])
            for p in pipelines
        }
        row = {**pair}
        for pipeline in pipelines:
            sm = summaries[pipeline]
            row[f"{pipeline}_drop"] = sm["loss_drop"]
            row[f"{pipeline}_per_epoch"] = sm["loss_drop_per_epoch"]
        if "native_baseline" in summaries and "native_lbd" in summaries:
            cmp = compare_pipeline_pair(
                summaries["native_baseline"],
                summaries["native_lbd"],
                min_abs_drop_diff=min_abs_drop_diff,
                winner_rel_margin=winner_rel_margin,
                flat_rel_margin=flat_rel_margin,
            )
        else:
            cmp = {"outcome": "no_clear_difference"}
        row.update(cmp)
        msg = f"n={n_cases}: " + " ".join(f"{p}={summaries[p]['loss_drop']:.4f}" for p in pipelines)
        if "outcome" in cmp:
            msg += f" outcome={cmp['outcome']}"
        print(msg)
        rows.append(row)
        if cmp["outcome"] in ("native_lbd", "native_baseline"):
            stop_reason = f"winner_{cmp['outcome']}"
            break
        if len(rows) >= 2 and rows[-2]["outcome"] == "no_clear_difference" and cmp["outcome"] == "no_clear_difference":
            stop_reason = "difference_disappeared"
            break

    report = {
        "sweep_id": sweep_id,
        "pipelines": list(pipelines),
        "epochs": int(epochs),
        "sizes_requested": sizes,
        "rows": rows,
        "stop_reason": stop_reason,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path = BENCHMARK_ROOT / "results" / f"sweep_{sweep_id}.json"
    out_path.write_text(json.dumps(report, indent=2))
    return report


def print_sweep_report(report: dict) -> None:
    #AI
    pipelines = report["pipelines"]
    print(f"=== sample sweep {report['sweep_id']} stop={report['stop_reason']} ===")
    for row in report["rows"]:
        if "native_baseline" in pipelines and "native_lbd" in pipelines:
            print(
                f"n={row['n_cases']:4d}  baseline={row['baseline_drop']:.4f}  "
                f"lbd={row['lbd_drop']:.4f}  diff={row['drop_diff_lbd_minus_baseline']:+.4f}  "
                f"rel={row['relative_diff']:+.2f}  {row['outcome']}"
            )
            continue
        parts = " ".join(f"{p}={row[f'{p}_drop']:.4f}" for p in pipelines)
        print(f"n={row['n_cases']:4d}  {parts}  {row['outcome']}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark det/nnDet training pipelines")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run one or more pipelines")
    run_p.add_argument(
        "--pipelines",
        default="native_baseline,native_lbd,retinanet",
        help="Comma-separated pipeline keys",
    )
    run_p.add_argument("--lbd-folder", type=Path, default=DEFAULT_LBD_FOLDER)
    run_p.add_argument("--n-cases", type=int, default=DEFAULT_N_CASES)
    run_p.add_argument("--case-ids", default=None, help="Comma-separated case ids")
    run_p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    run_p.add_argument("--batches-per-epoch", type=int, default=DEFAULT_BATCHES_PER_EPOCH)
    run_p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    run_p.add_argument("--run-id", default=None)
    run_p.add_argument("--run-name", default=None, help="W&B run name (nnDet pipelines)")
    run_p.add_argument("--wandb", action="store_true", help="Enable W&B + checkpoint summary (nnDet)")
    run_p.add_argument("--project", default=DEFAULT_DET3D_PROJECT, help="fran project / W&B mnemonic")
    run_p.add_argument("--gpu", type=int, default=None, help="CUDA device index (sets CUDA_VISIBLE_DEVICES)")

    rep_p = sub.add_parser("report", help="Summarize results vs IMPS/DIET references")
    rep_p.add_argument("--results-dir", type=Path, default=BENCHMARK_ROOT / "results")
    rep_p.add_argument("--epochs", type=int, default=None)

    eval_p = sub.add_parser("eval", help="Score an existing run from its CSV + update manifest/log")
    eval_p.add_argument("--tag", required=True, help="Pipeline tag, e.g. det3d_fast_lbd")
    eval_p.add_argument("--run-id", required=True)
    eval_p.add_argument("--epochs", type=int, default=None, help="Epoch window for scoring (default: manifest)")

    all_p = sub.add_parser("all", help="run then report")
    all_p.add_argument("--pipelines", default="native_baseline,native_lbd,retinaunet_v3")
    all_p.add_argument("--lbd-folder", type=Path, default=DEFAULT_LBD_FOLDER)
    all_p.add_argument("--n-cases", type=int, default=DEFAULT_N_CASES)
    all_p.add_argument("--case-ids", default=None)
    all_p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    all_p.add_argument("--batches-per-epoch", type=int, default=DEFAULT_BATCHES_PER_EPOCH)
    all_p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    all_p.add_argument("--run-id", default=None)
    all_p.add_argument("--run-name", default=None, help="W&B run name (nnDet pipelines)")
    all_p.add_argument("--wandb", action="store_true", help="Enable W&B + checkpoint summary (nnDet)")
    all_p.add_argument("--project", default=DEFAULT_DET3D_PROJECT, help="fran project / W&B mnemonic")
    all_p.add_argument("--gpu", type=int, default=None, help="CUDA device index (sets CUDA_VISIBLE_DEVICES)")

    sweep_p = sub.add_parser("sweep", help="Double sample size until winner or flat")
    sweep_p.add_argument("--lbd-folder", type=Path, default=DEFAULT_LBD_FOLDER)
    sweep_p.add_argument(
        "--sizes",
        default="16,32,64,128,256,512",
        help="Comma-separated n_cases targets (doubling ladder)",
    )
    sweep_p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    sweep_p.add_argument("--sweep-id", default=None)
    sweep_p.add_argument("--gpu", type=int, default=None)
    sweep_p.add_argument(
        "--pipelines",
        default=",".join(SWEEP_PIPELINES),
        help="Comma-separated pipelines for this sweep leg",
    )
    sweep_p.add_argument("--min-abs-drop-diff", type=float, default=0.05)
    sweep_p.add_argument("--winner-rel-margin", type=float, default=0.15)
    sweep_p.add_argument("--flat-rel-margin", type=float, default=0.10)

    args = parser.parse_args()
    if getattr(args, "gpu", None) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    BENCHMARK_LOGS.mkdir(parents=True, exist_ok=True)
    SCRATCH_DET_DATA.mkdir(parents=True, exist_ok=True)
    DET_MODELS.mkdir(parents=True, exist_ok=True)

    if args.command == "report":
        report = report_benchmarks(args.results_dir, n_epochs=args.epochs)
        print_report(report)
        print(f"report written: {args.results_dir / 'report.json'}")
        return

    if args.command == "eval":
        manifest_path = BENCHMARK_ROOT / "results" / f"{args.tag}_{args.run_id}.json"
        meta = json.loads(manifest_path.read_text())
        meta = finalize_benchmark_run(meta, epochs=args.epochs)
        print_run_verdict(meta)
        return

    if args.command == "sweep":
        sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
        pipelines = tuple(p.strip() for p in args.pipelines.split(",") if p.strip())
        for pipeline in pipelines:
            if pipeline not in PIPELINES:
                raise SystemExit(f"unknown pipeline {pipeline}; choose from {list(PIPELINES)}")
        sweep_id = args.sweep_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report = run_sample_sweep(
            lbd_folder=args.lbd_folder,
            sizes=sizes,
            epochs=args.epochs,
            sweep_id=sweep_id,
            gpu=args.gpu,
            pipelines=pipelines,
            min_abs_drop_diff=args.min_abs_drop_diff,
            winner_rel_margin=args.winner_rel_margin,
            flat_rel_margin=args.flat_rel_margin,
        )
        print_sweep_report(report)
        print(f"sweep written: {BENCHMARK_ROOT / 'results' / f'sweep_{sweep_id}.json'}")
        return

    case_ids = None
    if args.case_ids:
        case_ids = [c.strip() for c in args.case_ids.split(",") if c.strip()]
    case_ids = select_case_ids(args.lbd_folder, args.n_cases, case_ids)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pipelines = [p.strip() for p in args.pipelines.split(",") if p.strip()]

    for pipeline in pipelines:
        if pipeline not in PIPELINES:
            raise SystemExit(f"unknown pipeline {pipeline}; choose from {list(PIPELINES)}")
        print(f"=== running {pipeline} cases={len(case_ids)} epochs={args.epochs} ===")
        meta = run_pipeline(
            pipeline=pipeline,
            case_ids=case_ids,
            lbd_folder=args.lbd_folder,
            epochs=args.epochs,
            batches_per_epoch=args.batches_per_epoch,
            batch_size=args.batch_size,
            run_id=run_id,
            task_name=sweep_task_name(pipeline, len(case_ids))
            if PIPELINES[pipeline].kind == "native"
            else None,
            wandb=getattr(args, "wandb", False),
            run_name=getattr(args, "run_name", None),
            project_title=getattr(args, "project", DEFAULT_DET3D_PROJECT),
        )
        print(f"done {pipeline} csv={meta['csv_path']}")
        if meta.get("wandb"):
            print(f"  wandb run_name={meta.get('run_name')} train_dir={meta.get('train_dir')}")
        if "pass_fail" not in meta:
            print_run_verdict(meta)

    if args.command == "all":
        report = report_benchmarks(BENCHMARK_ROOT / "results", n_epochs=args.epochs)
        print_report(report)
        print(f"report written: {BENCHMARK_ROOT / 'results' / 'report.json'}")


if __name__ == "__main__":
    main()
