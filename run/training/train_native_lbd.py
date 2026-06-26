#!/usr/bin/env python3
"""Native nnDet LBD training — materialize LBD → instance seg → Datamodule → RetinaUNetV001.

Uses ``det3d.extra.nndet_native_lbd`` (``lm`` CC instance ids + native ``pre_trafo``).
Not the fast GpuTail / disk-box hybrid path.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from det3d.configs.parser import ConfigMakerDet
from det3d.extra.hybrid import resolve_lbd_fg_case_ids
from det3d.extra.nndet_native_lbd import (
    DEFAULT_DET_MODELS,
    DEFAULT_LBD_FOLDER,
    FOLD,
    PLAN_ID,
    SCRATCH_DET_DATA,
    TASK,
    compose_native_lbd_cfg,
    load_nndet_train_cfgs,
    materialize_lbd_nndet_task,
    run_native_training_loop,
    setup_nndet_env,
)
from fran.managers.project import Project
from fran.utils.folder_names import FolderNames


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def _parse_case_ids(text: str | None) -> list[str] | None:
    if text is None:
        return None
    out = [c.strip() for c in text.split(",") if c.strip()]
    return out or None


def _lbd_folder(project_title: str, plan_id: int) -> Path:
    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(int(plan_id))
    return Path(FolderNames(P, C.configs["plan_train"]).lbd_folder)


def _resolve_train_ids(project, conf, args) -> list[str]:
    explicit = _parse_case_ids(args.case_ids)
    if explicit is not None:
        return explicit
    return resolve_lbd_fg_case_ids(project, conf, split="train", n_cases=args.n_train)


def _resolve_val_ids(project, conf, train_ids: list[str], args) -> list[str]:
    if args.train_equals_val:
        return list(train_ids)
    explicit = _parse_case_ids(args.val_case_ids)
    if explicit is not None:
        return explicit
    return resolve_lbd_fg_case_ids(project, conf, split="valid", n_cases=args.n_val)


def _materialize_native(
    lbd_folder: Path,
    det_data: Path,
    train_ids: list[str],
    val_ids: list[str],
    *,
    train_equals_val: bool,
) -> dict:
    #AI
    if train_equals_val:
        return materialize_lbd_nndet_task(
            lbd_folder=lbd_folder,
            scratch_det_data=det_data,
            case_ids=train_ids,
            n_cases=len(train_ids),
            train_as_val=True,
        )

    from nndet.io.load import save_pickle

    all_ids = list(dict.fromkeys(list(train_ids) + list(val_ids)))
    mat = materialize_lbd_nndet_task(
        lbd_folder=lbd_folder,
        scratch_det_data=det_data,
        case_ids=all_ids,
        n_cases=len(all_ids),
        train_as_val=False,
    )
    splits = [{"train": list(train_ids), "val": list(val_ids)}]
    split_path = Path(mat["task_dir"]) / "preprocessed" / "splits_final.pkl"
    save_pickle(splits, split_path)
    mat["splits"] = splits
    return mat


def main(args) -> None:
    #AI
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))

    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)

    det_data = Path(args.det_data) if args.det_data else SCRATCH_DET_DATA
    det_models = Path(args.det_models) if args.det_models else DEFAULT_DET_MODELS
    setup_nndet_env(det_data=det_data, det_models=det_models)

    P = Project(args.project)
    C = ConfigMakerDet(P)
    C.setup(int(args.plan))
    conf = C.configs
    if args.fold is not None:
        conf["dataset_params"]["fold"] = int(args.fold)

    lbd_folder = Path(args.lbd_folder) if args.lbd_folder else _lbd_folder(args.project, int(args.plan))

    train_ids = _resolve_train_ids(P, conf, args)
    val_enabled = not args.no_val
    val_ids: list[str] = []
    if val_enabled:
        val_ids = _resolve_val_ids(P, conf, train_ids, args)

    exp_id = args.exp_id or args.run_name
    if not exp_id:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        exp_id = f"LIDCA-NATIVE-LBD-{stamp}"
    run_name = args.run_name or exp_id

    train_bpe = args.batches_per_epoch
    if train_bpe is None:
        train_bpe = max(1, (len(train_ids) + int(args.batch_size) - 1) // int(args.batch_size))
    val_bpe = args.val_batches_per_epoch
    if val_enabled and val_bpe is None and val_ids:
        val_bpe = max(1, (len(val_ids) + int(args.batch_size) - 1) // int(args.batch_size))
    if not val_enabled or not val_ids:
        val_bpe = 0

    print(
        f"native LBD lbd_folder={lbd_folder} train={len(train_ids)} val={len(val_ids)} "
        f"epochs={args.epochs} batch_size={args.batch_size} "
        f"train_bpe={train_bpe} val_bpe={val_bpe} "
        f"train_mode={args.train_mode} exp_id={exp_id} "
        f"gpu={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}",
        flush=True,
    )

    if args.train_mode != "resume" or args.rematerialize:
        mat = _materialize_native(
            lbd_folder,
            det_data,
            train_ids,
            val_ids,
            train_equals_val=args.train_equals_val,
        )
        print(f"materialized {len(mat['case_ids'])} cases -> {mat['images_tr']}", flush=True)
        if mat["sidecar_drift"]:
            print(f"sidecar drift (lm truth used): {mat['sidecar_drift']}", flush=True)

    from nndet.io.datamodule.bg_module import Datamodule
    from nndet.io.load import load_pickle
    from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001
    from omegaconf import OmegaConf

    fold = int(args.fold) if args.fold is not None else FOLD
    cfg = compose_native_lbd_cfg(
        det_data=det_data,
        fold=fold,
        max_epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_train_batches_per_epoch=int(train_bpe),
        num_val_batches_per_epoch=int(val_bpe),
        precision=args.precision,
    )
    plan = load_pickle(cfg.host.plan_path)
    data_dir = Path(cfg.host.preprocessed_output_dir) / plan["data_identifier"] / "imagesTr"

    augment_cfg = OmegaConf.to_container(cfg.augment_cfg, resolve=True)
    datamodule = Datamodule(
        augment_cfg=augment_cfg,
        plan=plan,
        data_dir=data_dir,
        fold=fold,
    )
    datamodule.setup()

    model_cfg, trainer_cfg = load_nndet_train_cfgs()
    trainer_cfg["swa_epochs"] = 0
    trainer_cfg["monitor_key"] = "val0_metric"
    trainer_cfg["monitor_mode"] = "max"

    module = RetinaUNetV001(model_cfg=model_cfg, trainer_cfg=trainer_cfg, plan=plan)

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    notes = args.notes or (
        f"native LBD materialize n_train={len(train_ids)} n_val={len(val_ids)} "
        f"ep={args.epochs} bs={args.batch_size}"
    )

    fit_out = run_native_training_loop(
        module,
        datamodule,
        trainer_cfg,
        plan,
        fold=fold,
        exp_id=exp_id,
        train_mode=args.train_mode,
        max_epochs=int(args.epochs),
        wandb=args.wandb,
        run_name=run_name,
        project_title=args.project,
        tags=tags,
        notes=notes,
        val_enabled=val_enabled and int(val_bpe) > 0,
        permanent_checkpoint_every_n_epochs=int(args.permanent_checkpoint_every_n_epochs),
        check_val_every_n_epoch=int(args.val_every_n_epochs),
        wandb_grid_epoch_freq=int(args.wandb_grid_epoch_freq),
        det3d_configs=conf,
    )
    print("done", fit_out["train_dir"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Native nnDet LBD — materialize instance seg + RetinaUNetV001"
    )
    parser.add_argument("--project", default="lidca")
    parser.add_argument("--plan", type=int, default=4)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--lbd-folder", default=None)
    parser.add_argument("--det-data", default=None, help="nnDet det_data root (materialized task)")
    parser.add_argument("--det-models", default=None, help="nnDet checkpoint root")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--train-mode", choices=("overwrite", "resume"), default="overwrite")
    parser.add_argument("--rematerialize", action="store_true", help="Rebuild imagesTr even on resume")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--exp-id", default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--precision", default="16")
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--case-ids", default=None)
    parser.add_argument("--val-case-ids", default=None)
    parser.add_argument("--train-equals-val", action="store_true")
    parser.add_argument("--batches-per-epoch", type=int, default=None)
    parser.add_argument("--val-batches-per-epoch", type=int, default=None)
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument("--wandb", type=str2bool, default=True)
    parser.add_argument("--val-every-n-epochs", type=int, default=20)
    parser.add_argument("--wandb-grid-epoch-freq", type=int, default=20)
    parser.add_argument("--permanent-checkpoint-every-n-epochs", type=int, default=100)
    parser.add_argument("--tags", default="native_lbd,nndet,materialize,Task012_LIDC_lbd")
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()
    main(args)
