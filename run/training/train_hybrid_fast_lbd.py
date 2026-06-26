#!/usr/bin/env python3
"""Hybrid fast LBD training — LBD HDF5 + disk boxes → GpuTail → nnDet RetinaUNetV001.

Same path as ``LIDCA-FAST-LBD-E500-AUG`` (``run_det3d_fast_training_loop`` in
``det3d.extra.nndet_det3d_fast_lbd_bk``). Not TrainerDet / not native materialize.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from det3d.configs.parser import ConfigMakerDet
from det3d.extra.nndet_det3d_fast_lbd_bk import (
    DEFAULT_FAST_DET_MODELS,
    DEFAULT_FAST_PLAN_ID,
    DEFAULT_FAST_PROJECT,
    resolve_lbd_fg_case_ids,
    run_det3d_fast_training_loop,
)
from fran.managers.project import Project


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def _parse_case_ids(text: str | None) -> list[str] | None:
    if text is None:
        return None
    out = [c.strip() for c in text.split(",") if c.strip()]
    return out or None


def _resolve_train_ids(project, conf, args) -> list[str]:
    explicit = _parse_case_ids(args.case_ids)
    if explicit is not None:
        return explicit
    return resolve_lbd_fg_case_ids(
        project, conf, split="train", n_cases=args.n_train
    )


def _resolve_val_ids(project, conf, train_ids: list[str], args) -> list[str]:
    if args.train_equals_val:
        return list(train_ids)
    explicit = _parse_case_ids(args.val_case_ids)
    if explicit is not None:
        return explicit
    return resolve_lbd_fg_case_ids(
        project, conf, split="valid", n_cases=args.n_val
    )


def main(args) -> None:
    #AI
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))

    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)

    P = Project(args.project)
    C = ConfigMakerDet(P)
    C.setup(int(args.plan))
    conf = C.configs
    if args.fold is not None:
        conf["dataset_params"]["fold"] = int(args.fold)

    train_ids = _resolve_train_ids(P, conf, args)
    val_enabled = not args.no_val
    val_ids = None
    if val_enabled:
        val_ids = _resolve_val_ids(P, conf, train_ids, args)

    exp_id = args.exp_id or args.run_name
    if not exp_id:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        exp_id = f"LIDCA-HYBRID-FAST-LBD-{stamp}"
    run_name = args.run_name or exp_id

    det_models = Path(args.det_models) if args.det_models else DEFAULT_FAST_DET_MODELS
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    notes = args.notes or (
        f"hybrid fast LBD n_train={len(train_ids)} "
        f"n_val={len(val_ids) if val_ids else 0} ep={args.epochs} bs={args.batch_size}"
    )

    print(
        f"hybrid fast LBD train={len(train_ids)} val={len(val_ids) if val_ids else 0} "
        f"epochs={args.epochs} batch_size={args.batch_size} "
        f"train_mode={args.train_mode} exp_id={exp_id} gpu={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}",
        flush=True,
    )
    if args.train_equals_val:
        print("train_equals_val: val uses same case list as train", flush=True)

    val_split = "train" if args.train_equals_val else "valid"

    fit_out = run_det3d_fast_training_loop(
        case_ids=train_ids,
        val_case_ids=val_ids,
        val_split=val_split,
        epochs=int(args.epochs),
        batches_per_epoch=args.batches_per_epoch,
        batch_size=int(args.batch_size),
        project_title=args.project,
        plan_id=int(args.plan),
        exp_id=exp_id,
        train_mode=args.train_mode,
        det_models=det_models,
        wandb=args.wandb,
        run_name=run_name,
        tags=tags,
        notes=notes,
        wandb_grid_epoch_freq=int(args.wandb_grid_epoch_freq),
        val_every_n_epochs=int(args.val_every_n_epochs),
        permanent_checkpoint_every_n_epochs=int(args.permanent_checkpoint_every_n_epochs),
        val_batches_per_epoch=args.val_batches_per_epoch,
        val_enabled=val_enabled,
        debug=args.debug,
        use_gpu_tail=not args.no_gpu_tail,
    )
    print("done", fit_out["train_dir"], flush=True)


# %%
#SECTION:-------------------- setup--------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid fast LBD → nnDet RetinaUNetV001 (GpuTail + disk boxes)"
    )
    parser.add_argument("--project", default=DEFAULT_FAST_PROJECT)
    parser.add_argument("--plan", type=int, default=DEFAULT_FAST_PLAN_ID)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA_VISIBLE_DEVICES index")
    parser.add_argument(
        "--train-mode",
        choices=("overwrite", "resume"),
        default="overwrite",
        help="overwrite: fresh weights; resume: model_last.ckpt in exp folder",
    )
    parser.add_argument("--run-name", default=None, help="W&B run name (default: --exp-id)")
    parser.add_argument(
        "--exp-id",
        default=None,
        help="nnDet train_dir name under det_models/det3d_fast_lbd/{exp_id}/fold0",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="Cap FG train cases (default: all FG train)",
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=None,
        help="Cap FG valid cases (default: all FG valid)",
    )
    parser.add_argument(
        "--case-ids",
        default=None,
        help="Comma-separated train case ids (overrides --n-train)",
    )
    parser.add_argument(
        "--val-case-ids",
        default=None,
        help="Comma-separated val case ids (overrides --n-val)",
    )
    parser.add_argument(
        "--train-equals-val",
        action="store_true",
        help="Use train case list for validation (overfit / parity smoke)",
    )
    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=None,
        help="Cap train steps per epoch (default: full loader)",
    )
    parser.add_argument("--val-batches-per-epoch", type=int, default=None)
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument("--wandb", type=str2bool, default=True)
    parser.add_argument("--val-every-n-epochs", type=int, default=5)
    parser.add_argument("--wandb-grid-epoch-freq", type=int, default=20)
    parser.add_argument("--permanent-checkpoint-every-n-epochs", type=int, default=100)
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument(
        "--no-gpu-tail",
        action="store_true",
        help="CPU item augs (DataManagerDetLBD) instead of GpuTail",
    )
    parser.add_argument("--det-models", default=None, help="det_models root (default: agent_rw benchmark)")
    parser.add_argument(
        "--tags",
        default="hybrid_fast_lbd,det3d_fast_lbd,gpu_tail,disk_boxes",
    )
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()
    main(args)
