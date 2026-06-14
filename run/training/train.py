#!/usr/bin/env python3
import argparse

import torch
from det3d.configs.parser import ConfigMakerDet
from det3d.preprocessing.run_build import build_from_plan
from det3d.trainers.trainerdet import TrainerDet
from fran.managers import Project
from fran.utils.misc import parse_devices


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def main():
    parser = argparse.ArgumentParser(description="MONAI 3D RetinaNet detection training")
    parser.add_argument("--project", required=True, dest="project_title")
    parser.add_argument("--plan", required=True, type=int)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--wandb", type=str2bool, default=True)
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--skip-json-build", type=str2bool, default=False)
    parser.add_argument("--val-every-n-epochs", type=int, default=None)
    parser.add_argument(
        "--batch-tfms",
        type=str2bool,
        default=True,
        help="Use DataManagerDualDetBTfms (GPU spatial tail via on_after_batch_transfer)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE — running on CPU")

    if not args.skip_json_build:
        _, configs = build_from_plan(args.project_title, args.plan)
    else:
        project = Project(args.project_title)
        config_maker = ConfigMakerDet(project)
        config_maker.setup(args.plan)
        configs = config_maker.configs

    devices = parse_devices(args.devices)
    trainer = TrainerDet(
        project_title=args.project_title,
        configs=configs,
        run_name=args.run_name,
    )
    trainer.setup(
        devices=devices,
        epochs=args.epochs,
        lr=args.lr,
        wandb=args.wandb,
        debug=args.debug,
        val_every_n_epochs=args.val_every_n_epochs,
        batch_tfms=args.batch_tfms,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
