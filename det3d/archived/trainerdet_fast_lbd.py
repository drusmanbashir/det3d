"""TrainerDet fast LBD scratch — approved triad: TrainerDet + DataManagerDualDetBTfms + RetinaUNetManager.

Same wiring as ``trainerdet.py``; LBD plan 4, optional case subset, GpuTail batch path.
Disk boxes + semantic seg via ``RetinaUNetManager`` → ``det3d_batch_to_nndet`` (no bypass loop).

REPL: run ``# %%`` cells top-to-bottom; ``Tm.fit()`` for full training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

DEFAULT_PROJECT = "lidca"
DEFAULT_PLAN_ID = 4
DEFAULT_N_CASES = 16


def resolve_lbd_case_ids(
    project,
    conf: dict,
    *,
    n_cases: int = DEFAULT_N_CASES,
    case_ids: Sequence[str] | None = None,
) -> list[str]:
    if case_ids is not None:
        return [str(c) for c in case_ids]

    import pandas as pd
    from fran.utils.folder_names import FolderNames

    fold = int(conf["dataset_params"]["fold"])
    ds = [x.strip() for x in conf["plan_train"]["datasources"].split(",") if x.strip()]
    train_ids, _ = project.get_train_val_case_ids(fold, ds, nnz_allowed=False)
    train_set = set(train_ids)
    lbd_folder = Path(FolderNames(project, conf["plan_train"]).lbd_folder)
    df = pd.read_csv(lbd_folder / "dataset_details.csv")
    rows = df[(df["has_fg"]) & (~df["bbox_empty"])].sort_values("case_id")
    pool = [str(c) for c in rows["case_id"] if str(c) in train_set]
    return pool[: int(n_cases)]


def run_trainerdet_fast_lbd(
    *,
    project_title: str = DEFAULT_PROJECT,
    plan_id: int = DEFAULT_PLAN_ID,
    fold: int = 0,
    epochs: int = 500,
    batch_size: int = 1,
    device_id: int = 0,
    batch_tfms: bool = True,
    wandb: bool = True,
    run_name: str = "DET3D-FAST-LBD-E500-FULL",
    tags: Sequence[str] | None = None,
    description: str = "TrainerDet fast LBD RetinaUNet full train",
    lr: float | None = None,
    debug: bool = False,
    train_indices: Sequence[str] | int | None = None,
    val_indices: Sequence[str] | int | None = None,
    val_sampling: float = 1.0,
    val_every_n_epochs: int = 20,
    wandb_grid_epoch_freq: int = 20,
    permanent_checkpoint_every_n_epochs: int = 100,
    n_cases: int | None = None,
    case_ids: Sequence[str] | None = None,
):
    from det3d.configs.parser import ConfigMakerDet
    from det3d.trainers.trainerdet import TrainerDet
    from fran.managers import Project

    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = C.configs
    conf["dataset_params"]["fold"] = int(fold)
    conf["model_params"]["arch"] = "retinaunet"

    if train_indices is None and n_cases is not None:
        train_indices = resolve_lbd_case_ids(P, conf, n_cases=n_cases, case_ids=case_ids)

    tag_list = list(tags) if tags else ["det3d_fast_lbd", "trainerdet", "disk_boxes"]
    Tm = TrainerDet(P.project_title, conf, run_name)
    Tm.setup(
        compiled=False,
        train_indices=train_indices,
        val_indices=val_indices,
        val_sampling=val_sampling,
        val_every_n_epochs=int(val_every_n_epochs),
        cbs=[],
        debug=debug,
        batch_size=int(batch_size),
        batch_tfms=batch_tfms,
        devices=[int(device_id)],
        epochs=int(epochs),
        profiler=False,
        wandb=wandb,
        wandb_grid_epoch_freq=int(wandb_grid_epoch_freq),
        tags=tag_list,
        description=description,
        lr=lr,
        permanent_checkpoint_every_n_epochs=int(permanent_checkpoint_every_n_epochs),
    )
    Tm.fit()
    return Tm


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="TrainerDet fast LBD full training")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--run-name", default="DET3D-FAST-LBD-E500-FULL")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-every-n-epochs", type=int, default=20)
    parser.add_argument("--wandb-grid-epoch-freq", type=int, default=20)
    parser.add_argument("--permanent-checkpoint-every-n-epochs", type=int, default=100)
    parser.add_argument("--n-cases", type=int, default=None, help="Limit train cases; omit for all")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--smoke-grid", action="store_true", help="2 ep, grid every val, 8 cases")
    args = parser.parse_args()

    if args.smoke_grid:
        run_trainerdet_fast_lbd(
            epochs=2,
            run_name=f"{args.run_name}-GRID-SMOKE",
            device_id=args.gpu,
            batch_size=args.batch_size,
            wandb=not args.no_wandb,
            val_every_n_epochs=1,
            wandb_grid_epoch_freq=1,
            permanent_checkpoint_every_n_epochs=2,
            n_cases=8,
            description="TrainerDet fast LBD wandb grid smoke",
            tags=["det3d_fast_lbd", "trainerdet", "grid_smoke"],
        )
        return

    run_trainerdet_fast_lbd(
        epochs=args.epochs,
        run_name=args.run_name,
        device_id=args.gpu,
        batch_size=args.batch_size,
        wandb=not args.no_wandb,
        val_every_n_epochs=args.val_every_n_epochs,
        wandb_grid_epoch_freq=args.wandb_grid_epoch_freq,
        permanent_checkpoint_every_n_epochs=args.permanent_checkpoint_every_n_epochs,
        n_cases=args.n_cases,
        description="TrainerDet fast LBD RetinaUNet full train (all indices when --n-cases omitted)",
    )


# %%
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
        raise SystemExit(0)
# SECTION:-------------------- setup --------------------------------------------------------------------------------------
    from fran.managers import Project

    from det3d.configs.parser import ConfigMakerDet

    project_title = DEFAULT_PROJECT
    plan_id = DEFAULT_PLAN_ID

    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = C.configs
    conf["dataset_params"]["fold"] = 0

# %%
# SECTION:-------------------- TRAINING --------------------------------------------------------------------------------------
    conf["model_params"]["arch"] = "retinaunet"

    bs = 1
    device_id = 0
    batch_tfms = True
    wandb = True
    run_name = "DET3D-FAST-LBD-E500-FULL"
    tags = ["det3d_fast_lbd", "trainerdet", "disk_boxes", "e500_full"]
    description = "TrainerDet fast LBD RetinaUNet (DualDetBTfms + disk boxes)"
    lr = None
    debug_ = False
    profiler = False
    compiled = False
    cbs = []
    wandb_grid_epoch_freq = 20
    val_every_n_epochs = 20
    n_cases = None
    case_ids = None
    train_indices = None
    val_indices = None
    val_sampling = 1.0
    epochs = 500
    permanent_checkpoint_every_n_epochs = 100

    from det3d.trainers.trainerdet import TrainerDet

    Tm = TrainerDet(P.project_title, conf, run_name)

# %%
    Tm.setup(
        compiled=compiled,
        train_indices=train_indices,
        val_indices=val_indices,
        val_sampling=val_sampling,
        val_every_n_epochs=val_every_n_epochs,
        cbs=cbs,
        debug=debug_,
        batch_size=bs,
        batch_tfms=batch_tfms,
        devices=[device_id],
        epochs=epochs,
        profiler=profiler,
        wandb=wandb,
        wandb_grid_epoch_freq=wandb_grid_epoch_freq,
        tags=tags,
        description=description,
        lr=lr,
        permanent_checkpoint_every_n_epochs=permanent_checkpoint_every_n_epochs,
    )

# %%
# SECTION:-------------------- fit --------------------------------------------------------------------------------------
    Tm.fit()

# %%
# SECTION:-------------------- inspect batch --------------------------------------------------------------------------------------
    N = Tm.N
    D = Tm.D
    tmt = D.train_manager
    tmv = D.valid_manager
    tmt.setup()
    tmv.setup()
    train_dl = tmt.dl
    val_dl = tmv.dl
    print(f"train: {tmt}")
    print(f"valid: {tmv}")
    print(f"train_indices: {train_indices}")

# %%
    batch = next(iter(train_dl))
    batch.keys()
    print(batch["image"].shape, batch["bbox"].shape, batch["lm"].shape)

# %%
# SECTION:-------------------- single train step --------------------------------------------------------------------------------------
    device = f"cuda:{device_id}"
    N = N.to(device)
    batch_dev = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
    losses, preds, nb = N._step_losses(batch_dev, 0, evaluation=False)
    {k: float(v) for k, v in losses.items()}

# %%
# SECTION:-------------------- single val step (det metrics) --------------------------------------------------------------------------------------
    val_batch = next(iter(val_dl))
    val_batch_dev = {k: v.to(device) if hasattr(v, "to") else v for k, v in val_batch.items()}
    vlosses, vpreds, vnb = N._step_losses(val_batch_dev, 0, evaluation=True)
    vpreds.keys()
