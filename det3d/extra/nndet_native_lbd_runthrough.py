"""Full native nnDetection circuit on det3d LBD — run one ``# %%`` cell at a time.

Helpers / APIs live in ``nndet_native_lbd.py`` (do not duplicate here).
Granular native↔det3d parity: ``nndet_parity_cp0_4.py``.

Prereq: ``conda activate dl``

Flow:
  0 verify LBD format roundtrip (optional)
  1 materialize LBD → nnDet imagesTr
  2 Hydra compose + load plan
  3 Datamodule + dataloaders
  4 inspect one train batch
  5 RetinaUNetV001
  6 pre_trafo on one batch
  7 manual training_step + optimizer step
  8 validation_step on one val batch
  9 full Lightning fit loop
 10 read det loss from checkpoints / metrics (optional)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch

from det3d.extra.nndet_native_lbd import (
    DEFAULT_DET_MODELS,
    DEFAULT_LBD_FOLDER,
    FOLD,
    PLAN_ID,
    SCRATCH_BATCH_SIZE,
    SCRATCH_DET_DATA,
    SCRATCH_EXP_ID,
    SCRATCH_MAX_EPOCHS,
    SCRATCH_N_CASES,
    SCRATCH_TRAIN_MODE,
    TASK,
    append_metrics_log,
    clear_cuda_scratch,
    inspect_nndet_batch,
    load_nndet_train_cfgs,
    materialize_lbd_nndet_task,
    nndet_batch_to_device,
    run_native_training_loop,
    scratch_compose_cfg,
    select_scratch_case_ids,
    setup_nndet_env,
    verify_lbd_format_roundtrip,
)
from utilz.helpers import pp
from utilz.imageviewers import ImageBBoxViewer, ImageMaskViewer

# --- edit for your run ---
LBD_FOLDER = DEFAULT_LBD_FOLDER
DET_DATA = SCRATCH_DET_DATA
DET_MODELS = DEFAULT_DET_MODELS
N_CASES = SCRATCH_N_CASES
CASE_IDS: Optional[List[str]] = None  # e.g. ["lidc_0067"] or None → first N FG cases
MAX_EPOCHS = SCRATCH_MAX_EPOCHS
BATCH_SIZE = SCRATCH_BATCH_SIZE
EXP_ID = SCRATCH_EXP_ID
WANDB = True
RUN_NAME: Optional[str] = None  # default EXP_ID; set e.g. LIDCA-NNDET-LBD
WANDB_PROJECT = "lidca"

print("LBD_FOLDER", LBD_FOLDER)
print("DET_DATA", DET_DATA)
print("N_CASES", N_CASES, "MAX_EPOCHS", MAX_EPOCHS)

# %%
#SECTION:--- 0 — env + optional format verify ---
    setup_nndet_env(det_data=DET_DATA, det_models=DET_MODELS)
    check_ids = select_scratch_case_ids(LBD_FOLDER, n_cases=min(4, N_CASES), case_ids=CASE_IDS)
    roundtrip = verify_lbd_format_roundtrip(LBD_FOLDER, check_ids)
    pp(roundtrip)

# %%
#SECTION:--- 1 — materialize LBD → nnDetection task tree ---
    mat = materialize_lbd_nndet_task(
        lbd_folder=LBD_FOLDER,
        scratch_det_data=DET_DATA,
        case_ids=CASE_IDS,
        n_cases=N_CASES,
    )
    print("task_dir", mat["task_dir"])
    print("images_tr", mat["images_tr"])
    print("cases", mat["case_ids"])
    if mat["sidecar_drift"]:
        print("sidecar drift vs lm (materialize uses lm truth):", mat["sidecar_drift"])

# %%
#SECTION:--- 2 — compose cfg + load plan ---
    from nndet.io.load import load_pickle

    cfg = scratch_compose_cfg(fold=FOLD, max_epochs=MAX_EPOCHS)
    plan_path = Path(str(cfg.host.plan_path))
    plan = load_pickle(plan_path)
    data_dir = Path(cfg.host.preprocessed_output_dir) / plan["data_identifier"] / "imagesTr"
    print("plan_path", plan_path)
    print("data_dir", data_dir)
    print("patch_size", plan["patch_size"], "plan_id", PLAN_ID)

# %%
#SECTION:--- 3 — Datamodule + train/val loaders ---
    from omegaconf import OmegaConf

    from nndet.io.datamodule.bg_module import Datamodule

    augment_cfg = OmegaConf.to_container(cfg.augment_cfg, resolve=True)
    datamodule = Datamodule(
        augment_cfg=augment_cfg,
        plan=plan,
        data_dir=data_dir,
        fold=FOLD,
    )
    datamodule.setup()
    train_gen = datamodule.train_dataloader()
    val_gen = datamodule.val_dataloader()
    print("train cases", len(datamodule.dataset_tr))
    print("val cases", len(datamodule.dataset_val))
    print("train case ids", list(datamodule.dataset_tr.keys())[:8], "...")
    train_iter = iter(train_gen)

# %%
#SECTION:--- 4 — inspect one native train batch ---
    train_batch = next(train_iter)
    pp(inspect_nndet_batch(train_batch))
    print("instance_mapping", train_batch["instance_mapping"])
    print("target unique", train_batch["target"].unique())
    ImageMaskViewer([train_batch["data"], train_batch["target"]], "train batch")

# %%
#SECTION:--- 5 — build RetinaUNetV001 ---
    from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001

    model_cfg, trainer_cfg = load_nndet_train_cfgs()
    trainer_cfg["num_train_batches_per_epoch"] = int(cfg.trainer_cfg.num_train_batches_per_epoch)
    clear_cuda_scratch()
    module = RetinaUNetV001(
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        plan=plan,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module = module.to(device)
    print(type(module.model).__name__, "params", sum(p.numel() for p in module.parameters()))

# %%
#SECTION:--- 6 — pre_trafo on one batch ---
    clear_cuda_scratch()
    module.train()
    train_batch_gpu = nndet_batch_to_device(train_batch, device)
    bb2 = module.pre_trafo(**train_batch_gpu)
    print("pre_trafo keys", sorted(bb2.keys()))
    bbox = bb2["boxes"][0].cpu()
    bbox_viz = torch.stack(
        [bbox[:, 0], bbox[:, 1], bbox[:, 4], bbox[:, 2], bbox[:, 3], bbox[:, 5]], dim=1
    )
    ImageBBoxViewer(bb2["data"], bbox_viz)

# %%
#SECTION:--- 7 — one training_step + manual optimizer step ---
    step_out = module.training_step(train_batch_gpu, batch_idx=0)
    pp({k: step_out[k] for k in step_out})
    opt_cfgs = module.configure_optimizers()
    optimizer = opt_cfgs[0][0]
    scheduler = opt_cfgs[1]["scheduler"]
    optimizer.zero_grad(set_to_none=True)
    step_out["loss"].backward()
    optimizer.step()
    scheduler.step()
    print("lr", optimizer.param_groups[0]["lr"])
    clear_cuda_scratch()

# %%
#SECTION:--- 8 — validation_step on one val batch ---
    val_batch = next(iter(val_gen))
    clear_cuda_scratch()
    module.eval()
    with torch.no_grad():
        val_out = module.validation_step(nndet_batch_to_device(val_batch, device), batch_idx=0)
    print("val", {k: val_out[k] for k in val_out if k != "loss"})
    print("val total", float(val_out["loss"]))
    clear_cuda_scratch()

# %%
#SECTION:--- 9 — full native Lightning training loop (W&B + checkpoints) ---
    fit_out = run_native_training_loop(
        module=module,
        datamodule=datamodule,
        trainer_cfg=trainer_cfg,
        plan=plan,
        fold=FOLD,
        exp_id=EXP_ID,
        train_mode=SCRATCH_TRAIN_MODE,
        max_epochs=MAX_EPOCHS,
        wandb=WANDB,
        run_name=RUN_NAME,
        project_title=WANDB_PROJECT,
        tags=["native_lbd", "runthrough", TASK],
        val_enabled=int(trainer_cfg.get("num_val_batches_per_epoch", 0)) > 0,
    )
    trainer = fit_out["trainer"]
    train_dir = fit_out["train_dir"]
    if fit_out.get("logger") is not None:
        print("wandb run", fit_out["run_name"], fit_out["logger"].experiment.url)
    append_metrics_log(train_dir, tag="native_lbd_runthrough", trainer=trainer)
    print("train_dir", train_dir)
    print("checkpoints", sorted(train_dir.glob("model_*.ckpt")))

# %%
#SECTION:--- 10 — optional: benchmark-style CLI (20 ep, n cases) ---
# Same pipeline as run/training/benchmark_det_pipelines.py native_lbd:
#
#   cd /home/ub/code/det3d
#   export MPLBACKEND=Agg
#   python run/training/benchmark_det_pipelines.py run \
#     --pipelines native_lbd --n-cases 16 --epochs 20 --gpu 0 \
#     --wandb --run-name NATIVE-LBD-RUNTHROUGH
#
# Without W&B:
#   export WANDB_MODE=disabled
#   python run/training/benchmark_det_pipelines.py run \
#     --pipelines native_lbd --n-cases 32 --epochs 20 \
#     --batches-per-epoch 16 --batch-size 1 --gpu 0 \
#     --run-id my_native_lbd_e20
#
# CSV: /s/agent_rw/nndet_benchmark/results/native_lbd_my_native_lbd_e20.csv
