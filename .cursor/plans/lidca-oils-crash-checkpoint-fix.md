# LIDCA-OILS crash + checkpoint fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Resume LIDCA-OILS run-through training without losing epoch progress on plateau, and survive GPU/tmux interruptions.

**Architecture:** Fix Lightning `ModelCheckpoint` wiring in `TrainerDet` so `last.ckpt` updates every train epoch regardless of `train0_loss` improving; keep top-k and snapshot callbacks unchanged. Resume via existing `local_train_det.sh` CLI on GPU 0 in tmux `dl`.

**Tech Stack:** PyTorch Lightning 2.x, `TrainerDetRT`, `local_train_det.sh`, tmux

---

## Investigation summary (2026-07-09)

### Crash symptom
- Log `/s/agent_rw/tmp/lidca_oils_resume_cli.log` ends abruptly at **epoch 427, step 28/328** (~12:56). No traceback, no `EXIT:` line.
- Earlier attempt `/s/agent_rw/tmp/lidca_oils_resume.log` had explicit **CUDA OOM** (GPU 0 shared with another process).

### Root cause A — checkpoint not persisting (code bug)
- Training ran epochs **420–426** (completed full 328/328 steps each).
- Canonical `last.ckpt` still **epoch 419** (`Modify: Jul 8 22:04`). Zero `.ckpt` files written anywhere under `/s/fran_storage` during today's run.
- Lightning `ModelCheckpoint.on_train_epoch_end` only calls `_save_last_checkpoint` when `_last_global_step_saved == trainer.global_step`, i.e. **after a successful top-k save**.
- With `monitor=train0_loss`, `save_top_k=2`, `every_n_epochs=5`, if epoch-end loss does not beat the top-2 best (`-0.3542` at epoch 419), **both top-k and last are skipped**.
- Logged epoch losses ~`-0.339` (worse than best) → **7 epochs of work lost on resume**.

### Root cause B — process death (environment)
- tmux `dl:oils_train` window closed ~12:56; likely SIGHUP / session kill, not Python exception.
- Mitigation: exclusive GPU 0 (`CUDA_VISIBLE_DEVICES=0`), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, no competing jobs on GPU 0.

---

## Task 1: Add unconditional last-epoch checkpoint callback

**Files:**
- Modify: `det3d/trainers/trainerdet.py` (`TrainerDet.init_cbs`)
- Modify: `det3d/trainers/trainerdet_rt.py` (only if run-through needs different cadence)

- [ ] In `TrainerDet.init_cbs`, after existing two `ModelCheckpoint` callbacks, add a **third** callback:

```python
ModelCheckpoint(
    dirpath=str(Path(checkpoint_from_model_id(self.run_name, normalize_keys=False)).parent)
    if self.run_name
    else None,
    save_top_k=0,
    save_last=True,
    every_n_epochs=1,
    monitor=None,
    enable_version_counter=False,
    auto_insert_metric_name=False,
    **self.checkpoint_kwargs,
)
```

- [ ] Resolve `dirpath` via existing fran helper `checkpoint_from_model_id(self.run_name, normalize_keys=False).parent` when `run_name` set; otherwise let Lightning resolve (new runs).
- [ ] Keep existing monitored callback (`every_n_epochs=5`, top-k) and snapshot callback (`every_n_epochs=100`) unchanged.
- [ ] Ensure `checkpoint_kwargs["save_on_train_epoch_end"] = True` remains set on `TrainerDetRT`.

**Verify:** ad-hoc smoke in `/s/agent_rw/tmp/` — resume LIDCA-OILS, run **1 full epoch**, confirm `last.ckpt` mtime and embedded `epoch` increment even if `train0_loss` does not improve.

---

## Task 2: Optional — explicit dirpath on all three callbacks

**Files:** `det3d/trainers/trainerdet.py`

- [ ] When `self.run_name` is set, pass the same canonical `dirpath` (`…/LIDCA-OILS/checkpoints`) to **all** ModelCheckpoint instances so saves never land under `lightning_logs/version_*` on fresh runs (W&B logger path drift).

---

## Task 3: Resume training via CLI

**Files:** use existing `det3d/run/training/local_train_det.sh` (no new script)

- [ ] Confirm GPU 0 free: `nvidia-smi`
- [ ] Launch in tmux:

```bash
tmux new-window -t dl -n oils_train \
  "source ~/mambaforge/etc/profile.d/conda.sh && conda activate dl && \
   export FRAN_CONF=/s/fran_storage/conf && \
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
   export CUDA_VISIBLE_DEVICES=0 && \
   /home/ub/code/det3d/det3d/run/training/local_train_det.sh \
     lidca 4 0 2 500 0 false false true 0.0 '' LIDCA-OILS \
     'resume LIDCA-OILS' '' true 1 '' false false false 100 1.0 runthrough retinaunet false '' \
   2>&1 | tee -a /s/agent_rw/tmp/lidca_oils_resume_cli.log"
```

- [ ] After epoch 420 completes, verify:
  - `last.ckpt` epoch ≥ 420
  - W&B run https://wandb.ai/drubashir/lidc/runs/LIDCA-OILS shows continued steps

---

## Task 4: Monitor to completion

- [ ] Tail log: `tail -f /s/agent_rw/tmp/lidca_oils_resume_cli.log`
- [ ] Expect ~73 epochs remaining (419 → 499) at ~3 min/epoch ≈ 3.5 h
- [ ] On clean exit, confirm `epoch0499-snapshot.ckpt` or final `last.ckpt` at epoch 499

---

## Out of scope (defer)

- CaseID recorder / whole-image val gate (see `.cursor/plans/lidca-quark-runthrough-resume.md` Phase B)
- W&B grid OOM at epoch 200 (not triggered this run; grid epochs 420/425 ran)
