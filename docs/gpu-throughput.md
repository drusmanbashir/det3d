# GPU throughput: LIDC plan 3 RetinaUNet

Measured 2026-11-19 on the RTX 5090 (Sinclair), LIDC plan 3: patch 256×192×112, batch 4, bf16-mixed, 134 training steps per epoch.

## Result

| State | s/step | Training epoch |
|---|---|---|
| Original run (2 train workers, val batches retained) | 1.75–1.80 | ~4.0 min |
| Fixed (8 workers, val batches released) | 0.56–0.60 | ~1.3 min |

Validation (every 2 epochs) takes about 45 s once fixed.

## Causes and fixes

1. **Data starvation.** `DataManagerDet._num_workers` used `min(4, max(2, bs // 8))`, which gives 2 workers at batch 4. Each step waited 0.63 s for data. Training workers now come from `training_num_workers` in each host's `$FRAN_CONF/config.yaml` (8 on Sinclair, 12 on andrena). 8 workers and 12 workers measured the same (0.585 s/step), with 0.01 s of data wait.
2. **cuDNN fallback after validation.** `RetinaUNetManager.validation_step` stashed every validation batch, GPU tensors included, in `_nndet_wandb_grid_val_batches`. Nothing cleared the list until the next validation started. With that memory held, cuDNN could not get workspace for the training convolutions. It switched to direct kernels and kept them for the rest of the run. `wgrad2d_grouped_direct_kernel` alone took about 1,040 ms per step. Training ran 3× slower from the first validation onward, and allocator retries rose from 2 to 57.
   - The stash is gone. Steps put predictions on `batch["pred"]` (fran's pattern), and the grid copies one 2D slice per drawn case, on grid epochs only.

## Validation cost (3090, 121 cases, 31 batches of 4)

| Part | Before | After |
|---|---|---|
| Model: nnDet loss + predict (GPU) | 17.9 s | 18.7 s |
| Seg dice: nnDet numpy evaluator → `SegmentationEvaluatorTorch` (GPU counts, nnDet's finish) | 15.9 s | 0.09 s |
| GPU→CPU copies for evaluators | 7.8 s | ≈0 |
| Grid stash on every validation | 4.0 s | 0 |
| Wall, including data wait | 53 s | 24 s |

GPU and numpy seg dice gave identical tp/fp/fn on all 31 batches. Training rows are back on the grid (`images/grid_train`), from the same forward with `evaluation=True` on the batches fran flags.

## Settings tested (5090, batch 4, sustained steps 10–50)

| Setting | s/step | Peak alloc | Verdict |
|---|---|---|---|
| Baseline, no W&B logger | 0.585 | 20.3 GB | Reference |
| W&B logger on | 0.585 | 20.3 GB | No cost |
| Progress bar off | 0.574 | 20.3 GB | 2% gain, not worth changing |
| cuDNN benchmark (`nndet.benchmark=1`) | 0.568 | 25.4 GB | 3% gain for 5 GB more memory; keep off (memory headroom is what cuDNN needs) |
| Channels-last 3D | 0.663 | 20.0 GB | Slower; reject |
| GPU batch transforms (`--batch-tfms true`) | 0.937 | 20.3 GB | Slower (0.39 s data wait); reject |
| `torch.compile` on encoder and decoder | 1.774 | 27.7 GB | Much slower and more memory; reject |
| 12 workers | 0.586 | 20.3 GB | Same as 8 |

## Remaining GPU cost (per step, before validation)

Total kernel time is about 0.49 s per step at 0.58 s wall time.

- **Convolution forward and backward:** about 290 ms. This is the useful work.
- **Layout conversions:** cuDNN's NCDHW↔NDHWC conversions take about 85 ms. Channels-last would remove them, but it measured slower on this cuDNN/Blackwell stack.
- **Copies:** `aten::copy_` (bf16 autocast casts) is about 84 ms, and device-to-device memcpy about 34 ms.
- **Normalisation:** BatchNorm and GroupNorm take about 60 ms.
- **Host-to-device transfers:** about 30 ms. Box and target tensors are not pinned.

No single fix remains that is both cheap and large. The next levers are a larger batch or patch from BoxC002 (the plan fit leaves 2 GB of buffer) and pinned target transfers.

## Per-GPU settings

The 5090 is about twice as fast as the 3090 per sample (batch 2: 0.32 s/step on the 5090, 0.63 s/step on the 3090), but both use the same settings. 8 workers feed both: the 5090 consumes about 6.8 samples/s and the 3090 about 3.3. The worker count is set per host in `$FRAN_CONF` (`training_num_workers`), because host RAM, not the GPU, limits it: 46 workers at batch 4 used about 65% of 121 GB and systemd-oomd killed the run. Patch and batch size already follow the planning card through BoxC002.

## Harness

`/s/agent_rw/tmp/det3d_gpu_profile/` (scratch):

- `bench_settings.py`: sustained s/step with settings passed as `key=value`. `mon=1 epochs=N` prints step rate, memory, allocator retries and clocks every 25 steps.
- `prof_stacks.py`: kernel time grouped by Python call stack for windows before and after validation.
