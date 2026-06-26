# RetinaUNet v3 — agent handoff

Joint 3D detection + segmentation on vendored nnDetection encoder/UFPN, trained through **MONAI** det (`RetinaNetDetector2`) and **FRAN-style** seg loss. No runtime `import nndet`.

**Config key:** `model_params.arch = retinaunet_v3`  
**Do not confuse with** `arch=retinaunet` → `RetinaUNetManager` + nnDetection `RetinaUNetV001` (separate path; see comparison below).

---

## Architecture (one forward)

```
images [B,C,D,H,W]
  → RetinaUNetFeatureExtractor (Encoder + UFPNModular)
  → head_maps → cls_head + reg_head  (MONAI RetinaNet det)
  → all_decoder_maps → RetinaUNetSegmenterHead (finest map → 1×1 seg logits)
```

| Component | Module |
|-----------|--------|
| Network | `RetinaUNetV3` — `det3d/detection/retinaunet.py` |
| Feature extractor | `det3d/detection/retinaunet_network.py` |
| Seg head (forward only) | `det3d/detection/arch/nndet/heads/segmenter.py` |
| Detector shell | `RetinaNetDetector2` — `det3d/detection/retinanet_detector2.py` |
| Joint train forward | `forward_train_joint` — `det3d/detection/retinanet_train.py` |
| Seg loss | `RetinaUNetSegLoss` — `det3d/evaluation/losses.py` |
| Lightning manager | `RetinaUNetManagerV3` — `det3d/managers/retinaunet_v3.py` |
| Factory | `resolve_detector_manager` / `build_detector_manager` — `det3d/managers/detector_factory.py` |
| Build from config | `create_retinaunet_v3_from_conf` — `det3d/architectures/create_detector.py` |

---

## vs `arch=retinaunet` (nnDetection)

| | **retinaunet_v3** | **retinaunet** (nnDet) |
|--|-------------------|------------------------|
| Manager | `RetinaUNetManagerV3` | `RetinaUNetManager` |
| Train API | `forward_train_joint` + MONAI detector | `model.train_step(...)` |
| Boxes | xyzxyz (native) | xyxyzz via `xyzxyz_exclusive_batch_to_nndet` |
| Val det | `compute_coco_metrics` | nnDet `BoxEvaluator` |
| Val seg | `compute_seg_dice` | nnDet `SegmentationEvaluator` |
| nnDet install | Not required | Required (`/home/ub/code/nnDetection`) |

---

## Environment

- Conda env: **`dl`**
- Repo: `/home/ub/code/det3d`
- Config excel: `~/code/fran/configurations/experiment_configs_det.xlsx`  
  Regenerate (adds `retinaunet_v3`, `lambda_dice`, `lambda_ce`):  
  `python run/configs/build_experiment_configs_det.py`
- Pipeline smoke protocol: `.cursor/rules/dl-pipeline-validation.mdc`

---

## Required config

### `model_params`

```python
configs["model_params"]["arch"] = "retinaunet_v3"
# retinaunet encoder keys (same as arch=retinaunet body):
# encoder_start_channels, encoder_conv_kernels, encoder_strides,
# decoder_levels, encoder_max_channels
configs["model_params"]["val_patch_size"] = [512, 512, 208]  # SW infer ROI
```

### `loss_params` (v3-specific)

```python
configs["loss_params"]["lambda_dice"] = 0.5
configs["loss_params"]["lambda_ce"] = 0.5
```

### `plan_train` (must exist before build)

Usually set by `build_from_plan` + `infer_det_labels_from_data_folder`. For synthetic smoke, set manually:

```python
configs["plan_train"]["fg_labels"] = [1]          # raw label values → 0-indexed in manager
configs["plan_train"]["w_cls"] = 1.0
configs["plan_train"]["w_reg"] = 1.0
configs["plan_train"]["detections_per_img"] = 25
configs["plan_train"]["n_input_channels"] = 1
configs["plan_train"]["spatial_dims"] = 3
```

Det loss/sampler for v3 follows `retinaunet` overrides in `det3d/detection/loss_config.py` (GIoU reg, batch_size_per_image=32, pos_fraction=0.33).

---

## Batch contract (DataManager)

Training/val batches must include:

| Key | Role |
|-----|------|
| `image` | `[B,C,D,H,W]` float tensor |
| `bbox` | xyzxyz boxes per sample (list or tensor per collate) |
| `label` | class labels per box (raw values; manager maps via `fg_labels`) |
| `lm` | instance label map for seg loss / val dice |

Manager maps labels: `label_to_idx = {v: i for i, v in enumerate(plan["fg_labels"])}`.  
MONAI targets use **0-indexed** class indices in `targets[i]["label"]`.

---

## Init and run

### 1. Synthetic smoke (no data)

```bash
conda run -n dl python -m det3d.extra.retinaunet_v3
```

Scratch file: `det3d/extra/retinaunet_v3.py` (`# %%` blocks for step-by-step).

### 2. Programmatic (manager + trainer)

```python
from det3d.configs.parser import ConfigMakerDet
from det3d.managers.detector_factory import build_detector_manager
from det3d.preprocessing.run_build import build_from_plan
from det3d.trainers.trainerdet import TrainerDet
from fran.managers import Project

project_title = "lidca"
plan_id = 1
P = Project(project_title)
C = ConfigMakerDet(P)
C.setup(plan_id)
conf = C.configs
conf["dataset_params"]["fold"] = 0
conf["model_params"]["arch"] = "retinaunet_v3"
conf["loss_params"]["lambda_dice"] = 0.5
conf["loss_params"]["lambda_ce"] = 0.5

_, conf = build_from_plan(project_title, plan_id, configs=conf)

T = TrainerDet(project_title=project_title, configs=conf, run_name="v3_smoke")
T.setup(
    devices=[0],
    batch_size=2,
    epochs=50,
    val_every_n_epochs=999,      # train-only smoke
    early_stopping=False,
    wandb=False,
    debug=True,
    train_indices=[0, 1, 2, 3],  # tiny subset — pass if supported in setup()
)
T.fit()
```

`TrainerDet` resolves manager via `resolve_detector_manager` → `RetinaUNetManagerV3` when `arch=retinaunet_v3`.

### 3. CLI

```bash
conda run -n dl python run/training/train.py \
  --project lidca \
  --plan 1 \
  --fold 0 \
  --arch retinaunet_v3 \
  --batch-size 2 \
  --epochs 50 \
  --val-every-n-epochs 5 \
  --devices 0
```

### 4. Detector only (no Lightning)

```python
from det3d.architectures.create_detector import create_detector_from_conf

detector, val_patch_size = create_detector_from_conf(conf)
# detector.network is RetinaUNetV3
# detector is RetinaNetDetector2
```

---

## Training step flow

1. `RetinaUNetManagerV3.training_step(batch)`
2. `forward_train_joint(detector, image, targets, seg_loss_fnc, lm, plan)`
3. Single `detector.network(images)` → pop `seg_logits` → MONAI det loss on cls/box heads
4. `RetinaUNetSegLoss(seg_logits, lm)` → Dice + CE
5. `combine_det_seg_loss_dict` → total = `w_cls*cls + w_reg*box + seg`

Logged keys (prefix `train0_`): `loss`, `classification`, `box_regression`, `loss_ce`, `loss_dice`.

---

## Validation

| Metric | Source |
|--------|--------|
| `val0_metric` | MONAI COCO (`compute_coco_metrics`) — detection |
| `val0_seg_dice` | `compute_seg_dice` — foreground dice on val batches |
| WandB grid | `WandbRetinaUNetImageGridCallback`; `adapt_nndet_boxes=False` (xyzxyz native) |

Det inference: MONAI sliding-window via `RetinaNetDetector2`.  
Seg on val: **per-batch full forward** on `batch["image"]` (no SW seg stitch yet).

---

## Logged metrics reference

```
train0_loss, train0_classification, train0_box_regression, train0_loss_ce, train0_loss_dice
val0_metric          # det (prog bar)
val0_seg_dice        # seg (prog bar)
val0_*               # COCO keys from compute_coco_metrics
```

---

## Smoke pass criteria

See `dl-pipeline-validation.mdc`:

1. No crash epoch 0.
2. `train0_loss` decreases on 2–8 fixed cases.
3. Optional phase-2: enable val → `val0_metric`, `val0_seg_dice`, WandB grid.

---

## Known gaps / not implemented

- Sliding-window **segmentation** inference + stitch (det SW works).
- `infer_det` / sidecar export for v3 seg preds.
- nnDet checkpoint load/save compatibility.
- `patch_stream` val special-casing (nnDet manager has this; v3 uses generic path).

---

## File map (quick grep)

```
det3d/detection/retinaunet.py              RetinaUNetV3, build_retinaunet_v3
det3d/detection/retinaunet_network.py        encoder/UFPN feature extractor
det3d/detection/retinanet_train.py          forward_train_joint
det3d/detection/retinanet_detector2.py      MONAI detector shell
det3d/evaluation/losses.py                  RetinaUNetSegLoss, combine_det_seg_loss_dict
det3d/evaluation/seg.py                     compute_seg_dice
det3d/managers/retinaunet_v3.py             RetinaUNetManagerV3
det3d/managers/retinanet.py                 parent (det val, optim, targets)
det3d/architectures/create_detector.py      create_retinaunet_v3_from_conf
det3d/managers/detector_factory.py          arch routing
det3d/trainers/trainerdet.py                WandB callback routing for v3
det3d/extra/retinaunet_v3.py                scratch / synthetic smoke
run/configs/build_experiment_configs_det.py   excel arch + seg loss params
```

---

## Related docs

- `det3d/detection/NNDET_PORT.md` — vendored nnDet arch provenance
- `.cursor/rules/dl-pipeline-validation.mdc` — train-only overfit smoke protocol
- `.cursor/rules/extra-scratch-workflow.mdc` — `# %%` scratch block convention
