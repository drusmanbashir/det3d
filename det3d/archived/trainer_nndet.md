# trainer_nndet — staged integration map

Scratch file: `/home/ub/code/det3d/det3d/extra/trainer_nndet.py`  
Sibling (MONAI path): `/home/ub/code/det3d/det3d/extra/trainer.py`

This doc maps each `# %%` stage to the **canonical source** it mirrors, and compares the **det3d+MONAI** training path vs **det3d data + nnDetection model/train**.

---

## Big picture

**det3d** owns preprocessing, HDF5 shards, LBD crops, collate, and GPU batch transforms. That pipeline is unchanged.

**nnDetection** owns RetinaUNet architecture (encoder/UFPN/ATSS head/GIoU/BCE/DiCE segmenter), anchor matching, loss, and SGD+poly-LR schedule. We import it at runtime from `/home/ub/code/nnDetection` — not the vendored encoder-only copy under `/home/ub/code/det3d/det3d/detection/arch/nndet/` (see `/home/ub/code/det3d/det3d/detection/NNDET_PORT.md`).

The adapter in stage 3 **skips** nnDetection's native `pre_trafo` (`FindInstances` → `Instances2Boxes`), because det3d already provides boxes and labels. We pass them straight into `BaseRetinaNet.train_step`.

---

## Stage map

<div style="overflow-x: auto;">

| Stage | What you run | Primary det3d source | Primary nnDetection source | Same in `trainer.py`? |
|-------|--------------|----------------------|----------------------------|----------------------|
| **0** setup | `ConfigMakerDet`, `Project`, fold | `/home/ub/code/det3d/det3d/configs/parser.py` (`ConfigMakerDet`) | — | Yes — identical |
| **1** dataloaders | `DataManagerDualDetBTfms`, `prepare_data`, `setup` | `/home/ub/code/det3d/det3d/managers/data/batch_tfms.py` (`DataManagerDetLBDBTfms`), `/home/ub/code/det3d/det3d/trainers/trainerdet.py` (`init_dm`, `normalize_plan_modes`) | — (nnDet uses `/home/ub/code/nnDetection/nndet/io/datamodule/bg_module.py` instead) | Partial — `trainer.py` uses `TrainerDet.setup()` which wraps same DM |
| **2** inspect batch | `next(train_dl)`, `transforms_batch` | `/home/ub/code/det3d/det3d/managers/data/collate.py` (`lbd_det_collate`), `/home/ub/code/det3d/det3d/transforms/gpu_det.py` (`BatchItemCompose` / GpuTail) | — | Yes — same as `trainer.py` TS block |
| **3** adapter | `det3d_batch_to_nndnet` | Batch keys: `image`, `bbox`, `label`, optional `mask` from collate | Target keys for `train_step`: `data`, `target_boxes`, `target_classes`, `target_seg` — see `/home/ub/code/nnDetection/nndet/core/retina.py` | No — `trainer.py` keeps MONAI keys (`image`, `bbox`, `label`) |
| **4** plan + cfg | synthesize `plan` or load `.pkl`; load `v001.yaml` | `/home/ub/code/det3d/det3d/detection/retinaunet_network.py` (`_plan_arch`), excel `plan_train` columns | `/home/ub/code/nnDetection/nndet/conf/train/v001.yaml`, planner output in `plan["architecture"]` + `plan["anchors"]` (normally from `/home/ub/code/nnDetection/nndet/planning/experiment/v001.py` + `BoxC002`) | No — `trainer.py` uses `create_detector_from_conf` / MONAI plan only |
| **5** build model | `RetinaUNetV001(...)` → `module.model` | — | `/home/ub/code/nnDetection/nndet/ptmodule/retinaunet/v001.py`, built via `from_config_plan` in `/home/ub/code/nnDetection/nndet/ptmodule/retinaunet/base.py` | No — `trainer.py` uses `RetinaNetManager` + `create_detector_from_conf` (`/home/ub/code/det3d/det3d/architectures/create_detector.py`) |
| **6** loss forward | `net.train_step(..., evaluation=False)` | — | `/home/ub/code/nnDetection/nndet/core/retina.py` (`BaseRetinaNet.train_step`); equivalent to body of `RetinaUNetModule.training_step` minus `pre_trafo` in `/home/ub/code/nnDetection/nndet/ptmodule/retinaunet/base.py` | No — `trainer.py` uses `forward_train_batched` (`/home/ub/code/det3d/det3d/detection/retinanet_train.py`) |
| **7** optim step | `module.configure_optimizers()` | `/home/ub/code/det3d/det3d/managers/det_schedule.py` (MONAI path: Adam/ReduceLROnPlateau) | `RetinaUNetModule.configure_optimizers` in `/home/ub/code/nnDetection/nndet/ptmodule/retinaunet/base.py` — SGD + `LinearWarmupPolyLR` (`/home/ub/code/nnDetection/nndet/training/learning_rate.py`) | No |
| **8** val forward | `train_step(..., evaluation=True)` | `/home/ub/code/det3d/det3d/evaluation/coco.py` (used by `RetinaUNetManager`) | `RetinaUNetModule.validation_step` + evaluators in `/home/ub/code/nnDetection/nndet/ptmodule/retinaunet/base.py` | Partial — `trainer.py` uses MONAI inferer + COCO metrics |
| **9** epoch loop | manual loop (commented) | `/home/ub/code/det3d/det3d/trainers/trainerdet.py` (`fit`) | `/home/ub/code/nnDetection/scripts/train.py` (`trainer.fit(module, datamodule)`) | Yes conceptually — `trainer.py` calls `Tm.fit()` |

</div>

---

## Batch format comparison

| Field | det3d (after collate + GpuTail) | nnDetection native dataloader | After stage-3 adapter |
|-------|--------------------------------|------------------------------|------------------------|
| Image | `batch["image"]` `[B,C,D,H,W]` fp16 from GpuTail | `batch["data"]` numpy→tensor | `data` from batch in-place |
| Boxes | `batch["bbox"]` list of `[N,6]` xyzxyz | from `Instances2Boxes` on instance seg | `target_boxes` |
| Labels | `batch["label"]` list of `[N]` 0-based fg | `batch["classes"]` | `target_classes` |
| Seg | optional `batch["mask"]` | `batch["target"]` instance map | `target_seg` (zeros if no mask) |

nnDetection's full pipeline would run `pre_trafo` in `RetinaUNetModule.training_step` (`FindInstances`, `Instances2Boxes`, `Instances2Segmentation` in `/home/ub/code/nnDetection/nndet/io/transforms/instances.py`). We bypass that because det3d boxes are already in world/patch coords.

---

## Model / loss comparison

| Piece | det3d path (`trainer.py`) | nnDetection path (`trainer_nndet.py`) |
|-------|---------------------------|---------------------------------------|
| Architecture | MONAI `RetinaNet` + ResNet-FPN, or vendored RetinaUNet body + MONAI detector shell | Full `RetinaUNetV001`: nnDet encoder, UFPN, ATSS matcher, BCE+GIoU head, DiCE segmenter |
| Detector wrapper | `RetinaNetDetector2` (`/home/ub/code/det3d/det3d/detection/retinanet_detector2.py`) | `BaseRetinaNet` (`/home/ub/code/nnDetection/nndet/core/retina.py`) — no MONAI wrapper |
| Train forward | `forward_train_batched` → MONAI anchor + cls/box loss | `BaseRetinaNet.train_step` → cls + reg + seg losses |
| Loss keys | `classification`, `box_regression` | `cls`, `reg`, `seg_ce`, `seg_dice` |
| Optimizer | Adam + ReduceLROnPlateau (via `configure_detection_optimizers`) | SGD + warmup poly LR per iteration |
| Manager class | `RetinaNetManager` / `RetinaUNetManager` (`/home/ub/code/det3d/det3d/managers/`) | `RetinaUNetV001` Lightning module (we only use `.model` + `configure_optimizers`) |

---

## Config / plan comparison

| Input | det3d excel `plan_train` | nnDetection `plan.pkl` |
|-------|--------------------------|------------------------|
| Patch size | `patch_size` | `plan["patch_size"]` |
| Encoder | `encoder_*`, `decoder_levels` | `plan["architecture"]` (from `D3V001` planner) |
| Anchors | `base_anchor_shapes` (MONAI) | `plan["anchors"]` (optimized by `BoxC002._plan_anchors`) |
| Matcher/sampler | excel columns → MONAI detector | `model_cfg.matcher_kwargs`, `head_sampler_kwargs` in `v001.yaml` (overwritten from excel in stage 4) |

Stage 4 **synthesizes** a minimal `plan` from excel when `PLAN_PATH` is unset. For production parity with a native nnDetection run, use a real `plan.pkl` from nnDetection preprocessing/planning.

---

## nnDetection data fingerprint & splits (LIDC / LUNA16)

Paper: [MICCAI 2021 / arXiv 2106.00817](https://arxiv.org/abs/2106.00817) — “Data Fingerprint” = dataset stats extracted at planning time (nnU-Net-style), then fed to heuristic rules for patch size, architecture, anchors, etc. Design overview: `/home/ub/code/nnDetection/docs/source/nnDetectionFunctionalDetails.svg`.

**Bottom line:** the fingerprint concept is published; **no frozen fingerprint or split bundle** for LIDC/LUNA16 is shipped in the nnDetection repo. You compute artifacts locally via `nndet_prep` + planning. Pretrained models: README § Pretrained models → **Coming Soon**.

### What the fingerprint is in code

After preprocessing (`nndet_prep`), nnDetection writes:

| File | Role |
|------|------|
| `{task}/preprocessed/properties/dataset_properties.pkl` | Fingerprint — spacing/shape stats, intensity stats, per-case instance props, class counts, IoU stats |
| `{task}/preprocessed/properties/props_per_case.pkl` | Per-case instance analysis cache |
| `{task}/preprocessed/properties/intensity_properties.pkl` | Intensity normalization stats |
| `{task}/preprocessed/{plan}.pkl` | Full plan (architecture + anchors + …), embeds `dataset_properties` |

Built by `DatasetAnalyzer` (`/home/ub/code/nnDetection/nndet/planning/analyzer.py`) from cropped `imagesTr` + labels. Key fields used downstream: `dim`, `class_dct`, `instance_props_per_patient`, `num_instances`, `all_ious`, `class_ious`, `intensity_properties` — see `/home/ub/code/nnDetection/nndet/planning/properties/` and anchor planner `BoxC002` (`/home/ub/code/nnDetection/nndet/planning/architecture/boxes/c002.py`).

Not the same as nnU-Net v2’s published fingerprint JSON (hashes/manifest). nnDetection has **no** equivalent static file for Task012/Task016.

### LUNA16 — Task016_Luna

| Item | Paper benchmark | In repo? | How to get it |
|------|-----------------|----------|---------------|
| Scans | 888 (test pool) | — | Official LUNA16 download |
| Split type | **Official** 10-fold CV | Algorithm yes, frozen file **no** | `projects/Task016_Luna/scripts/prepare.py` |
| `splits.json` | — | Generated at prep | Maps case id → fold id from subset folders |
| `splits_final.pkl` / `.json` | — | Generated at prep | 10 dicts `{train, val, test}` per fold |
| `dataset_properties.pkl` | — | **Local only** | After full `nndet_prep` on your Task016 tree |

Prep flow: README at `/home/ub/code/nnDetection/projects/Task016_Luna/README.md`. Run all 10 folds with `--sweep`; consolidate with `--no_model -c copy`. CPM eval via `prepare_eval_cpm.py`.

**vs det3d:** det3d LUNA excel/HDF5 shards and fold JSON are a separate pipeline — not byte-compatible with nnDetection’s `splits_final.pkl` unless you regenerate from the same official subset layout.

### LIDC — Task012_LIDC

| Item | Paper benchmark | In repo? | How to get it |
|------|-----------------|----------|---------------|
| Scans | 1035 (train pool) | — | LIDC-IDRI via TCIA |
| Split type | **Custom** (not LUNA official) | **Incomplete** | MIC preprocessing + missing split file |
| Raw conversion | — | External repo | [MIC-DKFZ/LIDC-IDRI-processing](https://github.com/MIC-DKFZ/LIDC-IDRI-processing) → `data_nrrd` + `characteristics.csv` |
| `prepare_mic.py` | — | Yes | `/home/ub/code/nnDetection/projects/Task012_LIDC/scripts/prepare_mic.py` |
| `splits_final.pkl` | — | **Not in repo** | README step 4 says copy from `projects/Task012_LIDC/` — that file is absent; script ends with `# TODO download custom split file` |
| `dataset_properties.pkl` | — | **Local only** | After `nndet_prep` on your Task012 tree |

Paper Table 1 (supplementary): LIDC split = **Custom**, cites MIC LIDC processing line ([9] in paper). Unlike Kits/Decathlon, **no Zenodo** drop for LIDC labels/splits in nnDetection.

**vs det3d:** det3d LIDC plan 1 (`dataset_fold0.json`, LBD shards, excel `plan_train`) is **not** the nnDetection custom split or fingerprint. Stage 4’s synthesized `plan` from excel approximates architecture/anchors only — not a reproduced MIC fingerprint.

### Reproducing nnDetection parity from det3d scratch

1. **LUNA16** — feasible split-wise: official data + `prepare.py` → your `dataset_properties.pkl` + `D3V001_3d.pkl` (or load existing if you already ran native prep).
2. **LIDC** — blocked on custom split unless you obtain MIC’s split file or reconstruct from their processing repo; fingerprint follows once prep completes.
3. **This scratch file** — intentionally keeps det3d `DataManagerDualDetBTfms`; adapter + optional `PLAN_PATH` let you align **model/plan** without replacing det3d data layout.

---

## Reference entry points

**det3d**
- Scratch MONAI: `/home/ub/code/det3d/det3d/extra/trainer.py`
- Hybrid DM pattern: `/home/ub/code/det3d/det3d/detection/luna16_training_dm_hybrid.py` (`setup_det_dataloaders`)
- Trainer wiring: `/home/ub/code/det3d/det3d/trainers/trainerdet.py`
- Collate: `/home/ub/code/det3d/det3d/managers/data/collate.py`

**nnDetection**
- Train CLI: `/home/ub/code/nnDetection/scripts/train.py`
- Train recipe: `/home/ub/code/nnDetection/nndet/conf/train/v001.yaml`
- Lightning module: `/home/ub/code/nnDetection/nndet/ptmodule/retinaunet/base.py`, `v001.py`
- Core forward+loss: `/home/ub/code/nnDetection/nndet/core/retina.py`
- Native data: `/home/ub/code/nnDetection/nndet/io/datamodule/bg_loader.py`

---

## VRAM: nnDetection vs MONAI RetinaNet (same det3d data)

Measured on this machine (train step + backward, AMP, LIDC plan 1):

| Forward patch | batch | peak VRAM | result |
|---------------|-------|-----------|--------|
| 192×192×80 (full) | 1 | ~29 GB | OK |
| 192×192×80 | 2+ | — | OOM |
| 128×128×64 (forward crop) | 1 | ~10 GB | OK |
| 128×128×64 | 2 | ~20 GB | OK |
| 96×96×48 | 4 | ~17 GB | OK |

MONAI RetinaNet at batch 8 on full patch is **expected to use far less VRAM**. That gap is real, not a bug in the scratch adapter.

**Why nnDetection is heavier**

1. **Anchor count** — nnDet places anchors on every cell of the finest FPN level. For 192³ input, level 0 alone is ~35M anchors (see log: `Generated 35389440 anchors on level 0`). MONAI RetinaNet uses a small fixed set of anchor shapes per FPN level (`AnchorGeneratorWithAnchorShape`), not a dense per-voxel grid.
2. **Matcher + HNM** — ATSS matching and hard-negative mining allocate tensors proportional to anchor count (sampler `batch_size_per_image=64` over huge pools).
3. **Segmentation head** — `RetinaUNetV001` includes DiCE segmenter on decoder features; MONAI RetinaNet path has cls/box only.
4. **Architecture** — full nnDet encoder/UFPN/heads vs MONAI ResNet-FPN + lighter detector shell.

**Patch size note:** lowering `SCRATCH_PATCH_SIZE` in excel/DM requires matching HDF5 shard folders (`src_212_212_88` etc.). Use `NNDET_FORWARD_PATCH_SIZE` instead to crop in the adapter without re-preprocessing.


Defined at top of `/home/ub/code/det3d/det3d/extra/trainer_nndet.py`:

- `torch._six` — removed in PyTorch 2.x; nnDetection still imports it
- `pytorch_lightning.core.memory.ModelSummary` — moved in Lightning 2.x

**Mixed precision (fran parity):** `TrainerDet` uses `bf16-mixed` for all archs. `det3d_batch_to_nndet` uses batch tensors in-place (no unwrap, no float32 cast, no `.to(device)`); Lightning transfers val batches to GPU before `validation_step`. Scratch infer: `fabric_infer.to_device(batch)` once at trainer boundary.
