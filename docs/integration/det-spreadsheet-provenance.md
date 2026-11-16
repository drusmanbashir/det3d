# Det spreadsheet: v001 integration (draft for approval)

Workbook: `/home/ub/code/fran/configurations/experiment_configs_det.xlsx`. Goal: move every `v001.yaml` setting into it. nnDet source: `/home/ub/code/nnDetection/nndet/conf/train/v001.yaml`.

## Rules

**Sheets are named by the code that reads them.**

| Sheet | A row belongs here when it is… | Config name in memory |
|---|---|---|
| `plans_det` | a value that changes per dataset (checked first) | `plan_train` |
| `model` | read while building or initialising the network | `model_params` |
| `loss` | read only when computing the training loss: matching, anchor sampling, loss terms | `loss_params` |
| `optim` | read in `configure_optimizers`: optimizer, learning rate and its schedule, SWA | `optim` |
| `trainer` | a Lightning `Trainer` or checkpoint argument | `trainer` |
| `data` | read by the data module: split, patch sampling, batch size, workers, cache | `dataset_params` |
| `augment` | a random training transform | `augment` |
| `inference` | used to turn network outputs into final boxes, in validation and prediction | `inference` |

**Names and columns:**
- Sheets have no `_params` suffix. The parser maps three of them back to the config names that fran code reads (`model`, `loss`, `data`).
- `lr`, `max_epochs` and `compiled` stay readable as `model_params[...]`, because fran reads them there (`fran/trainers/trainer.py:451`, `fran/trainers/helpers.py:20`).
- **Key names:**
  - A row that both architectures read (MONAI RetinaNet and nnDet RetinaUNet) keeps its flat det3d name; the bridge maps it to the `v001.yaml` path.
  - A row that only nnDet reads uses the `v001.yaml` path, e.g. `head_classifier_kwargs.prior_prob`. That is the same syntax as `nndet_train -o`.
- **Column use:**
  - `manual_value` is the value used now; new rows get the `v001.yaml` value.
  - The `nndet` column records the nnDet default and its source.
  - The tune columns hold the search space.
- **Zero overlap:** each setting lives in exactly one sheet across both workbooks.
- **Not moved:**
  - Infrastructure settings stay in `v001.yaml`: `gpus`, `accelerator`, `amp_backend`, `amp_level`, `dataloader`, `dataloader_kwargs`, `num_cached_per_thread`, `multiprocessing`.
  - Settings that are fixed in nnDet code and are not keys stay in code: no weight decay on norm layers (`get_params_no_wd_on_norm`), the SWA cycle learning rates (`initial_lr/10` to `/1000`).

## Rows

"Now" is the current det value; "new" means there is no row today.

### `plans_det` (unchanged columns, plus three moved here)

| Key | v001 / nnDet | Now |
|---|---|---|
| `spatial_dims` | `dim` in `dataset.json` | 3 (in `model_params`) |
| `n_input_channels` | `modalities` in `dataset.json` | 1 (in `model_params`) |
| `class_name` | `labels` in `dataset.json` | nodule (in `model_params`) |

It still owns `datasources`, `spacing`, `patch_dim0`, `patch_dim1` and the preprocessing columns.

### `model` (was `model_params`)

| Key | v001 / nnDet | Now | Note |
|---|---|---|---|
| `arch` | `module: RetinaUNetV001` | retinanet (tune: retinaunet) | |
| `encoder_start_channels` | 32 (BoxC002 constant) | 32 | `auto` or value → `plan_arch_overwrites` |
| `encoder_max_channels` | 320 (BoxC002 constant) | 320 | same |
| `encoder_conv_kernels` | planner | auto | same |
| `encoder_strides` | planner | auto | same |
| `decoder_levels` | planner | (1,2,3,4) | same |
| `base_anchor_shapes` | planner (anchor fit) | [[6,8,4],[8,6,5],[10,10,6]] | moved from `retinanet`; `auto` or value → `plan_anchors_overwrites` |
| `returned_layers` | — | [1,2] | MONAI only |
| `conv1_t_stride` | — | [2,2,1] | MONAI only |
| `decoder_kwargs.min_out_channels` | 8 | new | |
| `decoder_kwargs.upsampling_mode` | transpose | new | |
| `decoder_kwargs.num_lateral` | 1 | new | |
| `decoder_kwargs.norm_lateral` | False | new | |
| `decoder_kwargs.activation_lateral` | False | new | |
| `decoder_kwargs.num_out` | 1 | new | |
| `decoder_kwargs.norm_out` | False | new | |
| `decoder_kwargs.activation_out` | False | new | |
| `head_classifier_kwargs.num_convs` | 1 | new | |
| `head_classifier_kwargs.norm_channels_per_group` | 16 | new | |
| `head_classifier_kwargs.norm_affine` | True | new | |
| `head_classifier_kwargs.prior_prob` | 0.01 | new | |
| `head_regressor_kwargs.num_convs` | 1 | new | |
| `head_regressor_kwargs.norm_channels_per_group` | 16 | new | |
| `head_regressor_kwargs.norm_affine` | True | new | |
| `head_regressor_kwargs.learn_scale` | True | new | |

`encoder_kwargs` and `head_kwargs` are empty in `v001.yaml`; they get rows here only when a key is needed. nnDet's planner computes the kernels, strides, decoder levels and anchors, so they do not move to `plans_det`. Their rows retire when the BoxC002 step lands, and the values then come from the nnDet plan stored in the project folder.

### `loss` (was `loss_params`)

| Key | v001 / nnDet | Now | Note |
|---|---|---|---|
| `matcher_num_candidates` | `matcher_kwargs.num_candidates` 4 | 4 | |
| `matcher_center_in_gt` | `matcher_kwargs.center_in_gt` False | 0 | |
| `sampler_batch_size_per_image` | `head_sampler_kwargs.batch_size_per_image` 32 | 64 | differs |
| `balanced_sampler_pos_fraction` | `head_sampler_kwargs.positive_fraction` 0.33 | 0.3 | differs |
| `sampler_pool_size` | `head_sampler_kwargs.pool_size` 20 | 20 | |
| `sampler_min_neg` | `head_sampler_kwargs.min_neg` 1 | 16 | differs |
| `w_cls` | `head_classifier_kwargs.loss_weight` 1 | 1 | moved from `retinanet`; replaces nnDet `loss_weight` |
| `w_reg` | `head_regressor_kwargs.loss_weight` 1 | 1 | same |
| `head_classifier_kwargs.reduction` | mean | new | |
| `head_regressor_kwargs.reduction` | sum | new | |
| `segmenter_kwargs.dice_kwargs.batch_dice` | True | new | |
| `cls_loss` | — | bce (tune: focal) | MONAI only |
| `reg_loss` | — | smooth_l1 (tune: giou) | MONAI only |

### `optim` (new; takes the `nndet` and `scheduler_params` rows)

| Key | v001 / nnDet | Now | Note |
|---|---|---|---|
| `optimizer` | SGD (hardcoded in nnDet module) | SGD | |
| `lr` | `initial_lr` 0.01 | 0.01 (fran `model_params`) | also exposed as `model_params["lr"]` |
| `momentum` | `sgd_momentum` 0.9 | 0.9 | |
| `nesterov` | `sgd_nesterov` True | empty | |
| `weight_decay` | 3e-5 | 3e-5 | |
| `lr_scheduler` | warmup_poly (only option) | new; behaviour today = plateau | choice: `plateau` or `warmup_poly` |
| `warm_iterations` | 4000 | new | warmup_poly only |
| `warm_lr` | 1e-6 | new | warmup_poly only |
| `poly_gamma` | 0.9 | new | warmup_poly only |
| `scheduler_monitor` | — | train0_loss | plateau only |
| `scheduler_patience` | — | 30 | plateau only |
| `scheduler_mode` | — | min | plateau only |
| `scheduler_frequency` | — | 2 | plateau only |
| `use_swa` | on (always) | read in code, default off | new row |
| `swa_epochs` | 10 | read in code, default 10 | nnDet runs SWA for 10 extra epochs after `max_num_epochs` with a cyclic learning rate; det3d runs it inside the last 10 at a constant learning rate |

### `trainer` (new)

| Key | v001 / nnDet | Now | Note |
|---|---|---|---|
| `max_epochs` | `max_num_epochs` 50 | CLI, default 500 | also exposed as `model_params["max_epochs"]` |
| `num_train_batches_per_epoch` | 2500 | new | det3d's epoch is one full pass today |
| `num_val_batches_per_epoch` | 100 | new | |
| `val_every_n_epochs` | 1 (every epoch) | CLI, default 5 | |
| `precision` | 16 | bf16-mixed, hardcoded (`det3d/trainers/trainerdet.py:212`) | |
| `benchmark` | False | empty | moved from `nndet` |
| `deterministic` | False | empty | moved from `nndet` |
| `compiled` | — | 0 (fran `model_params`) | also exposed as `model_params["compiled"]` |
| `gradient_clip_val` | none | 1.0, hardcoded for RetinaUNet (`det3d/trainers/trainerdet.py:226`) | |
| `monitor_key` | `mAP_IoU_0.10_0.50_0.05_MaxDet_100` | new | |
| `monitor_mode` | max | new | |

### `data` (was `data_params`)

| Key | v001 / nnDet | Now | Note |
|---|---|---|---|
| `batch_size` | from plan (BoxC002 default 4) | 4 | `auto` = planner value |
| `oversample_foreground_percent` | 0.5 | fran `fgbg_ratio` 3 (75%) | replaces `fgbg_ratio`; det3d converts it to its positive:negative ratio |
| `fold` | `exp.fold` 0 (`conf/config.yaml`) | 0 | |
| `seed` | — | 0 | no det3d reader found |
| `validation_fraction` | — | 0.05 | no det3d reader found |
| `num_workers_train` | `num_threads` 12 (`det_num_threads`) | 2 | same setting under a different name |
| `num_workers_val` | — | 0 | |
| `valid_impl` | — | bbox_anchor (tune: patch_stream) | |
| `cache_rate` | — | 0 (fran) | moved from fran `dataset_params` |
| `ds_type` | — | None (fran) | moved from fran |
| `prezoom_scale` | — | 1 (fran) | moved from fran |

### `augment` (new)

| Key | v001 / nnDet | Now | Note |
|---|---|---|---|
| `augmentation` | base_more | — | choice: base_more, default, more, insane, noaug, mirror_only, base_mp, base_more_sol, base_more_sou |

fran's `transform_factors` (`det3d/managers/data/main.py:498`) and `affine3d` stay inherited until the augmentation pass merges them into this sheet. Until then, those sheets overlap with this one.

### `inference` (was `postproc_params`)

| Key | v001 / nnDet | Now | Note |
|---|---|---|---|
| `score_thresh` | 0 (module default, `nndet/ptmodule/retinaunet/base.py:431`); swept | 0.02 | |
| `nms_thresh` | 0.6; swept | 0.22 | |
| `detections_per_img` | 100 | 25 | |
| `topk_candidates_per_level` | 10000 | 1000 | |
| `remove_small_boxes` | 0.01; swept | 0 | |
| `infer_overlap` | — | code default 0.25 | new row |
| `infer_sw_batch_size` | — | code default 1 | new row |
| `predictor` | BoxPredictorSelective | — | record only; det3d has its own inference |

## Redundant after the move

- **Det sheets:** `nndet`, `scheduler_params` and `retinanet` are emptied. `model_params`, `loss_params`, `data_params` and `postproc_params` are renamed.
- **Det values:** `data_params.spacing` and `datasources` (`plans_det` owns both).
- **fran base sheets the det workbook stops inheriting:** `model_params`, `dataset_params`, `loss_params`, `resnet`. Keep `transform_factors`, `affine3d`, and the `plans_det` fallback, which `lidca` uses.
- **Code:**
  - `DET_SHEET_ONLY_KEYS` (`det3d/configs/parser.py`).
  - `build_nndet_trainer_cfg`'s copy from `model_params` (`det3d/configs/nndet_bridge.py`).
  - The YAML-only loading in `load_nndet_train_cfgs`.
  - The per-key mapping in `apply_det3d_plan_to_nndet_model_cfg` (`det3d/managers/helpers/nndet_retinaunet.py`).

## How the code reads the new keys

1. **Load** (`ConfigMakerDet`, `det3d/configs/parser.py`).
   - It reads the det workbook and inherits only `transform_factors`, `affine3d` and the fallback `plans_det` from fran.
   - It maps sheet names to config names (table above) and puts `lr`, `max_epochs` and `compiled` under `model_params`.
   - `plan_train` = global rows + the selected `plans_det` row.
2. **nnDet configs** (`det3d/configs/nndet_bridge.py`).
   - Load `v001.yaml` with OmegaConf.
   - Apply every dotted row as an override, the same as `nndet_train -o`.
   - Map the flat shared rows (matcher/sampler, `w_cls`/`w_reg`, optimizer) to their `v001.yaml` paths.
   - The result is `model_cfg` and `trainer_cfg` built from one source.
3. **Planning (BoxC002 step).** `D3V001.plan_experiment(RetinaUNetV001, model_cfg)` uses that same `model_cfg` for its trial networks. Any `model` row that is not `auto` becomes a `plan_arch_overwrites` / `plan_anchors_overwrites` entry. The plan pickle is the result.
4. **Network.** `RetinaUNetV001.from_config_plan(model_cfg, plan_arch, plan_anchors)`. MONAI RetinaNet reads the same flat rows it reads today.
5. **Optimizer** (`configure_detection_optimizers`, `det3d/managers/det_schedule.py`).
   - It builds SGD from `optim`.
   - With `lr_scheduler = plateau` it uses ReduceLROnPlateau, as today.
   - With `warmup_poly` it uses nnDet's `LinearWarmupPolyLR` and `get_params_no_wd_on_norm`, stepping every batch.
   - With `use_swa` on, SWA runs as configured.
6. **Trainer** (`TrainerDet`, `det3d/trainers/trainerdet.py`).
   - The Lightning `Trainer` gets `precision`, `benchmark`, `deterministic`, `gradient_clip_val`, `max_epochs` and `check_val_every_n_epoch`.
   - `limit_train_batches` and `limit_val_batches` come from the batches-per-epoch rows, which needs an endless sampler.
   - The checkpoint callback monitors `monitor_key` in `monitor_mode`.
7. **Data** (det3d data manager).
   - It reads `dataset_params` as today.
   - `oversample_foreground_percent` p becomes `RandCropExtendedBBoxd(pos=p, neg=1-p)` (`det3d/managers/data/main.py:841`). Today that crop gets `pos=fgbg_ratio`, `neg=1`.
8. **Inference.** The postprocessing rows go into the plan's architecture dict and the sliding-window predictor, as today.

## Sequence to BoxC002

`v001.yaml`'s planner `D3V001` calls `BoxC002`, a subclass of `BoxC001`. `BoxC001` supplies the 32/320 channel defaults.

1. **`ConfigMakerDet`** reads the `nndet_key` column into `configs["nndet_keys"]`.
2. **Bridge** builds `model_cfg` and `trainer_cfg` from `v001.yaml` plus every row whose `nndet_key` starts with `v001:`. Check that the training network is unchanged.
3. **Write `dataset_properties.pkl`**, BoxC002's one missing input (see [BoxC002.md](BoxC002.md)), into the project folder. Choice: write it from FRAN outputs, or run nnDet's own `ImageCropper` and `DatasetAnalyzer` on a temporary task folder (`det3d/run/nndet/generate_nndet_plan_silo.sh` already does that).
4. **`D3V001`** computes target spacing, axis order and median shape. Decide what happens when nnDet's target spacing differs from the `plans_det` spacing the FRAN data was preprocessed at: either feed in the FRAN spacing, or re-preprocess.
5. **`D3V001.plan_experiment` → `BoxC002`** with `RetinaUNetV001` and the step-2 `model_cfg`. This writes `D3V001_3d.pkl` to the project folder; `resolve_nndet_plan_path` already maps the project mnemonic to that file.
6. **`ConfigMakerDet` merges the stored plan into `plan_train`.** Then retire:
   - the rows and columns BoxC002 now owns (`encoder_*`, `decoder_levels`, `base_anchor_shapes`, `batch_size`, patch size)
   - `PlanAdvisorDet`, whose anchor suggestion BoxC002 replaces
7. **Afterwards:** the sheet renames, which will then touch fewer rows, and the training-side items (`warmup_poly`, SWA, precision, gradient clipping).

## Engine step: labelling rules, statistics and boxes

The "engine" here means nnDet's planner classes (`D3V001`, `BoxC002`) run on FRAN data.

- **Inputs, all read-only:**
  - the plan's label-bounded folder, built by `fran/run/preproc/analyze_resample.py --pipeline det`
  - the plan's labelling rules in `plans_det`: `remapping_source`, `remapping_lbd_rbd`, `remapping_train`, `fg_indices_exclude`, `ignore_labels_cc`, `dusting_mm`
  - the h5 file's per-case `spacing` and per-label histograms (`hist_labels`, `hu_min`)

  `global_properties.json` and the h5 file are not changed.
- **`DetDatasetProperties`** (det3d) builds nnDet's `dataset_properties` from those inputs:
  - **Boxes per case:** from the label-bounded folder's `bboxes/*.json`, which are already remapped and dusted.
  - **Classes and counts** (`class_dct`, `num_instances`): after remapping. Class names come from `metadata.yaml` labels merged by the remapping.
  - **`dim` and `modalities`:** from the dataset.
  - **`intensity_properties`** (mean, std, 0.5 and 99.5 percentiles): from the h5 histograms, summed over the labels that stay foreground after remapping, minus any excluded labels.
- **Gap:** only 166 of the 759 h5 cases have histograms, so the statistics cover only those cases until the h5 file is rebuilt.
- **Order:** each step uses only the outputs of earlier steps, so there is no loop.
  1. `D3V001._target_spacing_base` runs on raw data only: per-case spacings from the h5 file and per-case raw sizes from the image headers. The sizes are needed only for its anisotropy check.
  2. The label-bounded folder is built at that spacing. `FolderNames` takes the computed spacing.
  3. `BoxC002Det` takes that spacing as an input; it never infers spacing. It plans from the folder's boxes and its `dataset_details.csv` shapes.
- **Spacing in `plans_det`:** the column becomes `auto` by default (the engine's value). An explicit value stays possible as an override for hyperparameter search.

## Box convention

- **One convention for det3d:** MONAI `StandardMode`, i.e. `xyzxyz` with an exclusive end (MONAI `TO_REMOVE = 0`). It applies to sidecars, MONAI transforms, collate, evaluation, patch stitching and outputs.
- **Zero conversions is not possible without forking nnDet.** nnDet's box operations (anchor generator, box coder, IoU, NMS, ATSS matcher) are hardwired to `xyxyzz`, which is MONAI's `CornerCornerModeTypeC`.
- **Using nnDet's convention through MONAI is no better.** MONAI can name and convert `TypeC`, but it cannot compute in it:
  - all its box transforms (`AffineBox`, `ZoomBox`, `ResizeBox`, `FlipBox`, `ClipBoxToImage`, `SpatialCropBox`, `RotateBox90`, `BoxToMask`, `MaskToBox`) expect `StandardMode`
  - so do all its box utilities (IoU, GIoU, NMS, centers, area, clip, crop)
  - so do its `RetinaNetDetector`, box coder and ATSS matcher

  Storing boxes as `TypeC` would mean a conversion around every MONAI box operation.
- **The minimum is two conversion points**, both using MONAI's `convert_box_mode`:
  - `StandardMode` → `TypeC` entering nnDet (training targets, BoxC002 properties)
  - `TypeC` → `StandardMode` leaving nnDet (module predictions)
- **What that replaces:** the six hand-written helpers in `det3d/managers/helpers/nndet_retinaunet.py`, used in six files. Inference stitching moves to `StandardMode` using MONAI `box_iou` and `non_max_suppression`, instead of offsetting nnDet-format boxes.
- **`gt_box_mode` column:** goes once the bones sidecars, which use `cccwhd`, are rewritten in `StandardMode`.

## Hardcoded values the engine will provide

| Where | Key | Engine source |
|---|---|---|
| `plans_det` | `spacing` | `D3V001` target spacing from the h5 spacings |
| `plans_det` | `patch_dim0`, `patch_dim1`, `val_patch_size` | BoxC002 `patch_size` |
| `plans_det` | `gt_box_mode` | single box convention (after the bones rewrite) |
| `model_params` | `spatial_dims`, `n_input_channels` | `dataset_properties` `dim`, `modalities` |
| `model_params` | `class_name` | `class_dct` from the remapping |
| `model_params` | `encoder_start_channels`, `encoder_max_channels` | BoxC002 constants |
| `model_params` | `encoder_conv_kernels`, `encoder_strides`, `decoder_levels` | BoxC002 architecture |
| `retinanet` | `base_anchor_shapes` | BoxC002 anchors (MONAI RetinaNet could use them too) |
| `data_params` | `batch_size` | BoxC002 `batch_size` |
| `postproc_params` | `score_thresh`, `nms_thresh`, `remove_small_boxes` | nnDet sweep after training (later step); these stay as defaults until then |
| `global_properties.json` (read for det) | `intensity_clip_range`, `mean_fg`, `std_fg` | `intensity_properties` from the h5 histograms |
| code | `PlanAdvisorDet` defaults (anchors, patch, kernels, strides, spacing) | whole class retires |
| code | det3d's own architecture planning (`_plan_arch`, `plan_architecture_from_det3d`) | stored nnDet plan |
| code | `infer_det_labels_from_data_folder` | `class_dct` from the remapping |

## Direction

- The spreadsheet holds only **fixed keys**: values the operator sets.
- Values that nnDet classes compute (BoxC002, D3V001 and later steps) do not go in the spreadsheet. Each step stores them in fran/det3d-style files in the project folder, using nnDet datatypes where they fit (for example the plan pickle).
- `ConfigMakerDet` merges the spreadsheet with those stored results.
- Use nnDetection concepts and names wherever possible. This spreadsheet is only the starting point.

## Decisions (2026-11-15)

1. **Values reset to `v001.yaml`** in the current layout:
   - `balanced_sampler_pos_fraction` 0.3 → 0.33
   - `sampler_batch_size_per_image` 64 → 32
   - `sampler_min_neg` 16 → 1
   - `nesterov` → 1, `benchmark` → 0, `deterministic` → 0
   - new det `fgbg_ratio` = 1, the same as `oversample_foreground_percent` 0.5; it overrides fran's base value of 3

   Each changed cell's `notes` records its old value. Backup: `/s/agent_rw/tmp/det_xlsx_backup/experiment_configs_det.2026-11-15.xlsx`.
2. **Epochs stay as full passes over the data** for now. The batches-per-epoch rows wait.
3. **Anchors, kernels, strides and decoder levels stay out of `plans_det`**, because nnDet's planner computes them. Only fixed per-dataset keys (`spatial_dims`, `n_input_channels`, `class_name`) move there.
4. **New `nndet_key` column** on each det parameter sheet, naming the nnDet counterpart of each det-named row (`v001: trainer_cfg.sgd_momentum`, `plan: architecture.strides`, …). The prefix says where nnDet keeps it: `v001`, `config` (`config.yaml`), `dataset.json` or `plan`.

## Still open

- `warmup_poly` and SWA wiring. Until then `v001.yaml`'s schedule (warmup + poly) and SWA are not in effect; det3d uses plateau with SWA off.
- `precision` (bf16-mixed vs 16) and `gradient_clip_val` (1.0 vs none) are still hardcoded in `det3d/trainers/trainerdet.py`.
- The sheet renames and the bridge's OmegaConf merge for the nnDet-only rows. The new `model_cfg` rows are not in the workbook yet; they have no effect until the merge exists, and today's values already equal `v001.yaml` because the bridge loads it directly.
