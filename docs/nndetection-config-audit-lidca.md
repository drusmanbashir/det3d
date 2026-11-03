# LIDC configuration versus nnDetection

This describes the current `lidca` plan 4 path. It distinguishes the spreadsheet values, nnDetection's static `/home/ub/code/nnDetection/nndet/conf/train/v001.yaml` defaults, and values calculated by code. It is an audit of the current implementation, not a proposed merge policy.

## The short answer

`ConfigMakerDet` in `/home/ub/code/det3d/det3d/configs/parser.py` does **not** merge the saved nnDetection lung plan into the spreadsheet configuration. It merges Fran's `/home/ub/code/fran/configurations/experiment_configs.xlsx` with `/home/ub/code/fran/configurations/experiment_configs_det.xlsx`, reading each sheet's `manual_value` (the `tune_value` and `tune` columns do not select values here). Plan 4 then overrides the shared sheet defaults. The result is `configs["plan_train"]`, `configs["model_params"]`, and `configs["dataset_params"]`.

The spreadsheet's `model_params.arch` is **`retinanet`**. The nnDetection model is reached only if the run selects `retinaunet`, for example with `--arch retinaunet`. In that path, `det3d` loads only nnDetection's `model_cfg` and `trainer_cfg` sections from `/home/ub/code/nnDetection/nndet/conf/train/v001.yaml`. Those YAML defaults are the same for lungs and other datasets. `det3d` overrides selected values and builds a new nnDetection-compatible model plan from `plan_train`; it does not read `/s/fran_storage/nndet_conf/plans/lungs/D3V001_3d.pkl` during model construction.

**“Programmatic” here has two meanings:** `det3d` does calculate and transform some values at configuration and model-build time; nnDetection's dataset analysis, architecture planner, memory-based patch selection, and anchor optimization do **not** run in this training path.

| Top-level key in `/home/ub/code/nnDetection/nndet/conf/train/v001.yaml` | Read by `/home/ub/code/det3d/det3d/configs/nndet_bridge.py`? |
|---|---|
| `model_cfg` | Yes; copied, then six matcher/sampler fields are overwritten from `plan_train`. |
| `trainer_cfg` | Yes; copied, then selected fields are overwritten. Most training decisions belong to the outer `det3d` trainer. |
| `defaults`, `module`, `predictor`, `plan`, `planner`, `augment_cfg` | No; the bridge never passes these through nnDetection's configuration composer. |

## Effective LIDC plan 4 and where values come from

These values were checked by calling `ConfigMakerDet(Project("lidca")).setup(4)` without starting a trainer. The later training setup changes some values, as indicated below.

| Effective key | Value after `setup(4)` | Owner and runtime action |
|---|---|---|
| `model_params.arch` | `retinanet` | Detection sheet `model_params`; `--arch retinaunet` overrides it before manager selection. No nnDetection code runs on the default `retinanet` path. |
| `model_params.lr` | `0.01` | Inherited from Fran's base workbook; CLI `--lr` can change the training LR. |
| `model_params.max_epochs` | absent | Training setup inserts CLI `--epochs` (default 600); nnDetection's YAML default 50 is overwritten in the internal module. |
| `dataset_params.datasources` | `lidc, lidc2` | Detection `data_params` sheet; these are `det3d` data inputs. |
| `dataset_params.fold`, `batch_size`, `valid_impl` | `0`, `4`, `bbox_anchor` | Detection `data_params` sheet; training CLI can override fold and batch size. nnDetection's data loader is not used. |
| `dataset_params.spacing` | `[0.703125, 0.703125, 1.25]` | Detection `data_params` sheet. This is a distinct value from plan 4's spacing below. |
| `plan_train.spacing` | `(0.8, 0.8, 1.5)` | `plans_det` row 4 overrides the shared plan default. Not selected by nnDetection. |
| `plan_train.patch_size`, `src_dims` | `[128, 128, 64]`, `[140, 140, 70]` | `patch_dim0=128` and `patch_dim1=64` become a patch size; `det3d` calculates `src_dims` from it. `--patch-size` can override the patch size. |
| `plan_valid.val_patch_size` | `(128, 128, 64)` | `plans_det` row 4; kept separate from training patch size. |
| `plan_train.fg_labels` | absent at `ConfigMakerDet.setup` | `det3d` later infers labels from its prepared-data manifest and adds them to the train, validation, and test plans before model creation. |

## What nnDetection adds to a `retinaunet` model

The table covers every key under nnDetection's `model_cfg` in `/home/ub/code/nnDetection/nndet/conf/train/v001.yaml`. “Static” means it is copied from the YAML unchanged for all datasets. “Sheet” means the listed spreadsheet value is applied by `det3d` before nnDetection constructs the model.

| nnDetection `model_cfg` key | Effective value/source | Effect in the `retinaunet` path |
|---|---|---|
| `encoder_kwargs` | `{}`; static | Extra encoder arguments; none supplied. |
| `decoder_kwargs.min_out_channels` | `8`; static | Decoder minimum channels. |
| `decoder_kwargs.upsampling_mode` | `transpose`; static | Decoder upsampling method. |
| `decoder_kwargs.num_lateral` | `1`; static | Lateral decoder blocks. |
| `decoder_kwargs.norm_lateral` | `false`; static | Lateral normalization switch. |
| `decoder_kwargs.activation_lateral` | `false`; static | Lateral activation switch. |
| `decoder_kwargs.num_out` | `1`; static | Decoder output blocks. |
| `decoder_kwargs.norm_out` | `false`; static | Decoder output normalization switch. |
| `decoder_kwargs.activation_out` | `false`; static | Decoder output activation switch. |
| `head_kwargs` | `{}`; static | Extra detection-head arguments; none supplied. |
| `head_classifier_kwargs.num_convs` | `1`; static | Classifier depth. |
| `head_classifier_kwargs.norm_channels_per_group` | `16`; static | Classifier group normalization. |
| `head_classifier_kwargs.norm_affine` | `true`; static | Learned normalization scale/offset. |
| `head_classifier_kwargs.reduction` | `mean`; static | Classification loss reduction. |
| `head_classifier_kwargs.loss_weight` | `1.0`; static | Classification loss weight; spreadsheet `w_cls` is **not** forwarded here. |
| `head_classifier_kwargs.prior_prob` | `0.01`; static | Classifier bias initialization prior. |
| `head_regressor_kwargs.num_convs` | `1`; static | Box regressor depth. |
| `head_regressor_kwargs.norm_channels_per_group` | `16`; static | Regressor group normalization. |
| `head_regressor_kwargs.norm_affine` | `true`; static | Learned normalization scale/offset. |
| `head_regressor_kwargs.reduction` | `sum`; static | Box loss reduction. |
| `head_regressor_kwargs.loss_weight` | `1.0`; static | Box loss weight; spreadsheet `w_reg` is **not** forwarded here. |
| `head_regressor_kwargs.learn_scale` | `true`; static | Learned box regression scale. |
| `head_sampler_kwargs.batch_size_per_image` | `64`; sheet `loss_params.sampler_batch_size_per_image` replaces nnDetection's `32` | Number of sampled anchors per image, **not** data-loader batch size. |
| `head_sampler_kwargs.positive_fraction` | `0.3`; sheet `loss_params.balanced_sampler_pos_fraction` replaces `0.33` | Target positive fraction among sampled anchors. |
| `head_sampler_kwargs.pool_size` | `20`; sheet `loss_params.sampler_pool_size` | Hard-negative candidate pool multiplier. |
| `head_sampler_kwargs.min_neg` | `16`; sheet `loss_params.sampler_min_neg` replaces `1` | Minimum sampled negative anchors. |
| `segmenter_kwargs.dice_kwargs.batch_dice` | `true`; static | Compute segmentation Dice loss across the batch. |
| `matcher_kwargs.num_candidates` | `4`; sheet `loss_params.matcher_num_candidates` | Candidate anchors per feature level. |
| `matcher_kwargs.center_in_gt` | `false`; sheet `loss_params.matcher_center_in_gt=0` | Whether an anchor center must lie inside a target box. |
| `plan_arch_overwrites` | `{}`; static | nnDetection's final architecture override hook; unused. |
| `plan_anchors_overwrites` | `{}`; static | nnDetection's final anchor override hook; unused. |

The `RetinaUNetV001` class also fixes model choices outside YAML: instance-normalized convolutional blocks, a BCE classifier, a GIoU box regressor, ATSS box matching, hard-negative mining, and a foreground/background Dice segmenter. Those choices are generic to `RetinaUNetV001`; the LIDC spreadsheet does not choose them.

## Architecture and anchors: calculated by `det3d`, not imported from the nnDetection pickle

| nnDetection model-plan key | Current LIDC plan 4 source/calculation | Dataset dependence |
|---|---|---|
| `architecture.dim`, `in_channels` | Sheet `spatial_dims=3`, `n_input_channels=1` | Manually chosen for this dataset. |
| `architecture.start_channels`, `max_channels` | Sheet `encoder_start_channels=32`, `encoder_max_channels=320` | Shared manual settings, not nnDetection planning output. |
| `architecture.conv_kernels` | `encoder_conv_kernels=auto` becomes five `[3,3,3]` kernels in `det3d` code | Fixed fallback; no LIDC measurement. |
| `architecture.strides` | `encoder_strides=auto` becomes one `[1,1,1]` followed by four `[2,2,2]` strides | Fixed fallback; no LIDC measurement. |
| `architecture.decoder_levels` | Sheet `(1,2,3,4)` | Manual choice. |
| `architecture.fpn_channels`, `head_channels` | `start_channels × 4 = 128` for each | Calculated from a manual setting. |
| `architecture.classifier_classes` | Number of `fg_labels` inferred from the `det3d` data manifest | Dataset-dependent, calculated later. |
| `architecture.seg_classes` | `1` | Fixed by the bridge. |
| `architecture.score_thresh`, `nms_thresh`, `detections_per_img`, `topk_candidates`, `remove_small_boxes` | Sheet `0.02`, `0.22`, `25`, `1000`, `0` | Manual postprocessing settings, copied into the model. |
| `anchors.sizes`, `zsizes` | Sheet `base_anchor_shapes=[[6,8,4],[8,6,5],[10,10,6]]` becomes `sizes=(8,8,10,10)`, `zsizes=(4,5,6,6)`; last shape is repeated for the fourth decoder level | Calculated from manually supplied shapes. |
| `anchors.aspect_ratios`, `stride` | `(0.5,1.0,2.0)`, `1` | Fixed by the bridge. |
| `patch_size` | Sheet-derived `[128,128,64]` | Manual LIDC plan setting. |

The saved nnDetection lung pickle instead contains `target_spacing=[2.0,0.703125,0.703125]`, `patch_size=[128,128,64]`, **six** convolution stages, decoder levels `(2,3,4,5)`, and optimized `width`/`height`/`depth` anchors. Its dataset properties cover **608 `lidc` cases** and no `lidc2` cases. Its different spacing values and order, dataset membership, and anchor format make it a reference rather than an active source for this plan. `resolve_nndet_plan_path()` exists, but the RetinaUNet build passes `plan_path=None` and explicitly rejects a non-`None` path.

## What the nnDetection `trainer_cfg` YAML contributes

The entire `trainer_cfg` dictionary is copied into the nested nnDetection module, but `RetinaUNetManager.configure_optimizers()` uses `det3d`'s own SGD plus `ReduceLROnPlateau`. The nested module's nnDetection warmup/poly scheduler and its `configure_optimizers()` are not the optimizer hook used by the outer trainer. These values should therefore not be mistaken for the effective training policy.

| nnDetection `trainer_cfg` keys | Value in nested module | Effective `det3d` training source |
|---|---|---|
| `gpus`, `accelerator`, `precision`, `amp_backend`, `amp_level` | `1`, `null`, `16`, `native`, `O1` | Outer Fran/`det3d` Lightning trainer chooses devices and precision. |
| `deterministic`, `benchmark` | `false`, `false` | Static nnDetection defaults. The bridge checks `model_params`, so it does not apply the `nndet` sheet values to this copy. The outer trainer separately reads `plan_train.deterministic` and `plan_train.benchmark`; both become `false` from the sheet's blank values. |
| `monitor_key`, `monitor_mode` | nnDetection mAP key, `max` | `det3d` scheduler reads `plan_train.scheduler_monitor` (`train0_loss` for plan 4), `scheduler_mode=min`, `scheduler_patience=30`, and `scheduler_frequency=2`. |
| `max_num_epochs` | CLI epochs; default 600 | Outer trainer also uses CLI epochs. The YAML value `50` is replaced. |
| `num_train_batches_per_epoch`, `num_val_batches_per_epoch` | `2500`, `100` | Static nnDetection values stored in the nested module; `det3d` data loaders control iteration. |
| `initial_lr` | `model_params.lr=0.01` from Fran base workbook | Outer manager uses its `lr`, which can be changed by CLI `--lr`. |
| `sgd_momentum` | `model_params.momentum=0.9` from Fran base workbook | Outer manager reads `plan_train.momentum=0.9` (overlaid by `nndet` sheet). |
| `weight_decay` | `model_params.weight_decay=0` from Fran base workbook | Outer manager reads `plan_train.weight_decay=3e-5` (overlaid by `nndet` sheet). **The copies differ.** |
| `sgd_nesterov` | `true` from nnDetection YAML | Outer manager reads `plan_train.nesterov=None` and converts it to `false`. **The copies differ.** |
| `warm_iterations`, `warm_lr`, `poly_gamma`, `swa_epochs` | `4000`, `1e-6`, `0.9`, `10` | Nested nnDetection scheduler/SWA defaults; the outer manager uses `ReduceLROnPlateau`, so these do not select its schedule. |

## Spreadsheet keys that can be misread as active nnDetection settings

| Spreadsheet key | Actual `retinaunet` behavior |
|---|---|
| `nndet.optimizer=SGD` | Not read as a choice. `det3d` hardcodes SGD. |
| `nndet.momentum`, `nndet.weight_decay`, `nndet.nesterov` | Flattened into `plan_train`; the outer SGD reads these. The nested nnDetection `trainer_cfg` reads `model_params` instead, as shown above. |
| `nndet.benchmark`, `nndet.deterministic` | Flattened into `plan_train`; the outer Lightning trainer converts each blank value to `false`. The bridge does not copy these sheet values into the nested nnDetection `trainer_cfg`. |
| `retinanet.cls_loss=bce`, `retinanet.reg_loss=smooth_l1` | Not used to choose `RetinaUNetV001` losses. Its class fixes BCE and GIoU respectively; `smooth_l1` is **not** its box loss. |
| `retinanet.w_cls`, `retinanet.w_reg` | Not copied into the nnDetection head weights; its YAML uses `1.0` for both. |
| `retinanet.returned_layers`, `retinanet.conv1_t_stride` | Used by the separate RetinaNet path, not to select RetinaUNet encoder/decoder levels. |
| `model_params.encoder_*`, `decoder_levels` | Active for RetinaUNet through `det3d`'s `_plan_arch()`; `auto` uses `det3d` fixed defaults, not the nnDetection planner. |
| `postproc_params.*` | Active in the RetinaUNet model via `plan_architecture_from_det3d()`. |
| `data_params.*`, `plans_det.*`, Fran `dataset_params.*` | Control `det3d` data/preprocessing and its plan. nnDetection's `augment_cfg`, loader, splits, and planner are not loaded by the bridge. |
| `/home/ub/code/det3d/det3d/configs/nndet_param_map.yaml` | Documentation-style mapping only; no production Python code reads this file. Some entries describe intended rather than current behavior, especially `plan_pickle`, `monitor_key`, and classifier/regressor loss weights. |

The remaining detection spreadsheet fields also stay under `det3d` control:

| Sheet keys | Relationship to nnDetection in this path |
|---|---|
| `model_params.class_name` | Used by the outer manager for display names; nnDetection's evaluator internally names classes `class0`, etc. |
| `data_params.validation_fraction`, `seed`, `num_workers_train`, `num_workers_val` | Data split/loader settings; no nnDetection `augment_cfg` or loader is created. |
| `plans_det.mode`, `samples_per_file`, `expand_by`, `fg_indices_exclude`, `remapping_source`, `remapping_lbd_rbd`, `remapping_train`, `nnz_allowed`, `ignore_labels_cc`, `dusting_mm`, `gt_box_mode` | Preprocessing/label/box plan fields; none is passed to nnDetection's planner. |
| `plans_det.notes` | Spreadsheet note only. |
| Fran base workbook `transform_factors`, `affine3d`, `resnet`, and other `model_params` keys | Remain in the merged configuration where applicable; the nnDetection bridge does not read these sheets to configure `RetinaUNetV001`. |

## What happens programmatically, in order

1. `ConfigMakerDet` reads the base and detection workbooks, uses `manual_value`, merges same-named sheets, takes the `lungs` plan 4 row, parses its list-like values, calculates `patch_size` and `src_dims`, and adds Fran global intensity properties.
2. Training can change `arch`, fold, patch size, batch size, LR, and epochs from CLI. `det3d` prepares/loads its own data and infers `fg_labels` from its manifest.
3. Only with `arch=retinaunet`, the bridge loads static model and trainer dictionaries from `/home/ub/code/nnDetection/nndet/conf/train/v001.yaml`, applies the six matcher/sampler overrides, and turns `plan_train` into nnDetection's `architecture`, `anchors`, and `patch_size` dictionaries. nnDetection constructs the network from those dictionaries.
4. The outer `RetinaUNetManager` trains on `det3d` batches with `det3d` SGD and scheduler. No nnDetection dataset analyser, planning algorithm, GPU memory search, optimized-anchor search, or automatic parameter sweep is invoked in these steps.

## Source paths

| Purpose | Full path |
|---|---|
| Fran base spreadsheet | `/home/ub/code/fran/configurations/experiment_configs.xlsx` |
| Detection spreadsheet | `/home/ub/code/fran/configurations/experiment_configs_det.xlsx` |
| `ConfigMakerDet` | `/home/ub/code/det3d/det3d/configs/parser.py` |
| Spreadsheet reader | `/home/ub/code/fran/fran/configs/workbook.py` |
| nnDetection config bridge | `/home/ub/code/det3d/det3d/configs/nndet_bridge.py` |
| RetinaUNet model-plan bridge | `/home/ub/code/det3d/det3d/managers/helpers/nndet_retinaunet.py` |
| `det3d` architecture defaults | `/home/ub/code/det3d/det3d/detection/retinaunet_network.py` |
| Outer RetinaUNet manager | `/home/ub/code/det3d/det3d/managers/retinaunet.py` |
| Outer optimizer and scheduler | `/home/ub/code/det3d/det3d/managers/det_schedule.py` |
| Outer trainer | `/home/ub/code/det3d/det3d/trainers/trainerdet.py` |
| nnDetection `v001` YAML | `/home/ub/code/nnDetection/nndet/conf/train/v001.yaml` |
| nnDetection RetinaUNet implementation | `/home/ub/code/nnDetection/nndet/ptmodule/retinaunet/base.py` and `/home/ub/code/nnDetection/nndet/ptmodule/retinaunet/v001.py` |
| Saved nnDetection LIDC plan | `/s/fran_storage/nndet_conf/plans/lungs/D3V001_3d.pkl` |
| Plan-path registry | `/s/fran_storage/nndet_conf/nndet_plan_paths.yaml` |
| Manual mapping note | `/home/ub/code/det3d/det3d/configs/nndet_param_map.yaml` |

To inspect the keys in Neovim/IPython, use the REPL cells in `/home/ub/code/det3d/scratch/nndet_config_keys_scratch.py`. Its setup cell defaults to `lidca` plan 4. Change `project_title`, `plan_id`, or the config paths in that cell to inspect another project or file. The pad is excluded from Git by `/home/ub/code/det3d/.gitignore`.
