# Issues and bugs found in det3d / FRAN code

Found while tracing the pipeline for [../nndetection-integration.md](../nndetection-integration.md). Read from code on project `lidca`, lungs plan 4; nothing here has been changed in code. "Measured" means a value was computed or a model built; everything else is read from code. Effect on detection results has not been measured for any item.

Prefixes: `F/` = `/home/ub/code/fran/fran/`, `D/` = `/home/ub/code/det3d/det3d/`, `LA/` = `/home/ub/code/label_analysis/label_analysis/`, `N/` = `/home/ub/code/nnDetection/`. Sheet = `/home/ub/code/fran/configurations/experiment_configs_det.xlsx`.

## A. Bugs: the code does not do what it is written to do

| ID | Where | What happens | Consequence | Work group |
|---|---|---|---|---|
| A1 | `D/detection/retinaunet_network.py` `DEFAULT_STRIDES_3D`, read by `N/nndet/arch/encoder/modular.py` `Encoder` | The default list has five strides for five stages. The encoder expects four and reads entry `stage − 1`, so the leading `[1,1,1]` is applied to stage 1 and the last entry is never read | Network downsamples three times, not four (strides 1, 1, 2, 4, 8). First detection level is at full resolution: 14.4 million anchors per patch (measured), 12.6 million on that level | [2.4, 2.5](2-planning.md#24-stage-count-kernels-and-strides) |
| A2 | `D/inference/patch.py` `DetPatchRetinaUNet.swi_predictor` | Calls `offset_boxes` before `box_tile_weight`, so the tile-position weight is computed from crop coordinates against a tile-sized centre | Weights fall below 0.5 and go negative for tiles away from the crop origin (measured on a box at tile offset (256, 256, 128): −2.5 and −1.6, against 1.0 and 0.57 in tile coordinates). Weighted NMS then prefers the box from the tile nearest the origin | [6.4](6-inference.md#64-merging-tiles-within-one-set) |
| A3 | `F/preprocessing/globalproperties.py` `GlobalProperties.store_projectwide_properties` | Spacing array has one row per case in the project but only the sampled cases (at most 100) are filled; the rest stay zero | `spacing_median` is `[0, 0, 0]` for `lidca` (393 rows, 100 filled). No code reads the value | [1.1](1-data-analysis.md#11-size-spacing-and-crop-fingerprint) |
| A4 | `D/managers/helpers/nndet_retinaunet.py` `plan_anchors_from_det3d` with `N/nndet/core/boxes/anchors.py` `AnchorGenerator3D` | The three typed shapes become `zsizes (4,5,6,6)`; the generator gives every level the whole tuple | 12 anchors per position of which 3 are duplicates (z = 6 twice); the in-plane size is 8 on levels 0–1 and 10 on levels 2–3 whatever the level's stride. Duplicates enter each ATSS candidate set twice | [2.7](2-planning.md#27-anchors), [5.1](5-training.md#51-matching-and-sampling) |
| A5 | `D/configs/parser.py` `PlanAdvisorDet.suggest` | Reads boxes from `<project>/detection/fold0/bboxes/*.json`; sidecars are CSV files in the preprocessed folder. With boxes present it would pick three entries of the unsorted list by index | On `lidca` the list is always empty and the fixed anchors `[[6,8,4],[8,6,5],[10,10,6]]` are returned | [2.7](2-planning.md#27-anchors) |
| A6 | `D/transforms/gpu_det.py` `RandAffineBoxSyncd` | `RandAffined` is built without `padding_mode`; MONAI's default is `reflection` | Areas rotated or scaled in from outside the patch show mirrored image and mirrored label map, including mirrored lesions that have no box. Fraction of samples affected not measured | [4.4](4-data-loading.md#44-oversized-patch) |
| A7 | `F/run/preproc/analyze_resample.py` `_config_maker` | Calls `C.setup(plan, plan)`; `ConfigMakerDet.setup(plan_train, verbose=True)` takes the second value as `verbose` | Harmless today (a truthy plan id); the second argument does not select a validation plan | [3](3-preprocessing.md) |

## B. Sheet cells that do not reach the code that seems to use them

| ID | Cell | What actually happens | Work group |
|---|---|---|---|
| B1 | `plans_det.ignore_labels_cc` | Read only by the object-bounded path. The label-bounded path used by plan 4 reads `plan["ignore_labels"]`, which no column fills | [3.5](3-preprocessing.md#35-instances-and-ground-truth-boxes) |
| B2 | `retinanet.cls_loss`, `reg_loss`, `w_cls`, `w_reg` | Not read on the `retinaunet` path; losses are fixed by the nnDetection class with weight 1 | [5.2](5-training.md#52-losses) |
| B3 | `nndet.nesterov` (blank) | `bool(None)` gives plain momentum; the nested nnDetection configuration says Nesterov but is not the optimiser in use | [5.3](5-training.md#53-optimiser) |
| B4 | `nndet.weight_decay` 3e-5 versus base workbook `model_params.weight_decay` 0 | Two copies with different values; the outer optimiser uses 3e-5, the nested nnDetection configuration receives 0 | [5.3](5-training.md#53-optimiser) |
| B5 | `data_params.spacing` `[0.703125, 0.703125, 1.25]` | Not read on this path; preprocessing uses `plans_det.spacing` `(0.8, 0.8, 1.5)` | [2.1](2-planning.md#21-target-spacing) |
| B6 | `plans_det.val_patch_size` | Not used by the validation data manager, which crops the training patch size | [4.7](4-data-loading.md#47-validation-patches) |
| B7 | `data_params.num_workers_train`, `validation_fraction`, `seed` | No reader. `num_workers_val = 0` is read and raised to 2 | [4](4-data-loading.md) |
| B8 | `plan["use_swa"]` | Read by `TrainerDet.init_cbs`; no sheet has the key, so weight averaging cannot be switched on from the sheet | [5.5](5-training.md#55-weight-averaging) |
| B9 | Command-line `--patch-overlap` | Does not reach the `retinanet` inferer, which uses `plan["infer_overlap"]` (0.25) | [6.1](6-inference.md#61-tiling) |

## C. Two parts of the pipeline that disagree with each other

| ID | Parts | Disagreement | Work group |
|---|---|---|---|
| C1 | Box targets versus segmentation target | Components under 2 mm are dropped from the box list but stay in the stored label map; mirrored lesions from A6 have label but no box. The same voxels are foreground for the segmentation loss and background for the box classifier | [3.5](3-preprocessing.md#35-instances-and-ground-truth-boxes), [5.2](5-training.md#52-losses) |
| C2 | Training ground truth versus scorer ground truth | Training: components per class, face-connected, major axis ≥ 2 mm. `LA/run/score_froc.py`: components of any label, 26-connected, equivalent diameter ≥ 3 mm | [7.3](7-evaluation.md#73-froc) |
| C3 | Class-wise NMS versus class-agnostic scoring | NMS at every stage runs per class, so one lesion can survive once per class; the scorer ignores class and counts the second box as a false positive | [6.4](6-inference.md#64-merging-tiles-within-one-set), [7.3](7-evaluation.md#73-froc) |
| C4 | Training crop versus inference crop | Training crops are bounded by raw labels before remapping (raw label 1, discarded by plan 4, sets the extent; median 39% of the scan); inference uses a localiser lung crop with a 10 mm margin. Lesion-free cases never enter training | [3.1](3-preprocessing.md#31-crop) |
| C5 | Intensity statistics versus plan labels | `get_bbox_only_voxels` takes every voxel with a raw label above 0, so raw label 1 enters the clip-range percentiles although plan 4 maps it to background. Share not measured | [1.3](1-data-analysis.md#13-foreground-intensity-statistics) |
| C6 | Trained box lower bound versus scorer | `disk_bbox_to_nndet_xyxyzz` shifts the lower corner by −1 for training; saved predictions keep that convention while the scorer's ground truth uses the component's first voxel. One-voxel offset, inferred from code | [6.8](6-inference.md#68-restore-to-image-space) |
| C7 | Thresholds versus metrics | Score > 0.02 and 25 boxes per window are applied before boxes reach the in-training metric and the saved sidecars; FROC at 4–8 false positives per scan and the AP curve read the low-score tail that this removes | [6.3](6-inference.md#63-per-patch-filtering), [7.1](7-evaluation.md#71-in-training-metric) |
| C8 | Checkpoint ranking versus checkpoint used | Training ranks checkpoints by validation metric (or training loss in run-through mode); `checkpoint_from_model_id(sort_method="last")` in `F/trainers/helpers.py` takes the newest `.ckpt` by modification time | [5.6](5-training.md#56-checkpoint-choice) |

## D. Behaviour worth knowing, not defects

| ID | Where | Behaviour | Work group |
|---|---|---|---|
| D1 | `F/managers/project.py` `Project._create_folds` | `int(1 / 0.15)` gives six folds, consecutive slices of the sorted case ids, no shuffle. `lidca` folds hold 101, 99, 98, 137, 158, 160 cases; cause of the unequal sizes not found | [4.1](4-data-loading.md#41-splits) |
| D2 | `D/managers/data/main.py` `RandCropExtendedBBoxd` | A lesion in a scan with k lesions is drawn with weight 1/k; lesion-directed crops are an independent 0.75 draw each, so a batch can contain none | [4.3](4-data-loading.md#43-patch-sampling) |
| D3 | `D/transforms/gpu_det.py` box warp | Under rotation the box is the envelope of eight warped corners; at 0.2 rad a square side grows by about 18%. These enlarged boxes are the regression targets | [4.6](4-data-loading.md#46-boxes-after-augmentation) |
| D4 | `D/transforms/gpu_det.py` intensity transforms | `RandScaleIntensityd` and `RandShiftIntensityd` are a positive scale and a shift; the network's first instance normalisation removes both except where the patch is padded | [4.5](4-data-loading.md#45-augmentation-set) |
| D5 | `D/managers/data/main.py` validation chain | One patch per case centred on its first box; every scored lesion sits near the patch centre and no lesion-free patch is seen | [4.7](4-data-loading.md#47-validation-patches) |
| D6 | `LA/geometry_itk.py` `label_to_labelmap` | Connectivity is not set (ITK default, face connectivity); touching lesions of one class become one box | [3.5](3-preprocessing.md#35-instances-and-ground-truth-boxes) |
| D7 | `D/trainers/trainerdet.py` `TrainerDet.init_cbs` | If weight averaging is switched on, its rate is set to the initial 0.01, above whatever `ReduceLROnPlateau` has reached. Lightning internals not opened | [5.5](5-training.md#55-weight-averaging) |
| D8 | `D/managers/retinanet.py` validation | `val0_metric` on the `retinanet` path is the plain mean of eight COCO entries, computed with one class name for a three-class model | [7.1](7-evaluation.md#71-in-training-metric) |

## E. Suspected, not confirmed

| ID | Where | Suspicion | What would confirm it |
|---|---|---|---|
| E1 | `D/inference/post.py` `PackRetinaNetPredsd` | Reads `pred["box"]` while the detector's target key is set to `"bbox"`; the `retinanet` inference path may fail | One `retinanet` inference run |
| E2 | `D/inference/patch.py` `forward_tile` | Crops smaller than the patch go through the network unpadded; shapes not divisible by 8 may fail or misalign | One inference run on a small crop |
| E3 | `F/data/datasource.py` `Datasource.dump_to_h5` | With `return_voxels=False` it calls `.cpu()` on `None` and catches only `ValueError` | Run `update_datasources.py` without `--return-voxels` |
| E4 | `lidca` raw label 1 | Organ-sized label that plan 4 maps to background; its identity sets how C4 and C5 should be read | Open one raw label map |
| E5 | `det3d_batch_to_nndet` segmentation target | Label values 1–3 are passed with `seg_classes = 1`; upstream binarises inside the loss (`target[target > 0] = 1`), which was read in nnDetection's code but not exercised on det3d batches | One training step with an assertion on the target's unique values |
