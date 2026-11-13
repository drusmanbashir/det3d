# det3d versus nnDetection: algorithm comparison per work group

Scope: every work group of [../nndetection-integration.md](../nndetection-integration.md) whose "Exists" value is Yes, Partly or Different (28 groups; the No rows are skipped), example project `lidca`, lungs plan 4, det3d `retinaunet` path, nnDetection `RetinaUNetV001` + `D3V001` at local commit `09ff4ad`. Each section scores only the criteria that apply to that method, 1–5 per side: 5 = no defect found and the stronger design on that criterion, 4 = sound with a minor limitation, 3 = adequate with a material limitation, 2 = a demonstrable defect or a large disadvantage, 1 = defective or inert in effect. Totals compare the two sides within one row only. Every statement cites the file and symbol read; "(data)" marks numbers read from the lidca preprocessed CSV files or the saved nnDetection LIDC plan `/s/fran_storage/nndet_conf/plans/lungs/D3V001_3d.pkl`, not from code; "(trace)" marks a value taken from the trace notes; "code over notes" marks a place where code contradicts the integration docs or trace notes.

Prefixes: `N/` = `/home/ub/code/nnDetection/`, `F/` = `/home/ub/code/fran/fran/`, `D/` = `/home/ub/code/det3d/det3d/`, `LA/` = `/home/ub/code/label_analysis/label_analysis/`, `U/` = `/home/ub/mambaforge/envs/dl/lib/python3.13/site-packages/nnunet/` (nnU-Net 1.7.1, called by nnDetection), `M/` = `/home/ub/mambaforge/envs/dl/lib/python3.13/site-packages/monai/` (MONAI 1.5.2).

| # | Work group | nnDetection total | det3d total | Winner |
|---|---|---|---|---|
| 1.1 | Size, spacing and crop fingerprint | 10 | 3 | nnDetection |
| 1.3 | Foreground intensity statistics | 12 | 10 | nnDetection |
| 2.5 | Decoder levels | 16 | 8 | nnDetection |
| 2.6 | Channel widths | 5 | 5 | tie |
| 3.1 | Crop | 9 | 10 | det3d |
| 3.2 | Transpose | 8 | 9 | det3d |
| 3.3 | Resampling | 9 | 10 | det3d |
| 3.4 | Intensity normalisation | 7 | 8 | det3d |
| 3.5 | Instances and ground-truth boxes | 9 | 6 | nnDetection |
| 4.1 | Splits | 9 | 5 | nnDetection |
| 4.2 | Epoch length | 7 | 7 | tie |
| 4.3 | Patch sampling | 7 | 7 | tie |
| 4.4 | Oversized patch | 10 | 9 | nnDetection |
| 4.5 | Augmentation set | 10 | 10 | tie |
| 4.6 | Boxes after augmentation | 14 | 10 | nnDetection |
| 4.7 | Validation patches | 10 | 9 | nnDetection |
| 5.1 | Matching and sampling | 13 | 10 | nnDetection |
| 5.2 | Losses | 10 | 8 | nnDetection |
| 5.3 | Optimiser | 7 | 7 | tie |
| 5.5 | Weight averaging | 7 | 5 | nnDetection |
| 5.6 | Checkpoint choice | 8 | 5 | nnDetection |
| 6.1 | Tiling | 7 | 7 | tie |
| 6.3 | Per-patch filtering | 7 | 7 | tie |
| 6.4 | Merging tiles within one set | 9 | 5 | nnDetection |
| 6.8 | Restore to image space | 8 | 9 | det3d |
| 7.1 | In-training metric | 8 | 8 | tie |
| 7.2 | Whole-case validation scoring | 8 | 4 | nnDetection |
| 7.3 | FROC | 10 | 12 | det3d |

# Phase 1: data analysis
## 1.1 Size, spacing and crop fingerprint
| | nnDetection | det3d |
|---|---|---|
| Correctness | 5 | 2 |
| Adaptivity | 5 | 1 |
| **Total** | **10** | **3** |

**Winner:** nnDetection — its per-case sizes and spacings feed planning; det3d's stored spacing median is zero by construction and nothing reads it.

- **nnDetection**
  - Pro: `get_sizes_and_spacings_after_cropping`, `get_size_reduction_by_cropping` (N/nndet/planning/properties/medical.py) record every training case; `D3V001.determine_forward_backward_permutation` (N/nndet/planning/experiment/v001.py) reads `all_spacings` and `all_sizes`.
  - Con: sizes are taken after `crop_to_nonzero`, which on CT removes almost nothing (3.1), so the crop ratio is close to 1 and carries little information.
- **det3d**
  - Con (code over notes): `GlobalProperties.store_projectwide_properties` (F/preprocessing/globalproperties.py) allocates `np.zeros((len(self.case_ids), 3))` but fills only the ≤100 sampled rows of `self.case_properties`; with 393 rows the median is 0. This is the cause of lidca `spacing_median = [0, 0, 0]`, which the trace lists as UNCONFIRMED.
  - Con: no crop ratio; resampled shapes (`Preprocessor._store_dataset_summary`, F/preprocessing/preprocessor.py) exist only after the spacing is fixed; no file in D/ or F/ reads `spacing_median` (grep).
  - Pro: spacing is read from `fg_voxels.h5` attributes (`Datasource.dump_to_h5`, F/data/datasource.py), so no image is reloaded.

## 1.3 Foreground intensity statistics
| | nnDetection | det3d |
|---|---|---|
| Correctness | 5 | 3 |
| Statistical soundness | 4 | 3 |
| Efficiency | 3 | 4 |
| **Total** | **12** | **10** |

**Winner:** nnDetection — its foreground is the training-target foreground of every case; det3d's includes a label the plan discards and at most 100 cases.

- **nnDetection**
  - Pro: `get_voxels_in_foreground` (N/nndet/planning/properties/intensity.py) uses `seg > 0` after the crop (voxels outside the nonzero mask are −1), i.e. the instance voxels that become targets.
  - Pro: every case contributes; subsampling (`[::10]`) is within case, so between-case variation, the larger variance term of a pooled percentile, is fully represented.
  - Con: voxels are Python lists concatenated in `run_collect_intensity_properties` (`all_voxels += iv`), one object per voxel; systematic, not random, subsampling.
- **det3d**
  - Con: `SingleCaseAnalyzer.get_bbox_only_voxels` (F/preprocessing/datasetanalyzers.py) is `img[mask != bg_label]` on raw labels, so raw label 1, which plan 4 remaps to 0, enters the statistic; its share is UNCONFIRMED (`fg_voxels.h5` not mounted).
  - Con: `GlobalProperties.sample_cases` keeps ≤100 cases (`random.Random(42)`), so between-case variation is sampled from 100 cases.
  - Pro: voxels are stored once per datasource and concatenated as one NumPy array (`_retrieve_h5_voxels`).

# Phase 2: planning
## 2.5 Decoder levels
| | nnDetection | det3d |
|---|---|---|
| Correctness | 5 | 2 |
| Adaptivity | 4 | 1 |
| Efficiency | 4 | 1 |
| Robustness (tiny lesions) | 3 | 4 |
| **Total** | **16** | **8** |

**Winner:** nnDetection — levels follow the planned stages; det3d's stride list is read with an offset, so its typed levels sit at strides 1, 2, 4, 8 and carry 14.2× the anchors of the LIDC plan.

- **nnDetection**
  - Pro: `BoxC002._plan_architecture` (N/nndet/planning/architecture/boxes/c002.py) recomputes `decoder_levels_start = min(max(1, R − 4), 2)` after each memory replan.
  - Pro (data): LIDC plan levels (2, 3, 4, 5) at strides (2,4,4) to (8,32,32): 37,504 positions × 27 = 1,012,608 anchors per 128×128×64 patch.
  - Con: no full-resolution level; finest anchor spacing on LIDC is 4 × 2.8 × 2.8 mm (stride (2,4,4) at (2.0, 0.70, 0.70) mm).
- **det3d**
  - Con: `DEFAULT_STRIDES_3D` (D/detection/retinaunet_network.py) has five entries for five kernels; `Encoder` (N/nndet/arch/encoder/modular.py) applies `strides[stage_id - 1]` and documents one entry fewer than kernels. The leading `[1,1,1]` leaves stage 1 undownsampled and the fifth entry is never read: three downsamplings, absolute strides [1,1,2,4,8] (`encoder_abs_strides_from_plan` agrees).
  - Con: the typed `(1,2,3,4)` (`_plan_arch`) equals upstream's rule for five stages but, through that offset, gives 1,198,080 positions × 12 = 14,376,960 anchors per patch, 12.6 M of them on level 1.
  - Pro: the stride-1 level places anchor centres at every 0.8 mm voxel, the densest coverage for nodules of a few voxels.

## 2.6 Channel widths
| | nnDetection | det3d |
|---|---|---|
| Correctness | 5 | 5 |
| **Total** | **5** | **5** |

**Winner:** tie — identical constants.

- **nnDetection**
  - Pro: `BoxC001.create_default_settings` sets max 320 (N/nndet/planning/architecture/boxes/base.py); `BoxC002.create_default_settings` sets start 32 and FPN = head = 4 × start (c002.py).
  - Con: constants; the memory fit acts on patch size and depth only.
- **det3d**
  - Pro: `_plan_arch` (D/detection/retinaunet_network.py) reproduces 32 / 320 and `fpn_channels = start_channels * 4`.
  - Con: same constants; no memory check (2.3 is a No row).

# Phase 3: preprocessing
## 3.1 Crop
| | nnDetection | det3d |
|---|---|---|
| Statistical soundness | 3 | 3 |
| Efficiency | 2 | 4 |
| Robustness | 4 | 3 |
| **Total** | **9** | **10** |

**Winner:** det3d — on lidca its crop keeps a median 39% of the scan, cutting storage and I/O; its effect on false positives outside the crop is undecided without a whole-scan FP/scan measurement.

- **nnDetection**
  - Pro: `crop_to_nonzero` (N/nndet/preprocessing/crop.py) keeps the whole scan, so random patches cover every region the test-time tiling visits, and lesion-free cases stay in training.
  - Con: `create_nonzero_mask` tests `data != 0`; CT air is about −1000 HU, so on CT the crop removes almost nothing, yet `binary_fill_holes` runs over the full volume.
  - Con: random patches include air and table outside the body.
- **det3d**
  - Pro (data): LBD crops keep a median 39% (IQR 31–45%) of the resampled scan volume (LBD versus source `dataset_details.csv`); median crop (354, 246, 180) voxels.
  - Con (code over notes): `CropForegroundMinShaped` (F/transforms/spatialtransforms.py) runs before `Remap` (`tfms_keys` "…Crop,Remap…", F/preprocessing/labelbounded.py) with `select_fn=is_positive`, so the extent comes from raw labels the plan discards (lidc_0001: 3.9 M source foreground voxels, one box). The doc's "neighbourhood of annotated lesions" understates the region: no lidca crop equals the patch on all axes (data). Identity of that raw label: UNCONFIRMED.
  - Con: with `nnz_allowed = 0`, cases without labels are dropped (`_filter_cases_by_stats_nnz`, D/managers/data/main.py).

## 3.2 Transpose
| | nnDetection | det3d |
|---|---|---|
| Correctness | 5 | 5 |
| Robustness | 3 | 4 |
| **Total** | **8** | **9** |

**Winner:** det3d — it applies each case's direction cosines, so all cases share one anatomical frame; upstream applies one dataset-wide axis permutation, the identity on LIDC (data: `transpose_forward = [0, 1, 2]`).

- **nnDetection**
  - Pro: `GenericPreprocessor.transpose` (N/nndet/preprocessing/preprocessor.py) permutes image, label and both spacings with one `transpose_forward`; `restore_detection` inverts it.
  - Con: direction matrices are only stored (`itk_direction`, N/nndet/io/load.py); no reorientation step exists in load, crop or preprocessor, so cases with flipped directions enter mirrored (mitigated by 3-axis mirror augmentation, 4.5).
- **det3d**
  - Pro: `Orientationd(axcodes="RAS")` in `_NiftiResamplerBase` (F/preprocessing/fixed_spacing.py) flips and permutes per case from its affine; inference repeats it (`create_preprocess_transforms`, D/inference/patch.py) and `BoxRd` (D/inference/post.py) restores the source frame.
  - Con: axis order is fixed x, y, z regardless of anisotropy. Exact array transposition of `LoadSITKd` (F/transforms/imageio.py): UNCONFIRMED.

## 3.3 Resampling
| | nnDetection | det3d |
|---|---|---|
| Statistical soundness | 4 | 3 |
| Efficiency | 2 | 4 |
| Robustness (anisotropy) | 3 | 3 |
| **Total** | **9** | **10** |

**Winner:** det3d — equal anisotropy handling in effect, lower cost; cubic interpolation's fidelity gain is unmeasured on this data.

- **nnDetection**
  - Pro: image order 3 (`GenericPreprocessor.resample` → `resample_patient`, N/nndet/preprocessing/resampling.py); a cubic kernel attenuates small high-contrast structures less than a linear one. NaN set to 0 first.
  - Con (code over notes): `resample` passes `force_separate_z=False`; nnU-Net's `resample_patient` (U/preprocessing/preprocessing.py) then fixes `do_separate_z = False`, so the coarse axis is never resampled separately, contrary to the index and planning doc; `order_z_data=9999` is never reached.
  - Con: skimage spline on float64 copies (`resample_data_or_seg`: `data.astype(float)`, `resize(order=3, anti_aliasing=False)`) on CPU.
- **det3d**
  - Pro: MONAI `Spacingd` trilinear on torch tensors after `Dev` (F/preprocessing/fixed_spacing.py); label nearest, then `ResizeToTensord` forces equal image and label shapes.
  - Con: linear interpolation smooths small nodules more than cubic; no anti-aliasing when z is downsampled; no coarse-axis rule (same as upstream in effect).

## 3.4 Intensity normalisation
| | nnDetection | det3d |
|---|---|---|
| Statistical soundness | 4 | 4 |
| Efficiency | 3 | 4 |
| **Total** | **7** | **8** |

**Winner:** det3d — same clip bounds; the first block is conv → InstanceNorm (`ConvInstanceRelu`, N/nndet/arch/conv.py), which cancels any positive affine input map up to eps and padding, so min–max versus z-score does not change stage-0 output; det3d keeps HU on disk, so the clip can change without reprocessing.

- **nnDetection**
  - Pro: `normalize_ct` (N/nndet/preprocessing/preprocessor.py) z-scores with foreground mean and SD, so intensity augmentation strengths scale with tissue contrast (4.5).
  - Con: normalised float32 is written once (`run_process`: `data.astype(np.float32)`, `np.savez_compressed`); new bounds mean reprocessing every case.
  - Con: value 0 used for loader and convolution padding equals the foreground mean intensity; effect UNCONFIRMED.
- **det3d**
  - Pro: `ScaleIntensityRanged(..., 0, 1, clip=True)` at load (`DataManagerDetSource.create_transforms`, D/managers/data/main.py) and at inference (`norm_dtype_transforms`, D/inference/patch.py) from the checkpoint's saved config; HU stored as float16 (`H` step, F/preprocessing/labelbounded.py).
  - Con: value 0 used for padding equals −973 HU on lidca; `mean_fg`, `std_fg` are computed and unused. Effect of the padding value UNCONFIRMED.

## 3.5 Instances and ground-truth boxes
| | nnDetection | det3d |
|---|---|---|
| Correctness | 5 | 3 |
| Robustness | 4 | 3 |
| **Total** | **9** | **6** |

**Winner:** nnDetection — lesion identity comes from the instance mask; det3d merges touching same-class lesions and defines a lesion differently in training and scoring.

- **nnDetection**
  - Pro: `compute_candidates` (N/nndet/preprocessing/preprocessor.py) boxes each instance id of the resampled instance mask (`instances_to_boxes_np`); touching lesions stay separate; no size filter.
  - Pro: boxes and segmentation target come from the same mask after augmentation (4.6, 5.2).
  - Con: an instance lost in nearest-neighbour resampling disappears without a record (`np.unique` survivors only).
- **det3d**
  - Con: `DetectionBBoxStatsd` (F/transforms/bbox_stats.py) → `label_to_labelmap` (LA/geometry_itk.py) labels components per class with `BinaryImageToShapeLabelMapFilter`, connectivity not set (ITK default, face connectivity); touching lesions of one class become one box.
  - Con: `dust` (LA/geometry_itk.py) drops components with `major_axis` < 2.0 mm or `flatness == 0`; the scorer (7.3) uses 26-connectivity and a 3 mm equivalent diameter. The stored `lm` is not rewritten, so dusted components stay in the segmentation target.
  - Pro: physical-size filtering in mm removes isolated annotation specks before any box is stored (`to_voxel_detection_records`, D/geometry/lmg.py).

# Phase 4: data loading
## 4.1 Splits
| | nnDetection | det3d |
|---|---|---|
| Correctness | 4 | 3 |
| Statistical soundness | 5 | 2 |
| **Total** | **9** | **5** |

**Winner:** nnDetection — a seeded random permutation; det3d slices the sorted id order, so any structure in id order becomes fold structure.

- **nnDetection**
  - Pro: `BaseModule.create_new_split` (N/nndet/io/datamodule/base.py): `KFold(n_splits=5, shuffle=True, random_state=12345)` on sorted ids; deterministic, exchangeable folds, saved once and reused.
  - Con: case-level split; no grouping by patient.
- **det3d**
  - Con: `Project._create_folds` (F/managers/project.py) defaults to `shuffle=False`; folds are consecutive slices (`val_indices`, F/data/datasource.py) of the sorted case-id order.
  - Con: `folds = int(1 / pct_valid)` gives six folds (16.7% validation) for 0.15; only unassigned cases are split on later calls (`get_unassigned_cases`); lidca folds hold 101–160 cases (trace), cause UNCONFIRMED.
  - Pro: rows with the same case id share one `index` (`create_df_folds`), keeping multi-image cases in one fold.

## 4.2 Epoch length
| | nnDetection | det3d |
|---|---|---|
| Statistical soundness | 3 | 4 |
| Efficiency | 4 | 3 |
| **Total** | **7** | **7** |

**Winner:** tie — epoch length is a bookkeeping unit; its consequences are scored in 5.6 (and 5.4, a No row).

- **nnDetection**
  - Pro: `FixedSlimDataLoaderBase.__len__` = 2,500 batches (N/nndet/io/datamodule/bg_loader.py, N/nndet/conf/train/v001.yaml), independent of dataset size, as the per-iteration poly schedule requires.
  - Con: draws with replacement (`DataLoader3DFast.select`: `np.random.choice`), so per-case exposure varies.
- **det3d**
  - Pro: one crop per case per epoch, DataLoader `shuffle=True` (D/managers/data/main.py): reshuffling without replacement gives each case equal exposure per epoch.
  - Con: about 148 iterations per epoch on fold 0 (trace); per-epoch work in `TrainerDet.init_cbs` (D/trainers/trainerdet.py; checkpoint every 5 epochs, W&B grids) recurs about 17× more often per iteration than upstream's.

## 4.3 Patch sampling
| | nnDetection | det3d |
|---|---|---|
| Statistical soundness | 4 | 3 |
| Robustness (lesions cut by tiles) | 3 | 4 |
| **Total** | **7** | **7** |

**Winner:** tie — undecided without a measurement (per-lesion sensitivity stratified by lesions per case and by distance to tile borders).

- **nnDetection**
  - Pro: `DataLoader3DFast.select` draws foreground uniformly over all instances (`build_cache`), the unit FROC counts.
  - Pro: `DataLoader3DOffset.get_fg_crop` places the window so the whole box lies inside, at a random offset.
  - Con: the two random elements (`get_bg_crop`) are not lesion-free; cut lesions appear only by chance, while test-time tiles cut lesions.
- **det3d**
  - Pro: `sample_crop_center_from_extended_boxes` (D/transforms/crop_indices.py) rejects background centres inside any extended box (64 tries), giving lesion-free negatives.
  - Pro: extended boxes (`ExtendedBBoxesByPatchSizesd`, F/transforms/detection.py) require only one voxel of overlap, so lesions cut by the window border are trained, as produced by 20–25% overlap tiling (6.1).
  - Con: case first, then lesion (`RandCropExtendedBBoxd`, D/managers/data/main.py): a lesion in a case with k lesions is drawn with weight 1/k; foreground is a per-crop Bernoulli (0.75), so a batch may hold no lesion crop.

## 4.4 Oversized patch
| | nnDetection | det3d |
|---|---|---|
| Correctness | 4 | 2 |
| Efficiency | 2 | 5 |
| Robustness (borders) | 4 | 2 |
| **Total** | **10** | **9** |

**Winner:** nnDetection — real context fills rotated and shrunk windows; det3d fills them with reflected image content that can contain lesions without boxes.

- **nnDetection**
  - Pro: `get_patch_size` (N/nndet/io/augmentation/base.py) enlarges [128,128,64] to [249,204,170] for ±30° and scale 0.7; padding (`save_get_crop`, N/nndet/io/patching.py) occurs only at scan borders.
  - Con: the bound rotates the corner vector about one axis at a time, not the combined rotation, so it is approximate.
  - Con: 8.2× the voxels are read and held as float64 (`generate_train_batch`: `np.zeros(..., dtype=float)`), and 64% of samples are only centre-cropped (rotation and scaling each p 0.2).
- **det3d**
  - Pro: final-size HDF5 read and GPU affine (`RandAffineBoxSyncd`, D/transforms/gpu_det.py).
  - Con: `RandAffined` is built without `padding_mode`; the MONAI default is `reflection` (M/transforms/spatial/dictionary.py `RandAffined`), so out-of-patch areas show mirrored content, including mirrored lesions with no box; `lm` is mirrored too, so segmentation and box targets disagree there. Affected sample fraction: UNCONFIRMED.
  - Con: `prezoom_scale` (oversized min shape and extended-box columns) exists at 1.0.

## 4.5 Augmentation set
| | nnDetection | det3d |
|---|---|---|
| Statistical soundness | 4 | 3 |
| Efficiency | 2 | 4 |
| Robustness | 4 | 3 |
| **Total** | **10** | **10** |

**Winner:** tie — undecided without a measurement (exchange the two sets at fixed sampler and schedule).

- **nnDetection**
  - Pro: `BaseMoreAug.get_training_transforms` (N/nndet/io/augmentation/bg_aug.py): mirror per axis p 0.5, matching 8-fold mirror TTA; inverted gamma p 0.1; dummy-2D rotation when the patch is anisotropic (set in the LIDC plan, data).
  - Pro: batchgenerators passes the sampled "variance" as the normal's scale (`augment_gaussian_noise`), so noise SD ≤ 0.1 foreground SD (≈ 29 HU if the lidca SD of 285.6 HU applies).
  - Con: cubic spatial interpolation of the oversized patch in CPU workers; multiplicative brightness and contrast are near-affine and largely cancelled by the first InstanceNorm.
- **det3d**
  - Pro: all transforms on GPU at final size (`build_train_gpu_tail_compose`, D/transforms/gpu_det.py).
  - Con: one random axis flipped with p 0.1 (`RandAxisFlipd`); rotation ±0.20/±0.20/±0.15 rad and per-axis scale ±15% at p 0.3 (sheet `affine3d`, trace).
  - Con: `RandRandGaussianNoised` (F/transforms/intensitytransforms.py) SD up to 0.18 clip widths, ≤ 252 HU on lidca; `RandScaleIntensityd` (×0…2) and `RandShiftIntensityd` (±1) are positive-affine and cancelled by the first InstanceNorm except at padding.

## 4.6 Boxes after augmentation
| | nnDetection | det3d |
|---|---|---|
| Correctness | 5 | 2 |
| Efficiency | 4 | 5 |
| Robustness | 5 | 3 |
| **Total** | **14** | **10** |

**Winner:** nnDetection — boxes are recomputed from the voxels still visible; det3d's warped-corner envelope inflates rotated boxes and can keep boxes whose lesion left the patch.

- **nnDetection**
  - Pro: `Instances2Boxes` → `instances_to_boxes` (N/nndet/io/transforms/instances.py) takes min−1 / max+1 of each surviving instance after augmentation on the training device; truncated lesions get truncated boxes, vanished lesions none.
  - Pro: `Instances2Segmentation` builds the segmentation target from the same mask.
  - Con: one `nonzero` pass per instance per sample; a thin instance fragmented by nearest-neighbour rotation yields one box over all fragments.
- **det3d**
  - Pro: `_warp_box_corners` (D/transforms/gpu_det.py) is exact for flips and axis-aligned scaling, eight points per box.
  - Con: under rotation the axis-aligned envelope of eight corners exceeds the lesion; at 0.2 rad a square side grows by cos + sin ≈ 1.18; these boxes are GIoU targets.
  - Con: `ClipBoxToImagedWithDf(remove_empty=True)` removes only boxes empty after clipping; the warped `lm` is loaded on this path but not used for boxes.

## 4.7 Validation patches
| | nnDetection | det3d |
|---|---|---|
| Statistical soundness | 3 | 3 |
| Efficiency | 3 | 4 |
| Robustness | 4 | 2 |
| **Total** | **10** | **9** |

**Winner:** nnDetection — half its validation patches are random and the rest place lesions at random offsets; det3d validates only lesion-centred patches of positive cases.

- **nnDetection**
  - Pro: `val_dataloader` (N/nndet/io/datamodule/bg_module.py) reuses the training sampler with `patch_size_generator = patch_size`, 100 batches, half random.
  - Con: patches are redrawn every epoch, so the monitored metric carries sampling noise; no whole-case score during training.
- **det3d**
  - Pro: `BboxCenterCropSlicesd` (D/managers/data/main.py) is deterministic: the same patches every epoch, so epoch-to-epoch change reflects the model.
  - Con: one patch per case centred on `bbox[0]`; every validation lesion sits near the centre; lesion-free cases are filtered (`_filter_cases_by_stats_nnz`); other lesions outside that patch are not scored.

# Phase 5: training
## 5.1 Matching and sampling
| | nnDetection | det3d |
|---|---|---|
| Correctness | 5 | 4 |
| Statistical soundness | 4 | 4 |
| Efficiency | 4 | 2 |
| **Total** | **13** | **10** |

**Winner:** nnDetection — identical matcher and sampler code; det3d feeds them 14.2× more anchors, some duplicated.

- **nnDetection**
  - Pro: `ATSSMatcher.compute_matches` (N/nndet/core/boxes/matcher/atss.py) sets the positive threshold per GT at mean + SD of candidate IoUs; 4 × 27 = 108 candidates per level on the LIDC plan.
  - Con: full [N_gt × M] distance and IoU matrices; about 4 MB per matrix per GT at 1.0 M anchors.
  - Con: `min_neg=1`, 32 anchors per image (v001.yaml): a batch without positives trains on 2 negatives (`get_num_neg`, N/nndet/core/boxes/sampler.py).
- **det3d**
  - Con: `AnchorGenerator3D` copies the whole `zsizes` tuple to each level (N/nndet/core/boxes/anchors.py); `plan_anchors_from_det3d` (D/managers/helpers/nndet_retinaunet.py) gives `(4,5,6,6)`, so z = 6 anchors exist twice per ratio, enter each candidate set twice and shift the mean + SD threshold.
  - Con: the same matrices span 14.4 M anchors, about 57.5 MB per matrix per GT.
  - Pro: sheet sampler values (`apply_det3d_plan_to_nndet_model_cfg`): 256 sampled anchors per batch of four instead of 128 and ≥16 negatives; effect UNCONFIRMED without a measurement.

## 5.2 Losses
| | nnDetection | det3d |
|---|---|---|
| Correctness | 5 | 5 |
| Statistical soundness (target consistency) | 5 | 3 |
| **Total** | **10** | **8** |

**Winner:** nnDetection — the loss code is the same; det3d's segmentation and box terms see different lesion sets.

- **nnDetection**
  - Pro: `BaseRetinaNet.train_step` (N/nndet/core/retina.py): BCE on sampled anchors, GIoU on sampled positives; `DiCESegmenterFgBg.compute_loss` (N/nndet/arch/heads/segmenter.py) sets `target > 0` to 1, then 0.5 CE + 0.5 Dice.
  - Pro: box and mask targets derive from one augmented instance mask (4.6).
  - Con: all loss weights are constants (v001.yaml).
- **det3d**
  - Pro (code over notes): `disk_bbox_to_nndet_xyxyzz` (D/managers/helpers/nndet_retinaunet.py) subtracts 1 from indices 0, 1, 4 of xyxyzz, the lower corners x1, y1, z1 (trace says "x1, y1, z2"), reproducing upstream's min−1 / max+1 boxes.
  - Con: `target_seg = lm.squeeze(1).long()` (`det3d_batch_to_nndet`) keeps components the 2 mm dust removed from boxes (3.5) and reflected lesion copies (4.4): foreground for Dice and CE, background for BCE.

## 5.3 Optimiser
| | nnDetection | det3d |
|---|---|---|
| Statistical soundness | 4 | 3 |
| Robustness | 3 | 4 |
| **Total** | **7** | **7** |

**Winner:** tie — undecided without a measurement (det3d with Nesterov on and clipping off).

- **nnDetection**
  - Pro: `RetinaUNetModule.configure_optimizers` (N/nndet/ptmodule/retinaunet/base.py): SGD with Nesterov; `get_params_no_wd_on_norm` (N/nndet/training/optimizer/utils.py) exempts normalisation parameters from decay.
  - Con: no gradient clipping in N/nndet or N/scripts (grep); fp16 AMP with `terminate_on_nan=True` (N/scripts/train.py `_train`), so a loss spike propagates unclipped.
- **det3d**
  - Pro: `gradient_clip_val = 1.0` for `retinaunet` and bf16-mixed precision (`TrainerDet.setup`, D/trainers/trainerdet.py).
  - Con: `configure_detection_optimizers` (D/managers/det_schedule.py): `nesterov=bool(plan["nesterov"])` with a blank cell gives plain momentum; one parameter group, so InstanceNorm and GroupNorm gains decay too.

## 5.5 Weight averaging
| | nnDetection | det3d |
|---|---|---|
| Correctness | 3 | 3 |
| Statistical soundness | 4 | 2 |
| **Total** | **7** | **5** |

**Winner:** nnDetection — it averages ten snapshots in its default run; det3d's averaging is off and, when on, anneals toward the initial rate.

- **nnDetection**
  - Pro: `SWACycleLinear` (N/nndet/training/swa.py) from `configure_callbacks` (N/nndet/ptmodule/base_module.py): 10 extra epochs, one snapshot at each epoch start.
  - Con: `cyclic_linear_lr` (N/nndet/training/learning_rate.py) returns the "final" rate (1e-5) at cycle start and rises to the "initial" rate (1e-3) at cycle end; snapshots at epoch start (`BaseSWA.on_train_epoch_start`) therefore follow the high-rate end of a cycle in 9 of 10 cases.
  - Con: whether `model_last.ckpt` holds averaged weights: UNCONFIRMED (PL 1.4.2 internals).
- **det3d**
  - Con: `StochasticWeightAveraging` only if `plan["use_swa"]` (`TrainerDet.init_cbs`, D/trainers/trainerdet.py); no sheet has the key.
  - Con: when on, `swa_lrs = float(self.lr)`, the initial 0.01; Lightning's SWA scheduler anneals toward `swa_lrs`, i.e. back up after `ReduceLROnPlateau` has lowered the rate (Lightning internals not opened: UNCONFIRMED).

## 5.6 Checkpoint choice
| | nnDetection | det3d |
|---|---|---|
| Correctness | 4 | 3 |
| Robustness | 4 | 2 |
| **Total** | **8** | **5** |

**Winner:** nnDetection — selection by name; det3d inference loads the newest file by modification time, and its metric-ranked files are never read by default.

- **nnDetection**
  - Pro: sweep, consolidation and prediction load `model_last` by name (`get_loader_fn`, `load_final_model` asserts one match, N/nndet/inference/loading.py).
  - Con: `load_all_models` globs every `*.ckpt`; predicting from a fold directory loads `model_best` and `model_last` as a two-model ensemble.
- **det3d**
  - Con: `checkpoint_from_model_id(sort_method="last")` (F/trainers/helpers.py) takes the newest `*.ckpt` by `st_mtime`; copying or touching a file changes the model used.
  - Con: `TrainerDet.init_cbs` ranks by `val0_metric` on 4.7 patches, `TrainerDetRT` by training loss (D/trainers/trainerdet_rt.py); neither ranking reaches default inference.
  - Pro: falls back to the second-newest file when the newest cannot be loaded.

# Phase 6: inference
## 6.1 Tiling
| | nnDetection | det3d |
|---|---|---|
| Efficiency | 2 | 4 |
| Robustness (lesions at tile borders) | 5 | 3 |
| **Total** | **7** | **7** |

**Winner:** tie — 3× the windows buys whole-lesion containment up to P/2 per axis instead of P/5–P/4; undecided without per-lesion sensitivity for lesions longer than P/4.

- **nnDetection**
  - Pro: `Predictor.tile_case` (N/nndet/inference/predictor.py), `overlap=0.5`, `create_grid(mode="symmetric")` (N/nndet/io/patching.py): every lesion up to half the patch per axis lies whole in some tile; overhanging tiles are shifted inside.
  - Con: 144 windows for a (354, 246, 180) crop (LBD median) against 48 at overlap 0.2–0.25 (both counted with the library grid functions), before 8× mirroring.
- **det3d**
  - Pro: MONAI `SlidingWindowInferer` (F/inference/base.py) batches windows; `needs_swi` (D/inference/patch.py) skips tiling when the crop fits one patch.
  - Con: overlap 0.2 (`infer_det.py`) or 0.25 (`infer_cascade.py`): in z (64 slices at 1.5 mm) a lesion longer than 13–16 slices (19–24 mm) can straddle every window border.
  - Con: `forward_tile` passes crops smaller than the patch unpadded; preprocessing keys "L,E,S,O,Norm,Dtype" contain no divisibility pad; behaviour for shapes not divisible by 8: UNCONFIRMED.

## 6.3 Per-patch filtering
| | nnDetection | det3d |
|---|---|---|
| Statistical soundness | 4 | 3 |
| Efficiency | 3 | 4 |
| **Total** | **7** | **7** |

**Winner:** tie — undecided without a measurement (FROC with det3d's per-window values set to 0 / 0.6 / 100).

- **nnDetection**
  - Pro: `postprocess_detections_single_image` (N/nndet/core/retina.py) with defaults from `RetinaUNetModule.from_config_plan` (score > 0, top 10,000, NMS 0.6, 100): selection is left to whole-case steps.
  - Con: these defaults are neither planned nor swept; `probs.sort` sorts all anchor × class scores before the top-k cut.
- **det3d**
  - Pro: `plan_architecture_from_det3d` (D/managers/helpers/nndet_retinaunet.py) copies 0.02 / 1,000 / 0.22 / 25, so fewer boxes reach the tile merge.
  - Con: score > 0.02 and 25 boxes per window prune the low-score tail before windows meet; FROC at 4–8 FP/scan reads that tail.
  - Con: the same full sort runs over 14.4 M × 3 scores per window (2.5).

## 6.4 Merging tiles within one set
| | nnDetection | det3d |
|---|---|---|
| Correctness | 5 | 1 |
| Efficiency | 4 | 4 |
| **Total** | **9** | **5** |

**Winner:** nnDetection — det3d computes the position weight from crop coordinates, which gives weights below 0.5 and negative values and inverts the NMS ranking.

- **nnDetection**
  - Pro: `BoxEnsemblerSelective.process_batch` (N/nndet/inference/ensembler/detection.py) calls `_get_box_in_tile_weight` on tile-local centres before `_apply_offsets_to_boxes`; boxes are clipped to the patch first, so weights lie in [0.5, 1] (code over trace, which says centres outside the tile go below 0.5).
  - Pro: `batched_weighted_nms_model` (N/nndet/inference/detection/model.py) ranks by score × weight and returns raw scores.
  - Con: class-wise NMS (`batched_nms` with `idxs=labels`); one lesion may survive once per class.
- **det3d**
  - Con: `swi_predictor` (D/inference/patch.py) calls `offset_boxes` before `box_tile_weight`; the weight `1 − clamp(d/|c| − 0.5, min=0)` has no lower bound, so tiles whose centre lies more than 1.5 |c| = 144 voxels from the crop origin (|c| = 96 for 128×128×64) give negative weights.
  - Con: `batched_weighted_nms_model` then keeps, among overlapping duplicates, the box from the tile nearest the crop origin and, among negative weights, the lower score; the reported score is the raw one.
  - Con: class-wise NMS and a cap of 25 per crop (`filter_boxes`), while the scorer is class-agnostic (7.3).

## 6.8 Restore to image space
| | nnDetection | det3d |
|---|---|---|
| Correctness | 4 | 5 |
| Robustness | 4 | 4 |
| **Total** | **8** | **9** |

**Winner:** det3d — it replays the recorded resampling and orientation operations with exact shape ratios and writes world coordinates; upstream scales by the spacing ratio.

- **nnDetection**
  - Pro: `restore_detection` (N/nndet/inference/restore.py): permute back, scale, add crop offset; three vector operations.
  - Con: scaling uses `spacing_after_resampling / original_spacing`, while the resampled shape was rounded (`resample_patient`, U/preprocessing/preprocessing.py); error ≤ 0.5 resampled voxel at the far boundary. Output stays in array index space.
- **det3d**
  - Pro: `inverse_preprocess_points_from_ops` (D/inference/post.py) inverts `SpatialResample` by the recorded `orig_size` / current shape and `Orientation` by nibabel orientation transforms, LIFO over `applied_operations`.
  - Pro: `Offd` adds the localiser crop start, `BoxRd` restores the source orientation, `SaveDetOutputd` adds world boxes (`AffineBoxToWorldCoordinated`).
  - Con: saved `bbox_voxel_full` keeps the trained min−1 lower bound (5.2; `nndet_xyxyzz_to_xyzxyz` applies no ±1), while the scorer's GT uses `slice.start` (7.3): one voxel offset, inferred from code, not measured.

# Phase 7: evaluation
## 7.1 In-training metric
| | nnDetection | det3d |
|---|---|---|
| Correctness | 5 | 4 |
| Statistical soundness | 3 | 4 |
| **Total** | **8** | **8** |

**Winner:** tie — same `BoxEvaluator`; the differences come from the patches (4.7) and the per-patch thresholds (6.3).

- **nnDetection**
  - Pro: `RetinaUNetModule.evaluation_end` (N/nndet/ptmodule/retinaunet/base.py) with `BoxEvaluator.create(fast=True)` (N/nndet/evaluator/det.py): mAP over IoU 0.10–0.50 and FROC at 0.1 / 0.5.
  - Con: patches redrawn each epoch add sampling noise to the monitored key.
- **det3d**
  - Pro: `RetinaUNetManager.validation_step`, `on_validation_epoch_end` (D/managers/retinaunet.py) call the nested module's `box_evaluator` on fixed patches.
  - Con: detections pass the sheet thresholds first (score > 0.02, 25 per patch), cutting the precision–recall curve; with `patch_stream`, `val0_metric` becomes segmentation Dice.

## 7.2 Whole-case validation scoring
| | nnDetection | det3d |
|---|---|---|
| Correctness (provenance) | 4 | 2 |
| Efficiency | 4 | 2 |
| **Total** | **8** | **4** |

**Winner:** nnDetection — prediction, raw-state saving, sweep and scoring run as one recorded step per fold; det3d's two manual steps record neither run nor fold and keep no pre-filter boxes.

- **nnDetection**
  - Pro: `_train` with `--sweep` (N/scripts/train.py): `module.sweep` saves per-set states, `extract_results` re-fuses without re-running the network, `_evaluate` scores original and preprocessed space.
  - Con: opt-in flag; sweep predictions run 8 mirrors per case.
- **det3d**
  - Con: `D/run/inference/infer_det.py`, then `LA/run/score_froc.py` `main`, which writes no run or fold identifier.
  - Con: sidecars hold boxes after per-window and tile filtering (6.3, 6.4); any NMS change needs re-inference.
  - Pro: per-case JSON sidecars (`save_inference_sidecar`, D/utils/bbox_sidecar.py) carry voxel and world boxes for any scorer.

## 7.3 FROC
| | nnDetection | det3d |
|---|---|---|
| Correctness | 4 | 3 |
| Statistical soundness | 3 | 5 |
| Robustness | 3 | 4 |
| **Total** | **10** | **12** |

**Winner:** det3d — case bootstrap and paired differences give an interval for every CPM and every comparison, and ignored GT works where upstream's `compute` fails.

- **nnDetection**
  - Pro: `FROCMetric` (N/nndet/evaluator/detection/froc.py) pools all cases after per-class greedy matching (`matching_batch`, N/nndet/evaluator/detection/matching.py).
  - Con: `compute` drops ignored detections with `dt_matches[...].reshape(old_shape)`, which raises when any detection is ignored.
  - Con: point estimate only; `max_detections = 100` per case and class (`DetectionEvaluator`, N/nndet/evaluator/det.py).
- **det3d**
  - Pro: `score_froc.py` (LA/run/score_froc.py) reuses `matching_batch` and `compute_froc_curve_one_iou`, removes ignored detections per case (`case_arrays`), and adds a 10,000-sample case bootstrap with paired CPM differences (`main`).
  - Pro: GT components under 3 mm equivalent diameter are ignored, neither TP nor FP (`gt_boxes`).
  - Con: GT rule (26-connected, any label > 0, 3 mm) differs from the training targets (per class, face-connected, 2 mm major axis; 3.5); class-agnostic matching of class-wise NMS output counts a second-class duplicate on a matched lesion as FP.
