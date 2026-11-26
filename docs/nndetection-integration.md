# det3d versus nnDetection: what exists, what does not

Example throughout: project `lidca`, lungs plan 4, nnDetection `RetinaUNetV001` + `D3V001`.
Each row is one work group. "Detail" links the phase file; every phase file opens with a flowchart of both pipelines (owning file and class or method per step) and then one heading per work group.

Bugs and inconsistencies found in det3d / FRAN code during the trace: [integration/issues.md](integration/issues.md).

Scored comparison of the methods both sides implement (winner, pros and cons): [integration/comparison.md](integration/comparison.md).

Background: [nndetection-planning-algorithm.md](nndetection-planning-algorithm.md) (upstream planning in full), [nndetection-config-audit-lidca.md](nndetection-config-audit-lidca.md) (which sheet values reach the model).

Exists: **Yes** same method · **Partly** same idea, simpler or not wired · **Different** another method does the job · **No** absent.

| # | Phase | nnDetection step | det3d today | Exists | Detail |
|---|---|---|---|---|---|
| 1.1 | Data analysis | Per-case size, spacing and crop ratio collected into a dataset fingerprint | Per-case spacing and resampled shape summaries stored; no crop ratio; `spacing_median` stored as zeros on `lidca` | Partly | [1 § 1.1](integration/1-data-analysis.md#11-size-spacing-and-crop-fingerprint) |
| 1.2 | Data analysis | Instance box sizes, counts per class and overlaps, at original resolution, before planning | Per-instance rows written after preprocessing; nothing aggregates them | No | [1 § 1.2](integration/1-data-analysis.md#12-instance-box-statistics) |
| 1.3 | Data analysis | Foreground intensity mean, standard deviation, 0.5 / 99.5 percentiles | Same four statistics; only the percentiles are used | Yes | [1 § 1.3](integration/1-data-analysis.md#13-foreground-intensity-statistics) |
| 1.4 | Data analysis | Modality names select the normalisation scheme | Channel count typed in the sheet; one fixed scheme | No | [1 § 1.4](integration/1-data-analysis.md#14-modality-and-normalisation-scheme) |
| 2.1 | Planning | Target spacing from per-axis median, corrected for one coarse axis | Typed: `(0.8, 0.8, 1.5)` | No | [2 § 2.1](integration/2-planning.md#21-target-spacing) |
| 2.2 | Planning | Axis order: coarsest axis first, inverse saved | Fixed RAS x, y, z | No | [2 § 2.2](integration/2-planning.md#22-axis-order) |
| 2.3 | Planning | Patch size: isotropic in millimetres, shrunk to fit a GPU memory target at batch 4 | Typed: `[128, 128, 64]`; memory never consulted | No | [2 § 2.3](integration/2-planning.md#23-patch-size-and-memory-fit) |
| 2.4 | Planning | Stage count, per-axis kernels and strides from spacing and patch | `auto` gives five `[3,3,3]` stages for every dataset | No | [2 § 2.4](integration/2-planning.md#24-stage-count-kernels-and-strides) |
| 2.5 | Planning | Decoder levels that feed the head, from stage count | Typed `(1,2,3,4)`; a stride-list offset puts the first detection level at full resolution | Partly | [2 § 2.5](integration/2-planning.md#25-decoder-levels) |
| 2.6 | Planning | Channel widths 32 / 320, FPN and head 128 (constants upstream) | Same numbers | Yes | [2 § 2.6](integration/2-planning.md#26-channel-widths) |
| 2.7 | Planning | Anchors: three sizes per axis (27 per position), optimised on training boxes, scaled per level | Three typed shapes; 12 per position, same size on every level | No | [2 § 2.7](integration/2-planning.md#27-anchors) |
| 2.8 | Planning | Dummy-2D augmentation flag when the patch is strongly anisotropic | None | No | [2 § 2.8](integration/2-planning.md#28-dummy-2d-flag) |
| 2.9 | Planning | Second, low-resolution plan when boxes exceed the patch | None (upstream never trains it either) | No | [2 § 2.9](integration/2-planning.md#29-low-resolution-plan) |
| 3.1 | Preprocessing | Crop to the nonzero image region; whole scan kept | One crop per case around all raw labels (median 39% of the scan on `lidca`); lesion-free cases dropped | Different | [3 § 3.1](integration/3-preprocessing.md#31-crop) |
| 3.2 | Preprocessing | Transpose arrays, spacing and boxes to the planned axis order | Reorient to RAS | Different | [3 § 3.2](integration/3-preprocessing.md#32-transpose) |
| 3.3 | Preprocessing | Resample: cubic image, nearest label (separate coarse-axis rule switched off) | Trilinear image, nearest label | Partly | [3 § 3.3](integration/3-preprocessing.md#33-resampling) |
| 3.4 | Preprocessing | CT normalisation: clip to percentiles, subtract foreground mean, divide by foreground standard deviation, stored on disk | Clip to the same percentiles, rescale to 0–1, at load time | Different | [3 § 3.4](integration/3-preprocessing.md#34-intensity-normalisation) |
| 3.5 | Preprocessing | Instance mask with an instance → class map stored per case | Semantic mask; instances are connected components; small ones dropped | Different | [3 § 3.5](integration/3-preprocessing.md#35-instances-and-ground-truth-boxes) |
| 4.1 | Data loading | Five shuffled folds, fixed seed | Six sequential folds, no shuffle | Different | [4 § 4.1](integration/4-data-loading.md#41-splits) |
| 4.2 | Data loading | Epoch = 2,500 random batches | Epoch = one crop per case (about 148 batches) | Different | [4 § 4.2](integration/4-data-loading.md#42-epoch-length) |
| 4.3 | Data loading | Half of each batch holds a whole lesion at a random offset; lesion drawn uniformly over all lesions | 75% of crops overlap a lesion; case drawn first, then one of its lesions | Partly | [4 § 4.3](integration/4-data-loading.md#43-patch-sampling) |
| 4.4 | Data loading | Oversized patch cut first, rotated and scaled, then cropped to final size | Final-size patch augmented directly; `prezoom_scale` exists at 1.0 | Partly | [4 § 4.4](integration/4-data-loading.md#44-oversized-patch) |
| 4.5 | Data loading | Augmentation set: rotation, scale, noise, blur, brightness, contrast, two gammas, mirror on all axes | Flip on one axis, small affine, four intensity transforms at 0.1, blur and gamma at 0.15 | Partly | [4 § 4.5](integration/4-data-loading.md#45-augmentation-set) |
| 4.6 | Data loading | Boxes recomputed from the augmented instance mask | Boxes warped by their corners | Different | [4 § 4.6](integration/4-data-loading.md#46-boxes-after-augmentation) |
| 4.7 | Data loading | Validation on 100 random patches, half without a forced lesion | One lesion-centred patch per positive case | Different | [4 § 4.7](integration/4-data-loading.md#47-validation-patches) |
| 5.1 | Training | Anchor matching (ATSS) and hard-negative sampling | Same classes; sampler numbers typed in the sheet | Yes | [5 § 5.1](integration/5-training.md#51-matching-and-sampling) |
| 5.2 | Training | Losses: BCE + GIoU + foreground segmentation, all weight 1 | Same on `retinaunet`; sheet loss cells are not read on that path | Yes | [5 § 5.2](integration/5-training.md#52-losses) |
| 5.3 | Training | SGD with Nesterov; no weight decay on normalisation layers | Nesterov, separate weight-decay groups on `sota` | Yes | [5 § 5.3](integration/5-training.md#53-optimiser) |
| 5.4 | Training | Linear warm-up 4,000 iterations, then poly decay per iteration, 125,000 iterations | Same step schedulers on `sota`, with the decay horizon ending at the tail | Yes | [5 § 5.4](integration/5-training.md#54-learning-rate-schedule) |
| 5.5 | Training | Cyclic rate and epoch-start weight averaging | `auto` averages 10 of 60 epochs from scratch, zero of 50 with nnFoundation | Partly | [5 § 5.5](integration/5-training.md#55-weight-averaging) |
| 5.6 | Training | Final checkpoint is the model used everywhere | Averaged last for inference, raw best and permanent snapshots | Partly | [5 § 5.6](integration/5-training.md#56-checkpoint-choice) |
| 5.7 | Training | Five fold runs | One fold per run, no multi-fold runner | No | [5 § 5.7](integration/5-training.md#57-folds) |
| 6.1 | Inference | Tiling at 50% overlap, symmetric grid | MONAI sliding window at 20–25% overlap, only when the crop exceeds the patch | Partly | [6 § 6.1](integration/6-inference.md#61-tiling) |
| 6.2 | Inference | Eight mirrored variants per patch | None (placeholder file) | No | [6 § 6.2](integration/6-inference.md#62-test-time-mirroring) |
| 6.3 | Inference | Permissive per-patch filtering (score > 0, NMS 0.6, 100 boxes) | Strict typed values (0.02, 0.22, 25 boxes) | Different | [6 § 6.3](integration/6-inference.md#63-per-patch-filtering) |
| 6.4 | Inference | Position-in-tile weight and weighted NMS within one prediction set | Same functions reused; weight taken after the box offset, so it can go negative | Partly | [6 § 6.4](integration/6-inference.md#64-merging-tiles-within-one-set) |
| 6.5 | Inference | Weighted Box Clustering across mirrored variants and models | None; one prediction set per case | No | [6 § 6.5](integration/6-inference.md#65-fusion-across-sets) |
| 6.6 | Inference | Five ensembler parameters chosen by a sweep on validation predictions | Typed thresholds | No | [6 § 6.6](integration/6-inference.md#66-parameter-sweep) |
| 6.7 | Inference | Fold checkpoints consolidated and loaded as one ensemble | One checkpoint | No | [6 § 6.7](integration/6-inference.md#67-fold-ensemble) |
| 6.8 | Inference | Boxes restored to the original image grid | Inverse spacing, orientation and crop offset; world coordinates saved | Yes | [6 § 6.8](integration/6-inference.md#68-restore-to-image-space) |
| 7.1 | Evaluation | In-training metric: mean AP over IoU 0.10–0.50 on validation patches | Same evaluator on `retinaunet` | Yes | [7 § 7.1](integration/7-evaluation.md#71-in-training-metric) |
| 7.2 | Evaluation | Whole-case validation predictions written and scored after training | Separate inference run, then an offline scorer | Partly | [7 § 7.2](integration/7-evaluation.md#72-whole-case-validation-scoring) |
| 7.3 | Evaluation | FROC at IoU 0.1 and 0.5, pooled, 1/8–8 false positives per scan | Same FROC code at IoU 0.1, own ground-truth boxes, bootstrap added | Yes | [7 § 7.3](integration/7-evaluation.md#73-froc) |
| 7.4 | Evaluation | Full-mode AP over IoU 0.1–0.9, per class | None offline | No | [7 § 7.4](integration/7-evaluation.md#74-full-ap-and-per-class-metrics) |
| 7.5 | Evaluation | LUNA16: box centres to world-coordinate CSV for the official scorer | None; world boxes are saved per case | No | [7 § 7.5](integration/7-evaluation.md#75-luna16-official-scoring) |
