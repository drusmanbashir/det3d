# Parked items: nnDetection parity pathway

Goal: parity with nnDetection along Datasource → Planner (spacing + intensity) → LBD → BoxC002 → RetinaUNet.
Below are known issues that are not current blockers. Source: Codex audits of 2026-11-17 (code-read, nothing run).
F = fran/fran, D = det3d/det3d, N = nnDetection/nndet.

## Launch hygiene
- `--overwrite` does not overwrite HDF5 shards (`overwrite_hdf5_shards=False`, F/run/preproc/analyze_resample.py:314,344). Clear stale LIDC shards before the first run.
- Planned patch × prezoom must match the LBD crop manifest or the loader raises (D/managers/data/main.py:815).

## Legacy to retire (`Project.maybe_store_projectwide_properties`, `global_properties.json`)
- The FRAN config maker still injects legacy intensity keys (F/configs/parser.py).
- Stage A uses legacy mean, SD and clip bounds when `clip_center=True` (F/preprocessing/fixed_spacing.py).
- Standalone OBD preprocessing still uses legacy intensity scaling (D/preprocessing/object_bounded.py).
- Fold creation lives there: F/managers/project.py:1090. Project JSON metadata: F/managers/project.py:154,800,1165.

## Case selection and keying
- Stage-A query has no `test=0` filter, but folder identity excludes test cases: F/preprocessing/preprocessor.py:438 vs F/utils/folder_names.py:146.
- Bare case_id keys in shard lookup and BoxC002: D/managers/data/main.py:702,744; D/configs/nndet_planning.py:123,133. Datasources with the same prefix (lidc + lidc_short) collide.
- Parity runs need the same case list and folds: nnDet's old LIDC task had 608 cases and 5 folds; the project has 759 cases and 6 folds (F/managers/project.py:581; N/io/datamodule/base.py:126).
- Trainer overwrites classes from the folder manifest instead of validating them against the plan: D/trainers/trainerdet.py:289.

## Divergences by design or still undecided
- Resampling: trilinear vs nnDet cubic (F/preprocessing/fixed_spacing.py:130; N/preprocessing/preprocessor.py:379).
- Lung label 1 bounds the LBD crop (F/preprocessing/labelbounded.py:71,95).
- BoxC002 plans on post-LBD crop shapes; nnDet plans pre-resampling on nonzero-crop shapes (D/configs/nndet_planning.py:131,181; N/planning/experiment/base.py:219).
- D3V001 low-resolution plan is not produced.
- OBD has no auto-spacing, BoxC002 or trainer route (D/run/preproc/object_bounded.py:25).
- label_analysis `dust(remove_flat=True)` drops one-slice components from dataset_stats.csv.
- Per-label histograms cannot exclude dusted components inside a kept label.

## Training recipe (outside the planning pathway)
- Sampler, GPU augmentation, plateau schedule vs warmup/poly/SWA, bf16: D/managers/data/main.py:199,926; D/transforms/gpu_det.py:105,379; D/managers/det_schedule.py:5; D/trainers/trainerdet.py:212.

## Plan-variable inventory (Codex msg-20261117T215703-a5913f)
- Merged or imported labels (fran test plan 2 merge, import-only crops) add voxels that the Datasource statics don't hold. They need versioned external-mask geometry, or overlap summaries plus HU counts (F/preprocessing/imported.py:66,119,219).
- `remapping_train` is a fran loader mapping that RetinaUNet bypasses. `remapping_train_resnet` has no active reader. The `plans_dot` sheet is outside the active parsers.
- Remap order: `remapping_source` → LBD crop → `remapping_lbd_rbd` → component exclusion → dusting. A 2→0 map can still affect a crop taken before the remap.

## Astra review of uncommitted work (msg-20261117T233723-7e5163, 2026-11-17)
All latent for LIDC plan 1: the LIDC h5 and CSVs were rebuilt, and the LBD spacing baseline ran over all 759 cases.
- Existing H5 cases never gain `spatial_shape`/`affine`, and a missing `label_bboxes.csv` is never rebuilt when there are no new cases (F/data/datasource.py:222,362,412).
- Appending a case leaves the label_analysis CSV pair stale, because the cache returns it whenever both files exist (label_analysis/dataset_stats.py:143).
- A scan with no labels at all hits `max([])` at F/preprocessing/target_spacing.py:94, before the no-foreground fallback.
- Failed CSV rows for unselected cases still abort spacing (F/preprocessing/target_spacing.py:31).
- Fresh FRAN training and det source mode never load the resolved auto spacing (F/run/preproc/analyze_resample.py:54; D/configs/parser.py:370).
- RBD writer and loader derive different `src_dims`, so their folder hashes differ (F/preprocessing/labelbounded.py:275 vs F/configs/parser.py:245).
- FRAN's LBD completion check asks for a patch-derived shard tag, while the writer uses `lbd_crop_minimum` (F/configs/parser.py:96).
- Native BoxC002 anchor fitting reduces an empty tensor for tiny uniform-size cohorts (N/planning/architecture/boxes/base.py:414).
- The plan pickle is published before the centre-range columns are written (D/configs/nndet_planning.py:292).
- `from_nndet` turns a sub-voxel predicted box into an inverted box, because `remove_small_boxes` is 0 (D/managers/helpers/nndet_retinaunet.py:144; D/inference/patch.py:204).
- Sliding-window boxes clamped at a tile edge shift by one voxel, because tiles are offset in native space before `from_nndet` (D/inference/patch.py:259).
- label_analysis end-to-end stats replace `--ignore-labels` with registry organs and require a sibling `labels.yaml` (label_analysis/dataset_stats.py:330).
- Empty or organ-only cohorts break the whole_label export or plotting (label_analysis/geometry/multiproc.py:21; dataset_stats.py:193,240).
- nnDetection's `torch.amp` imports need a newer PyTorch than the README's pinned 1.11. The dl env is unaffected.

## Pending
- Parity harness design: /s/agent_rw/tmp/ccc/sessions/ccc-det3d-0a6ad8c8-2/packets/summary-msg-20261117T215514-a47082.md
- Plan-variable inventory (static vs per-plan): Codex handoff msg-20261117T215703-a5913f.
