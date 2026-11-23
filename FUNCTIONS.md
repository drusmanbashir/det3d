# FUNCTIONS.md
Scope: `det3d/planning/c002.py` `det3d/transforms/box_tracker.py` `det3d/run/e2e.sh`

## run

- `det3d/run/e2e.sh` — end to end: create the project if missing (`DATASOURCES`, `MNEMONIC`), fran `analyze_resample.py --pipeline det` (preprocess + BoxC002 plan), then `training/train.py`. Defaults `PROJECT=lidc PLAN=1 FOLD=0 ARCH=retinaunet`; env `PREPROC=0` skips preprocessing, `OVERWRITE=1` rebuilds; extra args go to train.py

## run/preprocessing

- `det3d/run/preprocessing/binarize_lbd_lms.py` — collapse LBD `lms/*.pt` to 0=bg / 1=lesion; set `label_org=1` on bbox sidecars (`--project`, named or numeric `--plan-id`, `--lbd-folder`, `--dry-run`); shell `binarize_lbd_lms.sh`

## run/preproc

- `det3d/run/preproc/object_bounded.py` — object-bounded detection preprocessing CLI (`--project`, named or numeric `--plan`, `--overwrite`, `--num-processes`, `--case-ids`, optional `--input-folder`)

Detection LBD preprocessing uses `fran/run/preproc/analyze_resample.py --pipeline det`. It saves target spacing before resampling, then fits the CT intensity statistics and BoxC002 and fills the centre ranges.

## det/preprocessing

- `object_bounded.py` — `ObjectBoundedDataGenerator` + `_OBJWorker`: fixed_spacing PT in; strict bbox crop (`expand_by=0`); compose `LoadT,Chan,Dev,Stats,N2P,AttachGT,Int`; patch size native — train `batch_size=1`, no collate/pad → `images/`, `lms/`, `bboxes/*_bboxN.json`
- `labelbounded.py` — scratch entry only; LBD preprocessing lives in fran `LabelBoundedDataGenerator` (`hdf5_shard_mode=det`)
- `helpers.py` — re-exports from fran
- `bbox_sidecar.py` — nbrhood symbols from fran; local JSON inference sidecar helpers remain

## det/utils

- `folder_names.py` — `obd_folder_from_plan(project, plan)` → `project.obd_folder/...`; `lbd_det_folder_from_plan(project, plan)` → `project.lbd_folder/...`

## det/geometry

- `lmg.py` — `DetectionLabelMapGeometryPT.to_voxel_detection_records`: ITK `[idx,idx,idx,size,size,size]` → xyzxyz voxel box
- `nndet_boxes.py` — `to_nndet` / `from_nndet`: exclusive xyzxyz ↔ nnDetection xyxyzz (padded minima); `offset_nndet_xyxyzz_boxes` tile → volume; `nndet_pred_to_vis` prediction dict → det3d keys

## det/transforms

- `bbox_stats.py` — `DetectionBBoxStatsd`: LMG GT on `data['bbox']`/`data['label']`; `AttachDetectionGTd`: post-crop LMG GT on `data['box']`/`data['label']` (OBD)
- `patch_size.py` — `NbrhoodsToPatchesOBDD`: strict lesion bbox N2P + 4D channel

## run/nndet

- `local_plan_sources.yaml` — local-only image/lm roots → nnDet task mapping (`liver|kidneys|pancreas|colon`; no downloads)
- `discover_local_plan_sources.py` — report on-disk sources + plan pkl status
- `stage_local_msd.py` — symlink local FRAN `images/` + `lms/` into nnDet task layouts (Decathlon raw or KiTS raw_splitted)
- `generate_nndet_plan_silo.sh` — ephemeral `nndet_prep` silo → `$nndet_conf/plans/{mnemonic}/D3V001_3d.pkl`; supports `all` and `--force`
- `generate_nndet_plans_local.sh` — run silo for every mnemonic in `local_plan_sources.yaml` (skips existing real pkls)

## det/architectures

- `create_detector.py` — `detector_arch_from_conf`, `create_detector_from_conf` (fran `create_network` pattern; retinanet | retinaunet)

## Libraries (import, not run)

- Detection plan file `plans/det3d_plan<id>.pkl`. FRAN preprocessing stores spacing, intensity and foreground labels. BoxC002 stores GPU-specific patch, batch and model at training start.
- `det3d/planning/c002.py` — `LBDBoxC002`, nnDetection's BoxC002 on LBD crops: measures the encoder used for training, according to `plans.encoder`. Stores the fit by GPU memory class.
- `det3d/transforms/box_tracker.py` — track LBD boxes through crops and spatial augmentation. The detection workbook defaults to `hull`, which uses `hull_xyz` and checks mask occupancy. `corner` warps box corners. `remeasure` rebuilds boxes from the warped `inst` map. For missing tracker data, rerun detection LBD with `--overwrite`.

Detection training and inference use FRAN's `NormaliseClipd.from_plan`. The workbook's `intensity_norm` defaults to `scale01` and also accepts `zscore_fg`. Set `intensity_properties` in the workbook or fit them during detection preprocessing.

## det/detection

- `retinanet_train.py` — `forward_train_batched`: training loss on DM-prebatched tensors (no detector preprocess)
- `loss_config.py` — `apply_detector_loss_plan`, `apply_detector_sampler_plan` from plan columns
- `retinaunet_network.py` — `build_retinaunet_feature_extractor` (nnDetection encoder/UFPN vendored)
- `nnfoundation_encoder.py` — `ResEncEncoder` (FRAN `resenc_encoder_kwargs` on the plan's kernels/strides, nnDetection encoder interface), `RetinaUNetV001NNF` (`--pipeline det-fm`; weights via FRAN `load_nnfoundation_encoder`)
- `NNDET_PORT.md` — provenance and MONAI-first decision table

## det/collate

- `collate.py` — `attach_targets`, `det_val_collate`, `obd_det_collate`, `lbd_det_collate` (flatten multi-crop lists → pad/stack); batch dict uses `bbox`/`label` lists + `targets`

## det/managers

- `data/main.py` — `DataManagerDet` family: `DataManagerDetSource` (LBD train: load image/mask, Norm, spatial aug + point sync, `lbd_det_collate`), `DataManagerDetLBD` (val full volumes), Whole/Patch/RBD/Short
- `data/dual.py` — `DataManagerDualDet`: paired train/valid detection managers for `TrainerDet`; `data/batch_tfms.py` adds the GPU batch tail (`DataManagerDualDetBTfms`)
- `data/transforms.py` — HDF5 shard/crop/case loaders (`LoadHDF5Det*`), box-centred crops (`RandCropExtendedBBoxd`, `CropDetPatchd`, `BboxCenterCropSlicesd`), `PadDetPatchd`
- `detector_factory.py` — `resolve_detector_manager`, `build_detector_manager` (retinanet | retinaunet)
- `retinaunet.py` — `RetinaUNetManager`: nnDetection RetinaUNetV001 built from the nested configs (the only place nnDet's flat architecture dict is made); det3d batch → nnDet targets; nnDet loss logging
- `det3d/managers/det_schedule.py` — SGD with step-based linear warmup and polynomial decay when the configuration supplies `warm_iterations`. For `det-fm`, the detection workbook’s `finetuning` sheet takes precedence. Blank selects ReduceLROnPlateau.

## run/training

- `det3d/run/training/train.py` delegates to FRAN's training CLI and selects `det` by default. For `--project lidc --plan sota --pipeline det-fm`, use `plans.encoder=resenc` and the detection workbook’s `finetuning` sheet. `--epochs` substitutes `auto` or a positive integer for `plans.epochs`. `--num-iterations-per-epoch` sets sampled training batches and `--limit-val-batches` caps smoke validation. Transfer uses `--transfer true --run-name {source_run} --arch retinaunet --source-ckpt last`.
- `train_det.sh` runs HPC Slurm det training through `train_retry.py --pipeline det`. Defaults are bones plan 1, RetinaNet and run-through. The optional fifth positional argument supplies `--epochs`.
- `local_train_det.sh` runs the local detection CLI. The optional fifth positional argument supplies `--epochs`.
- `submit_train_det.sh` submits det training to HPC. Usage is `submit_train_det.sh [project] [plan] [arch] [epochs]`.
## det3d/inference

- `post.py` — `PackRetinaNetPredsd`, `PackRetinaUNetPredsd`, `InvPreprocessBoxd`, `Offd`, `BoxRd`, `SaveDetOutputd`
- `patch.py` — `DetPatchRetinaNet`, `DetPatchRetinaUNet`, `DetPatchLBD`
- `cascade.py` — `DetBBoxCascadeInferer`, `DetSegBBoxCascadeInferer`
- `lbd.py` — `DetLBDRunner` (LBD `.pt` patch-only infer)
- `lbd_pt.py` — load LBD tensors and prepare full-volume patch data
- `patch.py` / `cascade.py` — re-exports of Det inferers
- `transforms.py` — legacy keyed transforms (superseded by post for cascade)
- `markups.py` — Slicer ROI Box `.mrk.json` from sidecar
- `visualize.py` — sidecar overlay viewers
- `retinaunet_nifti.py` — RetinaUNet tiled seg infer on LIDC2 nifti cases
- `mirror_tta.py` — `inference_tta` (dataset_params row), `mirror_tta_step` (nnDetection 8-mirror TTA, boxes + seg inverted) for `DetPatchRetinaUNet`
- `jaeger.py` — placeholder (phase 5)

## run/inference

- `_infer_common.py` — shared helpers for inference CLIs: `default_run_w`, `resolve_input_images`, `resolve_localiser_labels`
- `infer_cascade.py` — RetinaUNet cascade infer (`--run-p`, `--run-w`, `--lung-localiser`, `--folder`/`--dataset`, `--project lidca`)
- `infer_det.py` — cascade / LBD det infer (`--run-p`, `--arch`, `--run-w`, `--lbd-folder`, `--folder`/`--dataset`)
- `infer_retinaunet_lidc2_seg.py` — RetinaUNet seg infer on N LIDC2 nifti cases
- `view_retinaunet_seg_slicer.py` — open seg NIfTI outputs in 3D Slicer
- `view_preds.py` — view stored sidecars only (`--dir` or `--run-p`)
- `sidecar_to_mrk.py` — batch sidecar JSON → Slicer ROI Box `.mrk.json`

## run/tools

- `affine_voxel_world_demo.py` — numeric affine forward/inverse demos (toy, two-grid same world, preproc InvB trap); see `docs/affine-voxel-world.md`

