# FUNCTIONS.md

## run/preprocessing

- `backfill_sidecar_instances.py` — add ``instances`` {lm_cc_id: semantic_label} to existing `bboxes/*.json` from matching `lms/*.pt` (`--project`, `--plan-id`, `--lbd-folder`, `--dry-run`)
- `binarize_lbd_lms.py` — collapse LBD `lms/*.pt` to 0=bg / 1=lesion; set `label_org=1` on bbox sidecars (`--project`, `--plan-id`, `--lbd-folder`, `--dry-run`); shell `binarize_lbd_lms.sh`

## run/preproc

- `object_bounded.py` — object-bounded detection preprocessing CLI (`--project`, `--plan`, `--overwrite`, `--num-processes`, `--case-ids`, optional `--input-folder`)

Detection LBD preproc: `fran/run/preproc/analyze_resample.py --pipeline det` (ConfigMakerDet; `LabelBoundedDetDataGenerator` is a thin alias of fran `LabelBoundedDataGenerator` with `hdf5_shard_mode=det`).

## det/preprocessing

- `object_bounded.py` — `ObjectBoundedDataGenerator` + `_OBJWorker`: fixed_spacing PT in; strict bbox crop (`expand_by=0`); compose `LoadT,Chan,Dev,Stats,N2P,AttachGT,Int`; patch size native — train `batch_size=1`, no collate/pad → `images/`, `lms/`, `bboxes/*_bboxN.json`
- `labelbounded.py` — thin re-export: `LabelBoundedDetDataGenerator` → fran `LabelBoundedDataGenerator` (`hdf5_shard_mode=det`)
- `hdf5_shards_det.py`, `dataset_details.py`, `helpers.py` — re-exports from fran
- `bbox_sidecar.py` — nbrhood symbols from fran; local JSON inference sidecar helpers remain

## det/utils

- `folder_names.py` — `obd_folder_from_plan(project, plan)` → `project.obd_folder/...`; `lbd_det_folder_from_plan(project, plan)` → `project.lbd_folder/...`

## det/geometry

- `lmg.py` — `DetectionLabelMapGeometryPT.to_voxel_detection_records`: ITK `[idx,idx,idx,size,size,size]` → xyzxyz voxel box

## det/transforms

- `bbox_stats.py` — `DetectionBBoxStatsd`: LMG GT on `data['bbox']`/`data['label']`; `AttachDetectionGTd`: post-crop LMG GT on `data['box']`/`data['label']` (OBD)
- `patch_size.py` — `NbrhoodsToPatchesOBDD`: strict lesion bbox N2P + 4D channel

## run/planning

- `advise_det_plan.py` — offline PlanAdvisorDet compare table vs manual plan (`--project`, `--plan-id`)

## run/nndet

- `local_plan_sources.yaml` — local-only image/lm roots → nnDet task mapping (`liver|kidneys|pancreas|colon`; no downloads)
- `discover_local_plan_sources.py` — report on-disk sources + plan pkl status
- `stage_local_msd.py` — symlink local FRAN `images/` + `lms/` into nnDet task layouts (Decathlon raw or KiTS raw_splitted)
- `generate_nndet_plan_silo.sh` — ephemeral `nndet_prep` silo → `$nndet_conf/plans/{mnemonic}/D3V001_3d.pkl`; supports `all` and `--force`
- `generate_nndet_plans_local.sh` — run silo for every mnemonic in `local_plan_sources.yaml` (skips existing real pkls)

## det/architectures

- `create_detector.py` — `detector_arch_from_conf`, `create_detector_from_conf` (fran `create_network` pattern; retinanet | retinaunet)

## det/detection

- `retinanet_train.py` — `forward_train_batched`: training loss on DM-prebatched tensors (no detector preprocess)
- `loss_config.py` — `apply_detector_loss_plan`, `apply_detector_sampler_plan` from plan columns
- `retinaunet_network.py` — `build_retinaunet_feature_extractor` (nnDetection encoder/UFPN vendored)
- `NNDET_PORT.md` — provenance and MONAI-first decision table

## det/collate

- `collate.py` — `attach_targets`, `det_val_collate`, `obd_det_collate`, `lbd_det_collate` (flatten multi-crop lists → pad/stack); batch dict uses `bbox`/`label` lists + `targets`

## det/managers

- `data.py` — `DataManagerTrainDet` (LBD train: load image/mask, Norm, spatial aug + point sync, `lbd_det_collate`), `DataManagerTrainDetBTfms` / `DataManagerDetBTfms` (GPU batch tail), `DataManagerDetLBD` (val full volumes), `DataManagerDualDet` / `DataManagerDualDetBTfms`; `DataManagerDet` alias → TrainDet
- `detector_factory.py` — `resolve_detector_manager`, `build_detector_manager` (retinanet | retinaunet)
- `retinaunet.py` — `RetinaUNetManager` Lightning module (vendored UFPN + MONAI RetinaNet head)
- `det_schedule.py` — `configure_detection_optimizers` (SGD + ReduceLROnPlateau from plan `scheduler_*` / `nndet` keys)
- `retinanet_bk.py` — canonical `RetinaNetManager` (plan-driven loss/sampler)

## run/training

- `train.py` — shim → `fran/run/training/train.py` (prepends `--pipeline det` when omitted; `TrainerDet` / `TrainerDetTransfer`); standard: `--project lidca --plan 4 --arch retinaunet`; transfer: `--transfer true --run-name {source_run} --arch retinaunet --source-ckpt last`; also `--run-through`, `--resume-lr`, `--batch-tfms`
- `train_det.sh` — HPC Slurm det training (`train_retry.py --pipeline det`); defaults bones plan 1, retinanet, run-through, 500 epochs
- `local_train_det.sh` — local det training via `train.py` shim (PYTHONPATH + bones defaults)
- `submit_train_det.sh` — HPC submit wrapper (optional)
## run/archived/training

Retired nnDet migration / benchmark CLIs (superseded by `run/training/train.py` + `TrainerDet`):

- `benchmark_det_pipelines.py` — native nnDet + det3d pipeline benchmarks (`run|report|all|sweep`)
- `train_hybrid_fast_lbd.py` — hybrid fast LBD → nnDet RetinaUNetV001 (`det3d.archived.hybrid`)
- `train_native_lbd.py` — native nnDet LBD materialize + pre_trafo (`det3d.archived.nndet_native_lbd`)
- `hybrid.sh`, `hybrid_fast_lbd_benchmark.sh` — launchers for hybrid fast LBD CLI

## det3d/inference

- `post.py` — `PackRetinaNetPredsd`, `PackRetinaUNetPredsd`, `InvPreprocessBoxd`, `Offd`, `BoxRd`, `SaveDetOutputd`
- `patch.py` — `DetPatchRetinaNet`, `DetPatchRetinaUNet`, `DetPatchLBD`
- `cascade.py` — `DetBBoxCascadeInferer`, `DetSegBBoxCascadeInferer`
- `lbd.py` — `DetLBDRunner` (LBD `.pt` patch-only infer)
- `lbd_pt.py` — `load_lbd_pt`, `load_lbd_pt_patch_data`, `normalize_lbd_image`
- `patch.py` / `cascade.py` — re-exports of Det inferers
- `transforms.py` — legacy keyed transforms (superseded by post for cascade)
- `markups.py` — Slicer ROI Box `.mrk.json` from sidecar
- `visualize.py` — sidecar overlay viewers
- `retinaunet_nifti.py` — RetinaUNet tiled seg infer on LIDC2 nifti cases
- `jaeger.py`, `mirror_tta.py` — placeholders (phase 5)

## det3d/extra/archived

- `hybrid_lbd.py` — Luna16-hybrid RetinaNet on LBD `.pt` (archived)
- `hybrid_samples.py` — hybrid sample infer + viewer sidecars (archived)
- `hybrid_transfer.py` — hybrid fast-LBD ckpt path helpers (archived)

## run/inference

- `_infer_common.py` — shared helpers for inference CLIs: `default_run_w`, `resolve_input_images`, `resolve_localiser_labels`
- `infer_cascade.py` — RetinaUNet cascade infer (`--run-p`, `--run-w`, `--lung-localiser`, `--folder`/`--dataset`, `--project lidca`)
- `infer_det.py` — cascade / LBD det infer (`--run-p`, `--arch`, `--run-w`, `--lbd-folder`, `--folder`/`--dataset`)
- `infer_retinaunet_lidc2_seg.py` — RetinaUNet seg infer on N LIDC2 nifti cases
- `view_retinaunet_seg_slicer.py` — open seg NIfTI outputs in 3D Slicer
- `view_preds.py` — view stored sidecars only (`--dir` or `--run-p`)
- `sidecar_to_mrk.py` — batch sidecar JSON → Slicer ROI Box `.mrk.json`
- `view_det.py` — alias for `view_preds.py`

## run/tools

- `affine_voxel_world_demo.py` — numeric affine forward/inverse demos (toy, two-grid same world, preproc InvB trap); see `docs/affine-voxel-world.md`

## run/archived/inference

- `infer_lbd_pt.py` — hybrid Luna16 RetinaNet on LBD `.pt` (archived)
- `infer_hybrid_samples.py` — hybrid DM sample infer (archived)
- `view_hybrid_samples.py` — hybrid sample viewer (archived)

