# FUNCTIONS.md

## run/preprocessing

- `build_detection_json.py` — build MONAI detection JSON from lesion_stats.csv and fran fold splits
- `nndet_lidc_prep.py` — native nnDetection LIDC prep: FRAN nifti (`/media/UB/datasets/lidc_all`) → `Task012_LIDC` raw_splitted + fran splits + `nndet_prep` crop/analyze/plan/process (`--det-data`, `--skip-convert`, `--skip-prep`)

## run/preproc

- `object_bounded.py` — object-bounded detection preprocessing CLI (`--project`, `--plan`, `--overwrite`, `--num-processes`, `--case-ids`, optional `--input-folder`)
- `label_bounded.py` — label-bounded detection preprocessing CLI (same flags; one volume per case)

## det/preprocessing

- `object_bounded.py` — `ObjectBoundedDataGenerator` + `_OBJWorker`: fixed_spacing PT in; strict bbox crop (`expand_by=0`); compose `LoadT,Chan,Dev,Stats,N2P,AttachGT,Int`; patch size native — train `batch_size=1`, no collate/pad → `images/`, `lms/`, `bboxes/*_bboxN.json`
- `labelbounded.py` — `LabelBoundedDetDataGenerator` + `_LBDDetWorker`: fixed_spacing PT in → label crop/remap → `DetectionBBoxStatsd` (standard boxes) → `Stats,E,L,H` → `images/`, `masks/`, `bboxes/{case}.json` per case; postprocess writes `labels_all.json` + `dataset_details.csv`
- `dataset_details.py` — `dataset_details_from_mask_file`, `create_results_df_from_det_folder` (fran: `dataset_details_from_lm_file`, `create_results_df_from_lms_folder`); `write_dataset_details_csv`
- `bbox_sidecar.py` — `save_detection_sidecar` / `load_detection_sidecar`; `save_inference_sidecar` / `load_inference_sidecar`; `valid_detection_box` / `sidecar_bbox_empty`

## det/utils

- `folder_names.py` — `obd_folder_from_plan(project, plan)` → `project.obd_folder/...`; `lbd_det_folder_from_plan(project, plan)` → `project.lbd_folder/...`

## det/geometry

- `lmg.py` — `DetectionLabelMapGeometryPT.to_voxel_detection_records`: ITK `[idx,idx,idx,size,size,size]` → `gt_box_mode` voxel box

## det/transforms

- `bbox_stats.py` — `DetectionBBoxStatsd`: LMG GT on `data['bbox']`/`data['label']`; `AttachDetectionGTd`: post-crop LMG GT on `data['box']`/`data['label']` (OBD)
- `patch_size.py` — `NbrhoodsToPatchesOBDD`: strict lesion bbox N2P + 4D channel

## run/configs

- `build_experiment_configs_det.py` — build `~/code/fran/configurations/experiment_configs_det.xlsx` (model/loss/data params + plans_det)

## run/planning

- `advise_det_plan.py` — offline PlanAdvisorDet compare table vs manual plan (`--project`, `--plan-id`)

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
- `det_schedule.py` — `configure_detection_optimizers` (epoch_step | poly_iter)
- `retinanet_bk.py` — canonical `RetinaNetManager` (plan-driven loss/sampler)

## run/training

- `train.py` — Lightning det training entrypoint; `--arch retinanet|retinaunet`, `--nndet-forward-patch-size 128,128,64` (DM RandCrop patch; same DM for both arch), `--batch-tfms` → `TrainerDet.resolve_orchestrator_class`
- `nndet_env_dl.sh` — env vars for native nnDetection in conda `dl` (`det_data`, `det_models`, MLflow)
- `nndet_train_lidc.sh` — native nnDet LIDC training (`nndet_train Task012_LIDC`, forwards `-o` / `--sweep`)

## det3d/inference

- `patch.py` — `DetPatchInferer(BaseInferer)`: val-matched preprocess (`E,S,Norm,Dtype`), RetinaNetDetector forward
- `cascade.py` — `DetCascadeInferer` (RetinaNet bbox sidecar); `DetCascadeInfererRetinaUNet` (det+seg NIfTI)
- `transforms.py` — `OffsetBoxByBBoxd`, `ScaleBoxToCropNatived`, `PreservePreTfmBoxd`, `SaveInferenceSidecard`, `SaveInferenceMarkupsd`, `crop_around_boxes`; keyed post transforms for debug review
- `markups.py` — `bbox_world_ras_to_roi_lps`, `inference_sidecar_to_mrk_payload`, `save_inference_markups`; Slicer ROI Box `.mrk.json` from sidecar
- `visualize.py` — `view_inference_sidecar`, `save_sidecar_png`, `sidecar_pred_boxes`; load sidecar + overlay bboxes on slices
- `hybrid_lbd.py` — `build_hybrid_detector`, `infer_lbd_volume`, `save_lbd_pred_png`; Luna16-hybrid RetinaNet on LBD torch `.pt` volumes
- `hybrid_samples.py` — `run_hybrid_sample_infer`, `view_hybrid_sidecar`; infer 20 train + 20 val LBD volumes, save viewer JSON sidecars
- `retinaunet_nifti.py` — `run_lidc2_seg_infer`, `open_slicer_case`; RetinaUNet tiled seg infer on LIDC2 nifti cases (fixed_spacing .pt), export image+pred_seg NIfTI for Slicer
- `jaeger.py` — placeholder Jaeger logit aggregation (phase 5)
- `mirror_tta.py` — placeholder mirror TTA forward (phase 5)

## run/inference

- `infer_det.py` — cascade det infer CLI (`--run-p`, `--arch retinanet|retinaunet`, `--run-w`, `--lung-localiser`, `--project`, `--folder`/`--dataset`; `--lbd-folder` runs same DetPatchInferer on pre-cropped LBD `.pt`, no localiser)
- `infer_lbd_pt.py` — hybrid Luna16 RetinaNet on LBD torch `.pt` file or folder (`--model`, `--plan-json` or `--project`/`--plan-id`, `--input`/`--folder`, `--out-dir`; writes `{stem}_pred.png`)
- `infer_hybrid_samples.py` — hybrid DM infer on N train + N val cases (`--project`, `--plan-id`, `--model`, `--out-dir`; writes `{split}_{idx}_{case_id}.json` + `manifest.json`)
- `infer_retinaunet_lidc2_seg.py` — RetinaUNet seg infer on N LIDC2 nifti cases (`--ckpt`, `--out-dir`, `-n`, `--open-slicer`; writes `{case_id}_image.nii.gz`, `{case_id}_pred_seg.nii.gz`, `manifest.json`)
- `view_retinaunet_seg_slicer.py` — open seg NIfTI outputs in 3D Slicer (`--out-dir`, `--index`, `--list`)
- `view_hybrid_samples.py` — ImageBBoxViewer for hybrid sample sidecars (`--out-dir`, `--index`, `--show gt|pred|both`)
- `view_preds.py` — view stored sidecars only, no inference (`--index`, `--list`, `--dir` or `--run-p`; default `LIDC-TAINT` predictions)
- `sidecar_to_mrk.py` — batch sidecar JSON → Slicer ROI Box `.mrk.json` (`--dir` or `--run-p`, `--score-min`, `--overwrite`)
- `view_det.py` — alias for `view_preds.py`

