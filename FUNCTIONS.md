# FUNCTIONS.md

## run/preprocessing

- `build_detection_json.py` — build MONAI detection JSON from lesion_stats.csv and fran fold splits

## run/preproc

- `object_bounded.py` — object-bounded detection preprocessing CLI (`--project`, `--plan`, `--overwrite`, `--num-processes`, `--case-ids`, optional `--input-folder`)
- `label_bounded.py` — label-bounded detection preprocessing CLI (same flags; crops to `plan.lbd_crop_label`, one volume per case)

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

## det/detection

- `retinanet_train.py` — `forward_train_batched`: training loss on DM-prebatched tensors (no detector preprocess)

## det/collate

- `collate.py` — `attach_targets`, `det_val_collate`, `obd_det_collate`, `lbd_det_collate` (flatten multi-crop lists → pad/stack); batch dict uses `bbox`/`label` lists + `targets`

## det/managers

- `data.py` — `DataManagerTrainDet` (LBD train: load image/mask, Norm, spatial aug + point sync, `lbd_det_collate`), `DataManagerTrainDetBTfms` / `DataManagerDetBTfms` (GPU batch tail), `DataManagerDetLBD` (val full volumes), `DataManagerDualDet` / `DataManagerDualDetBTfms`; `DataManagerDet` alias → TrainDet

## run/training

- `train.py` — Lightning RetinaNet training entrypoint; `--batch-tfms` → `TrainerDet.resolve_orchestrator_class`

## det3d/inference

- `patch.py` — `DetPatchInferer(BaseInferer)`: val-matched preprocess (`E,S,Norm,Dtype`), RetinaNetDetector forward
- `cascade.py` — `DetCascadeInferer(CascadeInferer)`: TotalSeg localiser + det patch inferer; stepwise box postprocess
- `transforms.py` — `OffsetBoxByBBoxd`, `ScaleBoxToCropNatived`, `PreservePreTfmBoxd`, `SaveInferenceSidecard`, `crop_around_boxes`; keyed post transforms for debug review
- `visualize.py` — `view_inference_sidecar`, `save_sidecar_png`, `sidecar_pred_boxes`; load sidecar + overlay bboxes on slices
- `hybrid_lbd.py` — `build_hybrid_detector`, `infer_lbd_volume`, `save_lbd_pred_png`; Luna16-hybrid RetinaNet on LBD torch `.pt` volumes

## run/inference

- `infer_det.py` — cascade detection inference CLI (`--run-p`, `--run-w`, `--folder`/`--dataset`, `--debug`)
- `infer_lbd_pt.py` — hybrid Luna16 RetinaNet on LBD torch `.pt` file or folder (`--model`, `--plan-json` or `--project`/`--plan-id`, `--input`/`--folder`, `--out-dir`; writes `{stem}_pred.png`)
- `view_preds.py` — view stored sidecars only, no inference (`--index`, `--list`, `--dir` or `--run-p`; default `LIDC-TAINT` predictions)
- `view_det.py` — alias for `view_preds.py`

