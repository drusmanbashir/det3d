# FUNCTIONS.md

## run/preprocessing

- `build_detection_json.py` — build MONAI detection JSON from lesion_stats.csv and fran fold splits

## run/preproc

- `object_bounded.py` — object-bounded detection preprocessing CLI (`--project`, `--plan`, `--overwrite`, `--num-processes`, `--case-ids`, optional `--input-folder`)
- `label_bounded.py` — label-bounded detection preprocessing CLI (same flags; crops to `plan.lbd_crop_label`, one volume per case)

## det/preprocessing

- `object_bounded.py` — `ObjectBoundedDataGenerator` + `_OBJWorker`: fixed_spacing PT in; strict bbox crop (`expand_by=0`); compose `LoadT,Chan,Dev,Stats,N2P,AttachGT,Int`; patch size native — batch pad in `obd_det_collate` → `images/`, `lms/`, `bboxes/*_bboxN.json`
- `label_bounded.py` — `LabelBoundedDetDataGenerator` + `_LBDDetWorker`: fixed_spacing PT in → label crop/remap → `DetectionBBoxStatsd` → one `images/`, `lms/`, `bboxes/{case}.json` per case; optional HDF5 shards; postprocess writes `labels_all.json` only
- `bbox_sidecar.py` — `save_detection_sidecar` / `load_detection_sidecar`: RetinaNet-style `{box, label}` JSON sidecars

## det/utils

- `folder_names.py` — `obd_folder_from_plan(project, plan)` → `project.obd_folder/...`; `lbd_det_folder_from_plan(project, plan)` → `project.lbd_folder/...`

## det/geometry

- `lmg.py` — `DetectionLabelMapGeometryPT.to_voxel_detection_records`: ITK `[idx,idx,idx,size,size,size]` → `gt_box_mode` voxel box

## det/transforms

- `bbox_stats.py` — `AttachDetectionGTd`: post-crop LMG GT on `data['box']`/`data['label']`
- `patch_size.py` — `NbrhoodsToPatchesOBDD`: strict lesion bbox N2P + 4D channel

## det/detection

- `retinanet_train.py` — `forward_train_batched`: training loss on DM-prebatched tensors (no detector preprocess)

## det/managers

- `data.py` — `DataManagerDetOBD` (per-lesion patches), `DataManagerDetLBD` (full label-bounded volumes, shard-aware), `DataManagerDualDet` (OBD train + LBD val); `DataManagerDet` alias → OBD

## run/training

- `train.py` — Lightning RetinaNet training entrypoint
