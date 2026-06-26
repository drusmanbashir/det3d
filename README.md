# det

MONAI 3D RetinaNet lung nodule detection on custom datasets, orchestrated with fran-style Lightning engines.

## Quick start

```bash
# Build MONAI detection JSON from lesion_stats.csv
python run/preprocessing/build_detection_json.py --project test --plan 1

# Train (1 epoch smoke: pass --epochs 1)
python run/training/train.py --project test --plan 1 --epochs 1 --devices 0
```

## Config

- `plans_det` sheet in `fran/configurations/experiment_configs_det.xlsx`
- Ground truth: `label_analysis/lesion_stats.csv` per datasource (one row per lesion)
- Splits: fran `Project` folds when cases exist in `cases.db`; otherwise plan `validation_fraction`

**Design docs** (patch plans, HDF5 reuse, `src_dims`): [`docs/README.md`](docs/README.md)

## Layout

- `det/preprocessing/build_json.py` — lesion_stats → MONAI datalist JSON
- `det/managers/data.py` — `DataManagerDet` / `DataManagerDualDet` (LBD volumes from `labelbounded.py`; RandCrop train pipeline + `bboxes/`)
- `det/managers/retinanet.py` — `RetinaNetManager` LightningModule
- `det/trainers/trainer.py` — `TrainerDet` orchestrator
