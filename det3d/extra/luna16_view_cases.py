"""Shared val-case index for luna16 NIfTI vs hybrid LBD viewers."""
import json
from pathlib import Path

from det3d.detection.lidc_datalist import load_lidc_train_val
from fran.data.dataregistry import DS

N_VIEW = 10
PRED_DIR = Path("/s/agent_rw/tmp/luna16_training2_nifti_preds")
SCORE_SUFFIX = "_score030"


def select_val_entries(val_data, n=N_VIEW):
    n = min(int(n), len(val_data))
    n_head = n // 2
    n_tail = n - n_head
    indices = list(range(n_head)) + list(range(len(val_data) - n_tail, len(val_data)))
    return [val_data[i] for i in indices], indices


def resolve_nifti(path_str):
    stem = Path(path_str).name
    if not stem.endswith(".nii.gz"):
        stem = f"{Path(path_str).stem}.nii.gz"
    for ds in (DS.lidc, DS.lidc2):
        candidate = ds.folder / "images" / stem
        if candidate.is_file():
            return candidate
    return Path(path_str)


def sidecar_path(case_id, pred_dir=PRED_DIR, use_score030=True):
    pred_dir = Path(pred_dir)
    if use_score030:
        return pred_dir / f"{case_id}{SCORE_SUFFIX}.json"
    return pred_dir / f"{case_id}.json"


def list_val_cases(n=N_VIEW, project="lidca", plan_id=1, fold=0, pred_dir=PRED_DIR, use_score030=True):
    from det3d.inference.hybrid_samples import setup_hybrid_dm

    _, val_data, _meta = load_lidc_train_val(
        project_title="lidc",
        plan_id=1,
        ds_name="lidc_all",
        fold=0,
    )
    dm, _plan, _configs = setup_hybrid_dm(project, plan_id, fold=fold, batch_tfms=True)

    lbd_by_case = {}
    for row in dm.valid_manager.data:
        case_id = str(row["case_id"])
        if case_id not in lbd_by_case:
            lbd_by_case[case_id] = row

    cases = []
    val_sample, val_indices = select_val_entries(val_data, n=n)
    for val_idx, entry in zip(val_indices, val_sample):
        case_id = Path(entry["image"]).stem.replace(".nii", "")
        nifti_path = resolve_nifti(entry["image"])
        lbd_row = lbd_by_case[case_id]
        cases.append(
            {
                "val_idx": int(val_idx),
                "case_id": case_id,
                "nifti": nifti_path,
                "lbd_pt": Path(lbd_row["image"]),
                "lbd_row": lbd_row,
                "sidecar": sidecar_path(case_id, pred_dir=pred_dir, use_score030=use_score030),
            }
        )
    return cases, dm


def print_val_cases(cases):
    print(f"{'idx':>3}  {'val':>3}  {'case_id':<12}  {'nifti':<22}  {'lbd_pt':<22}  sidecar")
    for i, row in enumerate(cases):
        sidecar_name = row["sidecar"].name if row["sidecar"].is_file() else f"{row['case_id']}{SCORE_SUFFIX}.json (missing)"
        print(
            f"{i:>3}  {row['val_idx']:>3}  {row['case_id']:<12}  {row['nifti'].name:<22}  "
            f"{row['lbd_pt'].name:<22}  {sidecar_name}"
        )
