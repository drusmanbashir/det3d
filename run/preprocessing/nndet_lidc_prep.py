#!/usr/bin/env python3
"""Native nnDetection LIDC prep: FRAN nifti → Task012_LIDC → crop/analyze/plan/process.

Source layout (default /media/UB/datasets/lidc_all):
  images/lidc_XXXX.nii.gz
  lms/lidc_XXXX.nii.gz
  label_analysis/lesion_stats.csv

Writes under $det_data/Task012_LIDC/ (default det_data=/r/datasets/nndet_data):
  dataset.json, raw_splitted/, preprocessed/ (+ plan.pkl, splits_final.pkl)

Requires nnDetection on PYTHONPATH (/home/ub/code/nnDetection) and dl or nndet env.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

NNDET_ROOT = Path("/home/ub/code/nnDetection")
DEFAULT_DET_DATA = Path("/r/datasets/nndet_data")
DEFAULT_DET_MODELS = Path("/s/agent_rw/nndet_models")
DEFAULT_SOURCE = Path("/media/UB/datasets/lidc_all")
TASK_NAME = "Task012_LIDC"


def setup_nndet_env(det_data: Path, det_models: Path) -> None:
    os.environ["det_data"] = str(det_data)
    os.environ["det_models"] = str(det_models)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("det_num_threads", "4")
    os.environ.setdefault("det_verbose", "1")
    if str(NNDET_ROOT) not in sys.path:
        sys.path.insert(0, str(NNDET_ROOT))
    import nndet.compat  # noqa: F401


def group_lesion_rows(lesion_stats_csv: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with lesion_stats_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["processing_error"] in {"True", "true", "1"}:
                continue
            grouped[row["case_id"]].append(row)
    return grouped


def lm_rows_to_instance_seg(lm_arr: np.ndarray, lesion_rows: list[dict]) -> tuple[np.ndarray, dict]:
    final = np.zeros_like(lm_arr, dtype=np.uint8)
    instances: dict[str, int] = {}
    rix = 1
    for row in lesion_rows:
        org = int(row["label_org"])
        cc_id = int(row["label_cc"])
        mask = lm_arr == org
        cc, n = ndimage.label(mask)
        if cc_id < 1 or cc_id > n:
            continue
        inst = cc == cc_id
        if inst.sum() == 0:
            continue
        final[inst] = rix
        instances[str(rix)] = 0
        rix += 1
    return final, {"instances": instances}


def convert_lidc_to_raw_splitted(
    source_dir: Path,
    task_dir: Path,
    lesion_stats_csv: Path,
) -> list[str]:
    from nndet.io.load import save_json

    images_dir = source_dir / "images"
    lms_dir = source_dir / "lms"
    target_data_dir = task_dir / "raw_splitted" / "imagesTr"
    target_label_dir = task_dir / "raw_splitted" / "labelsTr"
    target_data_dir.mkdir(parents=True, exist_ok=True)
    target_label_dir.mkdir(parents=True, exist_ok=True)

    grouped = group_lesion_rows(lesion_stats_csv)
    case_ids = sorted(
        p.name[: -len(".nii.gz")]
        for p in images_dir.glob("*.nii.gz")
        if (lms_dir / p.name).is_file()
    )
    converted: list[str] = []

    for case_id in case_ids:
        image_path = images_dir / f"{case_id}.nii.gz"
        lm_path = lms_dir / f"{case_id}.nii.gz"
        img = sitk.ReadImage(str(image_path))
        shutil.copy2(image_path, target_data_dir / f"{case_id}_0000.nii.gz")

        lm = sitk.ReadImage(str(lm_path))
        lm_arr = sitk.GetArrayFromImage(lm)
        seg_arr, label_json = lm_rows_to_instance_seg(lm_arr, grouped[case_id])
        seg_itk = sitk.GetImageFromArray(seg_arr)
        seg_itk.CopyInformation(img)
        sitk.WriteImage(seg_itk, str(target_label_dir / f"{case_id}.nii.gz"))
        save_json(label_json, target_label_dir / f"{case_id}.json")
        converted.append(case_id)

    write_dataset_json(task_dir)
    converted_set = set(converted)
    for path in target_data_dir.glob("*_0000.nii.gz"):
        case_id = path.name[: -len("_0000.nii.gz")]
        if case_id not in converted_set:
            path.unlink()
    return converted


def paired_raw_splitted_case_ids(task_dir: Path) -> list[str]:
    images_dir = task_dir / "raw_splitted" / "imagesTr"
    labels_dir = task_dir / "raw_splitted" / "labelsTr"
    case_ids = []
    for path in sorted(images_dir.glob("*_0000.nii.gz")):
        case_id = path.name[: -len("_0000.nii.gz")]
        if (labels_dir / f"{case_id}.nii.gz").is_file() and (labels_dir / f"{case_id}.json").is_file():
            case_ids.append(case_id)
        else:
            path.unlink()
    return case_ids


def write_dataset_json(task_dir: Path) -> None:
    from nndet.io.load import save_json

    meta = {
        "name": "LIDC",
        "task": TASK_NAME,
        "target_class": None,
        "test_labels": False,
        "labels": {"0": "nodule"},
        "modalities": {"0": "CT"},
        "dim": 3,
    }
    save_json(meta, task_dir / "dataset.json")


def fran_lidc_splits(
    project_title: str,
    plan_id: int,
    fold: int,
    available_case_ids: set[str],
) -> list[dict]:
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers import Project
    from fran.preprocessing.preprocessor import resolve_plan_datasources
    from nndet.io.load import save_json, save_pickle

    project = Project(project_title=project_title)
    config_maker = ConfigMakerDet(project)
    config_maker.setup(plan_id)
    plan = config_maker.configs["plan_train"]
    datasources = resolve_plan_datasources(plan)
    ds_query = datasources if len(datasources) > 1 else datasources[0]
    train_ids, val_ids = project.get_train_val_case_ids(fold=fold, ds=ds_query)
    train_ids = [case_id for case_id in train_ids if case_id in available_case_ids]
    val_ids = [case_id for case_id in val_ids if case_id in available_case_ids]
    splits = [{"train": train_ids, "val": val_ids}]
    preprocessed_dir = Path(os.environ["det_data"]) / TASK_NAME / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(splits, preprocessed_dir / "splits_final.pkl")
    save_json(splits, preprocessed_dir / "splits_final.json")
    return splits


def run_nndet_preprocess(
    num_processes: int,
    num_processes_preprocessing: int,
    overwrite: bool,
) -> None:
    from copy import deepcopy

    from hydra import initialize_config_module
    from omegaconf import OmegaConf

    from nndet.utils.config import compose
    from scripts.preprocess import run

    initialize_config_module(config_module="nndet.conf", version_base="1.1")
    overrides = []
    if overwrite:
        overrides.append("prep.overwrite=True")
    cfg = compose(TASK_NAME, "config.yaml", overrides=overrides)
    run(
        OmegaConf.to_container(cfg, resolve=True),
        num_processes=num_processes,
        num_processes_preprocessing=num_processes_preprocessing,
    )


def main():
    parser = argparse.ArgumentParser(description="Native nnDetection LIDC planning + preprocessing")
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--det-data", type=Path, default=DEFAULT_DET_DATA)
    parser.add_argument("--det-models", type=Path, default=DEFAULT_DET_MODELS)
    parser.add_argument("--project", default="lidc")
    parser.add_argument("--plan", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-prep", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="overwrite cropped/preprocessed data")
    parser.add_argument("-np", "--num-processes", type=int, default=4)
    parser.add_argument("-npp", "--num-processes-preprocessing", type=int, default=3)
    args = parser.parse_args()

    setup_nndet_env(args.det_data, args.det_models)
    task_dir = args.det_data / TASK_NAME
    task_dir.mkdir(parents=True, exist_ok=True)

    lesion_stats_csv = args.source_dataset / "label_analysis" / "lesion_stats.csv"
    if args.skip_convert:
        converted = paired_raw_splitted_case_ids(task_dir)
        write_dataset_json(task_dir)
        splits = fran_lidc_splits(args.project, args.plan, args.fold, set(converted))
        print(f"raw_splitted {len(converted)} paired cases")
        print(f"split fold {args.fold}: train={len(splits[0]['train'])} val={len(splits[0]['val'])}")
    else:
        converted = convert_lidc_to_raw_splitted(args.source_dataset, task_dir, lesion_stats_csv)
        splits = fran_lidc_splits(args.project, args.plan, args.fold, set(converted))
        print(f"converted {len(converted)} cases")
        print(f"split fold {args.fold}: train={len(splits[0]['train'])} val={len(splits[0]['val'])}")

    if not args.skip_prep:
        run_nndet_preprocess(
            num_processes=args.num_processes,
            num_processes_preprocessing=args.num_processes_preprocessing,
            overwrite=args.overwrite,
        )
        plan_path = task_dir / "preprocessed" / "D3V001_3d.pkl"
        print(f"preprocessing done; plan at {plan_path}")


if __name__ == "__main__":
    main()
