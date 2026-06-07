import csv
import json
from ast import literal_eval
from pathlib import Path
from random import Random

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine
from tqdm import tqdm


def voxel_spacing_from_affine(affine):
    return np.linalg.norm(affine[:3, :3], axis=0)


def voxel_bbox_to_world_box(bbox, affine):
    starts = np.array(bbox[:3], dtype=np.float64)
    sizes_vox = np.array(bbox[3:], dtype=np.float64)
    spacing = voxel_spacing_from_affine(affine)
    center_vox = starts + 0.5 * sizes_vox
    center_world = apply_affine(affine, center_vox).tolist()
    size_world = (sizes_vox * spacing).tolist()
    return center_world + size_world


def load_lesion_rows(lesion_stats_csv, dusting_mm=None):
    rows = []
    with open(lesion_stats_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("processing_error", "False") in {"True", "true", "1"}:
                continue
            major_axis = float(row["major_axis"])
            if dusting_mm is not None and major_axis < dusting_mm:
                continue
            rows.append(row)
    return rows


def group_rows_by_case(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)
    return grouped


def split_case_ids_fallback(case_ids, validation_fraction, seed):
    case_ids = sorted(case_ids)
    rng = Random(seed)
    rng.shuffle(case_ids)
    n_validation = int(round(len(case_ids) * validation_fraction))
    if validation_fraction > 0 and n_validation == 0:
        n_validation = 1
    validation_ids = set(case_ids[:n_validation])
    return validation_ids


def build_detection_json(
    images_dir,
    lesion_stats_csv,
    dataset_json,
    train_case_ids=None,
    val_case_ids=None,
    validation_fraction=0.05,
    seed=0,
    foreground_class_id=0,
    image_path_prefix="",
    remapping_train=None,
    dusting_mm=None,
):
    images_dir = Path(images_dir)
    lesion_stats_csv = Path(lesion_stats_csv)
    dataset_json = Path(dataset_json)
    rows = load_lesion_rows(lesion_stats_csv, dusting_mm=dusting_mm)
    grouped = group_rows_by_case(rows)

    train_set = set(train_case_ids or [])
    val_set = set(val_case_ids or [])
    use_project_split = len(train_set) > 0 or len(val_set) > 0
    if use_project_split and len(train_set) == 0 and len(val_set) == 0:
        use_project_split = False

    if not use_project_split:
        val_set = split_case_ids_fallback(list(grouped.keys()), validation_fraction, seed)
        train_set = set(grouped.keys()) - val_set

    payload = {"training": [], "validation": []}
    errors = []
    cases_total = 0

    for case_id in tqdm(sorted(grouped.keys()), desc="build_detection_json"):
        image_path = images_dir / f"{case_id}.nii.gz"
        if not image_path.exists():
            errors.append({"case_id": case_id, "error": f"missing image {image_path}"})
            continue
        try:
            image = nib.load(str(image_path))
            boxes = []
            labels = []
            for row in grouped[case_id]:
                bbox = literal_eval(row["bbox"])
                boxes.append(voxel_bbox_to_world_box(bbox, image.affine))
                if remapping_train is None:
                    labels.append(foreground_class_id)
                else:
                    label_org = int(row["label_org"])
                    labels.append(remapping_train[label_org])
            if len(boxes) == 0:
                errors.append({"case_id": case_id, "error": "no boxes after filtering"})
                continue
            rel_image = str(Path(image_path_prefix) / image_path.name) if image_path_prefix else str(image_path)
            entry = {"image": rel_image, "box": boxes, "label": labels}
            split = "validation" if case_id in val_set else "training"
            if use_project_split and case_id not in train_set and case_id not in val_set:
                continue
            payload[split].append(entry)
            cases_total += 1
        except Exception as exc:
            errors.append({"case_id": case_id, "error": str(exc)})

    dataset_json.parent.mkdir(parents=True, exist_ok=True)
    dataset_json.write_text(json.dumps(payload, indent=4))
    summary = {
        "images_dir": str(images_dir),
        "lesion_stats_csv": str(lesion_stats_csv),
        "dataset_json": str(dataset_json),
        "cases_total": cases_total,
        "training_cases": len(payload["training"]),
        "validation_cases": len(payload["validation"]),
        "errors": len(errors),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "foreground_class_id": foreground_class_id,
        "train_case_ids": sorted(train_set),
        "validation_case_ids": sorted(val_set),
    }
    dataset_json.with_suffix(".summary.json").write_text(
        json.dumps({"summary": summary, "errors": errors}, indent=4)
    )
    if len(payload["training"]) == 0 and len(payload["validation"]) == 0:
        raise RuntimeError(f"detection json generation produced no cases ({len(errors)} errors)")
    return summary
