"""Build and load native nnDetection BoxC002 plans from FRAN LBD outputs."""

import hashlib
import json
from ast import literal_eval
import pickle
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from fran.configs.helpers import make_src_dims_from_patch_size, normalize_tree

from det3d.configs.nndet_bridge import load_nndet_train_cfgs
from det3d.managers.helpers.nndet_retinaunet import (
    apply_det3d_plan_to_nndet_model_cfg,
    ensure_nndet_importable,
    xyzxyz_exclusive_batch_to_nndet,
)
from fran.utils.bbox_sidecar import load_nbrhood_sidecar, nbrhood_df_to_detection_records
from fran.data.dataregistry import DS


def nndet_plan_path(project, plan_id):  #AI
    """Return the flat BoxC002 result path for a spreadsheet plan ID."""
    return Path(project.project_folder) / "plans" / f"{int(plan_id)}.pkl"


def nndet_plan_input_key(project, plan_train, lbd_folder):  #AI
    """Fingerprint selected source files and the final shapes and boxes for BoxC002."""
    lbd_folder = Path(lbd_folder)
    details_path = lbd_folder / "dataset_details.csv"
    digest = hashlib.sha256()
    recipe_keys = (
        "datasources",
        "spacing",
        "expand_by",
        "fg_indices_exclude",
        "ignore_labels",
        "ignore_labels_cc",
        "lm_group",
        "mode",
        "prezoom_scale",
        "spatial_dims",
        "n_input_channels",
        "encoder_start_channels",
        "encoder_max_channels",
        "matcher_num_candidates",
        "matcher_center_in_gt",
        "sampler_batch_size_per_image",
        "balanced_sampler_pos_fraction",
        "sampler_pool_size",
        "sampler_min_neg",
    )
    recipe = {key: plan_train[key] for key in recipe_keys if key in plan_train}
    recipe.update(
        {
            key: value
            for key, value in plan_train.items()
            if key.startswith(("remapping", "dusting"))
        }
    )
    recipe["preprocessing_patch_size"] = (
        plan_train["preprocessing_patch_size"]
        if "preprocessing_patch_size" in plan_train
        else plan_train["patch_size"]
    )
    normalized_recipe = normalize_tree(recipe, tuple_mode="list")
    encoded_recipe = json.dumps(
        normalized_recipe, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    digest.update(encoded_recipe.encode())
    details = pd.read_csv(details_path)
    source_datasets = [DS[name.strip()].ds for name in plan_train["datasources"].split(",")]
    ds_sql = ", ".join(f"'{name}'" for name in source_datasets)
    source_rows = project.sql_query(
        "SELECT ds, case_id, image, lm FROM datasources "
        f"WHERE test=0 AND ds IN ({ds_sql}) ORDER BY ds, case_id"
    )
    source_case_ids = {str(row[1]) for row in source_rows}
    details = details.loc[details["case_id"].astype(str).isin(source_case_ids)]
    missing = source_case_ids - set(details["case_id"].astype(str))
    if missing:
        raise ValueError(f"LBD outputs are missing {len(missing)} selected cases: {sorted(missing)}")
    digest.update(details.to_csv(index=False).encode())
    source_paths = sorted({Path(path) for row in source_rows for path in row[2:4]})
    lbd_paths = []
    for row in details.itertuples(index=False):
        stem = Path(str(row.fn_name)).stem
        lbd_paths.extend(
            [
                lbd_folder / "images" / f"{stem}.pt",
                lbd_folder / "lms" / f"{stem}.pt",
                lbd_folder / "bboxes" / f"{stem}.csv",
            ]
        )
    for path in tqdm(source_paths + lbd_paths, desc="Hash selected source and LBD files"):
        digest.update(str(path).encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_nndet_plan_from_lbd(project, plan_train, lbd_folder, *, spreadsheet_row=None, target_mem="RTX2080TI", gpu_id=0):  #AI
    """Run native D3V001/BoxC002 on final combined LBD shapes and boxes."""
    ensure_nndet_importable()
    from nndet.io.load import save_pickle
    from nndet.planning.architecture.boxes.c002 import BoxC002
    from nndet.planning.estimator import MemoryEstimatorDetection
    from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001
    lbd_folder = Path(lbd_folder)
    details = pd.read_csv(lbd_folder / "dataset_details.csv")
    spacing = np.asarray(plan_train["spacing"], dtype=float)
    source_datasets = [DS[name.strip()].ds for name in plan_train["datasources"].split(",")]
    ds_sql = ", ".join(f"'{name}'" for name in source_datasets)
    source_rows = project.sql_query(
        "SELECT ds, case_id FROM datasources "
        f"WHERE test=0 AND ds IN ({ds_sql}) ORDER BY ds, case_id"
    )
    source_case_ids = {str(row[1]) for row in source_rows}
    details = details.loc[details["case_id"].astype(str).isin(source_case_ids)]
    missing = source_case_ids - set(details["case_id"].astype(str))
    if missing:
        raise ValueError(f"LBD outputs are missing {len(missing)} selected cases: {sorted(missing)}")
    all_sizes = []
    cases = {}
    all_labels = set()
    for row in details.itertuples(index=False):
        case_id = str(row.case_id)
        if case_id in cases:
            raise ValueError(f"Duplicate LBD case ID: {case_id}")
        image_shape = np.asarray(literal_eval(row.shape) if isinstance(row.shape, str) else row.shape, dtype=int)
        if image_shape.size == 4 and image_shape[0] == 1:
            image_shape = image_shape[1:]
        sidecar_path = lbd_folder / "bboxes" / f"{Path(str(row.fn_name)).stem}.csv"
        sidecar = load_nbrhood_sidecar(sidecar_path)
        boxes, labels, _ = nbrhood_df_to_detection_records(sidecar)
        boxes = xyzxyz_exclusive_batch_to_nndet(
            np.stack([box.numpy() for box in boxes]) if boxes else np.empty((0, 6))
        ).cpu().numpy()
        labels = [int(label) for label in labels]
        all_labels.update(labels)
        all_sizes.append(image_shape)
        cases[case_id] = {
            "boxes": boxes,
            "original_spacing": spacing.copy(),
            "num_instances": dict(Counter(labels)),
        }
    if not cases:
        raise ValueError(f"No cases found in {lbd_folder / 'dataset_details.csv'}")
    if not all_labels:
        raise ValueError("BoxC002 requires foreground boxes in the selected LBD dataset")

    labels = sorted(all_labels)
    label_index = {label: index for index, label in enumerate(labels)}
    class_dct = {label_index[label]: str(label) for label in labels}
    counts = Counter()
    for case in cases.values():
        case_counts = {
            label_index[label]: count
            for label, count in case["num_instances"].items()
        }
        case["num_instances"] = case_counts
        counts.update(case_counts)
    properties = {
        "dim": 3,
        "modalities": {0: "CT"},
        "class_dct": class_dct,
        "num_instances": dict(counts),
        "all_ious": np.asarray([], dtype=np.float32),
        "class_ious": {label: np.asarray([], dtype=np.float32) for label in class_dct},
        "instance_props_per_patient": cases,
        "all_spacings": [spacing.copy() for _ in cases],
        "all_sizes": all_sizes,
    }
    model_cfg, _ = load_nndet_train_cfgs()
    model_cfg = apply_det3d_plan_to_nndet_model_cfg(model_cfg, plan_train)
    transpose = [0, 1, 2]
    transposed_spacing = spacing[transpose]
    resampled_shapes = [shape for shape in all_sizes]
    median_shape = np.median(np.vstack(resampled_shapes), axis=0)[transpose]

    with tempfile.TemporaryDirectory(prefix="boxc002-", dir="/s/agent_rw/tmp") as temp_dir:
        preprocessed_dir = Path(temp_dir)
        properties_dir = preprocessed_dir / "properties"
        properties_dir.mkdir()
        with (properties_dir / "dataset_properties.pkl").open("wb") as stream:
            pickle.dump(properties, stream)
        box = BoxC002(
            preprocessed_output_dir=preprocessed_dir,
            save_dir=preprocessed_dir / "analysis",
            network_cls=RetinaUNetV001,
            estimator=MemoryEstimatorDetection(target_mem=target_mem, gpu_id=gpu_id),
            model_cfg=model_cfg,
        )
        box.transpose_forward = transpose
        box.create_default_settings()
        box.process_properties()
        planned = box.plan(
            transpose_forward=transpose,
            target_spacing_transposed=transposed_spacing,
            median_shape_transposed=median_shape,
        )
    result = {
        "patch_size": [int(v) for v in planned["patch_size"]],
        "batch_size": int(planned["batch_size"]),
        "conv_kernels": planned["architecture"]["conv_kernels"],
        "strides": planned["architecture"]["strides"],
        "decoder_levels": planned["architecture"]["decoder_levels"],
        "anchors": planned["anchors"],
        "fg_labels": labels,
    }
    if spreadsheet_row is not None and str(spreadsheet_row["spacing"]).strip().lower() == "auto":
        result["target_spacing"] = spacing.tolist()
    out = nndet_plan_path(project, plan_train["plan_name"])
    out.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(result, out)
    return result


def load_nndet_plan(project, plan_id):  #AI
    """Load the computed BoxC002 fields for a plan, raising when it has not been generated."""
    with nndet_plan_path(project, plan_id).open("rb") as stream:
        result = pickle.load(stream)
    return result


def apply_nndet_plan(configs, computed):  #AI
    """Apply only planner-derived values to the spreadsheet-based runtime config."""
    for plan_key in ("plan_train", "plan_valid", "plan_test"):
        plan = configs[plan_key]
        plan["preprocessing_patch_size"] = list(plan["patch_size"])
        plan["patch_size"] = list(computed["patch_size"])
        plan["encoder_conv_kernels"] = computed["conv_kernels"]
        plan["encoder_strides"] = computed["strides"]
        plan["decoder_levels"] = tuple(computed["decoder_levels"])
        plan["nndet_anchors"] = computed["anchors"]
        plan["fg_labels"] = list(computed["fg_labels"])
        if "target_spacing" in computed:
            plan["spacing"] = list(computed["target_spacing"])
        plan["src_dims"] = make_src_dims_from_patch_size(plan["patch_size"])
    configs["dataset_params"]["batch_size"] = int(computed["batch_size"])
    return configs
