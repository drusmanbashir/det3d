"""Build ~/code/fran/configurations/experiment_configs_det.xlsx from plan spec."""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[2].parent / "fran" / "configurations" / "experiment_configs_det.xlsx"

PARAM_COLS = ["var_name", "tune_value", "tune_type", "tune", "manual_value", "notes"]

MODEL_PARAMS = [
    ("arch", "retinaunet", "choice", 0, "retinanet", "retinanet | retinaunet | retinaunet_v3"),
    ("spatial_dims", None, None, 0, 3, "both"),
    ("n_input_channels", None, None, 0, 1, "CT single channel"),
    ("class_name", None, None, 0, "nodule", "COCO metrics label only"),
    ("val_patch_size", None, None, 0, "[512,512,208]", "infer SW window"),
    ("gt_box_mode", None, None, 0, "cccwhd", "both"),
    ("returned_layers", "[2,3]", None, 0, "[1,2]", "retinanet only"),
    ("conv1_t_stride", None, None, 0, "[2,2,1]", "retinanet only"),
    ("base_anchor_shapes", "advisor output", None, 0, "[[6,8,4],[8,6,5],[10,10,6]]", "retinanet only"),
    ("encoder_start_channels", "16, 48", None, 0, 32, "retinaunet only"),
    ("encoder_conv_kernels", None, None, 0, "auto", "retinaunet only"),
    ("encoder_strides", None, None, 0, "auto", "retinaunet only"),
    ("decoder_levels", None, None, 0, "(1,2,3,4)", "retinaunet only"),
    ("encoder_max_channels", None, None, 0, 320, "retinaunet only"),
]

LOSS_PARAMS = [
    ("cls_loss", "focal", "choice", 0, "bce", "loss_params sheet"),
    ("reg_loss", "giou", "choice", 0, "smooth_l1", "retinanet default; retinaunet uses giou in code"),
    ("w_cls", None, None, 0, 1, "loss_params sheet"),
    ("w_reg", None, None, 0, 1, "loss_params sheet"),
    ("balanced_sampler_pos_fraction", "0.33", None, 0, 0.3, "loss_params sheet"),
    ("matcher_num_candidates", None, None, 0, 4, "loss_params sheet"),
    ("matcher_center_in_gt", None, None, 0, False, "loss_params sheet"),
    ("sampler_batch_size_per_image", "32", None, 0, 64, "retinanet default; retinaunet override in code"),
    ("sampler_pool_size", None, None, 0, 20, "loss_params sheet"),
    ("sampler_min_neg", "1", None, 0, 16, "retinanet default; retinaunet override in code"),
    ("lambda_dice", None, None, 0, 0.5, "retinaunet_v3 seg loss"),
    ("lambda_ce", None, None, 0, 0.5, "retinaunet_v3 seg loss"),
]

DATA_PARAMS = [
    ("datasources", None, None, 0, "lidc, lidc2", "plan row overrides"),
    ("fold", None, None, 0, 0, "scratch/CLI -> dataset_params"),
    ("batch_size", "2, 8", None, 0, 4, "scratch/CLI -> dataset_params"),
    ("spacing", None, None, 0, "[0.703125,0.703125,1.25]", "plan row overrides"),
    ("validation_fraction", None, None, 0, 0.05, "json-build fallback split"),
    ("seed", None, None, 0, 0, "json-build reproducibility"),
    ("num_workers_train", None, None, 0, 2, "DataLoader"),
    ("num_workers_val", None, None, 0, 0, "DataLoader"),
    ("valid_impl", "patch_stream", "choice", 0, "bbox_anchor", "valid dataloader: bbox_anchor | patch_stream"),
]

POSTPROC_PARAMS = [
    ("score_thresh", "0.01, 0.05", None, 0, 0.02, "infer box selector"),
    ("nms_thresh", "0.1, 0.3", None, 0, 0.22, "infer NMS"),
    ("detections_per_img", "10, 25, 50", None, 0, 25, "max boxes after NMS per volume"),
]

DET_RUNTIME_KEYS = {
    "max_epochs",
    "lr",
    "batch_size",
    "val_every_n_epochs",
    "fold",
    "seed",
    "validation_fraction",
    "num_workers_train",
    "num_workers_val",
}

DET_GLOBAL_PROP_KEYS = {
    "intensity_a_min",
    "intensity_a_max",
    "affine_lps_to_ras",
}

DET_SHEET_ONLY_KEYS = {
    "arch",
    "spatial_dims",
    "n_input_channels",
    "class_name",
    "val_patch_size",
    "gt_box_mode",
    "returned_layers",
    "conv1_t_stride",
    "base_anchor_shapes",
    "encoder_start_channels",
    "encoder_conv_kernels",
    "encoder_strides",
    "decoder_levels",
    "encoder_max_channels",
    "cls_loss",
    "reg_loss",
    "w_cls",
    "w_reg",
    "balanced_sampler_pos_fraction",
    "matcher_num_candidates",
    "matcher_center_in_gt",
    "sampler_batch_size_per_image",
    "sampler_pool_size",
    "sampler_min_neg",
    "lambda_dice",
    "lambda_ce",
    "score_thresh",
    "nms_thresh",
    "detections_per_img",
}

DET_PLAN_EXCLUDED_KEYS = DET_RUNTIME_KEYS | DET_GLOBAL_PROP_KEYS


def _lungs_plan_base():
    return {
        "mnemonic": "lungs",
        "notes": "",
        "datasources": "lidc, lidc2",
        "mode": "lbd",
        "spacing": "[0.703125,0.703125,1.25]",
        "samples_per_file": 1,
        "expand_by": 0,
        "fg_indices_exclude": 1,
        "remapping_source": None,
        "remapping_lbd_rbd": None,
        "remapping_train": None,
        "patch_dim0": 160,
        "patch_dim1": 96,
        "nnz_allowed": False,
        "ignore_labels_cc": 1,
        "dusting_mm": None,
    }


def _lungs_plan_rows():
    base = _lungs_plan_base()
    rows = []
    for plan_id in (1, 2):
        row = dict(base)
        row["id"] = plan_id
        for key in DET_PLAN_EXCLUDED_KEYS | DET_SHEET_ONLY_KEYS:
            row.pop(key, None)
        rows.append(row)
    return rows


PLANS_DET = pd.DataFrame(_lungs_plan_rows())


def _param_df(rows):
    return pd.DataFrame(rows, columns=PARAM_COLS)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        _param_df(MODEL_PARAMS).to_excel(writer, sheet_name="model_params", index=False)
        _param_df(LOSS_PARAMS).to_excel(writer, sheet_name="loss_params", index=False)
        _param_df(POSTPROC_PARAMS).to_excel(writer, sheet_name="postproc_params", index=False)
        _param_df(DATA_PARAMS).to_excel(writer, sheet_name="data_params", index=False)
        PLANS_DET.to_excel(writer, sheet_name="plans_det", index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
