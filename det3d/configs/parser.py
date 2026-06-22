import json

import numpy as np
import pandas as pd
from fran.configs.helpers import is_excel_None, make_src_dims_from_patch_size, normalize_tree
from fran.configs.parser import (
    KEYS_STR_TO_LIST,
    ConfigMaker,
    load_config_from_worksheet,
    parse_excel_dict,
)
from fran.configs.plan_parse import parse_plan_row
from fran.preprocessing import Mnemonics
from utilz.stringz import ast_literal_eval


DET_PLAN_LIST_KEYS = (
    "patch_size",
    "spacing",
    "fg_labels",
    "returned_layers",
    "conv1_t_stride",
    "base_anchor_shapes",
    "ignore_labels_cc",
    "decoder_levels",
    "fg_indices_exclude",
)

DET_RUNTIME_KEYS = frozenset(
    {
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
)

DET_GLOBAL_PROP_KEYS = frozenset(
    {
        "intensity_a_min",
        "intensity_a_max",
        "affine_lps_to_ras",
    }
)

DET_PLAN_EXCLUDED_KEYS = DET_RUNTIME_KEYS | DET_GLOBAL_PROP_KEYS

DET_SHEET_ONLY_KEYS = frozenset(
    {
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
        "score_thresh",
        "nms_thresh",
        "detections_per_img",
    }
)


def merge_param_dicts(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    merged.update(overlay)
    return merged


class PlanAdvisorDet:
    """Offline patch/anchor hints from det3d bbox sidecars; compare with manual plans."""

    def __init__(self, project):
        self.project = project

    def _bbox_sidecar_dir(self, fold: int):
        return self.project.project_folder / "detection" / f"fold{fold}" / "bboxes"

    def _load_box_sizes_mm(self, fold: int):
        bbox_dir = self._bbox_sidecar_dir(fold)
        if not bbox_dir.exists():
            return []
        sizes = []
        for path in bbox_dir.glob("*.json"):
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                continue
            boxes = data.get("boxes") or data.get("bbox")
            if boxes is None:
                continue
            for box in boxes:
                if len(box) >= 6:
                    w = abs(box[3] - box[0])
                    h = abs(box[4] - box[1])
                    d = abs(box[5] - box[2])
                    sizes.append([w, h, d])
        return sizes

    def suggest_anchors_from_boxes(self, box_sizes, n_anchors=3):
        if not box_sizes:
            return [[6, 8, 4], [8, 6, 5], [10, 10, 6]]
        arr = np.asarray(box_sizes, dtype=float)
        if len(arr) < n_anchors:
            reps = max(1, n_anchors // len(arr))
            arr = np.tile(arr, (reps, 1))[:n_anchors]
        idx = np.linspace(0, len(arr) - 1, n_anchors).astype(int)
        picked = arr[idx]
        anchors = []
        for row in picked:
            anchors.append([max(4, int(round(v))) for v in row])
        return anchors

    def suggest(self, fold: int, spacing=None, patch_size_hint=None) -> dict:
        spacing = spacing or [0.703125, 0.703125, 1.25]
        box_sizes = self._load_box_sizes_mm(fold)
        anchors = self.suggest_anchors_from_boxes(box_sizes)
        patch_size = patch_size_hint or [192, 192, 80]
        return {
            "patch_size": patch_size,
            "base_anchor_shapes": anchors,
            "encoder_conv_kernels": [[3, 3, 3]] * 5,
            "encoder_strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            "spacing": spacing,
        }

    def compare(self, manual: dict, suggested: dict) -> pd.DataFrame:
        keys = sorted(set(manual) | set(suggested))
        rows = []
        for key in keys:
            m = manual.get(key)
            s = suggested.get(key)
            rows.append(
                {
                    "var_name": key,
                    "manual_value": m,
                    "suggested_value": s,
                    "match": m == s,
                    "notes": "" if m == s else "advisor differs",
                }
            )
        return pd.DataFrame(rows)


class ConfigMakerDet(ConfigMaker):
    CONFIG_FILENAME = "experiment_configs_det.xlsx"
    PLANS_SHEET = "plans_det"
    WORKBOOK_SKIP_SHEETS = frozenset({"plans", "plans_dot", "plans_det"})
    PARAM_SHEETS = ("model_params", "loss_params", "postproc_params", "data_params")

    def resolve_configuration_filename(self):
        base = super().resolve_configuration_filename()
        det_path = base.parent / self.CONFIG_FILENAME
        if det_path.exists():
            return det_path
        return base

    def resolve_base_configuration_filename(self):
        return super().resolve_configuration_filename()

    @classmethod
    def load_workbook_configs(cls, settingsfilename):
        workbook = pd.ExcelFile(settingsfilename)
        configs_dict = {}
        for sheet in workbook.sheet_names:
            if sheet in cls.WORKBOOK_SKIP_SHEETS:
                continue
            configs_dict[sheet] = load_config_from_worksheet(settingsfilename, sheet)
        return configs_dict

    def __init__(self, project):
        self.project = project
        configuration_mnemonic = project.global_properties["mnemonic"]
        base_filename = self.resolve_base_configuration_filename()
        configuration_filename = self.resolve_configuration_filename()

        plans = pd.read_excel(
            configuration_filename,
            sheet_name=self.PLANS_SHEET,
            index_col="id",
            keep_default_na=False,
            na_values=["TRUE", "FALSE", ""],
        )
        plans = normalize_tree(plans)
        plans["mnemonic"] = plans["mnemonic"].map(Mnemonics.match)
        configuration_mnemonic_standardized = Mnemonics.match(configuration_mnemonic)
        self.plans = plans.loc[plans["mnemonic"] == configuration_mnemonic_standardized]
        self.plans = self.plans.drop(columns=["mnemonic"])
        self.plans.insert(0, "plan_id", self.plans.index)
        self.plans = self.plans.set_index("plan_id", drop=False)

        base_configs = self.load_workbook_configs(base_filename)
        det_configs = (
            self.load_workbook_configs(configuration_filename)
            if configuration_filename != base_filename
            else {}
        )
        self.configs = {}
        for key, val in base_configs.items():
            self.configs[key] = val
        for key, val in det_configs.items():
            if key == "data_params":
                self.configs.setdefault("dataset_params", {})
                self.configs["dataset_params"].update(val)
            elif key in self.configs and isinstance(self.configs[key], dict):
                self.configs[key].update(val)
            else:
                self.configs[key] = val
        self.configs = parse_excel_dict(self.configs, KEYS_STR_TO_LIST)
        self.advisor = PlanAdvisorDet(project)

    def param_defaults(self) -> dict:
        out = {}
        for sheet in self.PARAM_SHEETS:
            if sheet in self.configs:
                for key, val in self.configs[sheet].items():
                    if key not in DET_PLAN_EXCLUDED_KEYS:
                        out[key] = val
        return out

    def _parse_det_plan(self, plan: dict) -> dict:
        plan = parse_excel_dict(plan, DET_PLAN_LIST_KEYS)
        for key in DET_PLAN_LIST_KEYS:
            if key in plan and isinstance(plan[key], str):
                plan[key] = ast_literal_eval(plan[key])
        return parse_plan_row(plan)

    def _build_active_plan(self, plan_id: int) -> dict:
        plan_selected = dict(self.param_defaults())
        plan_row = dict(self.plans.loc[plan_id])
        for key in DET_PLAN_EXCLUDED_KEYS:
            plan_row.pop(key, None)
        for key in DET_SHEET_ONLY_KEYS:
            plan_row.pop(key, None)
        plan_selected.update(plan_row)
        for key in DET_PLAN_EXCLUDED_KEYS:
            plan_selected.pop(key, None)
        return self._parse_det_plan(plan_selected)

    def compare_plan_with_advisor(self, plan_id: int) -> pd.DataFrame:
        plan = self._parse_det_plan(dict(self.plans.loc[plan_id]))
        fold = int(self.configs["dataset_params"].get("fold", 0))
        spacing = plan.get("spacing")
        patch_size = plan.get("patch_size")
        suggested = self.advisor.suggest(fold=fold, spacing=spacing, patch_size_hint=patch_size)
        return self.advisor.compare(plan, suggested)

    def setup(self, plan_train: int, verbose=True):
        plan_valid = plan_train
        plan_test = plan_train
        self._set_active_plans(plan_train, plan_valid, plan_test)
        self.add_dataset_props()

    def _set_active_plans(self, plan_train, plan_valid, plan_test):
        for plan_id, suffix in [
            (plan_train, "train"),
            (plan_valid, "valid"),
            (plan_test, "test"),
        ]:
            self.configs[f"plan_{suffix}"] = self._build_active_plan(plan_id)
            self.configs[f"plan_{suffix}"]["plan_name"] = plan_id

        self.configs["plan_valid"]["patch_size"] = self.configs["plan_train"]["patch_size"]
        self.configs["plan_test"]["patch_size"] = self.configs["plan_train"]["patch_size"]
        for plan_key in ("plan_train", "plan_valid", "plan_test"):
            self._apply_src_dims_from_patch_size(self.configs[plan_key])

    @staticmethod
    def _apply_src_dims_from_patch_size(plan):
        patch_size = plan.get("patch_size")
        if is_excel_None(patch_size) or patch_size is None:
            return
        plan["src_dims"] = make_src_dims_from_patch_size(patch_size)

    def _assert_patch_fits_src_dims(self, plan):
        patch_size = plan.get("patch_size")
        if is_excel_None(patch_size) or patch_size is None:
            return
        src_dims = plan["src_dims"]
        for i, (ps, sd) in enumerate(zip(patch_size, src_dims)):
            if int(ps) > int(sd):
                raise ValueError(
                    f"patch_size[{i}]={ps} exceeds src_dims[{i}]={sd}; "
                    "RandCrop roi must fit inside shard volume"
                )

    
