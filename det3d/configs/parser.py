from pathlib import Path

import pandas as pd
from fran.configs.helpers import make_src_dims_from_patch_size
from fran.configs.parser import (
    KEYS_STR_TO_LIST,
    ConfigMaker,
    is_excel_None,
    load_config_from_worksheet,
    parse_excel_dict,
)
from fran.preprocessing import Mnemonics
from utilz.stringz import ast_literal_eval


DET_PLAN_LIST_KEYS = (
    "patch_size",
    "val_patch_size",
    "fg_labels",
    "returned_layers",
    "conv1_t_stride",
    "base_anchor_shapes",
    "ignore_labels_cc",
)


def load_config_from_workbook_det(settingsfilename):
    workbook = pd.ExcelFile(settingsfilename)
    skip = {"plans", "plans_dot", "plans_det"}
    configs_dict = {}
    for sheet in workbook.sheet_names:
        if sheet in skip:
            continue
        configs_dict[sheet] = load_config_from_worksheet(settingsfilename, sheet)
    return configs_dict


class ConfigMakerDet(ConfigMaker):
    def __init__(self, project):
        self.project = project
        configuration_mnemonic = project.global_properties["mnemonic"]
        configuration_filename = self.resolve_configuration_filename()
        plans = pd.read_excel(
            configuration_filename,
            sheet_name="plans_det",
            index_col="id",
            keep_default_na=False,
            na_values=["TRUE", "FALSE", ""],
        )
        plans["mnemonic"] = plans["mnemonic"].map(Mnemonics.match)
        configuration_mnemonic_standardized = Mnemonics.match(configuration_mnemonic)
        self.plans = plans.loc[plans["mnemonic"] == configuration_mnemonic_standardized]
        self.plans = self.plans.drop(columns=["mnemonic"])
        self.plans.insert(0, "plan_id", self.plans.index)
        self.plans = self.plans.set_index("plan_id", drop=False)
        configs = load_config_from_workbook_det(configuration_filename)
        self.configs = parse_excel_dict(configs, KEYS_STR_TO_LIST)

    def setup(self, plan_train: int, verbose=True):
        plan_valid = plan_train
        plan_test = plan_train
        self._set_active_plans(plan_train, plan_valid, plan_test)
        self.add_dataset_props()
        self._finalize_detection_plan()

    def _set_active_plans(self, plan_train, plan_valid, plan_test):
        for plan_id, suffix in [
            (plan_train, "train"),
            (plan_valid, "valid"),
            (plan_test, "test"),
        ]:
            plan_selected = dict(self.plans.loc[plan_id])
            plan_selected = parse_excel_dict(plan_selected, DET_PLAN_LIST_KEYS)
            for key in DET_PLAN_LIST_KEYS:
                if key in plan_selected and isinstance(plan_selected[key], str):
                    plan_selected[key] = ast_literal_eval(plan_selected[key])
            self.configs[f"plan_{suffix}"] = plan_selected
            self.configs[f"plan_{suffix}"]["plan_name"] = plan_id

    def _finalize_detection_plan(self):
        plan = self.configs["plan_train"]
        if is_excel_None(plan.get("mode")):
            plan["mode"] = "det"
        if is_excel_None(plan.get("spacing")):
            plan["spacing"] = [0.703125, 0.703125, 1.25]
        if is_excel_None(plan.get("remapping_source")):
            plan["remapping_source"] = None
        plan["expand_by"] = 0
        if is_excel_None(plan.get("expand_mode")):
            plan["expand_mode"] = "mm"
        if is_excel_None(plan.get("samples_per_file")):
            plan["samples_per_file"] = 1
        patch_size = plan.get("patch_size")
        if patch_size and is_excel_None(plan.get("patch_dim0")):
            plan["patch_dim0"] = int(patch_size[0])
            plan["patch_dim1"] = int(patch_size[1])
        if patch_size:
            plan["src_dims"] = make_src_dims_from_patch_size(patch_size)
        if is_excel_None(plan.get("foreground_class_id")):
            plan["foreground_class_id"] = 0
        ignore_labels_cc = plan.get("ignore_labels_cc")
        if is_excel_None(ignore_labels_cc):
            plan["ignore_labels"] = []
        elif isinstance(ignore_labels_cc, str):
            plan["ignore_labels"] = ast_literal_eval(ignore_labels_cc)
        elif isinstance(ignore_labels_cc, (int, float)):
            plan["ignore_labels"] = [int(ignore_labels_cc)]
        else:
            plan["ignore_labels"] = list(ignore_labels_cc)
        if is_excel_None(plan.get("fg_indices_exclude")):
            plan["fg_indices_exclude"] = list(plan["ignore_labels"])
        lbd_crop_label = plan.get("lbd_crop_label")
        if is_excel_None(lbd_crop_label) or lbd_crop_label == "":
            plan["lbd_crop_label"] = 1
        else:
            plan["lbd_crop_label"] = int(lbd_crop_label)
        if is_excel_None(plan.get("remapping_lbd_rbd")):
            plan["remapping_lbd_rbd"] = plan.get("remapping_source")
        fold = int(plan.get("fold", self.configs["dataset_params"].get("fold", 0)))
        plan["fold"] = fold
        self.configs["dataset_params"]["fold"] = fold
        if plan.get("batch_size") is not None:
            self.configs["dataset_params"]["batch_size"] = int(plan["batch_size"])
        dataset_json = (
            self.project.project_folder / "detection" / f"dataset_fold{fold}.json"
        )
        plan["dataset_json"] = str(dataset_json)
        if is_excel_None(plan.get("data_base_dir")):
            plan["data_base_dir"] = ""
        dusting_mm = plan.get("dusting_mm")
        if is_excel_None(dusting_mm) or dusting_mm == "":
            plan["dusting_mm"] = None
        else:
            plan["dusting_mm"] = float(dusting_mm)
        remapping = plan.get("remapping_train")
        if is_excel_None(remapping) or remapping == "":
            plan["remapping_train"] = None
        elif isinstance(remapping, str):
            plan["remapping_train"] = ast_literal_eval(remapping)
        self.configs["plan_valid"] = dict(plan)
        self.configs["plan_test"] = dict(plan)
