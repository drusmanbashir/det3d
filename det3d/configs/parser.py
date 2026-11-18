from pathlib import Path

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
from fran.utils.folder_names import FolderNames
from utilz.stringz import ast_literal_eval
from det3d.configs.nndet_planning import (
    apply_nndet_plan,
    load_nndet_plan,
    nndet_plan_path,
)



DET_PLAN_LIST_KEYS = (
    "patch_size",
    "val_patch_size",
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
        "box_tracker",
        "intensity_norm",
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
        "encoder_start_channels",
        "encoder_conv_kernels",
        "encoder_strides",
        "decoder_levels",
        "encoder_max_channels",
        "balanced_sampler_pos_fraction",
        "matcher_num_candidates",
        "matcher_center_in_gt",
        "sampler_batch_size_per_image",
        "sampler_pool_size",
        "sampler_min_neg",
        "cls_loss",
        "reg_loss",
        "w_cls",
        "w_reg",
        "returned_layers",
        "conv1_t_stride",
        "base_anchor_shapes",
        "score_thresh",
        "nms_thresh",
        "detections_per_img",
        "topk_candidates_per_level",
        "remove_small_boxes",
        "optimizer",
        "momentum",
        "weight_decay",
        "nesterov",
        "benchmark",
        "deterministic",
        "scheduler_monitor",
        "scheduler_patience",
        "scheduler_mode",
        "scheduler_frequency",
    }
)

DET_PLAN_ROW_KEYS = frozenset({"val_patch_size"})


class ConfigMakerDet(ConfigMaker):
    """Build experiment configs from the detection Excel workbook and plans_det sheet."""

    CONFIG_FILENAME = "experiment_configs_det.xlsx"
    PLANS_SHEET = "plans_det"
    WORKBOOK_SKIP_SHEETS = frozenset({"plans", "plans_dot", "plans_det"})
    PARAM_SHEETS = (
        "model_params",
        "nndet",
        "scheduler_params",
        "retinanet",
        "loss_params",
        "postproc_params",
        "data_params",
    )

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

    def _read_plans_det(self, path: Path) -> pd.DataFrame:
        plans = pd.read_excel(
            path,
            sheet_name=self.PLANS_SHEET,
            index_col="id",
            keep_default_na=False,
            na_values=["TRUE", "FALSE", ""],
        )
        plans = normalize_tree(plans)
        plans["mnemonic"] = plans["mnemonic"].map(Mnemonics.match)
        return plans

    def __init__(self, project):
        self.project = project
        configuration_mnemonic = project.global_properties["mnemonic"]
        base_filename = self.resolve_base_configuration_filename()
        configuration_filename = self.resolve_configuration_filename()
        configuration_mnemonic_standardized = Mnemonics.match(configuration_mnemonic)

        plans = self._read_plans_det(configuration_filename)
        mnemonic_plans = plans.loc[plans["mnemonic"] == configuration_mnemonic_standardized]
        if mnemonic_plans.empty and configuration_filename != base_filename:
            base_plans = self._read_plans_det(base_filename)
            mnemonic_plans = base_plans.loc[
                base_plans["mnemonic"] == configuration_mnemonic_standardized
            ]
        self.plans = mnemonic_plans
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
        dataset = self.configs["dataset_params"]
        self._workbook_batch = {key: dataset[key] for key in ("batch_size",) if key in dataset}
        self.configs["mnemonic"] = configuration_mnemonic_standardized
        self.configs["configurations_dir"] = str(self.resolve_configuration_filename().parent)

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
            if key in plan and isinstance(plan[key], str) and plan[key] != "auto":
                plan[key] = ast_literal_eval(plan[key])
        return parse_plan_row(plan)

    def _plan_row_extras(self, plan_id: int) -> dict:
        row = dict(self.plans.loc[plan_id])
        out = {}
        for key in DET_PLAN_ROW_KEYS:
            val = row.get(key)
            if is_excel_None(val) or val is None:
                continue
            if isinstance(val, str) and val.strip().startswith(("[", "(")):
                val = ast_literal_eval(val)
            out[key] = val
        return out

    def _build_active_plan(self, plan_id: int) -> dict:
        plan_selected = dict(self.param_defaults())
        plan_row = dict(self.plans.loc[plan_id])
        plan_row.pop("gt_box_mode", None)
        for key in DET_PLAN_EXCLUDED_KEYS:
            plan_row.pop(key, None)
        for key in DET_SHEET_ONLY_KEYS:
            plan_row.pop(key, None)
        plan_selected.update(plan_row)
        plan_selected.update(self._plan_row_extras(plan_id))
        for key in DET_PLAN_EXCLUDED_KEYS:
            plan_selected.pop(key, None)
        plan_selected["det_lbd"] = True
        plan_selected["prezoom_scale"] = float(self.configs["dataset_params"].get("prezoom_scale", 1.0))
        return self._parse_det_plan(plan_selected)

    def setup(self, plan_train: int, verbose=True, load_computed=True):
        """Load the workbook, optionally completing unresolved values from the saved fit."""
        dataset = self.configs["dataset_params"]
        dataset.pop("batch_size", None)
        dataset.update(self._workbook_batch)
        plan_valid = plan_train
        plan_test = plan_train
        self._set_active_plans(plan_train, plan_valid, plan_test)
        if load_computed:
            self._load_computed_plan()

    def _load_computed_plan(self):
        """Treat the saved fit as optional and allow it to contain only some fields."""
        path = nndet_plan_path(self.project, self.configs["plan_train"])
        try:
            computed = load_nndet_plan(path)
        except FileNotFoundError:
            computed = {}
        apply_nndet_plan(self.configs, computed)

    def _set_active_plans(self, plan_train, plan_valid, plan_test):
        for plan_id, suffix in [
            (plan_train, "train"),
            (plan_valid, "valid"),
            (plan_test, "test"),
        ]:
            self.configs[f"plan_{suffix}"] = self._build_active_plan(plan_id)
            self.configs[f"plan_{suffix}"]["plan_name"] = plan_id

        train_patch = self.configs["plan_train"].get("patch_size")
        for split in ("train", "valid", "test"):
            self.configs[f"plan_{split}"]["patch_size"] = train_patch
        val_extras = self._plan_row_extras(plan_valid)
        if "val_patch_size" in val_extras:
            self.configs["plan_valid"]["val_patch_size"] = val_extras["val_patch_size"]
        for plan_key in ("plan_train", "plan_valid", "plan_test"):
            self._apply_src_dims_from_patch_size(self.configs[plan_key])

    @staticmethod
    def _apply_src_dims_from_patch_size(plan):
        patch_size = plan["patch_size"]
        if patch_size == "auto" and plan["mode"] == "lbd":
            from fran.configs.defaults import lbd_crop_minimum

            plan["src_dims"] = list(lbd_crop_minimum(plan))
            return
        if is_excel_None(patch_size) or patch_size is None:
            return
        plan["src_dims"] = make_src_dims_from_patch_size(patch_size)

    def _assert_patch_fits_src_dims(self, plan):
        patch_size = plan.get("patch_size")
        if patch_size == "auto":
            return
        if is_excel_None(patch_size) or patch_size is None:
            return
        src_dims = plan["src_dims"]
        for i, (ps, sd) in enumerate(zip(patch_size, src_dims)):
            if int(ps) > int(sd):
                raise ValueError(
                    f"patch_size[{i}]={ps} exceeds src_dims[{i}]={sd}; "
                    "RandCrop roi must fit inside shard volume"
                )
# %%
if __name__ == '__main__':
#SECTION:-------------------- setup --------------------------------------------------------------------------------------
    from fran.configs.parser import ConfigMaker
    from fran.managers import Project

    P = Project("lidc")
    C = ConfigMakerDet(P)
    C.setup(3)
    plan = C.configs["plan_train"]
    F = FolderNames(P, plan)

# %%
