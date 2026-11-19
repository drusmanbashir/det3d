import pickle
from copy import deepcopy
from pathlib import Path

import pandas as pd
from fran.configs.helpers import is_excel_None, is_unspecified, normalize_tree
from fran.configs.plan_keys import normalize_ignore_labels
from fran.configs.parser import (
    KEYS_STR_TO_LIST,
    ConfigMaker,
    load_config_from_worksheet,
    parse_excel_dict,
)
from fran.configs.plan_parse import parse_plan_row
from fran.preprocessing import Mnemonics
from fran.preprocessing.plan_registry import StalePlanError
from fran.utils.folder_names import FolderNames
from utilz.stringz import ast_literal_eval



DET_PLAN_LIST_KEYS = (
    "patch_size",
    "val_patch_size",
    "spacing",
    "fg_indices_exclude",
    "model",
    "postprocess",
)



class ConfigMakerDet(ConfigMaker):
    """Build experiment configs from the detection workbook's plans sheet and parameter sheets."""

    CONFIG_FILENAME = "experiment_configs_det.xlsx"
    PLANS_SHEET = "plans"
    WORKBOOK_SKIP_SHEETS = frozenset({"plans", "plans_dot", "legend"})
    FIT_KEYS = ("spacing", "patch_size", "val_patch_size", "batch_size", "model", "fg_labels", "intensity_properties")
    UNPLANNED_PLAN_KEYS = ("postprocess", "val_patch_size")
    REQUEST_MODEL_KEYS = ("spatial_dims", "in_channels", "encoder_start_channels", "encoder_max_channels")

    @staticmethod
    def plan_request(configs):  #AI
        """Sheet values a plan is resolved from: plan_train (keys planning ignores excluded) and the model keys BoxC002 reads."""
        model = configs["model_params"]
        plan = {key: value for key, value in configs["plan_train"].items() if key not in ConfigMakerDet.UNPLANNED_PLAN_KEYS}
        request = {"plan_train": plan, "model_params": {key: model[key] for key in ConfigMakerDet.REQUEST_MODEL_KEYS}}
        return ConfigMakerDet.normalize_plan_request(request)

    @staticmethod
    def normalize_plan_request(request):  #AI
        """Compare stored plans by the effective ignore list across workbook versions."""
        plan = normalize_ignore_labels(request["plan_train"])
        return normalize_tree(dict(request, plan_train=plan), tuple_mode="list")

    @staticmethod
    def resolved_val_patch_size(plan, patch_size):  #AI
        """nnDetection validates on the training patch; a fixed workbook value is kept."""
        val = plan["val_patch_size"]
        return [int(v) for v in (patch_size if is_unspecified(val) else val)]

    def resolve_configuration_filename(self):
        det_path = super().resolve_configuration_filename().parent / self.CONFIG_FILENAME
        return det_path

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

    def _read_plans(self, path: Path) -> pd.DataFrame:
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

        plans = self._read_plans(configuration_filename)
        mnemonic_plans = plans.loc[plans["mnemonic"] == configuration_mnemonic_standardized]
        if mnemonic_plans.empty:
            raise ValueError(
                f"No plans for mnemonic {configuration_mnemonic_standardized!r} in {configuration_filename}"
            )
        self.plans = mnemonic_plans
        self.plans = self.plans.drop(columns=["mnemonic"])
        self.plans.insert(0, "plan_id", self.plans.index)
        self.plans = self.plans.set_index("plan_id", drop=False)

        base_configs = self.load_workbook_configs(base_filename)
        det_configs = self.load_workbook_configs(configuration_filename)
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
        self.configs["mnemonic"] = configuration_mnemonic_standardized
        self.configs["configurations_dir"] = str(self.resolve_configuration_filename().parent)

    def _parse_det_plan(self, plan: dict) -> dict:
        plan = parse_excel_dict(plan, DET_PLAN_LIST_KEYS)
        for key in DET_PLAN_LIST_KEYS:
            if key in plan and isinstance(plan[key], str) and plan[key] != "auto":
                plan[key] = ast_literal_eval(plan[key])
        return parse_plan_row(plan)

    def _build_active_plan(self, plan_id: int) -> dict:
        """Build one plan from its plans row; parameter sheets stay nested in configs."""
        plan_selected = dict(self.plans.loc[plan_id])
        plan_selected.pop("gt_box_mode", None)
        plan_selected["det_lbd"] = True
        plan_selected["prezoom_scale"] = float(self.configs["dataset_params"].get("prezoom_scale", 1.0))
        plan_selected = self._parse_det_plan(plan_selected)
        if plan_selected["postprocess"] == "auto":
            raise ValueError(
                f"Plan {plan_id}: postprocess = auto needs the postprocess sweep, which det3d does not have yet. "
                "Set {'score_thresh': ..., 'nms_thresh': ..., 'detections_per_img': ...} in the plans sheet."
            )
        return plan_selected

    def setup(self, plan_train: int, verbose=True, load_computed=True):
        """Load the workbook plan, then the resolved plan from det preprocessing when one exists."""
        plan_valid = plan_train
        plan_test = plan_train
        self._set_active_plans(plan_train, plan_valid, plan_test)
        if load_computed:
            self._load_computed_plan()

    def _load_computed_plan(self):
        """Use the resolved plan; refuse one resolved from different sheet values."""
        plan = self.configs["plan_train"]
        path = FolderNames(self.project, plan).plan_file
        if not path.is_file():
            return
        stored = pickle.loads(path.read_bytes())
        requested = self.plan_request(self.configs)
        old = self.normalize_plan_request(stored["requested"])
        if old != requested:
            missing = "<missing>"
            differences = [
                f"{sheet}.{key}: {old[sheet].get(key, missing)!r} -> {requested[sheet].get(key, missing)!r}"
                for sheet in requested
                for key in sorted(old[sheet].keys() | requested[sheet].keys())
                if old[sheet].get(key, missing) != requested[sheet].get(key, missing)
            ]
            raise StalePlanError(
                f"Plan {self.configs['plan_train']['plan_name']}: the sheet changed since det preprocessing "
                f"resolved {path}. Rerun detection preprocessing.\n" + "\n".join(differences)
            )
        val = self.resolved_val_patch_size(plan, stored["plan_train"].get("patch_size", plan["patch_size"]))
        if stored["plan_train"].get("val_patch_size") != val:
            raise StalePlanError(
                f"Plan {plan['plan_name']}: val_patch_size differs from {path}. "
                "Rerun detection preprocessing; only the stored value is refreshed."
            )
        self._apply_resolved_plan(stored["plan_train"])

    def _apply_resolved_plan(self, resolved):  #AI
        """Use the resolved plan's values in every split; the workbook's crop size stays the preprocessing identity."""
        for key in ("plan_train", "plan_valid", "plan_test"):
            plan = self.configs[key]
            patch = plan["patch_size"]
            spacing_auto = plan["spacing"] == "auto"
            plan.update(deepcopy({field: resolved[field] for field in self.FIT_KEYS if field in resolved}))
            plan["spacing_auto"] = spacing_auto
            if plan["mode"] == "lbd":
                plan["preprocessing_patch_size"] = patch

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
    print(F.folders["data_folder_source"])

# %%

    C.setup(1)
    plan = C.configs["plan_train"]
    F2 = FolderNames(P, plan)
    print(F2.folders["data_folder_source"])
# %%
