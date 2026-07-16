import os
from pathlib import Path

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


def merge_param_dicts(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    merged.update(overlay)
    return merged


def nndet_conf_root() -> Path:
    from utilz.fileio import load_yaml

    cfg = load_yaml(Path(os.environ["FRAN_CONF"]) / "config.yaml")
    return Path(cfg["nndet_conf"])


def nndet_plan_paths_file() -> Path:
    return nndet_conf_root() / "nndet_plan_paths.yaml"


def resolve_nndet_plan_path(mnemonic: str) -> Path:
    """Map project mnemonic → nnDet plan pickle via Mnemonics + nndet_plan_paths.yaml.

    lits / liver / litsmc → yaml key ``liver``; lidc / lungs / lung → ``lungs``.
    """
    from utilz.fileio import load_yaml

    canonical = Mnemonics.match(mnemonic)
    paths_file = nndet_plan_paths_file()
    paths = load_yaml(paths_file)
    if canonical not in paths:
        raise KeyError(
            f"No nnDet plan pickle for mnemonic {mnemonic!r} (canonical {canonical!r}). "
            f"Add {canonical}: <path/to/D3V001_3d.pkl> to {paths_file}. "
            f"Keys present: {sorted(paths)}"
        )
    return Path(paths[canonical])


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
        self.advisor = PlanAdvisorDet(project)
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
            if key in plan and isinstance(plan[key], str):
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
        val_extras = self._plan_row_extras(plan_valid)
        if "val_patch_size" in val_extras:
            self.configs["plan_valid"]["val_patch_size"] = val_extras["val_patch_size"]
        for plan_key in ("plan_train", "plan_valid", "plan_test"):
            self._apply_src_dims_from_patch_size(self.configs[plan_key])

    @staticmethod
    def _apply_src_dims_from_patch_size(plan):
        patch_size = plan["patch_size"]
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


# %%
if __name__ == "__main__":
    PROJECT = "lidca"
    PLAN_ID = 2
    RUN_NAME = "LIDCA-GYRO"
    CKPT = "/s/fran_storage/checkpoints/lidca/lidc/LIDCA-GYRO/checkpoints/last.ckpt"

# %%
# SECTION:--- ConfigMakerDet setup (same path as train / infer config) ---
    from fran.managers import Project

    P = Project(PROJECT)
    C = ConfigMakerDet(P)
    C.setup(PLAN_ID)
    conf = C.configs
    plan_train = conf["plan_train"]
    print("ConfigMakerDet project", PROJECT, "plan_id", PLAN_ID)

# %%
# SECTION:--- plan_train from ckpt vs ConfigMakerDet vs pickle ---
    import torch
    from copy import deepcopy
    from fran.inference.helpers import load_params
    from nndet.io.load import load_pickle
    from det3d.managers.helpers.nndet_retinaunet import plan_from_det3d

    params = load_params(RUN_NAME)
    ckpt_plan_train = params["configs"]["plan_train"]
    plan_train["fg_labels"] = ckpt_plan_train["fg_labels"]
    plan_train["labels_all"] = ckpt_plan_train["labels_all"]
    plan_path = resolve_nndet_plan_path(conf["mnemonic"])
    pkl_plan = load_pickle(str(plan_path))
    print("ckpt plan_train decoder_levels", ckpt_plan_train["decoder_levels"])
    print("excel plan_train decoder_levels", plan_train["decoder_levels"])
    print("fg_labels (from ckpt, as after TrainerDet label infer)", plan_train["fg_labels"])
    print("pickle  decoder_levels", pkl_plan["architecture"]["decoder_levels"])

# %%
# SECTION:--- WITH pickle (old broken path: partial override only) ---
    plan_with_pkl = deepcopy(pkl_plan)
    plan_with_pkl["patch_size"] = [int(v) for v in plan_train["patch_size"]]
    plan_with_pkl["architecture"]["classifier_classes"] = len(plan_train["fg_labels"])
    plan_with_pkl["architecture"]["seg_classes"] = 1
    print("WITH pickle decoder_levels", plan_with_pkl["architecture"]["decoder_levels"])

# %%
# SECTION:--- plan_train only (production build path) ---
    plan_train_only = plan_from_det3d(plan_train, plan_path=None)
    print("plan_train only decoder_levels", plan_train_only["architecture"]["decoder_levels"])

# %%
# SECTION:--- strict ckpt load: pickle path FAIL, plan_train only OK ---
    from det3d.configs.nndet_bridge import load_nndet_train_cfgs
    from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001

    model_cfg, trainer_cfg = load_nndet_train_cfgs()

    def _try_strict(plan, label):
        mod = RetinaUNetV001(model_cfg=model_cfg, trainer_cfg=trainer_cfg, plan=plan)
        sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
        prefix = "nndet_module.model._orig_mod."
        net_sd = {
            k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)
        }
        try:
            mod.model.load_state_dict(net_sd, strict=True)
            print(label, "strict load OK")
        except RuntimeError as exc:
            print(label, "strict load FAIL:", str(exc).split("\n")[0])

    from copy import deepcopy

    _try_strict(plan_with_pkl, "WITH pickle")
    _try_strict(plan_train_only, "plan_train only")
# end PythonMethodScratch

