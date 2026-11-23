from fran.configs.helpers import is_unspecified
from fran.configs.parser import ConfigMaker
from fran.configs.workbook import apply_finetuning, open_config_workbook


class ConfigMakerDet(ConfigMaker):
    """Read detection plans from the det workbook on top of the base fran parameter sheets."""

    CONFIG_FILENAME = "experiment_configs_det.xlsx"
    PIPELINE = "det3d"  # names the plan file, `FolderNames.plan_file`
    ENCODERS = (None, "resenc")
    FINETUNING_KEYS = ("checkpoint", "initial_lr", "momentum", "nesterov", "weight_decay", "warm_iterations", "warm_lr", "poly_gamma")
    SKIP_SHEETS = frozenset({"plans", "plans_dot", "legend"})
    FIT_KEYS = ("spacing", "fg_labels", "intensity_properties")
    GPU_FIT_KEYS = ("patch_size", "batch_size", "model")

    @staticmethod
    def resolved_val_patch_size(plan, patch_size):  #AI
        """nnDetection validates on the training patch; a fixed workbook value is kept."""
        val = plan["val_patch_size"]
        return [int(v) for v in (patch_size if is_unspecified(val) else val)]

    PLAN_OVERRIDES = {
        "dataset_params": ("valid_impl", "inference_overlap"),
        "postprocess": ("detections_per_img",),
    }

    def __init__(self, project):
        super().__init__(project)
        self.sheet_defaults = {sheet: self.configs[sheet].copy() for sheet in self.PLAN_OVERRIDES}

    def setup(
        self, plan_train, plan_valid=None, load_computed=True, gpu_key=None, *,
        load_gpu_fit=True, epochs=None, batch_size=None, pretrained=None,
    ):
        super().setup(
            plan_train, plan_valid, load_computed=load_computed, gpu_key=gpu_key,
            load_gpu_fit=load_gpu_fit, epochs=epochs, batch_size=batch_size, pretrained=pretrained,
        )
        for split in ("plan_train", "plan_valid", "plan_test"):
            plan = self.configs[split]
            if not is_unspecified(plan["patch_size"]):
                plan["val_patch_size"] = self.resolved_val_patch_size(plan, plan["patch_size"])
        plan = self.configs["plan_train"]
        for sheet, keys in self.PLAN_OVERRIDES.items():
            for key in keys:
                if key in self.plans.columns:
                    value = plan[key]
                    self.configs[sheet][key] = self.sheet_defaults[sheet].get(key) if value is None else value
        if "finetuning" in self.configs:
            apply_finetuning(self.configs)
        data = self.configs["dataset_params"]
        if self.resolve_epochs(self.configs)["valid"] is not None and data["valid_impl"] == "patch_stream":
            data["valid_impl"] = "bbox_anchor"

    def load_configs(self, workbook):
        """Load the base fran sheets, then merge the det sheets over them (data_params into dataset_params)."""
        base_path = self.resolve_configuration_filename().parent / ConfigMaker.CONFIG_FILENAME
        with open_config_workbook(base_path) as base_workbook:
            configs = super().load_configs(base_workbook)
        for key, val in super().load_configs(workbook).items():
            configs.setdefault("dataset_params" if key == "data_params" else key, {}).update(val)
        return configs

    def read_plan(self, plan_id, descendants=()):
        plan = super().read_plan(plan_id, descendants)
        plan["det_lbd"] = True
        return plan

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
