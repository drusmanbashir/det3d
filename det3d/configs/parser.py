from fran.configs.helpers import is_unspecified
from fran.configs.parser import ConfigMaker
from fran.utils.folder_names import FolderNames


class ConfigMakerDet(ConfigMaker):
    """Read detection plans from the det workbook on top of the base fran parameter sheets."""

    CONFIG_FILENAME = "experiment_configs_det.xlsx"
    PIPELINE = "det3d"  # names the plan file, `FolderNames.plan_file`
    SKIP_SHEETS = frozenset({"plans", "plans_dot", "legend"})
    FIT_KEYS = ("spacing", "fg_labels", "intensity_properties", "val_patch_size")  # val_patch_size: non-lbd plans
    GPU_FIT_KEYS = ("patch_size", "val_patch_size", "batch_size", "model")

    @staticmethod
    def resolved_val_patch_size(plan, patch_size):  #AI
        """nnDetection validates on the training patch; a fixed workbook value is kept."""
        val = plan["val_patch_size"]
        return [int(v) for v in (patch_size if is_unspecified(val) else val)]

    def load_configs(self, path):
        """Load the base fran sheets, then merge the det sheets over them (data_params into dataset_params)."""
        configs = super().load_configs(self.resolve_configuration_filename().parent / ConfigMaker.CONFIG_FILENAME)
        for key, val in super().load_configs(path).items():
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
