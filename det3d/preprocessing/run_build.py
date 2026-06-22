from fran.configs.helpers import is_excel_None
from fran.managers.project import Project
from fran.preprocessing.preprocessor import resolve_plan_datasources

from det3d.configs.parser import ConfigMakerDet
from det3d.preprocessing.build_json import build_detection_json
from det3d.preprocessing.paths import resolve_detection_paths


def build_from_plan(project_title, plan_id, configs=None):
    project = Project(project_title=project_title)
    if configs is None:
        config_maker = ConfigMakerDet(project)
        config_maker.setup(plan_id)
        configs = config_maker.configs
    plan = configs["plan_train"]
    dataset_params = configs["dataset_params"]
    images_dir, lesion_stats_csv, dataset_json = resolve_detection_paths(
        plan, project, fold=int(dataset_params["fold"])
    )
    datasources = resolve_plan_datasources(plan)
    fold = int(dataset_params["fold"])
    ds_query = datasources if len(datasources) > 1 else datasources[0]
    train_case_ids, val_case_ids = project.get_train_val_case_ids(
        fold=fold, ds=ds_query
    )
    summary = build_detection_json(
        images_dir=images_dir,
        lesion_stats_csv=lesion_stats_csv,
        dataset_json=dataset_json,
        train_case_ids=train_case_ids,
        val_case_ids=val_case_ids,
        validation_fraction=float(dataset_params.get("validation_fraction", 0.05)),
        seed=int(dataset_params.get("seed", 0)),
        image_path_prefix="",
        dusting_mm=plan.get("dusting_mm"),
    )
    plan["dataset_json"] = str(dataset_json)
    configs["plan_train"]["dataset_json"] = str(dataset_json)
    configs["plan_valid"]["dataset_json"] = str(dataset_json)
    return summary, configs
