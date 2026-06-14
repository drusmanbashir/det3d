import json
from pathlib import Path

from det3d.configs.parser import ConfigMakerDet
from det3d.preprocessing.build_json import build_detection_datalist
from det3d.preprocessing.paths import resolve_detection_paths
from fran.configs.helpers import is_excel_None
from fran.data.dataregistry import DS
from fran.managers import Project
from fran.preprocessing.preprocessor import resolve_plan_datasources
from fran.utils.misc import convert_remapping


def load_lidc_train_val(
    project_title="lidc",
    plan_id=1,
    ds_name="lidc",
    fold=None,
):
    project = Project(project_title=project_title)
    config_maker = ConfigMakerDet(project)
    config_maker.setup(plan_id)
    configs = config_maker.configs
    plan = configs["plan_train"]
    if fold is None:
        fold = int(configs["dataset_params"]["fold"])

    ds_folder = DS[ds_name].folder
    images_dir = ds_folder / "images"
    if is_excel_None(plan.get("lesion_stats_csv")):
        lesion_stats_csv = ds_folder / "label_analysis" / "lesion_stats.csv"
    else:
        lesion_stats_csv = plan["lesion_stats_csv"]

    _, _, dataset_json = resolve_detection_paths(plan, project)
    errors = []

    use_cached_json = False
    if dataset_json.is_file():
        payload = json.loads(dataset_json.read_text())
        training = payload.get("training", [])
        if training:
            sample_image = Path(training[0]["image"])
            if sample_image.parent == images_dir or str(sample_image).startswith(str(images_dir)):
                use_cached_json = True
                train_data = payload["training"]
                val_data = payload["validation"]

    if not use_cached_json:
        datasources = resolve_plan_datasources(plan)
        ds_query = datasources if len(datasources) > 1 else datasources[0]
        train_ids, val_ids = project.get_train_val_case_ids(fold=fold, ds=ds_query)
        remapping_train = plan.get("remapping_train")
        if remapping_train is not None and not isinstance(remapping_train, dict):
            remapping_train = convert_remapping(remapping_train)
        payload, errors, train_set, val_set = build_detection_datalist(
            images_dir=images_dir,
            lesion_stats_csv=lesion_stats_csv,
            train_case_ids=train_ids,
            val_case_ids=val_ids,
            foreground_class_id=int(plan.get("foreground_class_id", 0)),
            remapping_train=remapping_train,
            dusting_mm=plan.get("dusting_mm"),
        )
        train_data = payload["training"]
        val_data = payload["validation"]

    meta = {
        "plan": plan,
        "dataset_params": configs["dataset_params"],
        "images_dir": str(images_dir),
        "lesion_stats_csv": str(lesion_stats_csv),
        "dataset_json": str(dataset_json),
        "errors": len(errors),
        "train_cases": len(train_data),
        "val_cases": len(val_data),
    }
    return train_data, val_data, meta
