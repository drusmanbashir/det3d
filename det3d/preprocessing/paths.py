from pathlib import Path

from fran.configs.helpers import is_excel_None
from fran.data.dataregistry import DS
def datasource_folder(plan):
    ds_name = plan["datasources"].replace(" ", "").split(",")[0]
    return DS[ds_name].folder


def resolve_detection_paths(plan, project):
    ds_folder = datasource_folder(plan)
    images_dir = ds_folder / "images"
    if is_excel_None(plan.get("lesion_stats_csv")):
        lesion_stats_csv = ds_folder / "label_analysis" / "lesion_stats.csv"
    else:
        lesion_stats_csv = Path(plan["lesion_stats_csv"])
    fold = int(plan.get("fold", 0))
    dataset_json = (
        project.project_folder
        / "detection"
        / f"dataset_fold{fold}.json"
    )
    return images_dir, lesion_stats_csv, dataset_json
