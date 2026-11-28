from pathlib import Path
from fran.configs.plan_keys import organ_label
from utilz.fileio import load_json




def infer_det_labels_from_data_folder(dm, configs):
    dm.prepare_data()
    manifest = load_json(Path(dm.data_folder) / "manifest.json")
    labels_all = manifest["labels_all"]
    num_classes = max(labels_all) + 1
    fg_labels = [v for v in labels_all if v != 0] or [0]
    for plan_key in ("plan_train", "plan_valid", "plan_test"):
        configs[plan_key]["labels_all"] = labels_all
        configs[plan_key]["fg_labels"] = fg_labels
        configs[plan_key]["organ_label"] = organ_label(configs[plan_key])
    configs["model_params"]["num_classes"] = num_classes
