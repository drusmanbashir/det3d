from pathlib import Path

from utilz.fileio import load_json


def infer_det_labels_from_data_folder(dm, configs):
    dm.prepare_data()
    labels_all = load_json(Path(dm.data_folder) / "labels_all.json")
    num_classes = max(labels_all) + 1
    fg_labels = [v for v in labels_all if v != 0] or [0]
    for plan_key in ("plan_train", "plan_valid", "plan_test"):
        configs[plan_key]["labels_all"] = labels_all
        configs[plan_key]["fg_labels"] = fg_labels
    configs["model_params"]["num_classes"] = num_classes
