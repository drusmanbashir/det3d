"""Map ConfigMakerDet output onto nnDet trainer_cfg defaults."""

from copy import deepcopy
from pathlib import Path

import yaml

NNDET_ROOT = Path("/home/ub/code/nnDetection")
NNDET_TRAIN_CFG = NNDET_ROOT / "nndet/conf/train/v001.yaml"


def load_nndet_train_cfgs(cfg_path=NNDET_TRAIN_CFG):
    with open(cfg_path) as f:
        train_cfg = yaml.safe_load(f)
    return deepcopy(train_cfg["model_cfg"]), deepcopy(train_cfg["trainer_cfg"])


def build_nndet_trainer_cfg(configs: dict) -> dict:
    _, trainer_cfg = load_nndet_train_cfgs()
    trainer_cfg = deepcopy(trainer_cfg)
    model_params = configs["model_params"]
    if "benchmark" in model_params:
        trainer_cfg["benchmark"] = bool(model_params["benchmark"])
    if "deterministic" in model_params:
        trainer_cfg["deterministic"] = bool(model_params["deterministic"])
    trainer_cfg["initial_lr"] = float(model_params["lr"])
    trainer_cfg["weight_decay"] = float(model_params["weight_decay"])
    trainer_cfg["sgd_momentum"] = float(model_params["momentum"])
    if "nesterov" in model_params:
        trainer_cfg["sgd_nesterov"] = bool(model_params["nesterov"])
    return trainer_cfg
