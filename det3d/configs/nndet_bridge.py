"""Load nnDetection's RetinaUNet V001 configuration."""

from copy import deepcopy
from pathlib import Path

import yaml

NNDET_ROOT = Path("/home/ub/code/nnDetection")
NNDET_TRAIN_CFG = NNDET_ROOT / "nndet/conf/train/v001.yaml"


def load_nndet_train_cfgs(cfg_path=NNDET_TRAIN_CFG):
    with open(cfg_path) as f:
        train_cfg = yaml.safe_load(f)
    return deepcopy(train_cfg["model_cfg"]), deepcopy(train_cfg["trainer_cfg"])
