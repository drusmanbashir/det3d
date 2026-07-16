"""Fixture case dataclass."""

from dataclasses import dataclass

import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor


@dataclass
class InferFixtureCase:
    name: str
    image_full: MetaTensor
    lm_full: torch.Tensor
    bounding_box: list
    ignore_labels: list
    n_lesions: int
    source_image: str
    full_meta: dict
    lesion_boxes_full: np.ndarray
