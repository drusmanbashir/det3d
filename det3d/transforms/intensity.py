"""CT intensity transforms shared by detection training and inference."""

import torch
from fran.transforms.intensitytransforms import NormaliseClipd
from fran.transforms.misc_transforms import DummyTransform
from monai.transforms import Compose, EnsureTyped, ScaleIntensityRanged


def det_intensity_transform(dataset_params, plan, keys=("image",)):  #AI
    """Build the configured transform from the plan's fitted intensity statistics."""
    mode = dataset_params.get("intensity_norm", "scale01")
    stats = plan.get("intensity_properties")
    if stats is not None and stats != "auto":
        clip = [stats["percentile_00_5"], stats["percentile_99_5"]]
    else:
        raise ValueError("Set intensity_properties in the workbook or run detection preprocessing to fit them.")
    if mode == "scale01":
        transform = ScaleIntensityRanged(
            keys=keys, a_min=float(clip[0]), a_max=float(clip[1]),
            b_min=0.0, b_max=1.0, clip=True,
        )
    elif mode == "zscore_fg":
        transform = Compose([
            EnsureTyped(keys=keys, dtype=torch.float32),
            NormaliseClipd(keys=keys, clip_range=clip, mean=stats["mean"], std=stats["sd"]),
        ])
    else:
        raise ValueError(f"Unknown intensity_norm {mode!r}. Choose scale01 or zscore_fg.")
    return transform


def stored_intensity_transform(dataset_params, plan, keys=("image",)):  #AI
    """Normalisation for data read from the preprocessed store: a no-op when preprocessing
    already clipped and standardised it (fran DataManager._intensity_transform).
    .get: older plans lack the key (= not normalised on disk)."""
    if plan.get("preprocessing_clip_center", False):
        return DummyTransform(keys=keys)
    transform = det_intensity_transform(dataset_params, plan, keys=keys)
    return transform
