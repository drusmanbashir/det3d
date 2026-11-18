"""CT intensity transforms shared by detection training and inference."""

import torch
from fran.transforms.intensitytransforms import NormaliseClipd
from monai.transforms import Compose, EnsureTyped, ScaleIntensityRanged


def det_intensity_transform(dataset_params, plan, keys=("image",), *, legacy_checkpoint=False):  #AI
    """Build the configured transform from fitted statistics or a historical checkpoint's clip range."""
    mode = dataset_params.get("intensity_norm", "scale01")
    stats = plan.get("intensity_properties")
    if stats is not None and stats != "auto":
        clip = [stats["percentile_00_5"], stats["percentile_99_5"]]
    elif legacy_checkpoint and "intensity_norm" not in dataset_params and dataset_params.get("intensity_clip_range") is not None:
        clip = dataset_params["intensity_clip_range"]
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
