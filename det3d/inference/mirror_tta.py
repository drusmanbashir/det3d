"""nnDetection mirror test-time augmentation for RetinaUNet inference."""

import torch


def inference_tta(configs):  #AI
    """dataset_params inference_tta: nnDetection mirror count (8 = all 3D mirrors, 4 = the 2D ones).
    Blank, or absent in older checkpoints: 0, no TTA."""
    num_tta = configs["dataset_params"].get("inference_tta")
    num_tta = 0 if num_tta is None else int(num_tta)
    return num_tta


def mirror_tta_step(net, data, num_tta):  #AI
    """Run net.inference_step on every view from nnDetection's get_tta_transforms and map each
    prediction back with its inverse (seg flipped back, boxes mirrored back).

    Args:
        net: nnDetection network with inference_step(data) -> pred_boxes / pred_scores / pred_labels / pred_seg
        data: [N, C, *spatial] tile batch
        num_tta: mirror count; 0 runs the unchanged view only

    Returns:
        list[dict]: one inverted prediction per view; pred_boxes stay per view for the box merge
        Tensor: pred_seg averaged over views [N, C, *spatial]
    """
    import nndet.compat  # noqa: F401
    from nndet.inference.transforms import get_tta_transforms

    transforms, inverses = get_tta_transforms(num_tta, True)
    views = []
    for transform, inverse in zip(transforms, inverses):
        pred = net.inference_step(transform(data=data)["data"])
        views.append(inverse(**pred))
    seg = torch.stack([view["pred_seg"] for view in views]).mean(dim=0)
    return views, seg
