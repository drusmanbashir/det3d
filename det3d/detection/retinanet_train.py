import torch
from monai.apps.detection.utils.detector_utils import check_training_targets
from monai.apps.detection.utils.predict_utils import ensure_dict_value_to_list_


def validate_train_targets(detector, images: torch.Tensor, targets: list):
    return check_training_targets(
        images,
        targets,
        detector.spatial_dims,
        detector.target_label_key,
        detector.target_box_key,
    )


def forward_network_head(detector, images: torch.Tensor):
    dtype = next(detector.network.parameters()).dtype
    if images.dtype != dtype:
        images = images.to(dtype=dtype)
    head_outputs = detector.network(images)
    if isinstance(head_outputs, (tuple, list)):
        head_outputs = {
            detector.cls_key: head_outputs[: len(head_outputs) // 2],
            detector.box_reg_key: head_outputs[len(head_outputs) // 2 :],
        }
    else:
        ensure_dict_value_to_list_(head_outputs)
    return head_outputs


def build_train_anchors(detector, images: torch.Tensor, head_outputs: dict):
    detector.generate_anchors(images, head_outputs)
    num_anchor_locs_per_level = [
        x.shape[2:].numel() for x in head_outputs[detector.cls_key]
    ]
    for key in (detector.cls_key, detector.box_reg_key):
        head_outputs[key] = detector._reshape_maps(head_outputs[key])
    return head_outputs, num_anchor_locs_per_level


def compute_train_loss(
    detector, head_outputs: dict, targets: list, num_anchor_locs_per_level: list
):
    return detector.compute_loss(
        head_outputs, targets, detector.anchors, num_anchor_locs_per_level
    )


def forward_train_batched(detector, images: torch.Tensor, targets: list):
    """Training forward on DM-prebatched (B,C,D,H,W); skips RetinaNetDetector.preprocess_images."""
    targets = validate_train_targets(detector, images, targets)
    detector._check_detector_training_components()
    head_outputs = forward_network_head(detector, images)
    head_outputs, num_anchor_locs_per_level = build_train_anchors(
        detector, images, head_outputs
    )
    return compute_train_loss(
        detector, head_outputs, targets, num_anchor_locs_per_level
    )
