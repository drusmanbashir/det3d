import torch
from monai.apps.detection.utils.detector_utils import check_training_targets
from monai.apps.detection.utils.predict_utils import ensure_dict_value_to_list_


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


def forward_train_batched(detector, images: torch.Tensor, targets: list):
    """Training forward on DM-prebatched (B,C,D,H,W); skips RetinaNetDetector.preprocess_images."""
    targets = check_training_targets(
        images,
        targets,
        detector.spatial_dims,
        detector.target_label_key,
        detector.target_box_key,
    )
    detector._check_detector_training_components()
    head_outputs = forward_network_head(detector, images)
    head_outputs, num_anchor_locs_per_level = build_train_anchors(
        detector, images, head_outputs
    )
    return detector.compute_loss(
        head_outputs, targets, detector.anchors, num_anchor_locs_per_level
    )


def forward_train_joint(detector, images: torch.Tensor, targets: list, seg_loss_fnc, lm_batch, plan):
    """Single forward: MONAI det loss + RetinaUNet seg loss."""
    from det3d.evaluation.losses import combine_det_seg_loss_dict

    targets = check_training_targets(
        images,
        targets,
        detector.spatial_dims,
        detector.target_label_key,
        detector.target_box_key,
    )
    detector._check_detector_training_components()
    head_outputs = forward_network_head(detector, images)
    seg_key = detector.network.seg_key
    seg_logits = head_outputs.pop(seg_key)
    if isinstance(seg_logits, list):
        seg_logits = seg_logits[0]
    head_outputs, num_anchor_locs_per_level = build_train_anchors(
        detector, images, head_outputs
    )
    det_losses = detector.compute_loss(
        head_outputs, targets, detector.anchors, num_anchor_locs_per_level
    )
    seg_total = seg_loss_fnc(seg_logits, lm_batch)
    return combine_det_seg_loss_dict(
        det_losses, seg_total, seg_loss_fnc.loss_dict, detector, plan
    )
