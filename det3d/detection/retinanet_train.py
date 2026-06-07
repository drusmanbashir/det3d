import torch
from monai.apps.detection.utils.detector_utils import check_training_targets
from monai.apps.detection.utils.predict_utils import ensure_dict_value_to_list_


def forward_train_batched(detector, images: torch.Tensor, targets: list):
    """Training forward on DM-prebatched (B,C,D,H,W); skips RetinaNetDetector.preprocess_images."""
    spatial_dims = detector.spatial_dims
    targets = check_training_targets(
        images,
        targets,
        spatial_dims,
        detector.target_label_key,
        detector.target_box_key,
    )
    detector._check_detector_training_components()
    head_outputs = detector.network(images)
    if isinstance(head_outputs, (tuple, list)):
        tmp_dict = {
            detector.cls_key: head_outputs[: len(head_outputs) // 2],
            detector.box_reg_key: head_outputs[len(head_outputs) // 2 :],
        }
        head_outputs = tmp_dict
    else:
        ensure_dict_value_to_list_(head_outputs)
    detector.generate_anchors(images, head_outputs)
    num_anchor_locs_per_level = [x.shape[2:].numel() for x in head_outputs[detector.cls_key]]
    for key in (detector.cls_key, detector.box_reg_key):
        head_outputs[key] = detector._reshape_maps(head_outputs[key])
    return detector.compute_loss(
        head_outputs, targets, detector.anchors, num_anchor_locs_per_level
    )
