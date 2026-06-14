import torch
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.utils.detector_utils import check_training_targets
from monai.apps.detection.utils.predict_utils import ensure_dict_value_to_list_
from torch import Tensor


class RetinaNetDetector2(RetinaNetDetector):
    """
    RetinaNetDetector for DM-prebatched (B, C, D, H, W) training.

    Training with a batched Tensor skips preprocess_images (collate already padded).
    List inputs and eval/infer still use the parent forward (preprocess_images).
    """

    def forward(
        self,
        input_images: list[Tensor] | Tensor,
        targets: list[dict[str, Tensor]] | None = None,
        use_inferer: bool = False,
    ):
        if self.training and isinstance(input_images, Tensor):
            images = input_images

            targets = check_training_targets(
                images, targets, self.spatial_dims, self.target_label_key, self.target_box_key
            )
            self._check_detector_training_components()

            head_outputs = self.network(images)
            if isinstance(head_outputs, (tuple, list)):
                tmp_dict = {}
                tmp_dict[self.cls_key] = head_outputs[: len(head_outputs) // 2]
                tmp_dict[self.box_reg_key] = head_outputs[len(head_outputs) // 2 :]
                head_outputs = tmp_dict
            else:
                ensure_dict_value_to_list_(head_outputs)

            self.generate_anchors(images, head_outputs)
            num_anchor_locs_per_level = [x.shape[2:].numel() for x in head_outputs[self.cls_key]]

            for key in [self.cls_key, self.box_reg_key]:
                head_outputs[key] = self._reshape_maps(head_outputs[key])

            losses = self.compute_loss(head_outputs, targets, self.anchors, num_anchor_locs_per_level)
            return losses

        return super().forward(input_images, targets, use_inferer=use_inferer)


Detector2 = RetinaNetDetector2

