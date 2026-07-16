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

        seg_key = getattr(self.network, "seg_key", None)
        if seg_key is not None and not self.training and not use_inferer:
            network_forward = self.network.forward

            def det_head_forward(images):
                outputs = network_forward(images)
                outputs.pop(seg_key)
                return outputs

            self.network.forward = det_head_forward
            try:
                result = super().forward(input_images, targets, use_inferer=use_inferer)
            finally:
                self.network.forward = network_forward
            return result

        return super().forward(input_images, targets, use_inferer=use_inferer)

    def generate_anchors(self, images: Tensor, head_outputs: dict[str, list[Tensor]]) -> None:
        super().generate_anchors(images, head_outputs)
        device = head_outputs[self.cls_key][0].device
        self.anchors = [anchors.to(device) for anchors in self.anchors]

    def compute_cls_loss(
        self, cls_logits: Tensor, targets: list[dict[str, Tensor]], matched_idxs: list[Tensor]
    ) -> Tensor:
        device = cls_logits.device
        matched_idxs = [m.to(device) for m in matched_idxs]
        return super().compute_cls_loss(cls_logits, targets, matched_idxs)

    def compute_box_loss(
        self,
        box_regression: Tensor,
        targets: list[dict[str, Tensor]],
        anchors: list[Tensor],
        matched_idxs: list[Tensor],
    ) -> Tensor:
        matched_idxs = [m.to(anchors[i].device) for i, m in enumerate(matched_idxs)]
        return super().compute_box_loss(box_regression, targets, anchors, matched_idxs)


Detector2 = RetinaNetDetector2

