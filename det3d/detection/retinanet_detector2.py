import torch
from det3d.detection.retinanet_train import forward_train_batched
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
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
            return forward_train_batched(self, input_images, targets)
        return super().forward(input_images, targets, use_inferer=use_inferer)


Detector2 = RetinaNetDetector2
