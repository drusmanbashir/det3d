import torch
from det3d.detection.nndet_train import (
    build_nndet_retinaunet_module,
    det3d_batch_to_pre_trafo_input,
    forward_patch_size_from_configs,
    maybe_store_batch_grid_preds,
)
from det3d.managers.retinaunet import RetinaUNetManager
from det3d.utils.tensor import to_numpy


class RetinaUNetManagerV2(RetinaUNetManager):
    """nnDetection RetinaUNetV001 with native pre_trafo on det3d batches."""

    def _nndet_targets(self, batch):
        batch_pre = det3d_batch_to_pre_trafo_input(
            batch,
            forward_patch_size=self.forward_patch_size,
            fg_labels=self.plan["fg_labels"],
        )
        for key in ("data", "target"):
            batch_pre[key] = batch_pre[key].to(self.device)
        with torch.no_grad():
            batch_post = self.nndet_module.pre_trafo(**batch_pre)
        target_seg = batch_post["target"][:, 0]
        out = {
            "data": batch_post["data"],
            "target_boxes": batch_post["boxes"],
            "target_classes": batch_post["classes"],
            "target_seg": target_seg,
        }
        return out
