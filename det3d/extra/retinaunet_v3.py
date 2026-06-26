"""RetinaUNet v3 scratch — step blocks for pipeline validation (see .cursor/rules/dl-pipeline-validation.mdc)."""

from __future__ import annotations

import torch
from lightning.pytorch import LightningModule


class RetinaUNetV3Scratch(LightningModule):
  arch_key = "retinaunet_v3"
  seg_key = "seg_logits"


if __name__ == "__main__":
# %%
#SECTION:--- imports ---
    from det3d.architectures.create_detector import create_detector_from_conf
    from det3d.configs.parser import ConfigMakerDet
    from det3d.detection.retinanet_train import forward_train_joint
    from det3d.evaluation.losses import RetinaUNetSegLoss
    from det3d.managers.detector_factory import build_detector_manager
    from fran.managers import Project

# %%
#SECTION:--- config ---
    project_title = "lidca"
    plan_id = 1
    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = C.configs
    conf["dataset_params"]["fold"] = 0
    conf["model_params"]["arch"] = "retinaunet_v3"
    conf["loss_params"]["lambda_dice"] = 0.5
    conf["loss_params"]["lambda_ce"] = 0.5
    conf["plan_train"]["fg_labels"] = [1]
    conf["plan_train"]["w_cls"] = 1.0
    conf["plan_train"]["w_reg"] = 1.0
    conf["plan_train"]["detections_per_img"] = 25

# %%
#SECTION:--- build detector ---
    detector, val_patch_size = create_detector_from_conf(conf)
    print(detector, val_patch_size)

# %%
#SECTION:--- build manager ---
    manager = build_detector_manager("scratch", conf)
    manager.create_loss_fnc()
    print(manager.seg_loss_fnc)

# %%
#SECTION:--- synthetic forward + joint loss ---
    B, D, H, W = 1, 32, 32, 32
    images = torch.randn(B, 1, D, H, W)
    boxes = torch.tensor([[8.0, 8.0, 8.0, 16.0, 16.0, 16.0]])
    labels = torch.tensor([0], dtype=torch.long)
    targets = [{"label": labels, "bbox": boxes}]
    lm = torch.zeros(B, 1, D, H, W)
    lm[:, :, 8:16, 8:16, 8:16] = 1
    seg_loss_fnc = RetinaUNetSegLoss(lambda_dice=0.5, lambda_ce=0.5)
    detector.train()
    losses = forward_train_joint(
        detector,
        images,
        targets,
        seg_loss_fnc,
        lm,
        conf["plan_train"],
    )
    print({k: float(v) for k, v in losses.items() if torch.is_tensor(v)})

# %%
#SECTION:--- overfit smoke (train-only, tiny subset) ---
    # 1. train_indices = first 2–4 cases
    # 2. TrainerDet.setup(..., val_every_n_epochs=999, early_stopping=False)
    # 3. expect train0_loss down; train metrics improve on same batch
    # 4. optional: val0_metric / val0_seg_dice after enabling val

# end PythonMethodScratch
