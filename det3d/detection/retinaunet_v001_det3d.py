"""RetinaUNetV001 on det3d DataManager batches — skip native pre_trafo."""

from __future__ import annotations

from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001

from det3d.detection.nndet_train import (
    det3d_batch_to_nndet,
    maybe_store_batch_grid_preds,
)
from det3d.utils.tensor import sanitize_for_numpy


def fast_nndet_batch_to_device(nb, device):
    nb["data"] = nb["data"].to(device)
    nb["target_boxes"] = [b.to(device) for b in nb["target_boxes"]]
    nb["target_classes"] = [c.to(device) for c in nb["target_classes"]]
    nb["target_seg"] = nb["target_seg"].to(device)
    return nb


def log_nndet_det3d_step_losses(pl_module, losses, prefix):
    """RetinaUNetManager._log_losses parity: train0_* / val0_* per step + epoch."""
    cls_seg = losses["cls"] + losses["seg_ce"] + losses["seg_dice"]
    total = sum(losses.values())
    pl_module.log(
        f"{prefix}_cls_seg_loss",
        cls_seg,
        prog_bar=(prefix == "train0"),
        on_step=True,
        on_epoch=True,
        sync_dist=True,
    )
    pl_module.log(
        f"{prefix}_loss",
        total,
        prog_bar=False,
        on_step=True,
        on_epoch=True,
        sync_dist=True,
    )
    for key, val in losses.items():
        pl_module.log(
            f"{prefix}_{key}",
            val,
            on_step=(prefix == "train0"),
            on_epoch=True,
            sync_dist=True,
        )
    return total


class RetinaUNetV001Det3d(RetinaUNetV001):
    """nnDetection RetinaUNetV001 fed by det3d DM batches (bbox/label/lm, no pre_trafo)."""

    def __init__(self, model_cfg, trainer_cfg, plan, *, fg_labels):
        super().__init__(model_cfg=model_cfg, trainer_cfg=trainer_cfg, plan=plan)
        self._fg_labels = list(fg_labels)
        self._nndet_wandb_grid_val_batches = []

    def _det3d_batch_to_nndet(self, batch):
        return det3d_batch_to_nndet(
            batch,
            self._fg_labels,
        )

    def training_step(self, batch, batch_idx):
        nb = self._det3d_batch_to_nndet(batch)
        nb = fast_nndet_batch_to_device(nb, batch["image"].device)
        losses, _ = self.model.train_step(
            images=nb["data"],
            targets={
                "target_boxes": nb["target_boxes"],
                "target_classes": nb["target_classes"],
                "target_seg": nb["target_seg"],
            },
            evaluation=False,
            batch_num=batch_idx,
        )
        total = log_nndet_det3d_step_losses(self, losses, "train0")
        return {
            "loss": total,
            **{key: l.detach().item() for key, l in losses.items()},
        }

    def validation_step(self, batch, batch_idx):
        nb = self._det3d_batch_to_nndet(batch)
        nb = fast_nndet_batch_to_device(nb, batch["image"].device)
        targets = {
            "target_boxes": nb["target_boxes"],
            "target_classes": nb["target_classes"],
            "target_seg": nb["target_seg"],
        }
        losses, prediction = self.model.train_step(
            images=nb["data"],
            targets=targets,
            evaluation=True,
            batch_num=batch_idx,
        )
        prediction = sanitize_for_numpy(prediction)
        targets = sanitize_for_numpy(targets)
        maybe_store_batch_grid_preds(self, batch, prediction)
        self.evaluation_step(prediction=prediction, targets=targets)
        loss = sum(losses.values())
        return {
            "loss": float(loss.detach().item()),
            **{key: float(val.detach().item()) for key, val in losses.items()},
        }


def build_nndet_retinaunet_det3d_module(configs, num_train_batches):
    from pathlib import Path

    from det3d.configs.parser import resolve_nndet_plan_path
    from det3d.detection.nndet_train import (
        apply_det3d_plan_to_nndet_model_cfg,
        load_nndet_train_cfgs,
        plan_from_det3d,
    )

    plan_train = configs["plan_train"]
    plan_path = resolve_nndet_plan_path(
        configs["mnemonic"], Path(configs["configurations_dir"])
    )
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    model_cfg, trainer_cfg = load_nndet_train_cfgs()
    model_cfg = apply_det3d_plan_to_nndet_model_cfg(model_cfg, plan_train)
    trainer_cfg["num_train_batches_per_epoch"] = int(num_train_batches)
    plan = plan_from_det3d(plan_train, plan_path=str(plan_path))
    module = RetinaUNetV001Det3d(
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        plan=plan,
        fg_labels=plan_train["fg_labels"],
    )
    return module, plan
