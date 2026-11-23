import torch
from nndet.training.learning_rate import LinearWarmupPolyLR, linear_warm_up, poly_lr
from nndet.training.optimizer import get_params_no_wd_on_norm


class DetectionWarmupPolyLR(LinearWarmupPolyLR):
    """Use completed optimizer steps so the first step runs at warm_lr."""

    def get_lr(self):
        step = self.last_epoch
        if step < self.warm_iterations:
            return [
                linear_warm_up(step, warm_lr, self.warm_iterations, base_lr)
                for warm_lr, base_lr in zip(self.warm_lr, self.base_lrs)
            ]
        return [
            poly_lr(step - self.warm_iterations, base_lr, self.poly_iterations, self.poly_gamma)
            for base_lr in self.base_lrs
        ]


def _plateau_lr_scheduler_config(optimizer, scheduler_params: dict, monitor: str):  #AI
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        str(scheduler_params["scheduler_mode"]),
        patience=int(scheduler_params["scheduler_patience"]),
    )
    return {
        "scheduler": scheduler,
        "monitor": scheduler_params["scheduler_monitor"],
        "interval": "epoch",
        "frequency": int(scheduler_params["scheduler_frequency"]),
    }


def configure_detection_optimizers(
    model,
    configs: dict,
    lr: float,
    scheduler_warmup_holder: dict,
    trainer,
    monitor: str = "val0_metric",
):
    """nndet sheet warm_iterations set: nnDetection v001 (no decay on norm layers, LinearWarmupPolyLR
    stepped per iteration over max_epochs x batches per epoch). Blank (or row absent in older
    checkpoints): decay on every parameter and ReduceLROnPlateau."""
    nndet = configs["nndet"]
    weight_decay = float(nndet["weight_decay"])
    warm_iterations = nndet.get("warm_iterations")
    params = model.parameters() if warm_iterations is None else get_params_no_wd_on_norm(model, weight_decay)
    optimizer = torch.optim.SGD(
        params,
        lr,
        momentum=float(nndet["momentum"]),
        weight_decay=weight_decay,
        nesterov=bool(nndet["nesterov"]),
    )
    scheduler_warmup_holder["scheduler_warmup"] = None
    if warm_iterations is None:
        lr_scheduler = _plateau_lr_scheduler_config(optimizer, configs["scheduler_params"], monitor)
    else:
        scheduler = DetectionWarmupPolyLR(
            optimizer=optimizer,
            warm_iterations=int(warm_iterations),
            warm_lr=float(nndet["warm_lr"]),
            poly_gamma=float(nndet["poly_gamma"]),
            num_iterations=int(trainer.estimated_stepping_batches),
        )
        lr_scheduler = {"scheduler": scheduler, "interval": "step"}
    return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
