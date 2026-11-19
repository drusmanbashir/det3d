import torch


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
    params,
    configs: dict,
    lr: float,
    scheduler_warmup_holder: dict,
    monitor: str = "val0_metric",
):
    optimizer = torch.optim.SGD(
        params,
        lr,
        momentum=float(configs["nndet"]["momentum"]),
        weight_decay=float(configs["nndet"]["weight_decay"]),
        nesterov=bool(configs["nndet"]["nesterov"]),
    )
    scheduler_warmup_holder["scheduler_warmup"] = None
    return {
        "optimizer": optimizer,
        "lr_scheduler": _plateau_lr_scheduler_config(optimizer, configs["scheduler_params"], monitor),
    }
