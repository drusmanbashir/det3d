import torch


def _plateau_lr_scheduler_config(optimizer, plan: dict, monitor: str):  #AI
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        str(plan["scheduler_mode"]),
        patience=int(plan["scheduler_patience"]),
    )
    return {
        "scheduler": scheduler,
        "monitor": plan["scheduler_monitor"],
        "interval": "epoch",
        "frequency": int(plan["scheduler_frequency"]),
    }


def configure_detection_optimizers(
    params,
    plan: dict,
    lr: float,
    scheduler_warmup_holder: dict,
    monitor: str = "val0_metric",
):
    optimizer = torch.optim.SGD(
        params,
        lr,
        momentum=float(plan["momentum"]),
        weight_decay=float(plan["weight_decay"]),
        nesterov=bool(plan["nesterov"]),
    )
    scheduler_warmup_holder["scheduler_warmup"] = None
    return {
        "optimizer": optimizer,
        "lr_scheduler": _plateau_lr_scheduler_config(optimizer, plan, monitor),
    }
