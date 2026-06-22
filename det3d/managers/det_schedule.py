import torch


def configure_detection_optimizers(params, plan: dict, lr: float, scheduler_warmup_holder: dict):
    optimizer = torch.optim.SGD(
        params,
        lr,
        momentum=float(plan.get("momentum", 0.9)),
        weight_decay=float(plan.get("weight_decay", 3e-5)),
        nesterov=bool(plan.get("nesterov", True)),
    )
    lr_schedule = str(plan.get("lr_schedule", "epoch_step")).lower()
    if lr_schedule == "poly_iter":
        warm = int(plan.get("warm_iterations", 4000))
        gamma = float(plan.get("poly_gamma", 0.9))
        max_iter = int(plan.get("max_iterations", 125000))
        warm_lr = float(plan.get("warm_lr", 1e-6))

        def lr_lambda(step: int):
            if step < warm:
                return max(warm_lr / lr, step / max(warm, 1))
            progress = (step - warm) / max(max_iter - warm, 1)
            return max((1.0 - progress) ** gamma, 0.0)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scheduler_warmup_holder["scheduler_warmup"] = None
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    from det3d.transforms.warmup_scheduler import GradualWarmupScheduler

    after_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    scheduler_warmup_holder["scheduler_warmup"] = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=10, after_scheduler=after_scheduler
    )
    return optimizer
