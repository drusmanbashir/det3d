from copy import copy
from decimal import Decimal
from math import ceil

import torch
from fran.callback.debug_epoch_limit import DebugEpochBatchLimit
from lightning.pytorch.callbacks import ModelCheckpoint
from nndet.training.learning_rate import CycleLinear, LinearWarmupPolyLR, linear_warm_up, poly_lr
from nndet.training.optimizer import get_params_no_wd_on_norm


def resolve_tail_epochs(fraction, max_epochs, val_every_n_epochs):
    """Blank/auto: one sixth, floored to a validation interval. 0: off. (0, 1]: fraction of total epochs."""
    if fraction is None or fraction == "auto":
        target = max(val_every_n_epochs, max_epochs // 6)
        # Shorten the tail until its first update follows a validation epoch.
        tail = target - (target - max_epochs) % val_every_n_epochs
    else:
        assert type(fraction) in (int, float) and 0 <= fraction <= 1, (
            f"plans.swa_tail_fraction={fraction!r}: use auto, 0 (off), or a fraction in (0, 1]."
        )
        if fraction == 0:
            return 0
        tail = int(fraction * max_epochs)
    assert 0 < tail <= max_epochs and (max_epochs - tail) % val_every_n_epochs == 0, (
        f"plans.swa_tail_fraction={fraction!r} gives {tail} tail epochs of {max_epochs}; "
        f"the pre-tail boundary must land on a validation epoch (interval {val_every_n_epochs})."
    )
    return tail


class DetectionTail(ModelCheckpoint):
    """Average epoch-start parameters from start_epoch (zero-based). Write inference and resume weights to last.ckpt."""

    def __init__(self, start_epoch):
        super().__init__(save_top_k=0, save_last=True, save_on_train_epoch_end=True, enable_version_counter=False)
        self.start_epoch = start_epoch
        self.average = {}
        self.n_averaged = 0
        self._writing_last = False

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch < self.start_epoch + self.n_averaged:
            return
        assert epoch == self.start_epoch + self.n_averaged, "Missing SWA samples on resume"
        for name, param in pl_module.named_parameters(remove_duplicate=False):
            value = param.detach().cpu()
            if self.n_averaged == 0:
                self.average[name] = value.clone()
            else:
                self.average[name].lerp_(value, 1 / (self.n_averaged + 1))
        self.n_averaged += 1

    def state_dict(self):
        state = super().state_dict()
        state.update(average=self.average, n_averaged=self.n_averaged, start_epoch=self.start_epoch)
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        assert self.start_epoch == state_dict["start_epoch"], "SWA boundary changed on resume"
        self.average = {name: value.cpu() for name, value in state_dict["average"].items()}
        self.n_averaged = state_dict["n_averaged"]

    def on_train_epoch_end(self, trainer, pl_module):
        # Last advances even when the validation metric misses the top two or no snapshot is due.
        if self._should_skip_saving_checkpoint(trainer):
            return
        self._writing_last = True
        try:
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))
        finally:
            self._writing_last = False

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self._writing_last and self.n_averaged:
            checkpoint["current_model_state"] = checkpoint["state_dict"]
            checkpoint["state_dict"] = copy(checkpoint["current_model_state"])
            checkpoint["state_dict"].update(self.average)

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if trainer.state.fn != "fit":
            return
        if self.state_key not in checkpoint["callbacks"]:
            raise ValueError("Resume requires the same SWA schedule. For a sota checkpoint created without averaging, "
                             "set plans.swa_tail_fraction=0 or start a new run.")
        if "current_model_state" in checkpoint:
            pl_module.load_state_dict(checkpoint["current_model_state"])

    def on_train_end(self, trainer, pl_module):
        if self.n_averaged:
            with torch.no_grad():
                for name, param in pl_module.named_parameters():
                    param.copy_(self.average[name])


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


def _detection_optimizer(kind, params, lr, settings):
    weight_decay = float(settings["weight_decay"])
    if kind == "adamw":
        if settings["amsgrad"] not in (0, 1):
            raise ValueError(f"finetuning.amsgrad={settings['amsgrad']!r}: expected 0 or 1.")
        optimizer = torch.optim.AdamW(
            params, lr, weight_decay=weight_decay, betas=(settings["beta1"], settings["beta2"]),
            eps=settings["eps"], amsgrad=bool(settings["amsgrad"]),
        )
    elif kind == "sgd":
        optimizer = torch.optim.SGD(
            params, lr, momentum=float(settings["momentum"]), weight_decay=weight_decay,
            nesterov=bool(settings["nesterov"]),
        )
    else:
        raise ValueError(f"finetuning.optimizer={kind!r}: expected 'adamw' or 'sgd'.")
    return optimizer


def configure_detection_optimizers(
    model,
    configs: dict,
    lr: float,
    scheduler_warmup_holder: dict,
    trainer,
    monitor: str = "val0_metric",
):
    """For AdamW and SGD, end fractional warm-up on the first optimiser step at or after its target."""
    recipe = configs.get("finetuning")
    settings = configs["nndet"] if recipe is None else recipe
    kind = "sgd" if recipe is None else recipe["optimizer"]
    fraction = None if recipe is None else recipe["warmup_fraction"]
    warm_iterations = settings.get("warm_iterations")
    tail = next((cb for cb in trainer.callbacks if isinstance(cb, DetectionTail)), None)
    if warm_iterations is not None or fraction is not None or tail is not None:
        num_steps = int(trainer.estimated_stepping_batches)
    if fraction is not None or tail is not None:
        steps_per_epoch = num_steps // trainer.max_epochs
        for cb in trainer.callbacks:
            if isinstance(cb, DebugEpochBatchLimit) and cb.n < trainer.num_training_batches:
                steps_per_epoch = cb.n // trainer.accumulate_grad_batches
                num_steps = steps_per_epoch * trainer.max_epochs
        assert steps_per_epoch > 0, "The detection schedule requires at least one optimizer step per epoch."
    if fraction is not None:
        if type(fraction) not in (float, int) or not 0 <= fraction < 1:
            raise ValueError(f"finetuning.warmup_fraction={fraction!r}: expected a fraction in [0, 1).")
        warm_iterations = ceil(Decimal(str(fraction)) * num_steps)
        if tail is not None and tail.start_epoch and warm_iterations >= tail.start_epoch * steps_per_epoch:
            raise ValueError("finetuning.warmup_fraction must leave room for decay before plans.swa_tail_fraction.")
    params = model.parameters()
    if warm_iterations is not None or tail is not None:
        params = get_params_no_wd_on_norm(model, float(settings["weight_decay"]))
    optimizer = _detection_optimizer(kind, params, lr, settings)
    scheduler_warmup_holder["scheduler_warmup"] = None
    if tail is not None:
        assert warm_iterations is not None or tail.start_epoch == 0, (
            "plans.swa_tail_fraction requires nndet.warm_iterations for the pre-tail schedule."
        )
        cycle = CycleLinear(optimizer, steps_per_epoch, lr / 10, lr / 1000)
        scheduler = cycle
        if tail.start_epoch:
            pre_tail_steps = tail.start_epoch * steps_per_epoch
            decay = LinearWarmupPolyLR(
                optimizer, int(warm_iterations), float(settings["warm_lr"]), float(settings["poly_gamma"]), pre_tail_steps
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [decay, cycle], [pre_tail_steps])
        lr_scheduler = {"scheduler": scheduler, "interval": "step"}
    elif warm_iterations is None:
        lr_scheduler = _plateau_lr_scheduler_config(optimizer, configs["scheduler_params"], monitor)
    else:
        scheduler_class = DetectionWarmupPolyLR if fraction is None else LinearWarmupPolyLR
        scheduler = scheduler_class(
            optimizer=optimizer,
            warm_iterations=int(warm_iterations),
            warm_lr=float(settings["warm_lr"]),
            poly_gamma=float(settings["poly_gamma"]),
            num_iterations=num_steps,
        )
        if fraction is not None and warm_iterations == num_steps:
            scheduler.poly_iterations = 1  # The final scheduler check is after the last optimiser update.
        lr_scheduler = {"scheduler": scheduler, "interval": "step"}
    if "finetuning" in configs and warm_iterations is not None:
        print(
            f"pretrained=nnfoundation optimizer={kind}: initial_lr={lr:g}, warm_lr={settings['warm_lr']}, "
            f"warm_iterations={warm_iterations}, warmup_fraction={fraction}, total_iterations={num_steps}, "
            f"poly_gamma={settings['poly_gamma']}"
        )
    return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
