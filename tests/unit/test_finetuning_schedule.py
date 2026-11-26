import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from det3d.managers.det_schedule import DetectionTail, configure_detection_optimizers, resolve_tail_epochs
from fran.callback.debug_epoch_limit import DebugEpochBatchLimit
from fran.configs.parser import ConfigMaker
from fran.managers.unet_fm import UNetManagerFM


class ScheduleProbe(pl.LightningModule):
    def __init__(self, pipeline, fraction):
        super().__init__()
        self.pipeline = pipeline
        self.model = torch.nn.Linear(1, 1)
        self.lr = .01
        self.fm_config = {"lr_scheduler": "warmup_poly", "warmup_fraction": fraction}
        self.model_params = {"poly_exponent": 1.}
        self.trace = []

    def training_step(self, batch, batch_idx):
        lr = self.optimizers().param_groups[0]["lr"]
        tail = next((cb for cb in self.trainer.callbacks if isinstance(cb, DetectionTail)), None)
        self.trace.append((self.global_step, lr, tail.n_averaged if tail else 0))
        return self.model(torch.ones(1, 1)).sum()

    def configure_optimizers(self):
        if self.pipeline == "seg":
            opt = torch.optim.SGD(self.parameters(), lr=self.lr)
            return {"optimizer": opt, "lr_scheduler": UNetManagerFM._scheduler_from_config(self, opt)}
        cfg = {"finetuning": {**self.fm_config, "optimizer": "sgd", "warm_lr": .001,
                              "momentum": .9, "nesterov": True, "weight_decay": 0., "poly_gamma": 1.}}
        return configure_detection_optimizers(self.model, cfg, self.lr, {}, self.trainer)


def trainer(epochs, path, callbacks=(), **kwargs):
    return pl.Trainer(max_epochs=epochs, accelerator="cpu", devices=1, logger=False,
                      enable_checkpointing=bool(callbacks), enable_progress_bar=False,
                      enable_model_summary=False, default_root_dir=path, callbacks=list(callbacks), **kwargs)


@pytest.mark.parametrize("pipeline,cell,fraction,batches,accumulate,warm", [
    ("seg", "auto", .05, 250, 1, 12500), ("det", "auto", .032, 2500, 1, 4000),
    ("seg", 5, .25, 7, 2, 8), ("det", 5, .22, 7, 2, 5),
    ("seg", 150, .05, 9, 1, 72), ("det", 60, .032, 150, 1, 288),
    ("seg", None, .05, 5, 1, 250), ("det", None, .032, 5, 1, 160),
    ("seg", 100, .07, 10, 1, 70), ("det", 10, .07, 10, 1, 7),
])
def test_budget(tmp_path, capsys, pipeline, cell, fraction, batches, accumulate, warm):
    cfg = {"plan_train": {"epochs": cell, "pretrained": "nnfoundation"}}
    if pipeline == "det":
        cfg["nndet"] = {}
    plan = ConfigMaker.resolve_epochs(cfg)
    assert plan["train"] == (batches if cell == "auto" else None)
    model = ScheduleProbe(pipeline, fraction)
    owner = trainer(plan["epochs"], tmp_path, accumulate_grad_batches=accumulate)
    owner.strategy.connect(model)
    owner._data_connector.attach_data(model, train_dataloaders=DataLoader(range(batches)))
    model.trainer = owner
    sched = model.configure_optimizers()["lr_scheduler"]["scheduler"]
    log = capsys.readouterr().out
    if pipeline == "seg":
        per_epoch = (batches + accumulate - 1) // accumulate
        assert sched.warmup_epochs * per_epoch == warm
        assert f"{sched.warmup_epochs} epochs ({warm:,} iterations)" in log
    else:
        assert sched.warm_iterations == warm
        assert f"warm_iterations={warm}," in log
    print(log.strip())


def test_seg_epoch_transition(tmp_path, capsys):
    model = ScheduleProbe("seg", .25)
    owner = trainer(5, tmp_path)
    owner.fit(model, DataLoader(range(4)))
    assert [r for _, r, _ in model.trace] == pytest.approx(
        [0.005] * 4 + [.01] * 8 + [.01 * 2/3] * 4 + [.01/3] * 4)
    assert "2 epochs (8 iterations)" in capsys.readouterr().out


@pytest.mark.parametrize("tail_fraction", [0, .45])
def test_det_mid_epoch_resume_and_tail(tmp_path, tail_fraction):
    # 5 full passes of 4 batches. Warm-up ends at ceil(4.4) = step 5.
    # A 45% tail starts at the first whole epoch at or after step 11, i.e. 12.
    def callbacks(folder):
        marks = [ModelCheckpoint(dirpath=folder, filename="step-{step}", every_n_train_steps=2,
                                 save_top_k=-1, auto_insert_metric_name=False)]
        tail = resolve_tail_epochs(tail_fraction, 5, 1)
        if tail:
            marks.append(DetectionTail(5 - tail))
        return marks

    full = ScheduleProbe("det", .22)
    owner = trainer(5, tmp_path, callbacks(tmp_path/"full"))
    owner.fit(full, DataLoader(range(4)))
    steps, rates, averages = zip(*full.trace)
    assert steps == tuple(range(20))
    horizon = 12 if tail_fraction else 20
    assert rates[:6] == pytest.approx([.001 + .009*i/5 for i in range(1,6)] + [.01*(1-1/(horizon-5))])
    if tail_fraction:
        assert averages == (0,)*12 + (1,)*4 + (2,)*4
        assert rates[11] != rates[12] and rates[12] == pytest.approx(.00001)
    resumed = ScheduleProbe("det", .22)
    checkpoint = tmp_path/"full"/"step-2.ckpt"
    assert torch.load(checkpoint, weights_only=False)["global_step"] == 2
    trainer(5, tmp_path, callbacks(tmp_path/"resumed")).fit(
        resumed, DataLoader(range(4)), ckpt_path=checkpoint)
    assert resumed.trace == full.trace[2:]


def test_det_last_step_can_end_warmup(tmp_path):
    model = ScheduleProbe("det", .99)
    trainer(1, tmp_path).fit(model, DataLoader(range(2)))
    assert [lr for _, lr, _ in model.trace] == pytest.approx([.0055, .01])


def test_debug_epoch_boundary(tmp_path):
    model = ScheduleProbe("seg", .25)
    owner = trainer(5, tmp_path, [DebugEpochBatchLimit(3)], accumulate_grad_batches=2)
    owner.fit(model, DataLoader(range(12)))
    assert owner.global_step == 10
    assert [row[1] for row in model.trace] == pytest.approx(
        [.005]*3 + [.01]*6 + [.01*2/3]*3 + [.01/3]*3)
