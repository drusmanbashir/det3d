from __future__ import annotations

from pathlib import Path

import torch
from fran.callback.base import BatchSizeSafetyMargin
from fran.managers.wandb.wandb import WandbManager
from fran.trainers.helpers import (
    available_checkpoint_epochs_for_run,
    normalize_checkpoint_path,
    select_source_ckpt,
    switch_ckpt_keys,
)
from fran.trainers.trainer_rt import BatchSizeFinderRT
from lightning.pytorch.callbacks import EarlyStopping
from utilz.stringz import headline

from det3d.architectures.create_detector import arch_from_conf
from det3d.managers.detector_factory import resolve_detector_manager
from det3d.trainers.trainerdet import TrainerDet
from det3d.trainers.trainerdet_rt import TrainerDetRunThrough


class TrainerDetTransfer(TrainerDet):
    """Transfer-learning det trainer (mirrors fran TrainerTransfer)."""

    def __init__(
        self,
        project_title,
        configs,
        run_name=None,
        freeze=None,
        source_ckpt="interactive",
        resume_lr=None,
        ckpt=None,
        run_through: bool = False,
    ):
        assert freeze in [None, "encoder"], "Freeze either None or encoder"
        assert source_ckpt in ["interactive", "last"]
        assert run_name is not None or ckpt is not None, (
            "Please specificy a run or checkpoint to transfer learning from"
        )
        assert not (resume_lr is not None and run_name is None), (
            "resume_lr requires run_name for transfer"
        )
        super().__init__(
            project_title=project_title,
            configs=configs,
            run_name=None,
            ckpt_path=None,
        )
        self.run_through = bool(run_through)
        self.freeze = freeze
        self.source_run_name = run_name
        self.resume_lr = float(resume_lr) if resume_lr is not None else None
        self.ckpt_source = self.resolve_source_checkpoint(
            run_name=run_name,
            ckpt=ckpt,
            resume_lr=resume_lr,
            source_ckpt=source_ckpt,
        )
        self.ckpt = None
        if self.run_through:
            self.checkpoint_kwargs["save_on_train_epoch_end"] = True
            self.early_stopping_kwargs = {
                "monitor": "train0_loss",
                "mode": "min",
                "check_on_train_epoch_end": True,
            }

    def init_dm_unet(self, epochs, batch_size=None, override_dm_checkpoint=False):
        if override_dm_checkpoint:
            headline(
                "override_dm_checkpoint has no effect in transfer because no datamodule checkpoint is loaded."
            )
        self.configs["model_params"]["max_epochs"] = int(epochs)
        self.D = self.init_dm()
        source_manager = self.load_source_trainer(map_location="cpu")
        self.model_source = source_manager
        self.N = self.init_trainer(epochs)
        self.update_model()
        del self.model_source

    def resolve_source_checkpoint(
        self,
        run_name,
        ckpt,
        resume_lr,
        source_ckpt,
    ):
        if ckpt is not None:
            ckpt_path = Path(ckpt)
            if ckpt_path.exists() is False:
                raise RuntimeError(
                    f"No local checkpoint found for transfer source ckpt: {ckpt_path}"
                )
            raise NotImplementedError(
                "Explicit ckpt source selection for transfer is not implemented yet."
            )
        try:
            ckpts = available_checkpoint_epochs_for_run(run_name)
        except Exception as exc:
            raise RuntimeError(
                f"No local checkpoints found for transfer source run {run_name}."
            ) from exc
        if resume_lr is not None:
            return self.resolve_resume_lr_ckpt(
                run_name=run_name, resume_lr=resume_lr, ckpts=ckpts
            )
        source_ckpt_path = select_source_ckpt(run_name, source_ckpt)
        assert source_ckpt_path is not None, (
            f"No checkpoint found for source run: {run_name}"
        )
        return source_ckpt_path

    def resolve_resume_lr_ckpt(self, run_name: str, resume_lr: float, ckpts=None):
        if ckpts is None:
            ckpts = available_checkpoint_epochs_for_run(run_name)
        logger = WandbManager(
            project=self.project,
            run_id=run_name,
            wb_mode="online",
            log_model_checkpoints=False,
        )
        shifts = logger.lr_shift_epoch_map(run_id=run_name)
        if len(shifts) == 0:
            raise RuntimeError(
                f"No logged LR shifts found for transfer source run {run_name}; "
                f"cannot resolve resume_lr={resume_lr}."
            )
        deltas = (shifts["lr-Adam"].astype(float) - float(resume_lr)).abs()
        row = shifts.iloc[deltas.argsort()[:1]].iloc[0]
        shift_epoch = int(row["epoch"])
        after = [(epoch, ckpt) for epoch, ckpt in ckpts if epoch >= shift_epoch]
        if len(after) == 0:
            raise RuntimeError(
                f"No local checkpoint found at or after epoch {shift_epoch} for "
                f"source run {run_name}."
            )
        chosen = after[0][1]
        chosen = normalize_checkpoint_path(chosen)
        headline(
            "transfer resume_lr={} matched logged lr {} (prev_lr {}) at epoch {}; selected {}".format(
                resume_lr,
                row["lr-Adam"],
                row["prev_lr"],
                shift_epoch,
                chosen,
            )
        )
        return chosen

    def load_source_trainer(self, map_location: str = "cpu"):
        manager_cls = resolve_detector_manager(self.configs)
        try:
            source = manager_cls.load_from_checkpoint(
                self.ckpt_source,
                map_location=map_location,
                strict=True,
                weights_only=False,
            )
        except RuntimeError:
            switch_ckpt_keys(self.ckpt_source)
            source = manager_cls.load_from_checkpoint(
                self.ckpt_source,
                map_location=map_location,
                strict=True,
                weights_only=False,
            )
        headline(f"Source model loaded from checkpoint: {self.ckpt_source}")
        return source

    def update_model(self):
        self.report_head_mismatch()
        self.copy_weights()
        if self.freeze == "encoder":
            self.freeze_encoder()

    def freeze_encoder(self):
        arch = arch_from_conf(self.configs)
        if arch == "retinaunet":
            enc = self.N.net.encoder
        else:
            enc = self.N.detector.network.feature_extractor
        for param in enc.parameters():
            param.requires_grad = False

    def report_head_mismatch(self):
        old_n = len(self.model_source.plan["fg_labels"])
        new_n = len(self.configs["plan_train"]["fg_labels"])
        if old_n != new_n:
            headline(
                "Source fg_labels count ({0}) != target ({1}); detection heads "
                "will be skipped by shape and remain target-initialized.".format(
                    old_n, new_n
                )
            )

    def copy_weights(self):
        src_sd = self.model_source.state_dict()
        tgt_sd = self.N.state_dict()
        copied, skipped = 0, 0
        with torch.no_grad():
            for key, src_val in src_sd.items():
                tgt_val = tgt_sd.get(key)
                if tgt_val is None or tgt_val.shape != src_val.shape:
                    skipped += 1
                    continue
                tgt_sd[key].copy_(src_val)
                copied += 1
        self.N.load_state_dict(tgt_sd, strict=False)
        headline(
            f"Copied {copied} tensors from source model; skipped {skipped} tensors."
        )

    def init_cbs(
        self,
        extra_cbs,
        wandb,
        batchsize_finder,
        profiler,
        tags,
        description="",
        early_stopping=False,
        early_stopping_patience=30,
        lr_floor=None,
        permanent_checkpoint_every_n_epochs: int = 100,
        wandb_grid_epoch_freq: int = 5,
    ):
        cbs, logger, profiler = super().init_cbs(
            extra_cbs=extra_cbs,
            wandb=wandb,
            batchsize_finder=batchsize_finder,
            profiler=profiler,
            tags=tags,
            description=description,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            lr_floor=lr_floor,
            permanent_checkpoint_every_n_epochs=permanent_checkpoint_every_n_epochs,
            wandb_grid_epoch_freq=wandb_grid_epoch_freq,
        )
        if self.run_through and early_stopping:
            cbs = [cb for cb in cbs if not isinstance(cb, EarlyStopping)]
            early_stopping_cfg = {
                "patience": early_stopping_patience,
                "mode": "min",
                "min_delta": 0.0,
            }
            early_stopping_cfg.update(self.early_stopping_kwargs)
            cbs.append(EarlyStopping(**early_stopping_cfg))
        return cbs, logger, profiler

    def fit(self):
        try:
            self.trainer.fit(model=self.N, datamodule=self.D, ckpt_path=None)
        except KeyboardInterrupt:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.finish()
            except Exception:
                pass
            raise


class TrainerDetTransferRT(TrainerDetTransfer, TrainerDetRunThrough):
    """Transfer det training in run-through mode (mirrors fran TrainerTransferRT)."""

    def __init__(
        self,
        project_title,
        configs,
        run_name=None,
        freeze=None,
        source_ckpt="interactive",
        resume_lr=None,
        ckpt=None,
    ):
        super().__init__(
            project_title=project_title,
            configs=configs,
            run_name=run_name,
            freeze=freeze,
            source_ckpt=source_ckpt,
            resume_lr=resume_lr,
            ckpt=ckpt,
            run_through=True,
        )

    def init_cbs(
        self,
        extra_cbs,
        wandb,
        batchsize_finder,
        profiler,
        tags,
        description="",
        early_stopping=False,
        early_stopping_patience=30,
        lr_floor=None,
        permanent_checkpoint_every_n_epochs: int = 100,
        wandb_grid_epoch_freq: int = 5,
    ):
        cbs, logger, profiler = super().init_cbs(
            extra_cbs=extra_cbs,
            wandb=wandb,
            batchsize_finder=False,
            profiler=profiler,
            tags=tags,
            description=description,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            lr_floor=lr_floor,
            permanent_checkpoint_every_n_epochs=permanent_checkpoint_every_n_epochs,
            wandb_grid_epoch_freq=wandb_grid_epoch_freq,
        )
        if batchsize_finder:
            cbs[1:1] = [
                BatchSizeFinderRT(batch_arg_name="batch_size", mode="binsearch"),
                BatchSizeSafetyMargin(),
            ]
        return cbs, logger, profiler


if __name__ == "__main__":
# %%
# SECTION:-------------------- setup<--------------------------------------------------------------------------------------
    from fran.managers import Project
    from torch import Tensor
    from utilz.imageviewers import ImageBBoxViewer

    from det3d.configs.parser import ConfigMakerDet

    project_title = "lidca"
    plan_id = 4

    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = C.configs
    conf["dataset_params"]["fold"] = 0

# SECTION:-------------------- TRAINING --------------------------------------------------------------------------------------
# %%
    conf["model_params"]["arch"] = "retinanet"
    conf["model_params"]["arch"] = "retinaunet"
    print(conf["dataset_params"]["prezoom_scale"])
    print(conf["plan_train"]["patch_size"])# = [128,128,64]

    bs = 10
    device_id = 0
    batch_tfms = True
    batch_tfms = False
    wandb = True
    batchsize_finder= False
    run_name = None
    run_name = "LIDCA-DIET"
    tags = []
    description = "TrainerDet lidca retinanet"
    lr = None
    debug_ = False
    profiler = False
    compiled = False
    cbs = []
    wandb_grid_epoch_freq = 40
    val_every_n_epochs = 5
    train_indices = None
    val_indices = None
    val_sampling = 1.0
    epochs = 500

# %%
#SECTION:-------------------- TRAIN--------------------------------------------------------------------------------------
    Tm = TrainerDetTransfer(P.project_title, conf, run_name=run_name,  source_ckpt="last", resume_lr=None)
# %%
    Tm.setup(
        compiled=compiled,
        train_indices=train_indices,
        batch_tfms=batch_tfms,
        cbs=cbs,
        debug=debug_,
        batch_size=bs,
        devices=[device_id],
        epochs=600,
        batchsize_finder=batchsize_finder,
        wandb=wandb,
        wandb_grid_epoch_freq=wandb_grid_epoch_freq,
        tags=tags,
        description=description,
        lr=lr,
    )
#
    Tm.fit()

# %%
