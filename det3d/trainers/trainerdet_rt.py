from __future__ import annotations

from pathlib import Path
from typing import Optional

from fran.callback.base import BatchSizeSafetyMargin
from fran.trainers.trainer_rt import BatchSizeFinderRT, WandbLogBestCkptRT
from utilz.stringz import headline

from det3d.callback.wandb_det_grid import (
    WandbDetImageGridTrainCallback,
    WandbRetinaUNetImageGridTrainCallback,
)
from det3d.managers.data.labels import infer_det_labels_from_data_folder
from det3d.managers.data.run_through import DataManagerRTDet, DataManagerRTDetBTfms
from det3d.trainers.trainerdet import TrainerDet


class TrainerDetRunThrough(TrainerDet):
    """Run-through det training on full train split (mirrors fran TrainerRT)."""

    case_id_recorder_cls = None
    wandb_best_ckpt_cls = WandbLogBestCkptRT
    batchsize_finder_cls = BatchSizeFinderRT
    monitor_metric_name = "train0_loss"

    def __init__(
        self,
        project_title,
        configs,
        run_name=None,
        ckpt_path: Optional[str | Path] = None,
    ):
        super().__init__(
            project_title=project_title,
            configs=configs,
            run_name=run_name,
            ckpt_path=ckpt_path,
        )
        self.run_through = True
        self.checkpoint_kwargs["save_on_train_epoch_end"] = True
        self.early_stopping_kwargs = {
            "monitor": "train0_loss",
            "mode": "min",
            "check_on_train_epoch_end": True,
        }

    def wandb_grid_callback(self, wandb_grid_epoch_freq: int):
        """Build train-batch wandb grid callback for run-through (no val dataloader).

        Uses ``*TrainCallback`` variants that sample predictions from the train
        loop. For ``retinaunet``, selects the RetinaUNet grid and enables nnDet
        box layout adaptation (xyxyzz -> xyzxyz).
        """
        arch = self.configs["model_params"]["arch"]
        if arch == "retinaunet":
            kwargs = self.wandb_retinaunet_grid_cb_kwargs(wandb_grid_epoch_freq)
            kwargs["adapt_nndet_boxes"] = True
            return WandbRetinaUNetImageGridTrainCallback(**kwargs)
        return WandbDetImageGridTrainCallback(
            **self.wandb_grid_cb_kwargs(wandb_grid_epoch_freq)
        )

    def setup(
        self,
        batch_size=None,
        train_indices=None,
        val_indices=None,
        val_sampling: float = 1.0,
        logging_freq=25,
        lr=None,
        devices=1,
        compiled=None,
        wandb=True,
        profiler=False,
        debug: bool = False,
        debug_tfm_keys: str | None = None,
        cbs=[],
        tags=[],
        description="",
        epochs=600,
        batchsize_finder=False,
        override_dm_checkpoint=False,
        lr_floor=None,
        wandb_grid_epoch_freq: int = 5,
        permanent_checkpoint_every_n_epochs: int = 100,
        batch_tfms: bool = False,
        snapshot_freq: int = 1000,
    ):
        """Configure run-through det training on the full train split.

        Sets permanent-checkpoint and early-stopping kwargs for ``train0_loss``,
        then delegates to ``TrainerDet.setup`` (no validation loop in run-through).
        """
        self.permanent_checkpoint_every_n_epochs = int(
            permanent_checkpoint_every_n_epochs
        )
        self.snapshot_freq = int(snapshot_freq)
        return super().setup(
            batch_size=batch_size,
            train_indices=train_indices,
            val_indices=val_indices,
            val_sampling=val_sampling,
            logging_freq=logging_freq,
            lr=lr,
            devices=devices,
            compiled=compiled,
            wandb=wandb,
            profiler=profiler,
            debug=debug,
            debug_tfm_keys=debug_tfm_keys,
            cbs=cbs,
            tags=tags,
            description=description,
            epochs=epochs,
            batchsize_finder=batchsize_finder,
            override_dm_checkpoint=override_dm_checkpoint,
            early_stopping=False,
            lr_floor=lr_floor,
            wandb_grid_epoch_freq=wandb_grid_epoch_freq,
            permanent_checkpoint_every_n_epochs=permanent_checkpoint_every_n_epochs,
            batch_tfms=batch_tfms,
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
        early_stopping_monitor="train0_loss",
        early_stopping_mode="min",
        early_stopping_min_delta=0.0,
    ):
        self.permanent_checkpoint_every_n_epochs = int(
            permanent_checkpoint_every_n_epochs
        )
        self.early_stopping_kwargs.update(
            {
                "monitor": early_stopping_monitor,
                "mode": early_stopping_mode,
                "min_delta": float(early_stopping_min_delta),
            }
        )
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

    def resolve_orchestrator_class(self, batch_tfms=None):
        if batch_tfms is None:
            batch_tfms = self.batch_tfms
        if self.debug and self.debug_tfm_keys:
            raise NotImplementedError(
                "TrainerDetRunThrough does not support tfm_debug datamanagers"
            )
        return DataManagerRTDetBTfms if batch_tfms else DataManagerRTDet

    def init_dm_unet(self, epochs, batch_size, override_dm_checkpoint=False):
        if self.ckpt:
            if override_dm_checkpoint:
                headline(
                    "Run-through resume ignores override_dm_checkpoint because the datamodule is rebuilt from current Trainer configs."
                )
            headline(
                "Run-through resume: loading model checkpoint and rebuilding datamodule from current config."
            )
            self.D = self.init_dm()
            self.N = self.load_trainer()
            self.configs["model_params"] = self.N.model_params
        else:
            self.D = self.init_dm()
            self.N = self.init_trainer(epochs)
        headline(f"Data Manager initialized.\n {self.D}")

    def init_dm(self):
        dm_class = self.resolve_orchestrator_class(batch_tfms=self.batch_tfms)
        dm = dm_class(
            project_title=self.project.project_title,
            configs=self.configs,
            batch_size=self.configs["dataset_params"]["batch_size"],
            cache_rate=self.configs["dataset_params"]["cache_rate"],
            device=self.configs["dataset_params"].get("device", "cuda"),
            ds_type=self.configs["dataset_params"].get("ds_type"),
            train_indices=self.train_indices,
            debug=self.debug,
            batch_tfms=self.batch_tfms,
            snapshot_freq=self.snapshot_freq,
        )
        infer_det_labels_from_data_folder(dm=dm, configs=self.configs)
        return dm


TrainerDetRT = TrainerDetRunThrough


if __name__ == "__main__":
# SECTION:-------------------- setup<--------------------------------------------------------------------------------------
    from fran.managers import Project

    from det3d.configs.parser import ConfigMakerDet

    project_title = "lidca"
    plan_id = 4

    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = C.configs
    conf["dataset_params"]["fold"] = 0

# %%
# SECTION:-------------------- TRAINING --------------------------------------------------------------------------------------
    conf["model_params"]["arch"] = "retinaunet"
    print(conf["dataset_params"]["prezoom_scale"])

    bs = 2
    device_id = 0
    batch_tfms = False
    wandb = True
    tags = ["runthrough"]
    description = "native nndet pipeline"
    lr = None
    debug_ = False
    profiler = False
    compiled = False
    cbs = []
    batchsize_finder = False
    wandb_grid_epoch_freq = 5
    train_indices = None
    val_indices = None
    val_sampling = 1.0
    epochs = 500
    run_name = None
    run_name ="LIDCA-OILS"

# %%
    Tm = TrainerDetRT(P.project_title, conf, run_name=run_name)

# %%
    Tm.setup(
        compiled=compiled,
        train_indices=train_indices,
        val_indices=val_indices,
        val_sampling=val_sampling,
        cbs=cbs,
        debug=debug_,
        batch_size=bs,
        batch_tfms=batch_tfms,
        devices=[device_id],
        epochs=epochs,
        profiler=profiler,
        wandb=wandb,
        wandb_grid_epoch_freq=wandb_grid_epoch_freq,
        batchsize_finder=batchsize_finder,
        tags=tags,
        description=description,
        lr=lr,
    )

# %%
    Tm.fit()

# %%
    N = Tm.N
    D = Tm.D
    D.setup()
    D.prepare_data()
    tmt = D.train_manager

# %%
# %%
#SECTION:-------------------- TS--------------------------------------------------------------------------------------
    cb = Tm.get_callback("ModelCheckpoint")
    tr = Tm.trainer
    mon =cb._monitor_candidates(tr)
    cb._save_last_checkpoint(tr,mon)

    filepath = cb.format_checkpoint_name(mon, cb.CHECKPOINT_NAME_LAST)
