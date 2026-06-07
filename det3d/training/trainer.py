from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
from det.managers.retinanet import RetinaNetManager
from det.managers.data import DataManagerDualDet
from fran.callback.debug_epoch_limit import DebugEpochBatchLimit
from fran.callback.incremental import LRFloorStop
from fran.callback.wandb.wandb import WandbLogBestCkpt
from fran.managers.wandb.wandb import WandbManager
from fran.configs.helpers import normalize_logging_payload
from fran.managers import Project
from fran.trainers.helpers import checkpoint_from_model_id, switch_ckpt_keys
from fran.trainers.trainer import Trainer
from lightning.pytorch import Trainer as TrainerL
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.profilers import AdvancedProfiler
from utilz.stringz import headline


def _flatten_dict(d: dict, base: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{base}/{k}" if base else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


class TrainerDet(Trainer):
    monitor_metric_name = "val0_metric"

    def __init__(
        self,
        project_title,
        configs,
        run_name=None,
        ckpt_path: Optional[str | Path] = None,
    ):
        self.project = Project(project_title=project_title)
        self.configs = configs
        self.run_name = run_name
        if ckpt_path is not None:
            self.ckpt = Path(ckpt_path)
        else:
            self.ckpt = None if run_name is None else checkpoint_from_model_id(run_name)
        self.qc_configs(configs, self.project)
        self.checkpoint_kwargs = {}
        self.early_stopping_kwargs = {
            "monitor": self.monitor_metric_name,
            "mode": "max",
            "check_on_train_epoch_end": False,
        }

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
        val_every_n_epochs: int = None,
        cbs=[],
        tags=[],
        description="",
        epochs=None,
        batchsize_finder=False,
        override_dm_checkpoint=False,
        early_stopping=False,
        early_stopping_patience=30,
        lr_floor=None,
        wandb_grid_epoch_freq: int = 5,
        permanent_checkpoint_every_n_epochs: int = 100,
    ):
        if val_every_n_epochs is None:
            val_every_n_epochs = int(self.configs["plan_train"].get("val_every_n_epochs", 5))
        if epochs is None:
            epochs = int(self.configs["plan_train"].get("max_epochs", 300))

        self.val_every_n_epochs = int(val_every_n_epochs)
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.val_sampling = float(val_sampling)
        self.debug = bool(debug)
        self.maybe_alter_configs(batch_size, compiled)
        self.set_lr(lr)

        has_cuda = torch.cuda.is_available()
        if has_cuda:
            self.set_strategy(devices)
            trainer_devices = devices
            accelerator = "gpu"
            strategy = self.strategy
        else:
            self.devices = 1
            self.sync_dist = False
            self.strategy = "auto"
            trainer_devices = 1
            accelerator = "cpu"
            strategy = "auto"

        self.init_dm_unet(epochs, batch_size, override_dm_checkpoint)
        self.D.prepare_data()
        self.D.setup(stage="fit")

        cbs, logger, profiler = self.init_cbs(
            cbs=cbs,
            wandb=wandb,
            batchsize_finder=False,
            profiler=profiler,
            tags=tags,
            description=description,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            lr_floor=lr_floor,
            permanent_checkpoint_every_n_epochs=permanent_checkpoint_every_n_epochs,
        )
        self._ensure_local_ckpt_on_wandb_resume(logger)

        self.trainer = TrainerL(
            callbacks=cbs,
            accelerator=accelerator,
            devices=trainer_devices,
            precision="16-mixed" if has_cuda else 32,
            profiler=profiler,
            logger=logger,
            max_epochs=epochs,
            check_val_every_n_epoch=self.val_every_n_epochs,
            log_every_n_steps=logging_freq,
            num_sanity_val_steps=0,
            enable_checkpointing=True,
            default_root_dir=self.project.checkpoints_parent_folder,
            strategy=strategy,
        )

    def init_dm(self):
        plan = self.configs["plan_train"]
        batch_size = int(
            plan.get("batch_size", self.configs["dataset_params"].get("batch_size", 4))
        )
        self.configs["dataset_params"]["batch_size"] = batch_size
        cache_rate = self.configs["dataset_params"].get("cache_rate", 0.0)
        ds_type = self.configs["dataset_params"].get("ds_type")
        self.D = DataManagerDualDet(
            project_title=self.project.project_title,
            configs=self.configs,
            batch_size=batch_size,
            cache_rate=cache_rate,
            device=self.configs["dataset_params"].get("device", "cuda"),
            ds_type=ds_type,
            train_indices=self.train_indices,
            val_indices=self.val_indices,
            val_sampling=self.val_sampling,
            debug=self.debug,
        )
        return self.D

    def init_dm_unet(self, epochs, batch_size, override_dm_checkpoint=False):
        if self.ckpt:
            self.N = self.load_trainer()
            self.D = self.init_dm()
        else:
            self.D = self.init_dm()
            self.N = self.init_trainer(epochs)
        headline(f"Data Manager initialized.\n {self.D}")

    def init_trainer(self, epochs):
        return RetinaNetManager(
            project_title=self.project.project_title,
            configs=self.configs,
            lr=self.lr,
            sync_dist=self.sync_dist,
        )

    def load_trainer(self, map_location="cpu", **kwargs):
        try:
            return RetinaNetManager.load_from_checkpoint(
                self.ckpt, map_location=map_location, strict=True, **kwargs
            )
        except RuntimeError:
            switch_ckpt_keys(self.ckpt)
            return RetinaNetManager.load_from_checkpoint(
                self.ckpt, map_location=map_location, strict=True, **kwargs
            )

    def maybe_alter_configs(self, batch_size, compiled):
        if batch_size is not None:
            self.configs["plan_train"]["batch_size"] = int(batch_size)

    def set_lr(self, lr):
        if lr and not self.ckpt:
            self.lr = float(lr)
        elif lr and self.ckpt:
            self.lr = float(lr)
        elif lr is None and self.ckpt:
            self.state_dict = torch.load(self.ckpt, weights_only=False, map_location="cpu")
            self.lr = self.state_dict["lr_schedulers"][0]["_last_lr"][0]
        else:
            self.lr = float(self.configs["plan_train"]["lr"])

    def qc_configs(self, configs, project):
        return

    def init_cbs(
        self,
        cbs,
        wandb,
        batchsize_finder,
        profiler,
        tags,
        description="",
        early_stopping=False,
        early_stopping_patience=30,
        lr_floor=None,
        permanent_checkpoint_every_n_epochs: int = 100,
    ):
        cbs = []
        if self.debug:
            cbs += [DebugEpochBatchLimit(n=2)]

        cbs += [
            ModelCheckpoint(
                save_top_k=2,
                save_last=True,
                monitor=self.monitor_metric_name,
                mode="max",
                every_n_epochs=5,
                filename="{epoch}-{" + self.monitor_metric_name + ":.4f}",
                enable_version_counter=True,
                auto_insert_metric_name=True,
            ),
            ModelCheckpoint(
                save_top_k=-1,
                save_last=True,
                every_n_epochs=int(permanent_checkpoint_every_n_epochs),
                filename="epoch{epoch:04d}-snapshot",
                enable_version_counter=False,
                auto_insert_metric_name=False,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        if early_stopping:
            cbs += [
                EarlyStopping(
                    monitor=self.monitor_metric_name,
                    mode="max",
                    patience=int(early_stopping_patience),
                )
            ]

        if lr_floor is not None:
            cbs += [LRFloorStop(min_lr=lr_floor)]

        logger = None
        if wandb:
            logger = WandbManager(
                project=self.project,
                run_id=self.run_name,
                log_model_checkpoints=False,
                tags=tags,
                notes=description,
            )
            dm_cfg = {
                "dataset_params": normalize_logging_payload(
                    deepcopy(self.D.configs["dataset_params"])
                ),
                "plan_train": normalize_logging_payload(
                    deepcopy(self.D.configs["plan_train"])
                ),
            }
            flat_cfg = _flatten_dict(dm_cfg, base="configs/datamodule")
            logger.experiment.config.update(flat_cfg, allow_val_change=True)
            cbs += [WandbLogBestCkpt()]

        if profiler:
            profiler = AdvancedProfiler(
                dirpath=self.project.log_folder, filename="profiler"
            )
            cbs += [DeviceStatsMonitor(cpu_stats=True)]
        else:
            profiler = None

        return cbs, logger, profiler
