from __future__ import annotations

from fran.trainers.trainer_rt import BatchSizeFinderRT, WandbLogBestCkptRT

from det3d.callback.wandb_det_grid import (
    WandbDetImageGridTrainCallback,
    WandbRetinaUNetImageGridTrainCallback,
)
from det3d.managers.data.labels import infer_det_labels_from_data_folder
from det3d.managers.data.run_through import DataManagerRTDet, DataManagerRTDetBTfms
from det3d.trainers.trainerdet import TrainerDet


class TrainerDetRunThrough(TrainerDet):
    """Run-through det training on the full train split, without a validation loop."""

    wandb_best_ckpt_cls = WandbLogBestCkptRT
    batchsize_finder_cls = BatchSizeFinderRT
    batchsize_finder_init_val = 2
    monitor_metric_name = "train0_loss"
    monitor_mode = "min"
    run_through = True

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

    def resolve_orchestrator_class(self, batch_tfms=None):
        if batch_tfms is None:
            batch_tfms = self.batch_tfms
        if self.debug and self.debug_tfm_keys:
            raise NotImplementedError(
                "TrainerDetRunThrough does not support tfm_debug datamanagers"
            )
        return DataManagerRTDetBTfms if batch_tfms else DataManagerRTDet

    def init_dm_unet(self, batch_size, override_dm_checkpoint):
        """A run-through resume also adopts the checkpoint's model_params."""
        super().init_dm_unet(batch_size, override_dm_checkpoint)
        if self.ckpt:
            self.configs["model_params"] = self.N.model_params

    def init_dm(self):
        dm_class = self.resolve_orchestrator_class(batch_tfms=self.batch_tfms)
        dm = dm_class(
            project_title=self.project.project_title,
            configs=self.configs,
            batch_size=self.configs["plan_train"]["batch_size"],
            cache_rate=self.configs["dataset_params"]["cache_rate"],
            device=self.configs["dataset_params"].get("device", "cuda"),
            ds_type=self.configs["dataset_params"].get("ds_type"),
            train_indices=self.train_indices,
            debug=self.debug,
            batch_tfms=self.batch_tfms,
            snapshot_freq=self.case_id_recorder_freq,
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
