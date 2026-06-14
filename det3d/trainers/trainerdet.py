from copy import deepcopy
from pathlib import Path
from typing import Optional

from lightning.fabric import Fabric
import torch
from det3d.callback.case_recorder_det import CaseIDRecorderSnapshotDet
from det3d.detection.retinanet_train import forward_train_batched
from det3d.managers.retinanet_bk import RetinaNetManager
from det3d.managers.data import (
    DataManagerDualDet,
    DataManagerDualDetBTfms,
)
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
from utilz.imageviewers import ImageBBoxViewer
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
    case_id_recorder_cls = CaseIDRecorderSnapshotDet
    monitor_metric_name = "val0_metric"
    _DET_PIPELINE_MODES = frozenset({"det", "lbd"})

    def case_id_recorder_dl_idx(self) -> int:
        return 0

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
        self.ckpt = Path(ckpt_path) if ckpt_path is not None else None
        self._wandb_run_is_new = False if run_name is None else None
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
        batch_tfms: bool = True,
        case_id_recorder_freq: int = 50,
    ):
        if val_every_n_epochs is None:
            val_every_n_epochs = int(self.configs["plan_train"].get("val_every_n_epochs", 5))
        if epochs is None:
            epochs = int(self.configs["plan_train"].get("max_epochs", 300))

        self.val_every_n_epochs = int(val_every_n_epochs)
        self.case_id_recorder_freq = int(case_id_recorder_freq)
        assert self.case_id_recorder_freq % self.val_every_n_epochs == 0, (
            "case_id_recorder_freq must be divisible by val_every_n_epochs so "
            "snapshot validation can coincide with validation epochs."
        )
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.val_sampling = float(val_sampling)
        self.debug = bool(debug)
        self.batch_tfms = bool(batch_tfms)
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

        self.configs["plan_train"]["val_every_n_epochs"] = self.val_every_n_epochs
        self._resolve_run_ckpt(wandb=wandb)
        self.init_dm_unet(epochs, batch_size, override_dm_checkpoint)
        self.D.prepare_data()
        self.D.setup(stage="fit")
        headline(
            "Data module ready.\n"
            f"  train: {type(self.D.train_manager).__name__} — {self.D.train_manager}\n"
            f"  valid: {type(self.D.valid_manager).__name__} — {self.D.valid_manager}"
        )

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

        trainer_kwargs = dict(
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
        self.trainer = TrainerL(**trainer_kwargs)

    def resolve_orchestrator_class(self, batch_tfms=None):
        if batch_tfms is None:
            batch_tfms = self.batch_tfms
        return DataManagerDualDetBTfms if batch_tfms else DataManagerDualDet

    def normalize_plan_modes_for_det_pipeline(self):
        """Shim: ConfigMakerDet may set mode=det; DM infer only knows fran modes."""
        for key in ("plan_train", "plan_valid", "plan_test"):
            plan = self.configs[key]
            if plan["mode"] in self._DET_PIPELINE_MODES:
                plan["mode"] = "lbd"

    def qc_configs(self, configs, project):
        self.normalize_plan_modes_for_det_pipeline()

    def init_dm(self):
        self.normalize_plan_modes_for_det_pipeline()
        plan = self.configs["plan_train"]
        batch_size = int(
            plan.get("batch_size", self.configs["dataset_params"].get("batch_size", 64))
        )
        self.configs["dataset_params"]["batch_size"] = batch_size
        cache_rate = self.configs["dataset_params"].get("cache_rate", 0.0)
        ds_type = self.configs["dataset_params"].get("ds_type")
        dm_class = self.resolve_orchestrator_class()
        if "Det" not in dm_class.__name__:
            raise RuntimeError(
                f"TrainerDet orchestrator must be DataManagerDualDet*, got {dm_class.__name__}. "
                "Use: from det3d.trainers.trainerdet import TrainerDet"
            )
        dm_kwargs = dict(
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
            batch_tfms=self.batch_tfms,
        )
        self.D = dm_class(**dm_kwargs)
        return self.D

    def init_dm_unet(self, epochs, batch_size, override_dm_checkpoint=False):
        if self.ckpt:
            self.N = self.load_trainer()
            self.D = self.init_dm()
        else:
            self.D = self.init_dm()
            self.N = self.init_trainer(epochs)

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
            super().set_lr(None)
        else:
            self.lr = float(self.configs["plan_train"]["lr"])

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
        cbs = [
            self.case_id_recorder_cls(
                freq=self.case_id_recorder_freq,
                local_folder=str(self.project.log_folder / "case_recorder"),
                monitor_dl="both",
                dl_idx=self.case_id_recorder_dl_idx(),
            ),
        ]
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

    def setup_model_for_cuda(self, device=0, precision="16-mixed"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        if not hasattr(self, "N"):
            raise RuntimeError("Call setup() before setup_model_for_cuda().")
        fabric, model = self.setup_tm_for_cuda(
            device=device, precision=precision, wrap_inner_model=False
        )
        self.fabric_infer = fabric
        return model

    def setup_tm_for_cuda(
        self, device=0, precision="bf16-mixed", wrap_inner_model=True
    ):
        fabric = Fabric(
            accelerator="gpu",
            devices=[device],
            precision=precision,
        )

        if wrap_inner_model:
            self.N.detector.eval()
            self.N.detector = fabric.setup(self.N.detector)
            model = self.N.detector
        else:
            self.N.eval()
            self.N = fabric.setup(self.N)
            model = self.N

        return fabric, model


CaseIDRecorderDetRT = CaseIDRecorderSnapshotDet


# %%
if __name__ == "__main__":
#SECTION:-------------------- setup<--------------------------------------------------------------------------------------
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers import Project
    from utilz.helpers import pp

    P = Project("lidc")
    C = ConfigMakerDet(P)
    C.setup(1)
    conf = C.configs
    pp(conf["plan_train"])

# SECTION:-------------------- TRAINING --------------------------------------------------------------------------------------
# %%
    device_id = 0
    wandb = True
    run_name = "LIDC-TAINT"
    run_name = None
    run_name = "LIDC-TAINT4"
    description = "new changes after LIDC-TRAINT which failed"
    tags = []
    conf["dataset_params"]["fold"] = 0
    lr = None
    debug_ = False
    profiler = False
    cbs = []
    val_every_n_epochs = 2
    case_id_recorder_freq = 8
    train_indices = None
    val_indices = None
    val_sampling = 1.0
    epochs = None
    batch_size = 8
# SECTION:-------------------- TRAINING --------------------------------------------------------------------------------------
    Tm = TrainerDet(P.project_title, conf, None)
    if run_name is not None:
        Tm.run_name = run_name
# %%
    Tm.setup(
        train_indices=train_indices,
        val_indices=val_indices,
        val_sampling=val_sampling,
        val_every_n_epochs=val_every_n_epochs,
        cbs=cbs,
        debug=debug_,
        batch_size=batch_size,
        devices=[device_id],
        epochs=epochs,
        profiler=profiler,
        wandb=wandb,
        tags=tags,
        description=description,
        lr=lr,
        case_id_recorder_freq=case_id_recorder_freq,
    )
# %%
    Tm.fit()
# %%
#SECTION:-------------------- TS--------------------------------------------------------------------------------------
    N = Tm.N
    D = Tm.D
    tmt = D.train_manager
    tmv = D.valid_manager
# %%
    tmt.setup()
    tmv.setup()
    train_dl = tmt.dl
    val_dl = tmv.dl
    train_iter = iter(train_dl)
    tmt.ds[0]
# %%
    train_batch = next(train_iter)
    train_batch = tmt.transforms_batch(train_batch)
# %%
    batch = train_batch
    images = N._image_batch_tensor(batch)
    targets = N._targets_from_batch(batch)
    outputs = forward_train_batched(N.detector, images, targets)
    loss = N.w_cls * outputs[N.detector.cls_key] + outputs[N.detector.box_reg_key]
    N.log("train0_loss", loss, prog_bar=True, sync_dist=N.sync_dist)

# %%
    img = train_batch["image"]
    img.shape
    bbox = train_batch["bbox"]
    bbox
# %%
    n = 5
    im = img[n,0]
    box = bbox[n]
    print(box)
    ImageBBoxViewer(im, box)
# %%
    val_iter = iter(val_dl)
    val_batch = next(val_iter)
    img2 = val_batch["image"]
    bbox2 = val_batch["bbox"]
    box2 = bbox2[0]
    im2 = img2[0,0]
    ImageBBoxViewer(im2, box2)
# %%
    N = Tm.setup_model_for_cuda(device=device_id, precision="16-mixed")
    N.on_fit_start()
# %%
    val_batch = Tm.fabric_infer.to_device(val_batch)
# SECTION:-------------------- TRAIN STEP-BY-STEP ---------------------------------------------------------------------------
# %%
    from det3d.detection.retinanet_train import (
        build_train_anchors,
        compute_train_loss,
        forward_network_head,
        validate_train_targets,
    )

# %%
    N.detector.train()
# %%
    train_batch = Tm.fabric_infer.to_device(train_batch)
    images = N.train_images(train_batch)
    targets = N.train_targets(train_batch)
# %%
    targets = validate_train_targets(N.detector, images, targets)
    N.detector._check_detector_training_components()
# %%
    n = 6
    img = images[n, 0]
    tg = targets[n]
    bbox = tg['bbox']
    print(bbox.numel())
    ImageBBoxViewer(img, bbox)

# %%
    head_outputs = forward_network_head(N.detector, images)
# %%
    head_outputs, num_anchor_locs = build_train_anchors(N.detector, images, head_outputs)
# %%
    outputs = compute_train_loss(N.detector, head_outputs, targets, num_anchor_locs)
# %%
    train_loss, cls_loss, box_loss = N.train_total_loss(outputs)
# SECTION:-------------------- VAL STEP-BY-STEP -----------------------------------------------------------------------------
# %%
    N.on_validation_epoch_start()
# %%
    N.detector.eval()
    val_inputs = N.val_inputs(val_batch)
    val_targets = N.val_targets(val_batch)
    use_inferer = N.val_use_inferer(val_inputs)
# %%
    val_outputs = N.val_forward(val_inputs, use_inferer=use_inferer)
# %%
    N.val_outputs_all.extend(val_outputs)
    N.val_targets_all.extend(val_targets)
    N.on_validation_epoch_end()
# %%
    cb = Tm.get_callback("CaseIDRecorder")
    cb.dfs
# %%
