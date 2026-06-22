from ast import List
from torch import Tensor
from copy import deepcopy
from pathlib import Path
from typing import Optional
from nndet.core.retina import  BaseRetinaNet
import torch
from det3d.detection.nndet_train import xyzxyz_exclusive_batch_to_nndet
from fran.callback.debug_epoch_limit import DebugEpochBatchLimit
from fran.callback.incremental import LRFloorStop
from fran.configs.helpers import normalize_logging_payload
from fran.managers import Project
from fran.managers.wandb.wandb import WandbManager
from fran.trainers.helpers import switch_ckpt_keys
from fran.trainers.trainer import Trainer, _flatten_dict
from lightning.fabric import Fabric
from lightning.pytorch import Trainer as TrainerL
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.profilers import AdvancedProfiler
from utilz.imageviewers import ImageMaskViewer
from utilz.stringz import ast_literal_eval, headline

from det3d.architectures.create_detector import arch_from_conf
from det3d.callback.wandb_det_grid import (
    WandbDetImageGridCallback,
    WandbRetinaUNetImageGridCallback,
)
from det3d.managers.retinaunet import RetinaUNetManager
from det3d.configs.parser import ConfigMakerDet
from det3d.managers.data import DataManagerDualDet, DataManagerDualDetBTfms
from det3d.managers.detector_factory import resolve_detector_manager


class TrainerDet(Trainer):
    case_id_recorder_cls = None
    wandb_grid_cb_cls = WandbDetImageGridCallback
    monitor_metric_name = "val0_metric"
    _DET_PIPELINE_MODES = frozenset({"det", "lbd"})

    def case_id_recorder_dl_idx(self) -> int:
        return 0

    def wandb_grid_cb_kwargs(self, wandb_grid_epoch_freq: int) -> dict:
        val_patch_size = self.configs["model_params"]["val_patch_size"]
        if isinstance(val_patch_size, str):
            val_patch_size = ast_literal_eval(val_patch_size)
        plan = self.configs["plan_train"]
        kwargs = {
            "patch_size": [int(v) for v in val_patch_size],
            "epoch_freq": max(1, int(wandb_grid_epoch_freq)),
            "local_folder": str(self.project.log_folder / "wandb_grid"),
            "score_min": float(plan.get("wandb_grid_score_min", 0.3)),
            "score_mid_min": float(plan.get("wandb_grid_score_mid_min", 0.5)),
            "score_high_min": float(plan.get("wandb_grid_score_high_min", 0.8)),
            "tiny_side_px": int(plan.get("wandb_grid_tiny_side_px", 4)),
            "pred_top_k": plan.get("wandb_grid_pred_top_k", 5),
            "show_fg_heatmap": bool(plan.get("wandb_grid_show_fg_heatmap", True)),
        }
        return kwargs

    def wandb_retinaunet_grid_cb_kwargs(self, wandb_grid_epoch_freq: int) -> dict:
        plan = self.configs["plan_train"]
        kwargs = self.wandb_grid_cb_kwargs(wandb_grid_epoch_freq)
        kwargs["show_pred_seg"] = bool(plan.get("wandb_grid_show_pred_seg", True))
        kwargs["show_gt_seg"] = bool(plan.get("wandb_grid_show_gt_seg", True))
        return kwargs

    def wandb_grid_callback(self, wandb_grid_epoch_freq: int):
        if arch_from_conf(self.configs) == "retinaunet":
            return WandbRetinaUNetImageGridCallback(
                **self.wandb_retinaunet_grid_cb_kwargs(wandb_grid_epoch_freq)
            )
        return WandbDetImageGridCallback(
            **self.wandb_grid_cb_kwargs(wandb_grid_epoch_freq)
        )

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
        val_every_n_epochs: int = 5,
        cbs=[],
        tags=[],
        description="",
        epochs=600,
        batchsize_finder=False,
        override_dm_checkpoint=False,
        early_stopping=False,
        early_stopping_patience=30,
        lr_floor=None,
        wandb_grid_epoch_freq: int = 5,
        permanent_checkpoint_every_n_epochs: int = 100,
        batch_tfms: bool = True,
        case_id_recorder_freq: int = 50,
        nndet_forward_patch_size=None,
    ):
        self.val_every_n_epochs = int(val_every_n_epochs)
        self.max_epochs = int(epochs)
        self.case_id_recorder_freq = int(case_id_recorder_freq)
        if wandb_grid_epoch_freq is None:
            wandb_grid_epoch_freq = 5
        wandb_grid_epoch_freq = int(wandb_grid_epoch_freq)
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.val_sampling = float(val_sampling)
        self.debug = bool(debug)
        self.batch_tfms = bool(batch_tfms)
        self._resolve_run_ckpt(wandb=wandb)
        self.maybe_alter_configs(batch_size, compiled, nndet_forward_patch_size)
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

        self.init_dm_unet(self.max_epochs, batch_size, override_dm_checkpoint)
        self.D.prepare_data()
        self.D.setup(stage="fit")
        headline(
            "Data module ready.\n"
            f"  train: {type(self.D.train_manager).__name__} — {self.D.train_manager}\n"
            f"  valid: {type(self.D.valid_manager).__name__} — {self.D.valid_manager}"
        )

        cbs, logger, profiler = self.init_cbs(
            extra_cbs=cbs,
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
        self._ensure_local_ckpt_on_wandb_resume(logger)

        precision = "bf16-mixed"
        trainer_kwargs = dict(
            callbacks=cbs,
            accelerator=accelerator,
            devices=trainer_devices,
            precision=precision,
            profiler=profiler,
            logger=logger,
            max_epochs=self.max_epochs,
            check_val_every_n_epoch=self.val_every_n_epochs,
            log_every_n_steps=logging_freq,
            num_sanity_val_steps=0,
            enable_checkpointing=True,
            default_root_dir=self.project.checkpoints_parent_folder,
            strategy=strategy,
        )
        if arch_from_conf(self.configs) == "retinaunet":
            trainer_kwargs["gradient_clip_val"] = 1.0
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
        batch_size = int(self.configs["dataset_params"]["batch_size"])
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
        from det3d.managers.data.labels import infer_det_labels_from_data_folder

        infer_det_labels_from_data_folder(dm=self.D, configs=self.configs)
        return self.D

    def init_dm_unet(self, epochs, batch_size, override_dm_checkpoint=False):
        if self.ckpt:
            self.N = self.load_trainer()
            self.D = self.init_dm()
        else:
            self.D = self.init_dm()
            self.N = self.init_trainer(epochs)

    def init_trainer(self, epochs):
        detector = self.configs["model_params"]["arch"]
        if detector == "retinaunet":
            from det3d.managers.retinaunet import RetinaUNetManager

            manager_cls = RetinaUNetManager
        else:
            from det3d.managers.retinanet import RetinaNetManager

            manager_cls = RetinaNetManager
        N = manager_cls(
            project_title=self.project.project_title,
            configs=self.configs,
            lr=self.lr,
            sync_dist=self.sync_dist,
        )
        return N

    def load_trainer(self, map_location="cpu", **kwargs):
        manager_cls = resolve_detector_manager(self.configs)
        try:
            return manager_cls.load_from_checkpoint(
                self.ckpt, map_location=map_location, strict=True, **kwargs
            )
        except RuntimeError:
            switch_ckpt_keys(self.ckpt)
            return manager_cls.load_from_checkpoint(
                self.ckpt, map_location=map_location, strict=True, **kwargs
            )

    def maybe_alter_configs(self, batch_size, compiled, nndet_forward_patch_size=None):
        if batch_size is not None:
            self.configs["dataset_params"]["batch_size"] = int(batch_size)
        if compiled is not None:
            self.configs["model_params"]["compiled"] = bool(compiled)
        if nndet_forward_patch_size is not None:
            fps = [int(v) for v in nndet_forward_patch_size]
            plan = self.configs["plan_train"]
            plan["patch_size"] = fps
            plan["patch_dim0"] = fps[0]
            plan["patch_dim1"] = fps[1]
            self.configs["model_params"]["nndet_forward_patch_size"] = fps
            ConfigMakerDet(self.project)._assert_patch_fits_src_dims(plan)

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
        cbs = []
        if self.case_id_recorder_cls is not None:
            cbs.append(
                self.case_id_recorder_cls(
                    freq=self.case_id_recorder_freq,
                    local_folder=str(self.project.log_folder / "case_recorder"),
                    monitor_dl="both",
                    dl_idx=self.case_id_recorder_dl_idx(),
                )
            )
        if extra_cbs:
            cbs += list(extra_cbs)
        if self.debug:
            cbs += [DebugEpochBatchLimit(n=2)]

        ckpt_dir = (
            self.project.checkpoints_parent_folder
            / self.wandb_project_name().upper()
            / self.run_name
            / "checkpoints"
        )

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
                dirpath=str(ckpt_dir),
            ),
            ModelCheckpoint(
                save_top_k=-1,
                save_last=True,
                every_n_epochs=int(permanent_checkpoint_every_n_epochs),
                filename="epoch{epoch:04d}-snapshot",
                enable_version_counter=False,
                auto_insert_metric_name=False,
                dirpath=str(ckpt_dir),
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

        plan = self.configs["plan_train"]
        if bool(plan.get("use_swa", False)):
            cbs += [
                StochasticWeightAveraging(
                    swa_lrs=float(self.lr),
                    swa_epoch_start=self.max_epochs - int(plan.get("swa_epochs", 10)),
                )
            ]

        logger = None
        if wandb:
            logger = WandbManager(
                project=self.project,
                run_id=self.run_name,
                wandb_project_name=self.wandb_project_name(),
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
            cbs += [
                self.wandb_grid_callback(wandb_grid_epoch_freq),
                self.wandb_best_ckpt_cls(),
            ]

        if profiler:
            profiler = AdvancedProfiler(
                dirpath=self.project.log_folder, filename="profiler"
            )
            cbs += [DeviceStatsMonitor(cpu_stats=True)]
        else:
            profiler = None

        return cbs, logger, profiler

    def setup_model_for_cuda(self, device=0, precision="bf16-mixed"):
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


from det3d.callback.case_recorder_det import CaseIDRecorderSnapshotDet

CaseIDRecorderDetRT = CaseIDRecorderSnapshotDet


# %%
if __name__ == "__main__":
# SECTION:-------------------- setup<--------------------------------------------------------------------------------------
    from fran.managers import Project
    from torch import Tensor
    from utilz.imageviewers import ImageBBoxViewer

    from det3d.configs.parser import ConfigMakerDet

    project_title = "lidca"
    plan_id = 2

    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = C.configs
    conf["dataset_params"]["fold"] = 0

# SECTION:-------------------- TRAINING --------------------------------------------------------------------------------------
# %%
    conf["model_params"]["arch"] = "retinanet"
    conf["model_params"]["arch"] = "retinaunet"
    conf["plan_train"]["patch_size"] = [128, 128, 64]
    bs = 2
    device_id = 0
    batch_tfms = True
    batch_tfms = False
    wandb = True
    run_name = None
    run_name = "LIDCA-GYRO"
    tags = []
    description = "TrainerDet lidca retinanet"
    lr = None
    debug_ = False
    profiler = False
    compiled = False
    cbs = []
    wandb_grid_epoch_freq = 10
    val_every_n_epochs = 5
    case_id_recorder_freq = max(val_every_n_epochs, val_every_n_epochs * 2)
    train_indices = None
    val_indices = None
    val_sampling = 1.0
    epochs = 500
# SECTION:-------------------- TRAINING --------------------------------------------------------------------------------------
    Tm = TrainerDet(P.project_title, conf, run_name)
# %%
    Tm.setup(
        compiled=compiled,
        train_indices=train_indices,
        val_indices=val_indices,
        val_sampling=val_sampling,
        val_every_n_epochs=val_every_n_epochs,
        case_id_recorder_freq=case_id_recorder_freq,
        cbs=cbs,
        debug=debug_,
        batch_size=bs,
        batch_tfms=batch_tfms,
        devices=[device_id],
        epochs=epochs,
        profiler=profiler,
        wandb=wandb,
        wandb_grid_epoch_freq=wandb_grid_epoch_freq,
        tags=tags,
        description=description,
        lr=lr,
    )

# %%
# SECTION:-------------------- ii --------------------------------------------------------------------------------------
    Tm.fit()
# %%
# SECTION:-------------------- TS--------------------------------------------------------------------------------------
    N = Tm.N
    D = Tm.D
    tmt = D.train_manager
    tmv = D.valid_manager

# %%
    tmt.setup()
    tmv.setup()
    train_dl = tmt.dl
    train_iter = iter(train_dl)
# %%
    val_dl = tmv.dl
    val_iter = iter(val_dl)
# %%

    train_batch = next(train_iter)
    batch = train_batch
    img = batch["image"]
    box = batch["bbox"]
    img.shape

# %%
    val_batch = next(val_iter)
    batch = val_batch
    img = batch["image"]
    box = batch["bbox"]
    ImageBBoxViewer(img, box)

# %%

    N = Tm.setup_model_for_cuda(device=device_id, precision="16-mixed")
    # N.on_fit_start()
    batch = Tm.fabric_infer.to_device(batch)
    print(batch.keys())
# %%

    losses, preds, nb = N._step_losses(batch, 0,True)
# %%
    preds.keys()
    lm = preds["pred_seg"]
    batch_idx = 0
    ImageMaskViewer([img[0], lm[0,1,:]],'im')
# %%
# SECTION:-------------------- validation_step--------------------------------------------------------------------------------------  # T:block_meta|TrainerDet.validation_step
# %%
    SCORE_THRESH = 0.30  # was 0.02
    DETECTIONS_PER_IMG = 25  # was 100
    NMS_THRESH = 0.15  # was ~0.22; optional, tighter merge
    _saved = (N.net.score_thresh, N.net.detections_per_img, N.net.nms_thresh)
    N.net.score_thresh = SCORE_THRESH
    N.net.detections_per_img = DETECTIONS_PER_IMG
    N.net.nms_thresh = NMS_THRESH
# %%

    batch = batch
    R = N
# %%  # T:block_start|RetinaUNetManager._nndet_targets
# /home/ub/code/det3d/det3d/managers/retinaunet.py  # T:block_donor|/home/ub/code/det3d/det3d/managers/retinaunet.py
#SECTION:-------------------- _nndet_targets --------------------------------------------------------------------------------------  # T:block_meta|RetinaUNetManager._nndet_targets
    # requires R = RetinaUNetManager(...) in __main__  # T:requires_alias|R = RetinaUNetManager(...)
    from monai.data.box_utils import clip_boxes_to_image  # T:hoisted_import|from monai.data.box_utils import clip_boxes_to_image
    from monai.data.box_utils import clip_boxes_to_image
    data = batch["image"]
    forward_patch_size = R.plan["patch_size"]  # T:self_ref|forward_patch_size = self.plan["patch_size"]
    label_to_idx = {int(v): i for i, v in enumerate(R.plan["fg_labels"])}  # T:self_ref|label_to_idx = {int(v): i for i, v in enumerate(self.plan["fg_labels"])}
    target_seg_list = []
    target_boxes = []
    target_classes = []
    i = next(iter(range(data.shape[0])))  # T:loop_probe|for i in range(data.shape[0]):
    box = batch["bbox"][i]
    box = box.repeat(2,1)
    nndet_box = xyzxyz_exclusive_batch_to_nndet(box)  # T:indent|    nndet_box = xyzxyz_exclusive_batch_to_nndet(box)
    target_boxes.append(nndet_box)  # T:indent|    target_boxes.append(nndet_box)
    cls = batch["label"][i][: box.shape[0]]  # T:indent|    cls = batch["label"][i][: box.shape[0]]
    mapped = torch.tensor(  # T:indent|    mapped = torch.tensor(
        [label_to_idx[int(v.item())] for v in cls],  # T:indent|        [label_to_idx[int(v.item())] for v in cls],
        dtype=torch.long,  # T:indent|        dtype=torch.long,
        device=nndet_box.device,  # T:indent|        device=nndet_box.device,
    )  # T:indent|    )
    target_classes.append(mapped)  # T:indent|    target_classes.append(mapped)
    out = {
        "data": data,
        "target_boxes": target_boxes,
        "target_classes": target_classes,
        "target_seg": torch.stack(target_seg_list, 0),
    }
    _nndet_targets_result = out  # T:return|return out
#SECTION:-------------------- _nndet_targets end --------------------------------------------------------------------------------------  # T:block_meta_end|RetinaUNetManager._nndet_targets
    # end PythonMethodScratch  # T:block_end|RetinaUNetManager._nndet_targets
# %%
# SECTION:-------------------- _step_losses --------------------------------------------------------------------------------------  # T:block_meta|TrainerDet._step_losses
    nb = N._nndet_targets(batch)  # T:self_ref|nb = self._nndet_targets(batch)
    device_type = nb["data"].device.type
# %%
    with torch.autocast(device_type, enabled=device_type == "cuda" and not evaluation):
        losses, prediction = (
            N.net.train_step(  # T:self_ref|    losses, prediction = self.net.train_step(
                images=nb["data"],
                targets={
                    "target_boxes": nb["target_boxes"],
                    "target_classes": nb["target_classes"],
                    "target_seg": nb["target_seg"],
                },
                evaluation=evaluation,
                batch_num=batch_idx,
            )
        )
# %%
    _step_losses_result = (
        losses,
        prediction,
        nb,
    )  # T:return|return losses, prediction, nb
# SECTION:-------------------- _step_losses end --------------------------------------------------------------------------------------  # T:block_meta_end|TrainerDet._step_losses
    # end PythonMethodScratch  # T:block_end|TrainerDet._step_losses
# %%

    images = nb["data"]
    targets = {
        "target_boxes": nb["target_boxes"],
        "target_classes": nb["target_classes"],
        "target_seg": nb["target_seg"],
    }
    evaluation = True
    batch_num = 0
    B = N.net
# %%
# SECTION:-------------------- train_step --------------------------------------------------------------------------------------  # T:block_meta|TrainerDet.train_step
    """
    Perform a single training step (forward pass + loss computation)

    Args:
        images: batch of images
        targets: labels for training
            `target_boxes` (List[Tensor]): ground truth bounding boxes
                (x1, y1, x2, y2, (z1, z2))[X, dim * 2], X= number of ground
                truth boxes in image
            `target_classes` (List[Tensor]): ground truth class per box
                (classes start from 0) [X], X= number of ground truth
                boxes in image
            `target_seg`(Tensor): segmentation ground truth
                (only needed if :param:`segmenter`
                was provided in init) (classes start from 1, 0 background)
        evaluation (bool): compute final predictions (includes detection
            postprocessing)
        batch_num (int): batch index inside epoch

    Returns:
        torch.Tensor: final loss for back propagation
        Dict: predictions for metric calculation
            'pred_boxes': List[Tensor]: predicted bounding boxes for each
                image List[[R, dim * 2]]
            'pred_scores': List[Tensor]: predicted probability for the
                class List[[R]]
            'pred_labels': List[Tensor]: predicted class List[[R]]
            'pred_seg': Tensor: predicted segmentation [N, dims]
        Dict[str, torch.Tensor]: scalars for logging (e.g. individual
            loss components)
    """
    # import napari
    # with napari.gui_qt():
    #     viewer = napari.view_image(images.detach().cpu().numpy())
    #     viewer.add_labels(seg_targets[:, None].detach().cpu().numpy())
    target_boxes: List[Tensor] = targets["target_boxes"]
    target_classes: List[Tensor] = targets["target_classes"]
    target_seg: Tensor = targets["target_seg"]
    pred_detection, anchors, pred_seg = B(
        images
    )  # T:self_ref|pred_detection, anchors, pred_seg = self(images)
    labels, matched_gt_boxes = (
        B.assign_targets_to_anchors(  # T:self_ref|labels, matched_gt_boxes = self.assign_targets_to_anchors(
            anchors, target_boxes, target_classes
        )
    )
    losses = {}
    head_losses, pos_idx, neg_idx = (
        B.head.compute_loss(  # T:self_ref|head_losses, pos_idx, neg_idx = self.head.compute_loss(
            pred_detection, labels, matched_gt_boxes, anchors
        )
    )
    losses.update(head_losses)
    if B.segmenter is not None:  # T:self_ref|if self.segmenter is not None:
        losses.update(
            B.segmenter.compute_loss(pred_seg, target_seg)
        )  # T:self_ref|    losses.update(self.segmenter.compute_loss(pred_seg, target_seg))
    if evaluation:
        prediction = B.postprocess_for_inference(  # T:self_ref|    prediction = self.postprocess_for_inference(
            images=images,
            pred_detection=pred_detection,
            pred_seg=pred_seg,
            anchors=anchors,
        )
    else:
        prediction = None
    # self.save_matched_anchors(images=images, target_boxes=target_boxes,
    #                             anchors=anchors, pos_idx=pos_idx,
    #                             neg_idx=neg_idx, seg=seg_targets)
    train_step_result = losses, prediction  # T:return|return losses, prediction

# %%
# %%  # T:block_start|TrainerDet.postprocess_for_inference
# /home/ub/code/nnDetection/nndet/core/retina.py  # T:block_donor|/home/ub/code/nnDetection/nndet/core/retina.py
#SECTION:-------------------- postprocess_for_inference --------------------------------------------------------------------------------------  # T:block_meta|TrainerDet.postprocess_for_inference
    """
    Postprocess predictions for inference

    Args:
        images: input images
        pred_detection: detection predictions
        pred_seg: segmentation predictions
        anchors: anchors

    Returns:
        Dict: post processed predictions
            'pred_boxes': List[Tensor]: predicted bounding boxes for each
                image List[[R, dim * 2]]
            'pred_scores': List[Tensor]: predicted probability for
                the class List[[R]]
            'pred_labels': List[Tensor]: predicted class List[[R]]
            'pred_seg': Tensor: predicted segmentation [N, C, dims]
    """
    image_shapes = [images.shape[2:]] * images.shape[0]
    boxes, probs, labels = B.postprocess_detections(  # T:self_ref|boxes, probs, labels = self.postprocess_detections(
        pred_detection=pred_detection,
        anchors=anchors,
        image_shapes=image_shapes,
    )
    prediction = {"pred_boxes": boxes, "pred_scores": probs, "pred_labels": labels}
    if B.segmenter is not None:  # T:self_ref|if self.segmenter is not None:
        prediction["pred_seg"] = B.segmenter.postprocess_for_inference(pred_seg)["pred_seg"]  # T:self_ref|    prediction["pred_seg"] = self.segmenter.postprocess_for_inference(pred_seg)["pred_seg"]
    postprocess_for_inference_result = prediction  # T:return|return prediction
#SECTION:-------------------- postprocess_for_inference end --------------------------------------------------------------------------------------  # T:block_meta_end|TrainerDet.postprocess_for_inference
    # end PythonMethodScratch  # T:block_end|TrainerDet.postprocess_for_inference
# %%

# SECTION:--------------------end --------------------------------------------------------------------------------------


