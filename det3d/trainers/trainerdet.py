from ast import List
from torch import Tensor
from copy import deepcopy
from pathlib import Path
from typing import Optional
from nndet.core.retina import BaseRetinaNet
import torch
from det3d.detection.nndet_train import nndet_batch_to_xyzxyz, xyzxyz_exclusive_batch_to_nndet
from det3d.detection.preprocess_images import check_training_targets
from det3d.detection.retinanet_detector2 import RetinaNetDetector2
from det3d.detection.retinanet_train import forward_train_joint
from det3d.detection.retinaunet_v3 import RetinaUNetV3
from fran.callback.debug_epoch_limit import DebugEpochBatchLimit
from fran.callback.incremental import LRFloorStop
from fran.configs.helpers import normalize_logging_payload
from fran.managers import Project
from fran.managers.wandb.wandb import WandbManager
from fran.preprocessing.helpers import import_h5py
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
from utilz.imageviewers import ImageMaskBboxViewer, ImageMaskViewer
from utilz.stringz import ast_literal_eval, headline

from det3d.architectures.create_detector import arch_from_conf, create_retinaunet_v3_from_conf
from det3d.callback.wandb_det_grid import (
    WandbDetImageGridCallback,
    WandbRetinaUNetImageGridCallback,
)
from det3d.managers.retinaunet import RetinaUNetManager
from det3d.configs.parser import ConfigMakerDet
from det3d.managers.data import DataManagerDualDet, DataManagerDualDetBTfms
from det3d.managers.detector_factory import resolve_detector_manager


class TrainerDet(Trainer):
    wandb_grid_cb_cls = WandbDetImageGridCallback
    monitor_metric_name = "val0_metric"
    _DET_PIPELINE_MODES = frozenset({"det", "lbd"})

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
        arch = arch_from_conf(self.configs)
        if arch in ("retinaunet", "retinaunet_v3"):
            kwargs = self.wandb_retinaunet_grid_cb_kwargs(wandb_grid_epoch_freq)
            kwargs["adapt_nndet_boxes"] = arch == "retinaunet"
            return WandbRetinaUNetImageGridCallback(**kwargs)
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
        debug_tfm_keys: str | None = None,
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
        nndet_forward_patch_size=None,
    ):
        self.val_every_n_epochs = int(val_every_n_epochs)
        self.max_epochs = int(epochs)
        if wandb_grid_epoch_freq is None:
            wandb_grid_epoch_freq = 5
        wandb_grid_epoch_freq = int(wandb_grid_epoch_freq)
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.val_sampling = float(val_sampling)
        self.debug = bool(debug)
        self.debug_tfm_keys = debug_tfm_keys
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
        if self.debug and self.debug_tfm_keys:
            from det3d.managers.data.tfm_debug import (
                DataManagerDualDetBTfmsTfmDebug,
                DataManagerDualDetTfmDebug,
            )

            return (
                DataManagerDualDetBTfmsTfmDebug
                if batch_tfms
                else DataManagerDualDetTfmDebug
            )
        if self.configs["model_params"].get("pre_trafo", False):
            from det3d.managers.data.pre_trafo import (
                DataManagerDualDetPreTrafo,
                DataManagerDualDetPreTrafoBTfms,
            )

            return (
                DataManagerDualDetPreTrafoBTfms
                if batch_tfms
                else DataManagerDualDetPreTrafo
            )
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
        if self.debug and self.debug_tfm_keys:
            dm_kwargs["debug_tfm_keys"] = self.debug_tfm_keys
            n = self.train_indices if isinstance(self.train_indices, int) else 50
            dm_kwargs["debug_n_cases"] = int(n)
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
        from det3d.managers.detector_factory import resolve_detector_manager

        manager_cls = resolve_detector_manager(self.configs)
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
        if extra_cbs:
            cbs += list(extra_cbs)
        if self.debug and not self.debug_tfm_keys:
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
                **self.checkpoint_kwargs,
            ),
            ModelCheckpoint(
                save_top_k=-1,
                save_last=True,
                every_n_epochs=int(permanent_checkpoint_every_n_epochs),
                filename="epoch{epoch:04d}-snapshot",
                enable_version_counter=False,
                auto_insert_metric_name=False,
                **self.checkpoint_kwargs,
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


# %%
if __name__ == "__main__":
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

# %%
# SECTION:-------------------- TRAINING --------------------------------------------------------------------------------------
    from det3d.managers.data.tfm_debug import KEYS_ITEM_NO_SPATIAL

    conf["model_params"]["arch"] = "retinaunet"
    conf["affine3d"]["p"] = 0.0
    print(conf["dataset_params"]["prezoom_scale"])

    bs = 6
    device_id = 0
    batch_tfms = True
    wandb = True
    run_name = None

# %%
    tags = ["tfm-debug", "no-spatial", "instances-fix"]
    description = "TrainerDet tfm debug n=50 no spatial aug + sidecar instances"
    lr = None
    debug_ = True
    debug_tfm_keys = KEYS_ITEM_NO_SPATIAL
    profiler = False
    compiled = False
    cbs = []
    wandb_grid_epoch_freq = 20
    val_every_n_epochs = 2
    train_indices = 50
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
        cbs=cbs,
        debug=debug_,
        debug_tfm_keys=debug_tfm_keys,
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

    tmt.setup()
    tmv.setup()
    train_dl = tmt.dl
    train_iter = iter(train_dl)
# %%
    train_batch = next(train_iter)
    batch = train_batch
    batch.keys()
    img = batch['image']
    print(img.min(), img.max(), img.mean())
# %%
    batch['lm'].shape
    box = batch["bbox"]
# %%


    N = Tm.setup_model_for_cuda(device=device_id, precision="16-mixed")
    # N.on_fit_start()
    batch = Tm.fabric_infer.to_device(batch)
    print(batch.keys())
# %%
    images = batch["image"]

# %%
    R = N.detector
    debug = False
    script=None
    plan = Tm.configs["plan_train"]


    R= create_retinaunet_v3_from_conf(plan, script=script, debug=debug)


# %%
    images = batch["image"]
    R = N
# %%  # T:block_start|RetinaUNetV3.forward
    A = N.detector
    R = N.detector.network

# /home/ub/code/det3d/det3d/detection/retinaunet_v3.py  # T:block_donor|/home/ub/code/det3d/det3d/detection/retinaunet_v3.py
#SECTION:-------------------- forward --------------------------------------------------------------------------------------  # T:block_meta|RetinaUNetV3.forward
    # requires R = RetinaUNetV3(...) in __main__  # T:requires_alias|R = RetinaUNetV3(...)
    head_maps, all_maps = R.feature_extractor(images)  # T:self_ref|head_maps, all_maps = self.feature_extractor(images)
    [print(a.shape) for a in head_maps]
    [print(a.shape) for a in all_maps]
    pred_seg = R.segmenter(all_maps)  # T:self_ref|pred_seg = self.segmenter(all_maps)
    classification = R.classification_head(head_maps)  # T:self_ref|classification = self.classification_head(head_maps)
    box_regression = R.regression_head(head_maps)  # T:self_ref|box_regression = self.regression_head(head_maps)
    # return {
    #     R.cls_key: classification,
    #     R.box_reg_key: box_regression,
    #     R.seg_key: pred_seg["seg_logits"],
    # }
#SECTION:-------------------- forward end --------------------------------------------------------------------------------------  # T:block_meta_end|RetinaUNetV3.forward
    # end PythonMethodScratch  # T:block_end|RetinaUNetV3.forward
# %%
    from det3d.architectures.create_detector import _anchor_generator

    decoder_levels = plan.get("decoder_levels", (1, 2, 3, 4))
    if isinstance(decoder_levels, str):
        decoder_levels = ast_literal_eval(decoder_levels)
    anchor_generator = _anchor_generator (plan, decoder_levels)
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    net = build_retinaunet_v3(plan, num_anchors)
    if script:
        net = torch.jit.script(net)
    R = RetinaNetDetector2(network=net, anchor_generator=anchor_generator, debug=debug)


# %%
    outputs = forward_train_joint(
        N.detector,
        batch["image"],
        N._targets_from_batch(batch),
        N.seg_loss_fnc,
        batch["lm"],
        N.plan,
    )


# %%

# %%
    losses, preds, nb = N._step_losses(batch, 0, True) # %% preds.keys()
    lm = preds["pred_seg"]
    batch_idx = 0
    ImageMaskViewer([img[0], lm[0, 1, :]], "im")
    print(N.nndet_plan.keys())
    print(N.nndet_plan["anchors"].keys())
    plan2= N.nndet_plan
    plan2["anchors"]["width"]
# %%
    R = Tm.N
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
    R = N.nndet_module
    R.encoder
    R.box_evaluator
    R.classes




# %%
# %%  # T:block_start|RetinaUNetManager._step_losses
# %%
    img = nb['data']
    boxes = nb['target_boxes']
    box=  boxes[0]

    box_viz = nndet_batch_to_xyzxyz(box)
    im = img[0]
    ImageBBoxViewer(im, box_viz)

# %%

# /home/ub/code/det3d/det3d/managers/retinaunet.py  # T:block_donor|/home/ub/code/det3d/det3d/managers/retinaunet.py
#SECTION:-------------------- _step_losses --------------------------------------------------------------------------------------  # T:block_meta|RetinaUNetManager._step_losses
    # requires R = RetinaUNetManager(...) in __main__  # T:requires_alias|R = RetinaUNetManager(...)
    batch = batch
    batch_idx = 0
    evaluation = True
    nb = N._det3d_batch_to_nndet(batch)
    nb.keys()
    nb["target_classes"]
    nb["target_boxes"]
    targets = {
        "target_boxes": nb["target_boxes"],
        "target_classes": nb["target_classes"],
        "target_seg": nb["target_seg"],
    }

    losses, prediction = N.net.train_step(  # T:self_ref|losses, prediction = self.net.train_step(
        images=nb["data"],
        targets= targets,
                                          evaluation=evaluation,
        batch_num=batch_idx,
    )
    _step_losses_result = losses, prediction, nb  # T:return|return losses, prediction, nb
    prediction.keys()
    preds = prediction['pred_boxes'][0]
    scores = prediction['pred_scores'][0]
    nb['data'].min()
    nb['data'].max()
    nb['data'].mean()
    im.mean()
    im.min()
    box = preds[0]
    box_viz = nndet_batch_to_xyzxyz(box)
    ImageBBoxViewer(im, box_viz)
    from torch import Tensor

#SECTION:-------------------- train_step --------------------------------------------------------------------------------------  # T:block_meta|TrainerDet.train_step
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
    target_boxes = targets["target_boxes"]
    target_classes = targets["target_classes"]
    target_seg= targets["target_seg"]
    pred_detection, anchors, pred_seg = Tm(images)  # T:self_ref|pred_detection, anchors, pred_seg = self(images)
    labels, matched_gt_boxes = Tm.assign_targets_to_anchors(  # T:self_ref|labels, matched_gt_boxes = self.assign_targets_to_anchors(
        anchors, target_boxes, target_classes)
    losses = {}
    head_losses, pos_idx, neg_idx = Tm.head.compute_loss(  # T:self_ref|head_losses, pos_idx, neg_idx = self.head.compute_loss(
        pred_detection, labels, matched_gt_boxes, anchors)
    losses.update(head_losses)
    if Tm.segmenter is not None:  # T:self_ref|if self.segmenter is not None:
        losses.update(Tm.segmenter.compute_loss(pred_seg, target_seg))  # T:self_ref|    losses.update(self.segmenter.compute_loss(pred_seg, target_seg))
    if evaluation:
        prediction = Tm.postprocess_for_inference(  # T:self_ref|    prediction = self.postprocess_for_inference(
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
#SECTION:-------------------- _step_losses end --------------------------------------------------------------------------------------  # T:block_meta_end|RetinaUNetManager._step_losses
    # end PythonMethodScratch  # T:block_end|RetinaUNetManager._step_losses

# %%
    # /home/ub/code/det3d/det3d/managers/retinaunet.py  # T:block_donor|/home/ub/code/det3d/det3d/managers/retinaunet.py
    tfms="Ld,Rtr,L2,E,Norm,BoxToWorld,ToPoints,Zoom,Flip0,Flip1,Flip2,Rand90,Rot,AffinePts,ToBoxes,BoxClip,DelMask,IntensityTfms,Dtype"
    tfms = tmt.transforms_dict
    from fran.managers.data.main import RandCropByFlatIndicesd
# %%
    dici0 = tmt.data[0]
    Ld = tfms["Ld"]
    data = dici0

    import numpy as np
    dici = tfms["Ld"](dici0)
    dici
# %%
    R = tfms["Rtr"]
    dici = R(dici)
# %%
    data = dici
# %%  # T:block_start|RandCropByFlatIndicesd.__call__
# /home/ub/code/fran/fran/managers/data/main.py  # T:block_donor|/home/ub/code/fran/fran/managers/data/main.py
#SECTION:-------------------- __call__ --------------------------------------------------------------------------------------  # T:block_meta|RandCropByFlatIndicesd.__call__
    # requires R = RandCropByFlatIndicesd(...) in __main__  # T:requires_alias|R = RandCropByFlatIndicesd(...)
    d = dict(data)
    src_dims = tuple(int(v) for v in d[R.src_dims_key])  # T:self_ref|src_dims = tuple(int(v) for v in d[self.src_dims_key])
# %%
    fg = np.asarray(d[R.fg_indices_key], dtype=np.int64).reshape(-1)  # T:self_ref|fg = np.asarray(d[self.fg_indices_key], dtype=np.int64).reshape(-1)
    bg = np.asarray(d[R.bg_indices_key], dtype=np.int64).reshape(-1)  # T:self_ref|bg = np.asarray(d[self.bg_indices_key], dtype=np.int64).reshape(-1)
    out = []
    _ = next(iter(range(R.num_samples)))  # T:loop_probe|for _ in range(self.num_samples):
    sample = dict(d)  # T:indent|    sample = dict(d)
    pool, sample_is_fg = R._sample_pool(fg=fg, bg=bg)  # T:self_ref|    pool, sample_is_fg = self._sample_pool(fg=fg, bg=bg)
    print(sample_is_fg)
    print(pool.shape)
    sampled_flat_index = int(pool[R.R.randint(0, pool.size)])  # T:self_ref|    sampled_flat_index = int(pool[self.R.randint(0, pool.size)])
# %%
    center = tuple(  # T:indent|    center = tuple(
        int(v) for v in np.unravel_index(sampled_flat_index, src_dims)  # T:indent|        int(v) for v in np.unravel_index(sampled_flat_index, src_dims)
    )  # T:indent|    )
    print(center)
# %%
    crop_slices, crop_start, crop_end = R._compute_crop(center, src_dims)  # T:self_ref|    crop_slices, crop_start, crop_end = self._compute_crop(center, src_dims)
    sample["crop_center"] = center  # T:indent|    sample["crop_center"] = center
    sample["crop_slices"] = crop_slices  # T:indent|    sample["crop_slices"] = crop_slices
    sample["crop_start"] = crop_start  # T:indent|    sample["crop_start"] = crop_start
    sample["crop_end"] = crop_end  # T:indent|    sample["crop_end"] = crop_end
    sample["sample_is_fg"] = bool(sample_is_fg)  # T:indent|    sample["sample_is_fg"] = bool(sample_is_fg)
    sample["sampled_flat_index"] = sampled_flat_index  # T:indent|    sample["sampled_flat_index"] = sampled_flat_index
    out.append(sample)  # T:indent|    out.append(sample)
    __call___result = out  # T:return|return out

#SECTION:-------------------- __call__ end --------------------------------------------------------------------------------------  # T:block_meta_end|RandCropByFlatIndicesd.__call__
    # end PythonMethodScratch  # T:block_end|RandCropByFlatIndicesd.__call__

# %%
    fg = fg
    bg = bg
# %%  # T:block_start|RandCropByFlatIndicesd._sample_pool
# /home/ub/code/fran/fran/managers/data/main.py  # T:block_donor|/home/ub/code/fran/fran/managers/data/main.py
#SECTION:-------------------- _sample_pool --------------------------------------------------------------------------------------  # T:block_meta|RandCropByFlatIndicesd._sample_pool
    # requires R = RandCropByFlatIndicesd(...) in __main__  # T:requires_alias|R = RandCropByFlatIndicesd(...)
    choose_fg = R.R.rand() < R.pos / (R.pos + R.neg)  # T:self_ref|choose_fg = self.R.rand() < self.pos / (self.pos + self.neg)
    _sample_pool_result = bg, False  # T:return|return bg, False
#SECTION:-------------------- _sample_pool end --------------------------------------------------------------------------------------  # T:block_meta_end|RandCropByFlatIndicesd._sample_pool
    # end PythonMethodScratch  # T:block_end|RandCropByFlatIndicesd._sample_pool
# %%
    dici = tfms["L2"](dici[0])
    print(dici['image'].shape)
    im = dici['image'][0]
    ImageMaskViewer([im, im],'im')

# %%
    dici = tfms["E"](dici)
    print(dici['image'].shape)

    dici = tfms["Norm"](dici)
    print(dici['image'].shape)

    dici = tfms["BoxToWorld"](dici)
    print(dici['image'].shape)

    dici = tfms["ToPoints"](dici)
    print(dici['image'].shape)

    dici = tfms["Zoom"](dici)
    print(dici['image'].shape)

    dici = tfms["Flip0"](dici)
    print(dici['image'].shape)

    dici = tfms["Flip1"](dici)
    print(dici['image'].shape)

    dici = tfms["Flip2"](dici)
    print(dici['image'].shape)

    dici = tfms["Rand90"](dici)
    print(dici['image'].shape)

    dici = tfms["Rot"](dici)
    print(dici['image'].shape)

    dici = tfms["AffinePts"](dici)
    print(dici['image'].shape)

    dici = tfms["ToBoxes"](dici)
    print(dici['image'].shape)

    dici = tfms["BoxClip"](dici)
    print(dici['image'].shape)

    dici = tfms["DelMask"](dici)
    print(dici['image'].shape)

    dici = tfms["IntensityTfms"](dici)
    print(dici['image'].shape)

    dici = tfms["Dtype"](dici)
    print(dici['image'].shape)

# %%
#SECTION:-------------------- train_step end --------------------------------------------------------------------------------------  # T:block_meta_end|TrainerDet.train_step
    # end PythonMethodScratch  # T:block_end|TrainerDet.train_step




# %%

