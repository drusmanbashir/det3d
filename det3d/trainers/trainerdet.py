from ast import List
import os
from torch import Tensor
from pathlib import Path
from nndet.core.retina import BaseRetinaNet
import torch
from det3d.geometry.nndet_boxes import from_nndet
from det3d.detection.retinanet_detector2 import RetinaNetDetector2
from fran.callback.debug_epoch_limit import DebugEpochBatchLimit
from fran.managers.wandb.wandb import DET3D_WANDB_WORKSPACE
from fran.preprocessing.helpers import import_h5py
from fran.trainers.trainer import Trainer
from lightning.fabric import Fabric
from lightning.pytorch.callbacks import StochasticWeightAveraging
from utilz.imageviewers import ImageMaskBboxViewer, ImageMaskViewer
from utilz.stringz import ast_literal_eval, headline

from det3d.architectures.create_detector import arch_from_conf, val_patch_size_from_conf
from det3d.callback.wandb_det_grid import (
    WandbDetImageGridCallback,
    WandbRetinaUNetImageGridCallback,
)
from det3d.managers.retinaunet import RetinaUNetManager
from det3d.configs.parser import ConfigMakerDet
from det3d.managers.data import DataManagerDualDet, DataManagerDualDetBTfms
from det3d.managers.data.labels import infer_det_labels_from_data_folder
from det3d.managers.detector_factory import resolve_detector_manager


class TrainerDet(Trainer):
    """Detection trainer for RetinaNet / RetinaUNet on top of fran ``Trainer``.

    Wires ``DataManagerDualDet*`` orchestrators, detection W&B image grids, and
    ``val0_metric`` checkpointing. Normalizes ``plan_*`` modes ``det`` / ``lbd``
    to the label-bounded pipeline before DM setup.
    """

    case_id_recorder_cls = None
    monitor_metric_name = "val0_metric"
    monitor_mode = "max"
    monitor_format = ".4f"
    checkpoint_every_n_epochs = 5
    early_stopping_monitor = "val0_metric"
    debug_batch_limit = 2
    wandb_config_keys = ("dataset_params", "plan_train")
    _DET_PIPELINE_MODES = frozenset({"det", "lbd"})

    def __init__(self, project_title, configs, run_name=None, ckpt_path=None):
        """`run_name` is used verbatim; `ckpt_path` resumes from that file when no run is named."""
        super().__init__(project_title=project_title, configs=configs)
        self.run_name = run_name
        if ckpt_path is not None:
            self.ckpt = Path(ckpt_path)

    @property
    def manager_cls(self):
        return resolve_detector_manager(self.configs)

    def wandb_project_name(self) -> str:  #AI
        """Use the detection workspace by default and allow resuming in an older project."""
        return os.environ.get("DET3D_WANDB_PROJECT", DET3D_WANDB_WORKSPACE)

    def wandb_grid_cb_kwargs(self, wandb_grid_epoch_freq: int) -> dict:
        plan = self.configs["plan_train"]
        kwargs = {
            "patch_size": val_patch_size_from_conf(self.configs),
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
        """Base WandB grid kwargs plus RetinaUNet seg overlay toggles from plan_train."""
        plan = self.configs["plan_train"]
        kwargs = self.wandb_grid_cb_kwargs(wandb_grid_epoch_freq)
        kwargs["show_pred_seg"] = bool(plan.get("wandb_grid_show_pred_seg", True))
        kwargs["show_gt_seg"] = bool(plan.get("wandb_grid_show_gt_seg", True))
        return kwargs

    def wandb_grid_callback(self, wandb_grid_epoch_freq: int):
        arch = arch_from_conf(self.configs)
        if arch == "retinaunet":
            kwargs = self.wandb_retinaunet_grid_cb_kwargs(wandb_grid_epoch_freq)
            kwargs["adapt_nndet_boxes"] = True
            return WandbRetinaUNetImageGridCallback(**kwargs)
        return WandbDetImageGridCallback(
            **self.wandb_grid_cb_kwargs(wandb_grid_epoch_freq)
        )

    def setup(
        self,
        *,
        debug_tfm_keys: str | None = None,
        val_every_n_epochs: int = 5,
        batchsize_finder=False,
        batch_tfms: bool = True,
        snapshot_freq: int = 1000,
        **kwargs,
    ):
        """Detection defaults over ``Trainer.setup``.

        ``snapshot_freq`` is the run-through snapshot cadence; the batch-size
        finder only runs in run-through trainers. ``epochs`` None reads
        ``plans.epochs`` (``auto``: nnDetection v001). Early stopping follows
        ``model_params.early_stopping_patience`` (``auto``: none) unless a patience is passed.
        The remaining arguments and their defaults are inherited from ``Trainer.setup``.
        """
        self.debug_tfm_keys = debug_tfm_keys
        super().setup(
            val_every_n_epochs=int(val_every_n_epochs),
            batchsize_finder=batchsize_finder and self.run_through,
            batch_tfms=batch_tfms,
            case_id_recorder_freq=snapshot_freq,
            **kwargs,
        )
        self.D.prepare_data()
        self.D.setup(stage="fit")
        valid_mgr = getattr(self.D, "valid_manager", None)
        if valid_mgr is None:
            valid_mgr = getattr(self.D, "snapshot_manager", None)
        headline(
            "Data module ready.\n"
            f"  train: {type(self.D.train_manager).__name__} — {self.D.train_manager}\n"
            f"  valid: {type(valid_mgr).__name__} — {valid_mgr}"
        )

    def lightning_kwargs(self) -> dict:
        nndet = self.configs["nndet"]
        precision = nndet.get("precision")  # blank or absent in older checkpoints: bf16-mixed
        kwargs = dict(
            precision="bf16-mixed" if precision is None else str(precision),
            benchmark=bool(nndet["benchmark"]),
            deterministic=bool(nndet["deterministic"]),
        )
        limit_val_batches = self.configs["dataset_params"].get("limit_val_batches")
        if limit_val_batches is not None:
            kwargs["limit_val_batches"] = limit_val_batches
        if self.run_through:
            kwargs["limit_val_batches"] = 0
            kwargs["check_val_every_n_epoch"] = 1
        return kwargs

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
        return DataManagerDualDetBTfms if batch_tfms else DataManagerDualDet

    def normalize_plan_modes_for_det_pipeline(self):
        """Shim: ConfigMakerDet may set mode=det; DM infer only knows fran modes."""
        for key in ("plan_train", "plan_valid", "plan_test"):
            plan = self.configs[key]
            if plan["mode"] in self._DET_PIPELINE_MODES:
                plan["mode"] = "lbd"

    def qc_configs(self, configs, project):
        self.normalize_plan_modes_for_det_pipeline()
        arch = arch_from_conf(configs)
        if arch not in ("retinanet", "retinaunet"):
            raise ValueError(f"unsupported arch {arch!r}; use retinanet or retinaunet")

    def init_dm(self):
        self.normalize_plan_modes_for_det_pipeline()
        batch_size = int(self.configs["plan_train"]["batch_size"])
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
        infer_det_labels_from_data_folder(dm=self.D, configs=self.configs)
        return self.D

    def init_dm_unet(self, batch_size, override_dm_checkpoint):
        """A det resume takes the model from the checkpoint and rebuilds the datamodule from current configs."""
        if self.ckpt:
            self.N = self.load_trainer()
            self.D = self.init_dm()
        else:
            self.D = self.init_dm()
            self.N = self.init_trainer()

    def init_cbs(
        self,
        cbs,
        wandb,
        batchsize_finder,
        profiler,
        tags,
        description,
        early_stopping,
        early_stopping_patience,
        wandb_grid_epoch_freq,
    ):
        """Trainer callbacks plus optional SWA; a transform-debug run keeps full epochs."""
        cbs, logger, profiler = super().init_cbs(
            cbs=cbs,
            wandb=wandb,
            batchsize_finder=batchsize_finder,
            profiler=profiler,
            tags=tags,
            description=description,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            wandb_grid_epoch_freq=wandb_grid_epoch_freq,
        )
        if self.debug_tfm_keys:
            cbs = [cb for cb in cbs if not isinstance(cb, DebugEpochBatchLimit)]
        plan = self.configs["plan_train"]
        if bool(plan.get("use_swa", False)):
            cbs += [
                StochasticWeightAveraging(
                    swa_lrs=float(self.lr),
                    swa_epoch_start=self.max_epochs - int(
                        plan.get("swa_epochs", ConfigMakerDet.fixed()["DET_SWA_EPOCHS"])
                    ),
                )
            ]
        return cbs, logger, profiler

    def setup_model_for_cuda(self, device=0, precision="bf16-mixed"):
        """Put the whole manager on one GPU through Fabric, for interactive inference."""
        self.fabric_infer = Fabric(accelerator="gpu", devices=[device], precision=precision)
        self.N.eval()
        self.N = self.fabric_infer.setup(self.N)
        return self.N


# %%
if __name__ == "__main__":
# SECTION:-------------------- setup<--------------------------------------------------------------------------------------
    from fran.managers import Project
    from torch import Tensor
    from utilz.imageviewers import ImageBBoxViewer

    from det3d.configs.parser import ConfigMakerDet

    project_title = "lidc"
    plan_id = 3

    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = C.configs
    conf["dataset_params"]["fold"] = 0

# %%
# SECTION:-------------------- TRAINING --------------------------------------------------------------------------------------
    from det3d.managers.data.tfm_debug import KEYS_ITEM_NO_SPATIAL

    conf["model_params"]["arch"] = "retinanet"
    print(conf["dataset_params"]["prezoom_scale"])

    bs = 8
    device_id = 0
    batch_tfms = False
    wandb = True
    run_name = "LIDCA-GYRO"
    run_name = None

# %%
    tags = []
    description = "upgradefrom LIDCA-GYRO using nndet params"
    lr = None
    debug_ = False
    debug_tfm_keys = KEYS_ITEM_NO_SPATIAL
    profiler = False
    compiled = False
    cbs = []
    wandb_grid_epoch_freq = 20
    val_every_n_epochs = 2
    train_indices = 50
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
    plan = Tm.configs["plan_train"]
    R = N.nndet_module

# %%
    losses, preds, nb = N.step_losses(batch, 0, True) # %% preds.keys()
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

    box_viz = from_nndet(box)
    im = img[0]
    ImageBBoxViewer(im, box_viz)

# %%

# /home/ub/code/det3d/det3d/managers/retinaunet.py  # T:block_donor|/home/ub/code/det3d/det3d/managers/retinaunet.py
#SECTION:-------------------- _step_losses --------------------------------------------------------------------------------------  # T:block_meta|RetinaUNetManager._step_losses
    # requires R = RetinaUNetManager(...) in __main__  # T:requires_alias|R = RetinaUNetManager(...)
    batch = batch
    batch_idx = 0
    evaluation = True
    nb = N.det3d_batch_to_nndet(batch, N.plan["fg_labels"])
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
    box_viz = from_nndet(box)
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

