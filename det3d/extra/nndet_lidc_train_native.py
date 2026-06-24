"""Native nnDetection LIDC training scratch — step through # %% blocks, no epoch loop.

Prerequisite: run/preprocessing/nndet_lidc_prep.py (or equivalent native prep).

Uses nnDetection Datamodule + RetinaUNetV001 (not det3d DataManager).
Run blocks in IPython / VS Code interactive — do not run the full file at once.
"""
from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import List

from nndet.core.retina import BaseRetinaNet
import torch
import yaml

from utilz.helpers import pp
from utilz.imageviewers import ImageBBoxViewer, ImageMaskViewer

NNDET_ROOT = Path("/home/ub/code/nnDetection")
DEFAULT_DET_DATA = Path("/r/datasets/nndet_data")
DEFAULT_DET_MODELS = Path("/s/agent_rw/nndet_models")
TASK = "Task012_LIDC"
FOLD = 0
SCRATCH_BATCH_SIZE = 1


def setup_nndet_env(det_data: Path = DEFAULT_DET_DATA, det_models: Path = DEFAULT_DET_MODELS) -> None:
    os.environ["det_data"] = str(det_data)
    os.environ["det_models"] = str(det_models)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("det_num_threads", "2")
    os.environ.setdefault("det_verbose", "1")
    if str(NNDET_ROOT) not in sys.path:
        sys.path.insert(0, str(NNDET_ROOT))
    import nndet.compat  # noqa: F401


def load_nndet_train_cfgs(cfg_path=None):
    cfg_path = cfg_path or NNDET_ROOT / "nndet/conf/train/v001.yaml"
    with open(cfg_path) as handle:
        train_cfg = yaml.safe_load(handle)
    return deepcopy(train_cfg["model_cfg"]), deepcopy(train_cfg["trainer_cfg"])


def clear_cuda_scratch() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def inspect_nndet_batch(batch) -> dict:
    out = {"keys": sorted(batch.keys())}
    out["data"] = tuple(batch["data"].shape)
    out["data_dtype"] = str(batch["data"].dtype)
    out["data_device"] = str(batch["data"].device)
    if "target" in batch:
        out["target"] = tuple(batch["target"].shape)
        out["target_dtype"] = str(batch["target"].dtype)
    if "boxes" in batch:
        out["boxes"] = [tuple(b.shape) for b in batch["boxes"]]
    if "classes" in batch:
        out["classes"] = [tuple(c.shape) for c in batch["classes"]]
    return out


def nndet_batch_to_device(batch, device):
    """DL loader uses float64; model needs float32 on device."""
    out = dict(batch)
    out["data"] = out["data"].float().to(device)
    out["target"] = out["target"].float().to(device)
    return out


# %%
if __name__ == "__main__":
    setup_nndet_env()

# %%

    B = BaseRetinaNet
#SECTION:--- stage 0 — compose cfg + load plan ---
    from hydra import initialize_config_module
    from nndet.io.load import load_pickle
    from nndet.utils.config import compose

    initialize_config_module(config_module="nndet.conf", version_base="1.1")
    cfg = compose(
        TASK,
        "config.yaml",
        overrides=[
            f"exp.fold={FOLD}",
            f"+augment_cfg.batch_size={SCRATCH_BATCH_SIZE}",
            "augment_cfg.multiprocessing=False",
            "augment_cfg.num_train_batches_per_epoch=4",
            "augment_cfg.num_val_batches_per_epoch=2",
            "trainer_cfg.num_train_batches_per_epoch=4",
            "trainer_cfg.num_val_batches_per_epoch=2",
            "trainer_cfg.precision=32",
        ],
    )
    plan_path = Path(str(cfg.host.plan_path))
    plan = load_pickle(plan_path)
    data_dir = Path(cfg.host.preprocessed_output_dir) / plan["data_identifier"] / "imagesTr"
    print("plan_path", plan_path)
    print("data_dir", data_dir)
    print("patch_size", plan["patch_size"], "batch_size", plan["batch_size"])

# %%
#SECTION:--- stage 1 — native Datamodule + train loader ---
    from omegaconf import OmegaConf

    from nndet.io.datamodule.bg_module import Datamodule

    augment_cfg = OmegaConf.to_container(cfg.augment_cfg, resolve=True)
    datamodule = Datamodule(
        augment_cfg=augment_cfg,
        plan=plan,
        data_dir=data_dir,
        fold=FOLD,
    )
    datamodule.setup()
    train_gen = datamodule.train_dataloader()
    val_gen = datamodule.val_dataloader()
    print("train cases", len(datamodule.dataset_tr))
    print("val cases", len(datamodule.dataset_val))

    iteri = iter(train_gen)
# %%
#SECTION:--- stage 2 — inspect one native batch (pre pre_trafo) ---
    train_batch = next(iteri)
    pp(inspect_nndet_batch(train_batch))
    lm= train_batch["target"]
    train_batch.keys()
    pp(train_batch["instance_mapping"])
    print(lm.unique())
    img = train_batch["data"]
    ImageMaskViewer([img, lm],'im')

# %%

#SECTION:--- stage 3 — build RetinaUNetV001 ---
    from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001

    model_cfg, trainer_cfg = load_nndet_train_cfgs()
    trainer_cfg["num_train_batches_per_epoch"] = int(cfg.trainer_cfg.num_train_batches_per_epoch)
    clear_cuda_scratch()
    module = RetinaUNetV001(
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        plan=plan,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module = module.to(device)
    n_params = sum(p.numel() for p in module.parameters())
    print(type(module.model).__name__, "params", n_params)

# %%
#SECTION:--- stage 4 — training_step on one batch (includes pre_trafo) ---
    clear_cuda_scratch()
    module.train()
    train_batch_gpu = nndet_batch_to_device(train_batch, device)
    
    pp(train_batch_gpu.keys())
    ba = train_batch_gpu
    ba.keys()
    img = ba['data']
    lm = ba['target']
    print(img.shape)
    print(lm.shape)
    print(lm.unique())
# %%
    ImageMaskViewer([img,lm])
# %%

    bb2 = module.pre_trafo(**train_batch_gpu)
# %%
    print(bb2.keys())
    bb2['classes']
    torch.save(bb2, "bb2.pt")
    bbox = bb2["boxes"]
    lms = bb2['target']
    print(lms.unique())
    img2 = bb2['data']
    img2.shape
    img2 = img2.to("cpu")
    bbox = bbox[0].to("cpu")
    bbox = bb2["boxes"][0]  # cuda ok for stack; .cpu() before viewer if needed
# %%
    #nndet uses [x1, y1, x2, y2, z1, z2] convention; ImageBBoxViewer uses [x1, y1, x2, y2, z1, z2]
    #             0  1    2   3  4   5
    bbox_viz = torch.stack(
        [bbox[:, 0], bbox[:, 1], bbox[:, 4], bbox[:, 2], bbox[:, 3], bbox[:, 5]], dim=1
    ).cpu()
    bbox_list = [int(a) for a in bbox_viz[0].tolist()]
    slcs = (slice(bbox_list[0],bbox_list[3]), slice(bbox_list[1],bbox_list[4]), slice(bbox_list[2],bbox_list[5]))

# %%
    im2=img[slcs]
    ImageMaskViewer([im2, im2],'im')
    ImageBBoxViewer(img, bbox_viz)
    # ImageBBoxViewer(img, bbox)

# %%
#SECTION:--- stage 5 — manual backward + optimizer step ---
    opt_cfgs = module.configure_optimizers()
    optimizer = opt_cfgs[0][0]
    scheduler = opt_cfgs[1]["scheduler"]
    optimizer.zero_grad(set_to_none=True)
    step_out["loss"].backward()
    optimizer.step()
    scheduler.step()
    print("lr", optimizer.param_groups[0]["lr"])
    clear_cuda_scratch()

# %%
#SECTION:--- stage 6 — validation_step on one val batch ---
    val_batch = next(iter(val_gen))
    clear_cuda_scratch()
    module.eval()
    with torch.no_grad():
        val_out = module.validation_step(nndet_batch_to_device(val_batch, device), batch_idx=0)
    print("val", {k: val_out[k] for k in val_out if k != "loss"})
    print("val total", float(val_out["loss"]))
    clear_cuda_scratch()

# %%
    batch = train_batch_gpu
    batch_idx = 0
# %%  # T:block_start|RetinaUNetV001.training_step
# /home/ub/code/nnDetection/nndet/ptmodule/retinaunet/base.py  # T:block_donor|/home/ub/code/nnDetection/nndet/ptmodule/retinaunet/base.py
#SECTION:-------------------- training_step --------------------------------------------------------------------------------------  # T:block_meta|RetinaUNetV001.training_step
    """
    Computes a single training step
    See :class:`BaseRetinaNet` for more information
    """
    with torch.no_grad():
        batch = module.pre_trafo(**batch)  # T:self_ref|    batch = self.pre_trafo(**batch)
    losses, _ = module.model.train_step(  # T:self_ref|losses, _ = self.model.train_step(
        images=batch["data"],
        targets={
            "target_boxes": batch["boxes"],
            "target_classes": batch["classes"],
            "target_seg": batch["target"][:, 0],  # Remove channel dimension
        },
        evaluation=False,
        batch_num=batch_idx,
    )
    loss = sum(losses.values())
    training_step_result = {"loss": loss, **{key: l.detach().item() for key, l in losses.items()}}  # T:return|return {"loss": loss, **{key: l.detach().item() for key, l in losses.items()}}

# %%

    images = bb2['data']
    targets = {
        "target_boxes": bb2["boxes"],
        "target_classes": bb2["classes"],
        "target_seg": bb2["target"][:, 0],  # Remove channel dimension
            }
    evaluation = True
    batch_num = 0
# %%
    from torch import Tensor
# %%  # T:block_start|BaseRetinaNet.train_step
    B = module
# /home/ub/code/nnDetection/nndet/core/retina.py  # T:block_donor|/home/ub/code/nnDetection/nndet/core/retina.py
#SECTION:-------------------- train_step --------------------------------------------------------------------------------------  # T:block_meta|BaseRetinaNet.train_step
    # requires B = BaseRetinaNet(...) in __main__  # T:requires_alias|B = BaseRetinaNet(...)
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
    pred_detection, anchors, pred_seg = B(images)  # T:self_ref|pred_detection, anchors, pred_seg = self(images)
    labels, matched_gt_boxes = B.assign_targets_to_anchors(  # T:self_ref|labels, matched_gt_boxes = self.assign_targets_to_anchors(
        anchors, target_boxes, target_classes)
    losses = {}
    head_losses, pos_idx, neg_idx = B.head.compute_loss(  # T:self_ref|head_losses, pos_idx, neg_idx = self.head.compute_loss(
        pred_detection, labels, matched_gt_boxes, anchors)
    losses.update(head_losses)
    if B.segmenter is not None:  # T:self_ref|if self.segmenter is not None:
        losses.update(B.segmenter.compute_loss(pred_seg, target_seg))  # T:self_ref|    losses.update(self.segmenter.compute_loss(pred_seg, target_seg))
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
#SECTION:-------------------- train_step end --------------------------------------------------------------------------------------  # T:block_meta_end|BaseRetinaNet.train_step
    # end PythonMethodScratch  # T:block_end|BaseRetinaNet.train_step

