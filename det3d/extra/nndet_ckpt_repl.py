"""Load trained nnDetection LIDC checkpoint — step through # %% blocks in REPL.

Compare native nnDetection forward/preds vs det3d RetinaUNet plugin.
Run one cell at a time (IPython / Cursor interactive / nvim <leader>sa blocks).

Prereq: conda env nndet (or dl with nndet on PYTHONPATH), LIDC prep + trained fold0 ckpt.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

NNDET_ROOT = Path("/home/ub/code/nnDetection")
DEFAULT_FOLD_DIR = Path(
    "/r/datasets/nndet_models/Task012_LIDC/RetinaUNetV001_D3V001_3d/fold0"
)
DEFAULT_DET_DATA = Path("/r/datasets/nndet_data")
TASK = "Task012_LIDC"
FOLD = 0
SCRATCH_BATCH_SIZE = 1
CKPT_TAG = "best"  # best | last
DET3D_PROJECT = None  # e.g. "lidc_nndet" — stage 7 only

# Populated by stages; import after running stage 0+1: `from det3d.extra.nndet_ckpt_repl import CTX`
CTX: dict = {}


def setup_nndet_env(
    det_data: Path = DEFAULT_DET_DATA,
    det_models: Path = DEFAULT_FOLD_DIR.parent.parent,
) -> None:
    import os

    os.environ["det_data"] = str(det_data)
    os.environ["det_models"] = str(det_models)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("det_num_threads", "2")
    os.environ.setdefault("det_verbose", "1")
    if str(NNDET_ROOT) not in sys.path:
        sys.path.insert(0, str(NNDET_ROOT))
    import nndet.compat  # noqa: F401


def clear_cuda() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_fold_checkpoint(fold_dir: Path, ckpt_tag: str = "best"):
    from omegaconf import OmegaConf

    from nndet.io.load import load_pickle
    from nndet.ptmodule import MODULE_REGISTRY

    fold_dir = Path(fold_dir)
    cfg = OmegaConf.to_container(OmegaConf.load(fold_dir / "config_resolved.yaml"), resolve=True)
    plan = load_pickle(fold_dir / "plan.pkl")
    ckpts = sorted(fold_dir.glob(f"model_{ckpt_tag}*.ckpt"))
    ckpt_path = ckpts[0]
    module = MODULE_REGISTRY[cfg["module"]](
        model_cfg=cfg["model_cfg"],
        trainer_cfg=cfg["trainer_cfg"],
        plan=plan,
    )
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    load_info = module.load_state_dict(state)
    module.eval()
    module.float()
    return module, cfg, plan, ckpt_path, load_info


def nndet_plan_to_det3d(plan: dict) -> dict:
    arch = plan["architecture"]
    return {
        "spatial_dims": int(plan["network_dim"]),
        "n_input_channels": int(arch["in_channels"]),
        "encoder_start_channels": int(arch["start_channels"]),
        "encoder_max_channels": int(arch["max_channels"]),
        "decoder_levels": arch["decoder_levels"],
        "encoder_conv_kernels": "auto",
        "encoder_strides": "auto",
        "fg_labels": [1],
        "patch_size": [int(v) for v in plan["patch_size"]],
    }


def pre_trafo_batch(module, batch: dict) -> dict:
    with torch.no_grad():
        return module.pre_trafo(**batch)


def batch_targets(batch: dict, device: torch.device) -> dict:
    return {
        "target_boxes": [b.to(device) for b in batch["boxes"]],
        "target_classes": [c.to(device) for c in batch["classes"]],
        "target_seg": batch["target"][:, 0].to(device),
    }


def tensor_stats(t: torch.Tensor, name: str = "") -> dict:
    t = t.detach().float().cpu()
    return {
        "name": name,
        "shape": tuple(t.shape),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "min": float(t.min()),
        "max": float(t.max()),
    }


def pred_summary(pred: dict, score_thresh: float = 0.1) -> str:
    boxes = pred["pred_boxes"][0].detach().cpu()
    scores = pred["pred_scores"][0].detach().cpu()
    labels = pred["pred_labels"][0].detach().cpu()
    keep = scores >= score_thresh
    lines = [
        f"n_det (score>={score_thresh}): {int(keep.sum())} / {len(scores)}",
        f"top scores: {scores[scores.argsort(descending=True)[:5]].tolist()}",
    ]
    if keep.any():
        lines.append(f"top boxes:\n{boxes[keep][scores[keep].argsort(descending=True)[:3]]}")
    if "pred_seg" in pred:
        seg = pred["pred_seg"]
        if seg.dim() == 5:
            seg = seg[:, 0]
        fg = (seg > 0.5).float().mean().item()
        lines.append(f"seg fg frac: {fg:.4f}")
    return "\n".join(lines)


class ForwardCapture:
    """Register hooks on nnDetection BaseRetinaNet; read `.tensors` after forward."""

    def __init__(self):
        self.tensors: dict = {}
        self._handles = []

    def _save(self, name):
        def hook(_mod, _inp, out):
            if torch.is_tensor(out):
                self.tensors[name] = out.detach()
            elif isinstance(out, (list, tuple)):
                self.tensors[name] = [
                    x.detach() if torch.is_tensor(x) else x for x in out
                ]
            else:
                self.tensors[name] = out

        return hook

    def register(self, net) -> ForwardCapture:
        enc = net.encoder
        dec = net.decoder
        self._handles.append(enc.register_forward_hook(self._save("encoder_out")))
        self._handles.append(dec.register_forward_hook(self._save("decoder_out")))
        self._handles.append(net.head.register_forward_hook(self._save("head_out")))
        if net.segmenter is not None:
            self._handles.append(
                net.segmenter.register_forward_hook(self._save("seg_out"))
            )
        return self

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def fpn_head_levels(net, images: torch.Tensor) -> list:
    fpn = net.decoder(net.encoder(images))
    return [fpn[i] for i in net.decoder_levels]


def compare_feat_list(a: list, b: list, prefix: str = "") -> list:
    rows = []
    for i, (ta, tb) in enumerate(zip(a, b)):
        ta = ta.detach().float().cpu()
        tb = tb.detach().float().cpu()
        row = {
            "level": i,
            "nndet_shape": tuple(ta.shape),
            "det3d_shape": tuple(tb.shape),
        }
        if ta.shape == tb.shape:
            diff = (ta - tb).abs()
            row["max_abs_diff"] = float(diff.max())
            row["mean_abs_diff"] = float(diff.mean())
        rows.append(row)
        print(prefix, row)
    return rows


def _boxes_on_ax(ax, boxes, z, color, lw=1.5):
    if boxes is None or boxes.numel() == 0:
        return
    b = boxes.detach().cpu().numpy()
    for row in b:
        x1, y1, z1, x2, y2, z2 = row[:6]
        if z1 <= z <= z2:
            ax.plot(
                [x1, x2, x2, x1, x1],
                [y1, y1, y2, y2, y1],
                color=color,
                linewidth=lw,
            )


def show_patch_slices(
    images,
    z_indices=None,
    gt_boxes=None,
    pred_boxes=None,
    pred_scores=None,
    score_thresh=0.1,
    cmap="gray",
):
    """Matplotlib grid: axial slices with GT (lime) and pred (red) boxes."""
    img = images[0, 0].detach().cpu().numpy()
    d = img.shape[0]
    if z_indices is None:
        z_indices = [d // 4, d // 2, 3 * d // 4]
    pred = pred_boxes
    if pred_scores is not None and pred is not None:
        keep = pred_scores[0].detach().cpu() >= score_thresh
        pred = pred[0][keep]
    else:
        pred = pred[0] if pred is not None else None
    gt = gt_boxes[0] if gt_boxes is not None else None

    n = len(z_indices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, z in zip(axes, z_indices):
        ax.imshow(img[z], cmap=cmap)
        _boxes_on_ax(ax, gt, z, "lime", lw=2)
        _boxes_on_ax(ax, pred, z, "red", lw=1.5)
        ax.set_title(f"z={z}")
        ax.axis("off")
    fig.tight_layout()
    return fig


# %%
if __name__ == "__main__":
    setup_nndet_env()

# %%
#SECTION:--- stage 0 — load checkpoint + plan ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module, cfg, plan, ckpt_path, load_info = load_fold_checkpoint(DEFAULT_FOLD_DIR, CKPT_TAG)
    module = module.to(device)
    net = module.model
    print("ckpt", ckpt_path)
    print("load", load_info)
    print("patch", plan["patch_size"], "batch", plan["batch_size"])
    CTX.update(module=module, net=net, cfg=cfg, plan=plan, device=device, ckpt_path=ckpt_path)

# %%
#SECTION:--- stage 1 — native val dataloader (one patch batch) ---
    from nndet.io.datamodule.bg_module import Datamodule

    augment_cfg = dict(cfg["augment_cfg"])
    augment_cfg["batch_size"] = SCRATCH_BATCH_SIZE
    augment_cfg["multiprocessing"] = False
    augment_cfg["num_val_batches_per_epoch"] = 2
    data_dir = Path(cfg["host"]["preprocessed_output_dir"]) / plan["data_identifier"] / "imagesTr"
    datamodule = Datamodule(
        augment_cfg=augment_cfg,
        plan=plan,
        data_dir=data_dir,
        fold=FOLD,
    )
    datamodule.setup()
    val_gen = datamodule.val_dataloader()
    raw_batch = next(iter(val_gen))
    case_ids = raw_batch["case_id"]
    print("case", case_ids)
    print("data", raw_batch["data"].shape, "target", raw_batch["target"].shape)
    CTX.update(datamodule=datamodule, val_gen=val_gen, raw_batch=raw_batch, case_ids=case_ids)

# %%
#SECTION:--- stage 2 — pre_trafo + targets (native nnDetection path) ---
    batch = pre_trafo_batch(module, raw_batch)
    images = batch["data"].to(device=device, dtype=torch.float32)
    targets = batch_targets(batch, device)
    print("boxes", [b.shape for b in batch["boxes"]])
    print("classes", [c.tolist() for c in batch["classes"]])
    CTX.update(batch=batch, images=images, targets=targets)

# %%
#SECTION:--- stage 3 — forward + intermediate tensors ---
    capture = ForwardCapture().register(net)
    clear_cuda()
    net.eval()
    with torch.no_grad():
        pred_det, anchors, pred_seg = net(images)
        fpn_levels = fpn_head_levels(net, images)
    capture.tensors["fpn_head_levels"] = [t.detach() for t in fpn_levels]
    print("head keys", pred_det.keys())
    print("anchors", [a.shape for a in anchors])
    print("fpn levels", [tuple(t.shape) for t in fpn_levels])
    for k, v in capture.tensors.items():
        if isinstance(v, list):
            print(k, [tuple(x.shape) for x in v if torch.is_tensor(x)])
        elif torch.is_tensor(v):
            print(k, tuple(v.shape))
    CTX.update(capture=capture, pred_det=pred_det, anchors=anchors, pred_seg=pred_seg)

# %%
#SECTION:--- stage 4 — train_step eval + postprocessed preds ---
    clear_cuda()
    with torch.no_grad():
        losses, prediction = net.train_step(
            images=images,
            targets=targets,
            evaluation=True,
            batch_num=0,
        )
        infer_pred = net.inference_step(images)
    print("losses", {k: float(v) for k, v in losses.items()})
    print(pred_summary(prediction))
    print("--- inference_step ---")
    print(pred_summary(infer_pred))
    CTX.update(losses=losses, prediction=prediction, infer_pred=infer_pred)

# %%
#SECTION:--- stage 5 — visualize axial slices + boxes ---
    fig = show_patch_slices(
        images,
        gt_boxes=targets["target_boxes"],
        pred_boxes=infer_pred["pred_boxes"],
        pred_scores=infer_pred["pred_scores"],
        score_thresh=0.1,
    )
    plt.show()
    CTX["fig_slices"] = fig

# %%
#SECTION:--- stage 6 — det3d RetinaUNet feature extractor (untrained body) ---
    from det3d.detection.retinaunet import build_retinaunet
    from det3d.detection.retinaunet_network import build_retinaunet_feature_extractor

    det3d_plan = nndet_plan_to_det3d(plan)
    det3d_fe = build_retinaunet_feature_extractor(det3d_plan).to(device).eval()
    n_anchors = net.anchor_generator.num_anchors_per_location()[0]
    det3d_net = build_retinaunet(det3d_plan, num_anchors=n_anchors).to(device).eval()
    with torch.no_grad():
        det3d_feats = det3d_fe(images)
        det3d_heads = det3d_net(images)
    print("det3d fpn", [tuple(t.shape) for t in det3d_feats])
    print("det3d head keys", det3d_heads.keys())
    compare_feat_list(fpn_levels, det3d_feats, prefix="fe")
    CTX.update(det3d_plan=det3d_plan, det3d_fe=det3d_fe, det3d_net=det3d_net, det3d_feats=det3d_feats)

# %%
#SECTION:--- stage 7 — optional: same case via det3d DataManager ---
    if DET3D_PROJECT is None:
        print("set DET3D_PROJECT to run this stage")
    else:
        from fran.managers.project import Project

        from det3d.configs.parser import ConfigMakerDet
        from det3d.detection.nndet_train import det3d_batch_to_nndet, nndet_pred_to_vis
        from det3d.managers.data.batch_tfms import DataManagerDetLBDBTfms
        from det3d.managers.retinaunet import RetinaUNetManager

        project = Project(DET3D_PROJECT)
        conf = ConfigMakerDet(project).conf
        conf["dataset_params"]["batch_size"] = SCRATCH_BATCH_SIZE
        dm = DataManagerDetLBDBTfms(conf, project, fold=FOLD, mode="train")
        dm.prepare_data()
        dm.setup()
        det3d_batch = dm.transforms_batch(next(iter(dm.train_dataloader())))
        det3d_nb = det3d_batch_to_nndet(
            det3d_batch, forward_patch_size=list(plan["patch_size"])
        )
        with torch.no_grad():
            det3d_losses, det3d_pred = net.train_step(
                images=det3d_nb["data"],
                targets={
                    "target_boxes": det3d_nb["target_boxes"],
                    "target_classes": det3d_nb["target_classes"],
                    "target_seg": det3d_nb["target_seg"],
                },
                evaluation=True,
                batch_num=0,
            )
        vis = nndet_pred_to_vis(det3d_pred)
        print("det3d batch image", det3d_batch["image"].shape)
        print("nndet data", det3d_nb["data"].shape)
        print(pred_summary(det3d_pred))
        CTX.update(det3d_batch=det3d_batch, det3d_nb=det3d_nb, det3d_pred=det3d_pred, det3d_vis=vis)

# %%
#SECTION:--- stage 8 — REPL snippets (copy into console) ---
# Compare loaded ckpt vs fresh random init on same patch:
#   losses_rand, _ = CTX["net"].train_step(CTX["images"], CTX["targets"], True, 0)
#   print({k: (float(losses[k]), float(losses_rand[k])) for k in losses})
#
# Napari (if installed):
#   import napari; v=napari.Viewer(); v.add_image(CTX["images"][0,0].cpu().numpy()); v.add_labels(CTX["batch"]["target"][0,0].cpu().numpy())
#
# Hook cleanup:
#   CTX["capture"].close()

# end PythonMethodScratch
