"""RetinaUNet cascade: patch + cascade post chains and LM-as-pred adapter."""

from copy import deepcopy

import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.transforms.post.dictionary import Invertd

from det3d.extra.infer_debug.fixtures._case import InferFixtureCase
from det3d.geometry.lmg import voxel_start_size_to_xyzxyz
from det3d.inference.post import BoxRd, InvPreprocessBoxd, Offd, PackRetinaUNetPredsd
from fran.data.dataset import FillBBoxPatchesd
from fran.inference.helpers import load_params
from fran.transforms.inferencetransforms import MakeWritabled, RenameDictKeys, SqueezeListofListsd
from fran.transforms.spatialtransforms import RestoreOriginalOrientationd
from label_analysis.geometry_pt import LabelMapGeometryPT
from utilz.stringz import ast_literal_eval

PATCH_POST_KEYS = "Pack,SqL,InvP,InvPreBox"
CASCADE_POST_KEYS_SAFE = "SqL,MR,W,F,R,Off,BoxR"


def build_patch_preprocess(params):
    clip = params["configs"]["dataset_params"]["intensity_clip_range"]
    spacing = ast_literal_eval(params["configs"]["plan_train"]["spacing"])
    return Compose(
        [
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            Spacingd(keys=["image"], pixdim=spacing),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=float(clip[0]),
                a_max=float(clip[1]),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=["image"], dtype=torch.float16),
        ]
    )


def build_patch_post_dict(preprocess_compose):
    return {
        "Pack": PackRetinaUNetPredsd(),
        "SqL": SqueezeListofListsd(keys=["bounding_box"]),
        "InvP": Invertd(keys=["pred"], transform=preprocess_compose, orig_keys=["image"]),
        "InvPreBox": InvPreprocessBoxd(),
    }


def build_cascade_post_dict(run_p):
    return {
        "SqL": SqueezeListofListsd(keys=["bounding_box"]),
        "MR": RenameDictKeys(new_keys=["pred"], keys=[run_p]),
        "W": MakeWritabled(keys=["pred"]),
        "F": FillBBoxPatchesd(),
        "R": RestoreOriginalOrientationd(keys=["pred"]),
        "Off": Offd(box_keys=["pred_box"]),
        "BoxR": BoxRd(box_key="pred_box"),
    }


def forward_lm_spatial(lm_crop, spacing):
    d = {"image": lm_crop}
    if lm_crop.ndim == 3:
        d = EnsureChannelFirstd(keys=["image"], channel_dim="no_channel")(d)
    d = Spacingd(keys=["image"], pixdim=spacing, mode="nearest")(d)
    d = Orientationd(keys=["image"], axcodes="RAS")(d)
    return d["image"]


def _nbrhood_bboxes(lm, ignore_labels):
    L = LabelMapGeometryPT(li=lm, ignore_labels=ignore_labels, compute_feret=False)
    out = []
    for _, row in L.nbrhoods.iterrows():
        out.append(np.array(voxel_start_size_to_xyzxyz(row["bbox"]), dtype=np.float64))
    return np.stack(out, axis=0) if out else np.zeros((0, 6))


def _pick_boxes_by_gt(lmg_boxes, gt_boxes):
    gt_c = (gt_boxes[:, :3] + gt_boxes[:, 3:]) / 2
    lmg_c = (lmg_boxes[:, :3] + lmg_boxes[:, 3:]) / 2
    selected = []
    used = set()
    for gc in gt_c:
        best_i, best_d = None, np.inf
        for i, lc in enumerate(lmg_c):
            if i in used:
                continue
            d = float(np.linalg.norm(gc - lc))
            if d < best_d:
                best_d, best_i = d, i
        used.add(best_i)
        selected.append(lmg_boxes[best_i])
    return np.stack(selected, axis=0)


def boxes_preproc_from_lm(lm_ch, case: InferFixtureCase, img_crop_native=None):
    boxes = _nbrhood_bboxes(lm_ch, case.ignore_labels)
    if boxes.shape[0] == case.n_lesions:
        return torch.tensor(boxes, dtype=torch.float32)
    if boxes.shape[0] > case.n_lesions:
        boxes = _pick_boxes_by_gt(boxes, case.lesion_boxes_full)
        return torch.tensor(boxes, dtype=torch.float32)
    raise ValueError(
        f"LMG on preproc lm: {boxes.shape[0]} boxes, expected {case.n_lesions}"
    )


def build_fake_patch_batch(
    case: InferFixtureCase,
    img_crop,
    lm_crop,
    *,
    run_p: str,
    params,
    preprocess_compose,
):
    spacing = ast_literal_eval(params["configs"]["plan_train"]["spacing"])
    pre = preprocess_compose({"image": img_crop})
    img_preproc = pre["image"]
    img_batch = img_preproc.unsqueeze(0)

    if lm_crop.ndim == 3:
        lm_crop = lm_crop.unsqueeze(0)
    lm_mt = MetaTensor(lm_crop.contiguous(), meta=deepcopy(case.image_full.meta))
    lm_preproc = forward_lm_spatial(lm_mt, spacing)
    lm_ch = lm_preproc[0] if lm_preproc.ndim == 4 else lm_preproc
    stitched_seg = lm_ch.float().unsqueeze(0).unsqueeze(0)

    boxes_preproc = boxes_preproc_from_lm(lm_ch, case, img_crop)
    n = boxes_preproc.shape[0]

    return {
        "image": img_batch,
        "bounding_box": [case.bounding_box],
        "stitched_seg": stitched_seg,
        "merged_boxes": boxes_preproc,
        "merged_labels": torch.zeros(n, dtype=torch.long),
        "merged_scores": torch.ones(n, dtype=torch.float32),
        "source_image": case.source_image,
        "_run_p": run_p,
        "_crop_native_shape": tuple(int(v) for v in img_crop.shape[-3:]),
        "_full_meta": case.full_meta,
    }


def patch_batch_to_cascade_item(patch_out, case: InferFixtureCase, run_p: str):
    pred = patch_out["pred"].detach().cpu()
    if pred.ndim == 5:
        pred = pred[0]
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
    return {
        run_p: pred,
        "pred_box": patch_out["pred_box"].detach().cpu(),
        "pred_label": patch_out["pred_label"].detach().cpu(),
        "pred_score": patch_out["pred_score"].detach().cpu(),
        "bounding_box": case.bounding_box,
        "source_image": case.source_image,
        "crop_spatial_shape": patch_out.get("_crop_native_shape"),
        "full_meta": case.full_meta,
    }


def load_run_params(run_p: str, project_title=None):
    return load_params(run_p)
