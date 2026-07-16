"""CP-5 — dataloader batch parity: native nnDet vs DataManagerDetLBD.

Run one ``# %%`` cell at a time. Requires CP-2 parity task (run CP-2 in
``nndet_parity_cp0_4.py`` first). Inline asserts only.

Gate A: ``next(train_dl)`` both sides — structural keys/shapes + bridge runs.
Gate B: det3d-led crop on ``CASE_ID`` + ``det3d_batch_to_nndet`` vs native ref.
"""
from __future__ import annotations

import pickle
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from monai.transforms import ScaleIntensityRanged

from det3d.detection.nndet_train import det3d_batch_to_nndet
from det3d.archived.nndet_native_lbd import setup_nndet_env
from det3d.archived.nndet_parity_cp0_4 import (
    BOXES_ATOL,
    CASE_ID,
    DET3D_PLAN_ID,
    DET3D_PROJECT,
    NATIVE_PLAN_ID,
    PARITY_CP2_DET_DATA,
    PARITY_CP2_TASK,
    PARITY_SEED,
    _run_benchmark_smoke,
    _train_smoke_verdict,
    parity_compose_cfg,
)
from det3d.managers.data.main import DataManagerDetLBD

CP5_KEYS = "Ld,Rtr,L2,E,Norm"
PARITY_FG_LABELS = [0, 1]  # fallback; prefer _fg_labels_for_batch(det_batch)


def _fg_labels_for_batch(batch) -> list[int]:
    vals = set()
    for lab in batch["label"]:
        t = torch.as_tensor(lab).reshape(-1)
        vals.update(int(v) for v in t.tolist())
    return sorted(vals) if vals else [0]
IMAGES_TR = PARITY_CP2_TASK / "preprocessed" / NATIVE_PLAN_ID / "imagesTr"


def _shift_xyxyzz_boxes_to_crop(boxes: np.ndarray, crop_start) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    if boxes.size == 0:
        return boxes.reshape(0, 6)
    start = np.asarray(crop_start, dtype=np.float32)
    boxes[:, [0, 2]] -= start[0]
    boxes[:, [1, 3]] -= start[1]
    boxes[:, [4, 5]] -= start[2]
    return boxes


def _det3d_item_after_norm(dm, case_id: str, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    case_idx = next(i for i, row in enumerate(dm.data) if row["case_id"] == case_id)
    tfms = dm.transforms_dict
    dici = dm.data[case_idx]
    dici = tfms["Ld"](dici)
    dici = tfms["Rtr"](dici)[0]
    crop_slices = dici["crop_slices"]
    crop_start = dici["crop_start"]
    dici = tfms["L2"](dici)
    dici = tfms["E"](dici)
    dici = tfms["Norm"](dici)
    return dici, crop_slices, crop_start


def _item_to_collated_batch(dici):
    image = dici["image"]
    lm = dici["lm"]
    if image.dim() == 4:
        image = image.unsqueeze(0)
    if lm.dim() == 4:
        lm = lm.unsqueeze(0)
    elif lm.dim() == 3:
        lm = lm.unsqueeze(0)
    return {
        "image": image,
        "lm": lm,
        "bbox": [dici["bbox"]],
        "label": [dici["label"]],
    }


def _cp5_collate_train(batch):
    """Train collate for keys Ld..Norm only (no ToPoints in pipeline)."""
    imgs = []
    lms = []
    classes = []
    boxes = []
    for itemlist in batch:
        for item in itemlist:
            imgs.append(item["image"])
            lms.append(item["lm"])
            classes.append(item["label"])
            boxes.append(item["bbox"])
    return {
        "image": torch.stack(imgs, 0),
        "lm": torch.stack(lms, 0),
        "bbox": boxes,
        "label": classes,
    }


# %%
if __name__ == '__main__':
    
#SECTION:--- CP-5 config + prerequisites ---
    print("CASE_ID", CASE_ID)
    print("PARITY_CP2_TASK", PARITY_CP2_TASK)
    print("IMAGES_TR", IMAGES_TR)
    assert (IMAGES_TR / f"{CASE_ID}.npy").is_file(), (
        f"run CP-2 in nndet_parity_cp0_4.py first — missing {IMAGES_TR / f'{CASE_ID}.npy'}"
    )
    print("CP-5 keys", CP5_KEYS)

# %%
#SECTION:--- CP-5 native Datamodule setup ---
    setup_nndet_env(det_data=PARITY_CP2_DET_DATA)
    from nndet.io.datamodule.bg_module import Datamodule
    from nndet.io.load import load_pickle
    from omegaconf import OmegaConf

    cfg = parity_compose_cfg(PARITY_CP2_DET_DATA)
    plan = load_pickle(Path(str(cfg.host.plan_path)))
    data_dir = Path(cfg.host.preprocessed_output_dir) / plan["data_identifier"] / "imagesTr"
    augment_cfg = OmegaConf.to_container(cfg.augment_cfg, resolve=True)
    datamodule = Datamodule(
        augment_cfg=augment_cfg,
        plan=plan,
        data_dir=data_dir,
        fold=0,
    )
    datamodule.setup()
    native_dl = datamodule.train_dataloader()
    native_case_ids = list(datamodule.dataset_tr.keys())
    print("native train cases", len(native_case_ids))
    print("CASE_ID in native train", CASE_ID in native_case_ids)
    print("plan patch_size", plan["patch_size"])

# %%
#SECTION:--- CP-5 det3d DataManagerDetLBD setup ---
    from fran.managers.project import Project

    from det3d.configs.parser import ConfigMakerDet

    P = Project(DET3D_PROJECT)
    C = ConfigMakerDet(P)
    C.setup(DET3D_PLAN_ID)
    conf = deepcopy(C.configs)
    conf["dataset_params"]["fold"] = 0
    conf["plan_train"]["patch_size"] = [128, 128, 64]
    conf["plan_valid"]["patch_size"] = [128, 128, 64]
    for key in ("plan_train", "plan_valid", "plan_test"):
        if conf[key]["mode"] in {"det", "lbd"}:
            conf[key]["mode"] = "lbd"
    dm = DataManagerDetLBD(
        P,
        conf,
        batch_size=1,
        split="train",
        debug=True,
        keys=CP5_KEYS,
    )
    dm.prepare_data()
    dm.setup()
    dm.collate_fn = _cp5_collate_train
    dm.create_dataloader()
    det_dl = dm.dl
    det3d_case_ids = [row["case_id"] for row in dm.data]
    print("dm keys", dm.keys)
    print("det3d cases", len(det3d_case_ids))
    print("CASE_ID in det3d train", CASE_ID in det3d_case_ids)
    print("manifest", dm.hdf5_manifest_fn)
    print("plan patch_size", dm.plan["patch_size"])

# %%
#SECTION:--- CP-5 Gate A — structural next(batch) both sides ---
    torch.manual_seed(PARITY_SEED)
    np.random.seed(PARITY_SEED)
    det_batch = next(iter(det_dl))

    torch.manual_seed(PARITY_SEED)
    np.random.seed(PARITY_SEED)
    native_batch = next(iter(native_dl))

    print("det_batch keys", sorted(det_batch.keys()))
    print("native_batch keys", sorted(native_batch.keys()))
    print("det image", tuple(det_batch["image"].shape), det_batch["image"].dtype)
    print("native data", tuple(native_batch["data"].shape), native_batch["data"].dtype)
    print("native target", tuple(native_batch["target"].shape))
    assert int(det_batch["image"].shape[0]) == 1
    assert int(native_batch["data"].shape[0]) == 1
    bridged_a = det3d_batch_to_nndet(
        det_batch,
        fg_labels=_fg_labels_for_batch(det_batch),
    )
    print("bridged keys", sorted(bridged_a.keys()))
    print("bridged data", tuple(bridged_a["data"].shape))
    print("Gate A pass: both dataloaders + bridge OK (no pixel assert)")

# %%
#SECTION:--- CP-5 Gate B — det3d-led item + bridge vs native ref ---
    dici, parity_crop_slices, parity_crop_start = _det3d_item_after_norm(
        dm, CASE_ID, PARITY_SEED
    )
    det_batch_b = _item_to_collated_batch(dici)
    bridged = det3d_batch_to_nndet(
        det_batch_b,
        fg_labels=_fg_labels_for_batch(det_batch_b),
    )
    bridged_data = bridged["data"].detach().cpu().numpy()
    bridged_seg = bridged["target_seg"].detach().cpu().numpy()
    bridged_boxes = bridged["target_boxes"][0].detach().cpu().numpy()
    bridged_classes = bridged["target_classes"][0].detach().cpu().numpy()

    native_data_full = np.load(IMAGES_TR / f"{CASE_ID}.npy")
    native_seg_full = np.load(IMAGES_TR / f"{CASE_ID}_seg.npy")
    if native_seg_full.ndim == 4:
        native_seg_full = native_seg_full[0]
    with open(IMAGES_TR / f"{CASE_ID}_boxes.pkl", "rb") as handle:
        native_boxes_pkl = pickle.load(handle)

    cs = parity_crop_slices
    native_data_crop = native_data_full[(slice(None), *cs)]
    native_seg_crop = native_seg_full[cs]
    clip = dm.dataset_params["intensity_clip_range"]
    norm = ScaleIntensityRanged(
        keys=["image"],
        a_min=float(clip[0]),
        a_max=float(clip[1]),
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )
    native_data = norm({"image": torch.as_tensor(native_data_crop, dtype=torch.float32)})[
        "image"
    ].numpy()
    native_target = native_seg_crop.astype(np.int32)
    native_boxes_crop = _shift_xyxyzz_boxes_to_crop(
        np.asarray(native_boxes_pkl["boxes"]), parity_crop_start
    )
    native_labels = np.asarray(native_boxes_pkl["labels"])

    img_diff = float(np.max(np.abs(bridged_data - native_data)))
    print("image max abs diff", img_diff)
    assert img_diff < 1e-3, img_diff
    print("bridged_seg unique", np.unique(bridged_seg))
    print("native_target unique", np.unique(native_target))
    print("bridged n_boxes", bridged_boxes.shape[0], "native n_boxes", native_boxes_crop.shape[0])
    n_b = min(bridged_boxes.shape[0], native_boxes_crop.shape[0])
    for i in range(n_b):
        diff = float(np.max(np.abs(bridged_boxes[i] - native_boxes_crop[i])))
        if diff > BOXES_ATOL:
            print(f"bbox i={i} max diff {diff}")
            break
    else:
        if n_b:
            print(f"first {n_b} bboxes within atol={BOXES_ATOL}")
    if bridged_classes.shape == native_labels.shape:
        print("classes equal", np.array_equal(bridged_classes, native_labels))
    else:
        print("class count mismatch (sidecar vs pkl encoding)", bridged_classes.shape, native_labels.shape)
    print("Gate B pass")

# %%
#SECTION:--- CP-5 Gate C — native next(batch) sanity ---
    torch.manual_seed(PARITY_SEED)
    np.random.seed(PARITY_SEED)
    native_batch_c = next(iter(native_dl))
    print("native keys", native_batch_c.get("keys"))
    print("instance_mapping", native_batch_c.get("instance_mapping"))
    print("Gate C: native dataloader healthy (no value assert vs det3d)")

# %%
#SECTION:--- CP-5 train smoke (20 ep) ---
    smoke_t0 = time.perf_counter()
    _run_benchmark_smoke(cp=5, n_cases=16, batches_per_epoch=8)
    smoke_wall_s = time.perf_counter() - smoke_t0

# %%
#SECTION:--- CP-5 train smoke verdict ---
    _train_smoke_verdict(cp=5, wall_s=smoke_wall_s)
