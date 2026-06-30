"""Granular native nnDet ↔ det3d parity REPL — checkpoints CP-0 through CP-4.

Run one ``# %%`` cell at a time. Inline asserts only (no compare_* wrappers).
Train smoke: 20-epoch ``native_lbd`` after each CP tensor gate.

See ``/s/agent_rw/nndet_benchmark/NATIVE_NNDET_HANDOFF.md`` § CP map.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from det3d.extra.lbd_nndet_materialize import (
    DUSTING_THRESHOLD,
    _spacing_from_lm_meta,
    instance_seg_to_nndet_boxes_pkl,
    lm_pt_to_instance_seg,
)
from det3d.extra.nndet_native_lbd import (
    NATIVE_DATASET_JSON_SRC,
    NATIVE_PLAN_SRC,
    NNDET_ROOT,
    TASK,
    setup_nndet_env,
)
from det3d.inference.hybrid_lbd import load_lbd_pt
from det3d.preprocessing.hdf5_shards_det import (
    ensure_hdf5_shards_for_plan,
    read_case_from_shard,
    shard_path_for_case,
)
from det3d.utils.bbox_sidecar import bbox_sidecar_path, load_detection_sidecar

CASE_ID = "lidc_0067"
LBD_FOLDER = Path(
    "/r/datasets/preprocessed/lidca/lbd/spc_080_080_150_rlb40c36831_rlb40c36831_ex000"
)
PARITY_SEED = 42
PARITY_ROOT = Path("/s/agent_rw/nndet_benchmark/parity")
BENCHMARK_ROOT = Path("/s/agent_rw/nndet_benchmark")
DET3D_PROJECT = "lidca"
DET3D_PLAN_ID = 4
NATIVE_PLAN_ID = "D3V001_3d"
PARITY_CP2_DET_DATA = PARITY_ROOT / "cp2"
PARITY_CP2_TASK = PARITY_CP2_DET_DATA / TASK
TRAIN_SMOKE_LOG = PARITY_ROOT / "train_smoke_log.jsonl"
BOXES_ATOL = 0.5

img_path = LBD_FOLDER / "images" / f"{CASE_ID}.pt"
lm_path = LBD_FOLDER / "lms" / f"{CASE_ID}.pt"
bbox_path = bbox_sidecar_path(LBD_FOLDER / "bboxes", CASE_ID)


def parity_compose_cfg(det_data: Path, fold: int = 0, max_epochs: int = 2):
    from hydra import initialize_config_module
    from nndet.utils.config import compose

    initialize_config_module(config_module="nndet.conf", version_base="1.1")
    return compose(
        TASK,
        "config.yaml",
        overrides=[
            f"host.parent_data={det_data}",
            f"exp.fold={fold}",
            "+augment_cfg.batch_size=1",
            "augment_cfg.multiprocessing=False",
            "augment_cfg.num_train_batches_per_epoch=4",
            "augment_cfg.num_val_batches_per_epoch=2",
            "trainer_cfg.num_train_batches_per_epoch=4",
            "trainer_cfg.num_val_batches_per_epoch=2",
            "trainer_cfg.precision=32",
            f"trainer_cfg.max_num_epochs={int(max_epochs)}",
        ],
    )


def _symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    os.symlink(src, dst)


def _append_train_smoke_log(row: dict) -> None:
    TRAIN_SMOKE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_SMOKE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _train_smoke_csv(cp: int) -> Path:
    return BENCHMARK_ROOT / "results" / f"native_lbd_parity_cp{cp}_e20.csv"


def _run_benchmark_smoke(cp: int, n_cases: int, batches_per_epoch: int, gpu: int = 0) -> Path:
    run_id = f"parity_cp{cp}_e20"
    cmd = [
        sys.executable,
        "run/training/benchmark_det_pipelines.py",
        "run",
        "--pipelines",
        "native_lbd",
        "--n-cases",
        str(n_cases),
        "--epochs",
        "20",
        "--batches-per-epoch",
        str(batches_per_epoch),
        "--batch-size",
        "1",
        "--gpu",
        str(gpu),
        "--run-id",
        run_id,
    ]
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("WANDB_MODE", "disabled")
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd="/home/ub/code/det3d", env=env, check=True)
    wall_s = time.perf_counter() - t0
    print(f"CP-{cp} train smoke wall_s={wall_s:.1f}")
    return _train_smoke_csv(cp)


def _train_smoke_verdict(cp: int, wall_s: float | None = None) -> None:
    csv_path = _train_smoke_csv(cp)
    df = pd.read_csv(csv_path)
    ep1 = float(df.loc[df["epoch"] == 1, "train_det_loss"].iloc[0])
    ep20 = float(df.loc[df["epoch"] == df["epoch"].max(), "train_det_loss"].iloc[0])
    drop = ep1 - ep20
    passed = drop > 0.3
    print(f"CP-{cp} ep1={ep1:.3f} ep20={ep20:.3f} drop={drop:.3f} pass={passed}")
    _append_train_smoke_log(
        {
            "cp": cp,
            "run_id": f"parity_cp{cp}_e20",
            "ep1": ep1,
            "ep20": ep20,
            "drop": drop,
            "wall_s": wall_s,
            "pass": passed,
        }
    )
    assert drop > 0.3, "train_det_loss did not improve enough"


# %%
#SECTION:--- config + paths ---
    print("CASE_ID", CASE_ID)
    print("LBD_FOLDER", LBD_FOLDER)
    print("img_path", img_path, img_path.is_file())
    print("lm_path", lm_path, lm_path.is_file())
    print("bbox_path", bbox_path, bbox_path.is_file())
    print("PARITY_SEED", PARITY_SEED)

# %%
#SECTION:--- shard prerequisite (det3d CP-0+) ---
    from fran.managers.project import Project

    from det3d.configs.parser import ConfigMakerDet

    P = Project(DET3D_PROJECT)
    C = ConfigMakerDet(P)
    C.setup(DET3D_PLAN_ID)
    plan_src_dims = C.configs["plan_train"]["src_dims"]
    shards_dir, linked = ensure_hdf5_shards_for_plan(LBD_FOLDER, plan_src_dims)
    manifest_fn = shards_dir / "manifest.json"
    ext_bbox_fn = LBD_FOLDER / "extended_bboxes" / f"{CASE_ID}.json"
    print("shards_dir", shards_dir, "linked", linked)
    print("manifest", manifest_fn, manifest_fn.is_file())
    print("extended_bboxes", ext_bbox_fn, ext_bbox_fn.is_file())
    assert manifest_fn.is_file(), f"missing shard manifest {manifest_fn}"
    assert ext_bbox_fn.is_file(), f"missing extended bbox {ext_bbox_fn}"
    shard_path = shard_path_for_case(manifest_fn, CASE_ID)
    print("shard_path", shard_path)

# %%
#SECTION:--- CP-0 native — load LBD .pt ---
    img_t = load_lbd_pt(img_path)
    lm_t = load_lbd_pt(lm_path)
    while img_t.dim() > 3:
        img_t = img_t.squeeze(0)
    while lm_t.dim() > 3:
        lm_t = lm_t.squeeze(0)
    print("img_t", tuple(img_t.shape), img_t.dtype, float(img_t.min()), float(img_t.max()))
    print("lm_t", tuple(lm_t.shape), lm_t.dtype, int(lm_t.min()), int(lm_t.max()))

# %%
#SECTION:--- CP-0 det3d — read HDF5 shard ---
    h5_case = read_case_from_shard(shard_path, CASE_ID)
    img_h5 = h5_case["image"]
    lm_h5 = h5_case["lm"]
    print("img_h5", img_h5.shape, img_h5.dtype, float(img_h5.min()), float(img_h5.max()))
    print("lm_h5", lm_h5.shape, lm_h5.dtype, int(lm_h5.min()), int(lm_h5.max()))

# %%
#SECTION:--- CP-0 compare — image + lm ---
    assert img_t.shape == torch.as_tensor(img_h5).shape
    img_diff = float(np.max(np.abs(img_t.cpu().numpy() - img_h5)))
    print("img max abs diff", img_diff)
    assert img_diff < 1e-4, img_diff
    assert lm_t.shape == torch.as_tensor(lm_h5).shape
    lm_diff = float(np.max(np.abs(lm_t.cpu().numpy().astype(np.int64) - lm_h5)))
    print("lm max abs diff", lm_diff)
    assert lm_diff == 0.0, lm_diff

# %%
#SECTION:--- CP-0 train smoke (20 ep) ---
    smoke_t0 = time.perf_counter()
    _run_benchmark_smoke(cp=0, n_cases=8, batches_per_epoch=4)
    smoke_wall_s = time.perf_counter() - smoke_t0

# %%
#SECTION:--- CP-0 train smoke verdict ---
    _train_smoke_verdict(cp=0, wall_s=smoke_wall_s)

# %%
#SECTION:--- CP-1a native — LMG dust -> instance seg ---
    seg_np, mapping, _L = lm_pt_to_instance_seg(lm_t, dusting_threshold=DUSTING_THRESHOLD)
    print("seg_np", seg_np.shape, seg_np.dtype)
    print("unique seg", np.unique(seg_np))
    print("n_instances", len(mapping))

# %%
#SECTION:--- CP-1a det3d — HDF5 stores raw lm (inspect only) ---
    print("shard lm unique", np.unique(lm_h5))
    print("note: det3d HDF5 lm is raw label map, not dusted instance seg")

# %%
#SECTION:--- CP-1b native — instances_to_boxes_np -> boxes pkl ---
    boxes_pkl = instance_seg_to_nndet_boxes_pkl(seg_np, mapping)
    boxes_np = np.asarray(boxes_pkl["boxes"])
    print("native n_boxes", len(boxes_pkl["instances"]))
    print("native instances", boxes_pkl["instances"])
    print("native boxes shape", boxes_np.shape)

# %%
#SECTION:--- CP-1b det3d — shard bbox + sidecar json ---
    shard_bbox = h5_case["bbox"]
    shard_label = h5_case["label"]
    sidecar_boxes, sidecar_labels, _ = load_detection_sidecar(bbox_path)
    sidecar_t = (
        torch.stack(sidecar_boxes)
        if sidecar_boxes
        else torch.zeros((0, 6), dtype=torch.float32)
    )
    print("shard bbox", shard_bbox.shape, "shard label", shard_label.shape)
    print("sidecar bbox", tuple(sidecar_t.shape))
    assert shard_bbox.shape == sidecar_t.shape
    if shard_bbox.size:
        shard_sidecar_diff = float(np.max(np.abs(shard_bbox - sidecar_t.numpy())))
        print("shard vs sidecar max diff", shard_sidecar_diff)

# %%
#SECTION:--- CP-1b compare — native lm truth vs shard/sidecar ---
    boxes_atol = BOXES_ATOL
    print("native n_boxes", len(boxes_pkl["instances"]), "shard n_boxes", shard_bbox.shape[0])
    if shard_bbox.shape[0] != boxes_np.shape[0]:
        print("WARN CP-1b: box count mismatch (shard uses sidecar json, native uses lm truth)")
    n_compare = min(boxes_np.shape[0], shard_bbox.shape[0])
    for i in range(n_compare):
        diff = float(np.max(np.abs(boxes_np[i] - shard_bbox[i])))
        if diff > boxes_atol:
            print(f"first box delta > {boxes_atol} at i={i}: max_diff={diff}")
            break
    else:
        if n_compare and boxes_np.shape == shard_bbox.shape:
            print("boxes allclose within", boxes_atol)

# %%
#SECTION:--- CP-1c native — properties + spacing ---
    data_np = img_t.detach().cpu().numpy().astype(np.float32)
    if data_np.ndim == 3:
        data_np = data_np[np.newaxis, ...]
    spacing = _spacing_from_lm_meta(lm_t)
    properties = OrderedDict(
        {
            "original_size_of_raw_data": np.array(data_np.shape[1:], dtype=np.int32),
            "original_spacing": spacing.copy(),
            "spacing_after_resampling": spacing.copy(),
            "size_after_cropping": np.array(data_np.shape[1:], dtype=np.int32),
            "size_after_resampling": np.array(data_np.shape[1:], dtype=np.int32),
            "instances": dict(mapping),
            "classes": np.unique(seg_np),
            "use_nonzero_mask_for_norm": False,
        }
    )
    print("spacing", spacing)
    print("size", properties["original_size_of_raw_data"])

# %%
#SECTION:--- CP-1c det3d — shard attrs ---
    print("image_shape attr", h5_case["image_shape"])
    print("lm_shape attr", h5_case["lm_shape"])
    assert list(properties["original_size_of_raw_data"]) == h5_case["image_shape"]

# %%
#SECTION:--- CP-1 train smoke (20 ep) ---
    smoke_t0 = time.perf_counter()
    _run_benchmark_smoke(cp=1, n_cases=8, batches_per_epoch=4)
    smoke_wall_s = time.perf_counter() - smoke_t0

# %%
#SECTION:--- CP-1 train smoke verdict ---
    _train_smoke_verdict(cp=1, wall_s=smoke_wall_s)

# %%
#SECTION:--- CP-2 native — materialize one case to parity task tree ---
    from nndet.io.load import load_pickle, save_pickle

    task_dir = PARITY_CP2_TASK
    preprocessed = task_dir / "preprocessed"
    images_tr = preprocessed / NATIVE_PLAN_ID / "imagesTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    parity_plan = dict(load_pickle(NATIVE_PLAN_SRC))
    parity_plan["patch_size"] = np.array([128, 128, 64], dtype=np.int64)
    save_pickle(parity_plan, preprocessed / f"{NATIVE_PLAN_ID}.pkl")
    dataset_meta = json.loads(NATIVE_DATASET_JSON_SRC.read_text())
    dataset_meta["task"] = TASK
    (task_dir / "dataset.json").write_text(json.dumps(dataset_meta, indent=2))
    np.save(images_tr / f"{CASE_ID}.npy", data_np)
    np.save(images_tr / f"{CASE_ID}_seg.npy", seg_np[np.newaxis, ...].astype(np.int32))
    save_pickle(properties, images_tr / f"{CASE_ID}.pkl")
    save_pickle(boxes_pkl, images_tr / f"{CASE_ID}_boxes.pkl")
    splits = [{"train": [CASE_ID], "val": [CASE_ID]}]
    save_pickle(splits, preprocessed / "splits_final.pkl")
    print("native images_tr", images_tr)
    for name in (
        f"{CASE_ID}.npy",
        f"{CASE_ID}_seg.npy",
        f"{CASE_ID}.pkl",
        f"{CASE_ID}_boxes.pkl",
    ):
        p = images_tr / name
        print(name, p.is_file())

# %%
#SECTION:--- CP-2 det3d — shard manifest + extended bbox ---
    manifest = json.loads(manifest_fn.read_text())
    case_ids_manifest = set()
    for shard_info in manifest["shards"]:
        case_ids_manifest.update(str(c) for c in shard_info["case_ids"])
    print("manifest shards", len(manifest["shards"]))
    print("CASE_ID in manifest", CASE_ID in case_ids_manifest)
    print("extended_bbox", ext_bbox_fn.is_file())
    assert CASE_ID in case_ids_manifest

# %%
#SECTION:--- CP-2 compare — native files + det3d on-disk ---
    assert (images_tr / f"{CASE_ID}.npy").is_file()
    assert (images_tr / f"{CASE_ID}_seg.npy").is_file()
    assert shard_path.is_file()
    assert ext_bbox_fn.is_file()
    print("CP-2 pass: native task tree + det3d shard layout OK")

# %%
#SECTION:--- CP-2 train smoke (20 ep) ---
    smoke_t0 = time.perf_counter()
    _run_benchmark_smoke(cp=2, n_cases=16, batches_per_epoch=8)
    smoke_wall_s = time.perf_counter() - smoke_t0

# %%
#SECTION:--- CP-2 train smoke verdict ---
    _train_smoke_verdict(cp=2, wall_s=smoke_wall_s)

# %%
#SECTION:--- CP-3 setup — nnDet env for native Datamodule ---
    setup_nndet_env(det_data=PARITY_CP2_DET_DATA)

# %%
#SECTION:--- CP-3 native — Datamodule.setup ---
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
    native_case_ids = list(datamodule.dataset_tr.keys())
    print("plan patch_size", plan["patch_size"])
    print("augment transforms", [t.__class__.__name__ for t in augment_cfg["transforms"]])
    print("len dataset_tr", len(datamodule.dataset_tr))
    print("CASE_ID in native train", CASE_ID in native_case_ids)

# %%
#SECTION:--- CP-3 det3d — DataManagerDetLBDBTfms ---
    from det3d.managers.data.batch_tfms import DataManagerDetLBDBTfms

    conf = deepcopy(C.configs)
    conf["dataset_params"]["fold"] = 0
    conf["plan_train"]["patch_size"] = [128, 128, 64]
    conf["plan_valid"]["patch_size"] = [128, 128, 64]
    for key in ("plan_train", "plan_valid", "plan_test"):
        if conf[key]["mode"] in {"det", "lbd"}:
            conf[key]["mode"] = "lbd"
    dm = DataManagerDetLBDBTfms(P, conf, batch_size=1, split="train", debug=True)
    dm.prepare_data()
    dm.setup()
    tmt = dm
    det3d_case_ids = [row["case_id"] for row in tmt.data]
    print("tmt keys", tmt.keys)
    print("tmt data_folder", tmt.data_folder)
    print("manifest", tmt.hdf5_manifest_fn)
    print("plan patch_size", tmt.plan["patch_size"])
    print("CASE_ID in det3d train", CASE_ID in det3d_case_ids)

# %%
#SECTION:--- CP-3 compare — patch_size + case presence ---
    assert list(plan["patch_size"]) == list(tmt.plan["patch_size"])
    assert CASE_ID in native_case_ids
    assert CASE_ID in det3d_case_ids
    print("CP-3 pass: patch_size match, case in both pipelines")

# %%
#SECTION:--- CP-3 train smoke (20 ep) ---
    smoke_t0 = time.perf_counter()
    _run_benchmark_smoke(cp=3, n_cases=16, batches_per_epoch=8)
    smoke_wall_s = time.perf_counter() - smoke_t0

# %%
#SECTION:--- CP-3 train smoke verdict ---
    _train_smoke_verdict(cp=3, wall_s=smoke_wall_s)

# %%
#SECTION:--- CP-4 det3d — find case index + Ld ---
    case_idx = next(i for i, row in enumerate(tmt.data) if row["case_id"] == CASE_ID)
    tfms = tmt.transforms_dict
    dici = tmt.data[case_idx]
    dici = tfms["Ld"](dici)
    print("after Ld keys", sorted(dici.keys()))
    print("hdf5_shard_path", dici.get("hdf5_shard_path"))

# %%
#SECTION:--- CP-4 det3d — Rtr (seeded) ---
    torch.manual_seed(PARITY_SEED)
    np.random.seed(PARITY_SEED)
    dici = tfms["Rtr"](dici)[0]
    parity_crop_slices = dici["crop_slices"]
    parity_crop_start = dici["crop_start"]
    print("after Rtr keys", sorted(dici.keys()))
    print("crop_center", dici.get("crop_center"))
    print("crop_slices", parity_crop_slices)
    print("crop_start", parity_crop_start)

# %%
#SECTION:--- CP-4 det3d — L2 ---
    dici = tfms["L2"](dici)
    print("after L2 image", tuple(dici["image"].shape))
    print("after L2 lm", tuple(dici["lm"].shape))
    print("after L2 bbox", tuple(dici["bbox"].shape))

# %%
#SECTION:--- CP-4 det3d — E + Norm ---
    dici = tfms["E"](dici)
    dici = tfms["Norm"](dici)
    det_image = dici["image"]
    det_lm = dici["lm"]
    det_bbox = dici["bbox"]
    det_label = dici["label"]
    if torch.is_tensor(det_image):
        det_image_np = det_image.detach().cpu().numpy()
    else:
        det_image_np = np.asarray(det_image)
    if torch.is_tensor(det_lm):
        det_lm_np = det_lm.detach().cpu().numpy()
    else:
        det_lm_np = np.asarray(det_lm)
    if torch.is_tensor(det_bbox):
        det_bbox_np = det_bbox.detach().cpu().numpy()
    else:
        det_bbox_np = np.asarray(det_bbox)
    print("det_image", det_image_np.shape, det_image_np.min(), det_image_np.max())
    print("det_lm unique", np.unique(det_lm_np))
    print("det_bbox", det_bbox_np.shape)

# %%
#SECTION:--- CP-4 native — load full case from CP-2 imagesTr ---
    native_data_full = np.load(images_tr / f"{CASE_ID}.npy")
    native_seg_full = np.load(images_tr / f"{CASE_ID}_seg.npy")
    if native_seg_full.ndim == 4:
        native_seg_full = native_seg_full[0]
    with open(images_tr / f"{CASE_ID}_boxes.pkl", "rb") as handle:
        import pickle

        native_boxes_pkl = pickle.load(handle)
    print("native_data_full", native_data_full.shape)
    print("native_seg_full", native_seg_full.shape)

# %%
#SECTION:--- CP-4 native — apply det3d crop_slices ---
    cs = parity_crop_slices
    native_data_crop = native_data_full[(slice(None), *cs)]
    native_seg_crop = native_seg_full[cs]
    print("native_data_crop", native_data_crop.shape)
    print("native_seg_crop", native_seg_crop.shape)
    assert native_data_crop.shape[1:] == det_image_np.shape[1:]

# %%
#SECTION:--- CP-4 native — intensity norm (match det3d ScaleIntensityRanged) ---
    from monai.transforms import ScaleIntensityRanged

    clip = dm.dataset_params["intensity_clip_range"]
    norm = ScaleIntensityRanged(
        keys=["image"],
        a_min=float(clip[0]),
        a_max=float(clip[1]),
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )
    normed = norm({"image": torch.as_tensor(native_data_crop, dtype=torch.float32)})
    native_data = normed["image"].numpy()
    native_target = native_seg_crop.astype(np.int32)
    print("native_data", native_data.shape, float(native_data.min()), float(native_data.max()))
    print("native_target unique", np.unique(native_target))

# %%
#SECTION:--- CP-4 compare — image ---
    img_diff = float(np.max(np.abs(det_image_np - native_data)))
    print("image max abs diff", img_diff)
    assert img_diff < 1e-3, img_diff

# %%
#SECTION:--- CP-4 compare — lm / target ---
    print("det_lm unique", np.unique(det_lm_np))
    print("native_target unique", np.unique(native_target))

# %%
#SECTION:--- CP-4 compare — bbox (xyzxyz vs xyxyzz) ---
    native_boxes = np.asarray(native_boxes_pkl["boxes"])
    if native_boxes.size and det_bbox_np.size:
        native_xyzxyz = native_boxes.copy()
        native_xyzxyz[:, [0, 1, 2, 3, 4, 5]] = native_boxes[:, [0, 1, 4, 2, 3, 5]]
        n_b = min(native_xyzxyz.shape[0], det_bbox_np.shape[0])
        print("native n_boxes", native_xyzxyz.shape[0], "det n_boxes", det_bbox_np.shape[0])
        for i in range(n_b):
            diff = float(np.max(np.abs(native_xyzxyz[i] - det_bbox_np[i])))
            if diff > BOXES_ATOL:
                print(f"bbox i={i} max diff {diff}")
                break
        else:
            print(f"first {n_b} bboxes within atol={BOXES_ATOL}")

# %%
#SECTION:--- CP-4 train smoke (20 ep) ---
    smoke_t0 = time.perf_counter()
    _run_benchmark_smoke(cp=4, n_cases=16, batches_per_epoch=8)
    smoke_wall_s = time.perf_counter() - smoke_t0

# %%
#SECTION:--- CP-4 train smoke verdict ---
    _train_smoke_verdict(cp=4, wall_s=smoke_wall_s)
