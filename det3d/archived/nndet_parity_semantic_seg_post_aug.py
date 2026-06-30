"""Semantic target_seg post-aug gate — precomputed helper vs native Instances2Segmentation.

Same production aug as disk-box gate (``DataManagerDetLBDBTfms`` + GpuTail).
Run one ``# %%`` cell at a time.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from det3d.detection.nndet_train import (
    _instance_mapping_for_item,
    _lm_seg_volume,
    det3d_semantic_target_seg_from_batch,
    ensure_nndet_importable,
)
from det3d.extra.nndet_parity_cp0_4 import (
    CASE_ID,
    DET3D_PLAN_ID,
    DET3D_PROJECT,
    LBD_FOLDER,
    PARITY_SEED,
)
from det3d.extra.nndet_parity_disk_boxes_post_aug import (
    _fg_labels_from_manifest,
    find_case_idx,
    run_disk_box_pipeline,
    setup_parity_dm,
)


def reference_semantic_native(
    lm_tensor: torch.Tensor,
    label: torch.Tensor,
    instances: dict | None,
    fg_labels: list[int],
) -> torch.Tensor:
    ensure_nndet_importable()
    from nndet.io.transforms.instances import FindInstances, Instances2Segmentation

    vol = _lm_seg_volume(lm_tensor)
    target = vol.float().unsqueeze(0).unsqueeze(0)
    mapping = _instance_mapping_for_item(
        vol, label, instances=instances, fg_labels=fg_labels
    )
    batch_pre = {"target": target, "instance_mapping": [mapping]}
    find = FindInstances(instance_key="target", save_key="present_instances")
    batch_find = find(**batch_pre)
    i2s = Instances2Segmentation(
        instance_key="target",
        map_key="instance_mapping",
        present_instances="present_instances",
    )
    batch_seg = i2s(**batch_find)
    return batch_seg["target"][0, 0].long()


def precomputed_semantic_from_batch(
    batch: dict,
    fg_labels: list[int],
    instances: dict | None,
) -> torch.Tensor:
    label = batch["label"][0] if isinstance(batch["label"], list) else batch["label"]
    lm = batch["lm"]
    if lm.dim() == 5:
        lm = lm[0]
    vol = _lm_seg_volume(lm)
    target = vol.float().unsqueeze(0).unsqueeze(0)
    mapping = _instance_mapping_for_item(
        vol, label, instances=instances, fg_labels=fg_labels
    )
    batch_pre = {"target": target, "instance_mapping": [mapping]}
    return det3d_semantic_target_seg_from_batch(batch_pre)[0].long()


def gate_semantic_seg_post_aug(
    dm,
    case_id: str,
    seed: int,
    fg_labels: list[int],
    *,
    instances: dict | None = None,
) -> bool:
    batch = run_disk_box_pipeline(dm, find_case_idx(dm, case_id), seed)
    label = batch["label"][0] if isinstance(batch["label"], list) else batch["label"]
    lm = batch["lm"]
    if lm.dim() == 5:
        lm = lm[0]
    sem_native = reference_semantic_native(lm, label, instances, fg_labels)
    sem_pre = precomputed_semantic_from_batch(batch, fg_labels, instances)
    print(
        "semantic unique native=",
        sem_native.unique(sorted=True).tolist(),
        "precomputed=",
        sem_pre.unique(sorted=True).tolist(),
    )
    approved = torch.equal(sem_native, sem_pre)
    print("SEMANTIC_SEG_GATE", "APPROVED" if approved else "REJECTED")
    return approved


# %%
if __name__ == "__main__":
#SECTION:--- config ---
    print("CASE_ID", CASE_ID)
    print("PARITY_SEED", PARITY_SEED)
    FG_LABELS = _fg_labels_from_manifest()
    print("FG_LABELS", FG_LABELS)
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# %%
#SECTION:--- setup DM ---
    dm = setup_parity_dm(batch_size=1, debug=True, device=DEVICE)
    print(dm)

# %%
#SECTION:--- semantic post-aug gate ---
    from det3d.extra.nndet_parity_disk_boxes_post_aug import load_sidecar_instances

    INST = load_sidecar_instances(CASE_ID)
    APPROVED = gate_semantic_seg_post_aug(
        dm, CASE_ID, PARITY_SEED, FG_LABELS, instances=INST
    )
    if not APPROVED:
        print("REJECTED: fix instance_mapping / lm before fast-path train")
    else:
        print("APPROVED: det3d_semantic_target_seg_from_batch matches Instances2Segmentation")
