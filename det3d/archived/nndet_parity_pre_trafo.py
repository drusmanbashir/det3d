"""CP-pre_trafo — compare nnDet pre_trafo vs det3d adapter on the same batch.

Load your own collated batch (``image``, ``lm``, ``bbox``, ``label``, ``instances``),
then:

* **Branch A:** ``det3d_batch_to_pre_trafo_input`` → ``FindInstances`` →
  ``Instances2Boxes`` → ``Instances2Segmentation`` (native ``RetinaUNetModule.pre_trafo``)
* **Branch B:** ``det3d_batch_to_nndet`` (skip-pre_trafo adapter / current trainer path)

Gates:

* **Pre:** ``data``, instance ``target``, ``instance_mapping`` from det3d batch builder
* **Post:** ``data``, ``target_boxes``, ``target_classes``; ``target_seg`` semantic vs instance
  (print + optional semantic remap on det3d lm)

Run one ``# %%`` cell at a time. See ``/s/agent_rw/nndet_benchmark/NATIVE_NNDET_HANDOFF.md``.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from det3d.detection.nndet_train import (
    det3d_batch_to_nndet,
    det3d_batch_to_pre_trafo_input,
    det3d_semantic_target_seg_from_batch,
    ensure_nndet_importable,
)
from det3d.archived.nndet_parity_cp0_4 import (
    BOXES_ATOL,
    CASE_ID,
    DET3D_PLAN_ID,
    DET3D_PROJECT,
    LBD_FOLDER,
    PARITY_SEED,
)
from det3d.utils.bbox_sidecar import bbox_sidecar_path, load_detection_sidecar

FORWARD_PATCH_SIZE = [128, 128, 64]
DATA_ATOL = 1e-3
BATCH_SOURCE = "cp5_pipeline"  # cp5_pipeline | manual
STRICT_BOX_PARITY = False  # True only when bbox list matches lm instances (not raw sidecar)


def _stack_batch_dim(x: torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(x)
    if t.dim() == 3:
        return t.unsqueeze(0)
    if t.dim() == 4 and int(t.shape[0]) != 1:
        return t.unsqueeze(0)
    return t


def make_collated_batch(
    image: torch.Tensor,
    lm: torch.Tensor,
    bbox,
    label,
    instances=None,
) -> dict:
    """Build det3d collate dict from user tensors (batch size 1 or stacked B)."""
    image = _stack_batch_dim(image).float()
    lm = _stack_batch_dim(lm)
    if isinstance(bbox, list):
        bbox_list = [torch.as_tensor(b, dtype=torch.float32) for b in bbox]
    else:
        b = torch.as_tensor(bbox, dtype=torch.float32)
        bbox_list = [b] if b.dim() == 2 else [b[i] for i in range(b.shape[0])]
    if isinstance(label, list):
        label_list = [torch.as_tensor(l, dtype=torch.long).reshape(-1) for l in label]
    else:
        lab = torch.as_tensor(label, dtype=torch.long).reshape(-1)
        label_list = [lab] if lab.dim() == 1 else [lab[i] for i in range(lab.shape[0])]
    n = int(image.shape[0])
    if len(bbox_list) != n or len(label_list) != n:
        raise ValueError(
            f"batch dim mismatch: image B={n}, bbox={len(bbox_list)}, label={len(label_list)}"
        )
    batch = {
        "image": image,
        "lm": lm,
        "bbox": bbox_list,
        "label": label_list,
    }
    if instances is not None:
        batch["instances"] = instances if isinstance(instances, list) else [instances]
    return batch


def build_nndet_pre_trafo_compose():
    ensure_nndet_importable()
    from nndet.io.transforms import (
        Compose,
        FindInstances,
        Instances2Boxes,
        Instances2Segmentation,
    )

    return Compose(
        FindInstances(instance_key="target", save_key="present_instances"),
        Instances2Boxes(
            instance_key="target",
            map_key="instance_mapping",
            box_key="boxes",
            class_key="classes",
            present_instances="present_instances",
        ),
        Instances2Segmentation(
            instance_key="target",
            map_key="instance_mapping",
            present_instances="present_instances",
        ),
    )


def run_pre_trafo_stepped(batch_pre: dict, sidecar_boxes=None, verbose: bool = True) -> dict:
    """Apply FindInstances → Instances2Boxes → Instances2Segmentation with prints."""
    ensure_nndet_importable()
    from nndet.io.transforms.instances import (
        FindInstances,
        Instances2Boxes,
        Instances2Segmentation,
    )

    find_instances = FindInstances(instance_key="target", save_key="present_instances")
    instances2boxes = Instances2Boxes(
        instance_key="target",
        map_key="instance_mapping",
        box_key="boxes",
        class_key="classes",
        present_instances="present_instances",
    )
    instances2seg = Instances2Segmentation(
        instance_key="target",
        map_key="instance_mapping",
        present_instances="present_instances",
    )

    if verbose:
        print_pre_state("before pre_trafo (nnDet input)", batch_pre)

    batch_find = find_instances(**batch_pre)
    if verbose:
        print_pre_state("after FindInstances", batch_find)

    batch_boxes = instances2boxes(**batch_find)
    if verbose:
        print_pre_state("after Instances2Boxes", batch_boxes)

    batch_post = instances2seg(**batch_boxes)
    if verbose:
        print_pre_state("after Instances2Segmentation (pre_trafo done)", batch_post)

    if sidecar_boxes is not None and verbose:
        for i, (sb, tb) in enumerate(zip(sidecar_boxes, batch_post["boxes"])):
            sbnp = torch.as_tensor(sb).detach().cpu().numpy()
            tbnp = tb.detach().cpu().numpy()
            print(f"[{i}] sidecar bbox vs Instances2Boxes shapes {sbnp.shape} vs {tbnp.shape}")
            if sbnp.size and tbnp.size:
                print(f"    sidecar xyzxyz=\n{sbnp}\n    pre_trafo xyxyzz=\n{tbnp}")

    return batch_post


def pre_trafo_train_targets(batch_post: dict) -> dict:
    return {
        "data": batch_post["data"],
        "target_boxes": batch_post["boxes"],
        "target_classes": batch_post["classes"],
        "target_seg": batch_post["target"][:, 0],
    }


def det3d_semantic_seg_from_batch(batch_pre: dict) -> torch.Tensor:
    """Map det3d lm instance ids → semantic seg (same rule as Instances2Segmentation)."""
    return det3d_semantic_target_seg_from_batch(batch_pre)


def print_pre_state(label: str, batch: dict) -> None:
    lines = [f"=== {label} ==="]
    if "data" in batch:
        lines.append(f"data shape={tuple(batch['data'].shape)} dtype={batch['data'].dtype}")
    if "target" in batch:
        t = batch["target"]
        t0 = t[0, 0] if t.dim() >= 4 else t[0]
        inst = t0.long().unique(sorted=True)
        inst = inst[inst > 0].tolist()
        lines.append(f"target shape={tuple(t.shape)} instance_ids={inst}")
    if "instance_mapping" in batch:
        lines.append(f"instance_mapping={batch['instance_mapping']}")
    if "present_instances" in batch:
        lines.append(
            "present_instances="
            f"{[[int(v) for v in x] for x in batch['present_instances']]}"
        )
    if "boxes" in batch:
        for i, boxes in enumerate(batch["boxes"]):
            cls = batch["classes"][i] if "classes" in batch else None
            bnp = boxes.detach().cpu().numpy()
            lines.append(f"boxes[{i}] shape={tuple(boxes.shape)}")
            lines.append(f"    {bnp}")
            if cls is not None:
                lines.append(f"classes[{i}]={cls.detach().cpu().tolist()}")
    print("\n".join(lines))


def print_train_targets(label: str, targets: dict) -> None:
    lines = [f"=== {label} ===", f"data shape={tuple(targets['data'].shape)}"]
    for i in range(len(targets["target_boxes"])):
        seg = targets["target_seg"][i]
        lines.append(
            f"[{i}] boxes {tuple(targets['target_boxes'][i].shape)} "
            f"classes {targets['target_classes'][i].tolist()} "
            f"target_seg unique {seg.unique(sorted=True).tolist()}"
        )
    print("\n".join(lines))


def assert_pre_input(batch: dict, batch_pre: dict, fg_labels: list[int]) -> None:
    rebuilt = det3d_batch_to_pre_trafo_input(
        batch, forward_patch_size=FORWARD_PATCH_SIZE, fg_labels=fg_labels
    )
    assert tuple(rebuilt["data"].shape) == tuple(batch_pre["data"].shape)
    assert torch.allclose(rebuilt["data"], batch_pre["data"], atol=DATA_ATOL)
    assert tuple(rebuilt["target"].shape) == tuple(batch_pre["target"].shape)
    assert torch.equal(rebuilt["target"].long(), batch_pre["target"].long())
    assert rebuilt["instance_mapping"] == batch_pre["instance_mapping"]
    print("Pre gate pass: det3d_batch_to_pre_trafo_input stable")


def assert_post_parity(
    targets_pre: dict,
    targets_det: dict,
    batch_pre: dict,
    boxes_atol: float = BOXES_ATOL,
    data_atol: float = DATA_ATOL,
    strict_boxes: bool = STRICT_BOX_PARITY,
) -> None:
    assert tuple(targets_pre["data"].shape) == tuple(targets_det["data"].shape)
    img_diff = float(torch.max(torch.abs(targets_pre["data"] - targets_det["data"])).item())
    print("post data max abs diff", img_diff)
    assert img_diff < data_atol, img_diff

    sem_det = det3d_semantic_seg_from_batch(batch_pre)
    n = len(targets_pre["target_boxes"])
    assert n == len(targets_det["target_boxes"])
    for i in range(n):
        sp = targets_pre["target_seg"][i].long()
        sd = targets_det["target_seg"][i].long()
        ss = sem_det[i].long()
        print(
            f"[{i}] target_seg unique pre(semantic)={sp.unique(sorted=True).tolist()} "
            f"det(adapter lm ids)={sd.unique(sorted=True).tolist()} "
            f"lm→semantic={ss.unique(sorted=True).tolist()}"
        )
        assert torch.equal(sp, ss), "pre_trafo semantic seg vs lm-derived semantic mismatch"

    for i in range(n):
        pb = targets_pre["target_boxes"][i].detach().cpu().numpy()
        db = targets_det["target_boxes"][i].detach().cpu().numpy()
        cb = targets_pre["target_classes"][i].detach().cpu().numpy()
        cd = targets_det["target_classes"][i].detach().cpu().numpy()
        print(f"[{i}] n_boxes pre_trafo={pb.shape[0]} adapter={db.shape[0]}")
        if pb.shape[0] != db.shape[0]:
            msg = "box count mismatch (sidecar bbox vs lm Instances2Boxes)"
            if strict_boxes:
                raise AssertionError(msg)
            print(f"    WARN {msg}")
            continue
        for j in range(pb.shape[0]):
            diff = float(np.max(np.abs(pb[j] - db[j])))
            if diff > boxes_atol:
                print(f"    WARN box {j} max diff {diff} (atol={boxes_atol})")
            else:
                print(f"    box {j} max diff {diff}")
        print(f"    classes pre={cb.tolist()} adapter={cd.tolist()}")
        if strict_boxes:
            assert np.array_equal(cb, cd), (cb, cd)

    print("Post gate pass: data + semantic seg" + (" + strict boxes" if strict_boxes else ""))


def batch_with_adapter_boxes_from_pre_trafo(
    det3d_batch: dict, batch_post: dict, fg_labels: list[int]
) -> dict:
    """Replace sidecar bbox with nnDet xyxyzz boxes inverted to det3d xyzxyz exclusive."""
    from det3d.detection.nndet_train import nndet_batch_to_xyzxyz

    idx_to_label = {i: int(v) for i, v in enumerate(fg_labels)}
    out = dict(det3d_batch)
    out["bbox"] = [
        nndet_batch_to_xyzxyz(batch_post["boxes"][i]) for i in range(len(batch_post["boxes"]))
    ]
    out["label"] = [
        torch.tensor(
            [idx_to_label[int(v)] for v in batch_post["classes"][i].tolist()],
            dtype=torch.long,
        )
        for i in range(len(batch_post["classes"]))
    ]
    return out


def load_batch_from_cp5_pipeline(case_id: str, seed: int, fg_labels: list[int]) -> dict:
    """Ld→Rtr→L2→E→Norm item + sidecar bbox/label/instances → collated batch."""
    from det3d.configs.parser import ConfigMakerDet
    from det3d.archived.nndet_parity_cp5_dataloader import (
        CP5_KEYS,
        _det3d_item_after_norm,
        _item_to_collated_batch,
    )
    from det3d.managers.data.main import DataManagerDetLBD
    from fran.managers.project import Project

    boxes_list, labels_list, instances = load_detection_sidecar(
        bbox_sidecar_path(LBD_FOLDER / "bboxes", case_id)
    )
    label = torch.stack(labels_list, 0) if labels_list else torch.zeros(0, dtype=torch.long)
    bbox = torch.stack(boxes_list, 0) if boxes_list else torch.zeros(0, 6)

    P = Project(DET3D_PROJECT)
    C = ConfigMakerDet(P)
    C.setup(DET3D_PLAN_ID)
    conf = deepcopy(C.configs)
    conf["dataset_params"]["fold"] = 0
    conf["plan_train"]["patch_size"] = list(FORWARD_PATCH_SIZE)
    conf["plan_valid"]["patch_size"] = list(FORWARD_PATCH_SIZE)
    for key in ("plan_train", "plan_valid", "plan_test"):
        if conf[key]["mode"] in {"det", "lbd"}:
            conf[key]["mode"] = "lbd"
    dm = DataManagerDetLBD(
        P, conf, batch_size=1, split="train", debug=True, keys=CP5_KEYS
    )
    dm.prepare_data()
    dm.setup()
    dici, _, _ = _det3d_item_after_norm(dm, case_id, seed)
    batch = _item_to_collated_batch(dici)
    batch["bbox"] = [bbox]
    batch["label"] = [label]
    batch["instances"] = [instances]
    batch["_fg_labels"] = fg_labels
    return batch


def _fg_labels_from_manifest() -> list[int]:
    from utilz.fileio import load_json

    manifest = load_json(LBD_FOLDER / "manifest.json")
    labels_all = manifest["labels_all"]
    return [int(v) for v in labels_all if int(v) != 0] or [0]


# %%
# %%
if __name__ == '__main__':
#SECTION:-------------------- setup --------------------------------------------------------------------------------------
    
#SECTION:--- config ---
    print("CASE_ID", CASE_ID)
    print("PARITY_SEED", PARITY_SEED)
    print("FORWARD_PATCH_SIZE", FORWARD_PATCH_SIZE)
    print("BATCH_SOURCE", BATCH_SOURCE)
    FG_LABELS = _fg_labels_from_manifest()
    print("FG_LABELS", FG_LABELS)

# %%
#SECTION:--- load batch (image, lm, bbox, label, instances) ---
    if BATCH_SOURCE == "cp5_pipeline":
        det3d_batch = load_batch_from_cp5_pipeline(CASE_ID, PARITY_SEED, FG_LABELS)
    elif BATCH_SOURCE == "manual":
        # Example — set tensors then uncomment:
        # det3d_batch = make_collated_batch(image, lm, bbox, label, instances=instances)
        raise ValueError(
            "BATCH_SOURCE='manual': assign det3d_batch = make_collated_batch(...) above, "
            "then re-run this cell"
        )
    else:
        raise ValueError(f"unknown BATCH_SOURCE={BATCH_SOURCE}")

    print("batch keys", sorted(det3d_batch.keys()))
    print("image", tuple(det3d_batch["image"].shape))
    print("lm", tuple(det3d_batch["lm"].shape))
    print("bbox[0]", tuple(det3d_batch["bbox"][0].shape))
    print("label[0]", det3d_batch["label"][0].tolist())
    print("instances[0]", det3d_batch["instances"][0])

# %%
#SECTION:--- pre gate — det3d_batch_to_pre_trafo_input ---
    batch_pre = det3d_batch_to_pre_trafo_input(
        det3d_batch,
        forward_patch_size=FORWARD_PATCH_SIZE,
        fg_labels=FG_LABELS,
    )
    print_pre_state("pre_trafo input (from det3d batch)", batch_pre)
    assert_pre_input(det3d_batch, batch_pre, FG_LABELS)

# %%
#SECTION:--- branch A — nnDet pre_trafo (stepped) ---
    batch_post = run_pre_trafo_stepped(batch_pre, sidecar_boxes=det3d_batch["bbox"])
    targets_pre = pre_trafo_train_targets(batch_post)
    print_train_targets("train_step targets from pre_trafo", targets_pre)

# %%
#SECTION:--- branch B — det3d adapter (skip pre_trafo) ---
    targets_det = det3d_batch_to_nndet(
        det3d_batch,
        fg_labels=FG_LABELS,
    )
    print_train_targets("train_step targets from det3d_batch_to_nndet", targets_det)

# %%
#SECTION:--- post gate — compare branches ---
    assert_post_parity(targets_pre, targets_det, batch_pre)

# %%
#SECTION:--- optional strict box parity (lm-aligned adapter boxes) ---
    if not STRICT_BOX_PARITY:
        print("Set STRICT_BOX_PARITY=True or run this cell for lm-aligned adapter test")
    batch_lm_boxes = batch_with_adapter_boxes_from_pre_trafo(det3d_batch, batch_post, FG_LABELS)
    targets_det_lm = det3d_batch_to_nndet(
        batch_lm_boxes,
        fg_labels=FG_LABELS,
    )
    assert_post_parity(
        targets_pre,
        targets_det_lm,
        batch_pre,
        strict_boxes=True,
    )
    print("Strict box parity pass (adapter uses pre_trafo boxes)")

# %%
#SECTION:--- optional — composed pre_trafo vs stepped ---
    compose = build_nndet_pre_trafo_compose()
    batch_compose = compose(**batch_pre)
    targets_compose = pre_trafo_train_targets(batch_compose)
    for key in ("data", "target_seg"):
        a = targets_pre[key] if key != "target_seg" else targets_pre["target_seg"]
        b = targets_compose[key] if key != "target_seg" else targets_compose["target_seg"]
        if key == "data":
            assert torch.allclose(a, b, atol=DATA_ATOL)
        else:
            assert torch.equal(a.long(), b.long())
    for i in range(len(targets_pre["target_boxes"])):
        assert torch.allclose(
            targets_pre["target_boxes"][i], targets_compose["target_boxes"][i], atol=1e-4
        )
        assert torch.equal(
            targets_pre["target_classes"][i], targets_compose["target_classes"][i]
        )
    print("Compose pre_trafo matches stepped pre_trafo")
