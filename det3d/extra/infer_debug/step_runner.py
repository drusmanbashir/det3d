"""Stepwise transform chain runner with per-stage snapshots."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch

from det3d.extra.infer_debug.metrics import (
    batch_n_boxes,
    batch_pred_shape,
    first_box_xyzxyz,
    fg_mask_equal,
    tensor_fg_count,
)


@dataclass
class StageRecord:
    """Per-stage snapshot of prediction metrics along a transform chain."""

    chain: str
    key: str
    pred_fg: int | None
    n_boxes: int | None
    pred_shape: tuple | None
    box0: list | None


def _pred_tensor(batch):
    if "pred" in batch:
        return batch["pred"]
    for k, v in batch.items():
        if isinstance(k, str) and ("LIDCA" in k or k.endswith("-QUARK")):
            return v
    return None


def snapshot_batch(batch, chain: str, key: str) -> StageRecord:
    pred = _pred_tensor(batch)
    fg = tensor_fg_count(pred)
    return StageRecord(
        chain=chain,
        key=key,
        pred_fg=fg,
        n_boxes=batch_n_boxes(batch),
        pred_shape=batch_pred_shape(batch),
        box0=first_box_xyzxyz(batch),
    )


def run_transform_chain(
    batch,
    transforms: dict,
    keys: str,
    *,
    chain_name: str,
    skip_keys=frozenset({"S", "Sav", "SavM"}),
):
    records = []
    d = dict(batch)
    for key in keys.split(","):
        key = key.strip()
        if not key or key in skip_keys:
            continue
        d = transforms[key](d)
        records.append(snapshot_batch(d, chain_name, key))
    return d, records


def write_stage_report(records: list[StageRecord], path: Path | None = None) -> str:
    lines = ["chain\tkey\tpred_fg\tn_boxes\tpred_shape\tbox0"]
    prev_fg = None
    for r in records:
        line = f"{r.chain}\t{r.key}\t{r.pred_fg}\t{r.n_boxes}\t{r.pred_shape}\t{r.box0}"
        lines.append(line)
        if prev_fg is not None and r.pred_fg is not None and r.pred_fg != prev_fg:
            lines.append(f"# fg delta after {r.key}: {r.pred_fg - prev_fg:+d}")
        prev_fg = r.pred_fg
    text = "\n".join(lines)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    return text


def compare_stage_to_ref(records: list[StageRecord], ref_fg: int, ref_boxes: int | None = None):
    diverged = []
    for r in records:
        if r.pred_fg is not None and r.pred_fg != ref_fg:
            diverged.append((r.key, "pred_fg", r.pred_fg, ref_fg))
        if ref_boxes is not None and r.n_boxes is not None and r.n_boxes != ref_boxes:
            diverged.append((r.key, "n_boxes", r.n_boxes, ref_boxes))
    return diverged


def save_stage_seg(batch, path: Path, full_meta=None):
    from monai.data.meta_tensor import MetaTensor

    pred = _pred_tensor(batch)
    if pred is None:
        return
    t = pred.detach().cpu()
    while t.ndim > 3:
        t = t[0]
    if t.ndim == 3:
        t = t.unsqueeze(0)
    meta = deepcopy(full_meta) if full_meta is not None else deepcopy(pred.meta)
    out = MetaTensor((t > 0).to(torch.uint8).contiguous(), meta=meta)
    from monai.transforms import SaveImage

    saver = SaveImage(output_dir=str(path.parent), output_postfix="", separate_folder=False)
    saver({"pred": out}, meta_data={"pred": {"filename_or_obj": str(path)}})
