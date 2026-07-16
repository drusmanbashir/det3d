"""Batch metrics for inference post stage comparison."""

import numpy as np
import torch
from label_analysis.geometry_pt import LabelMapGeometryPT


def tensor_fg_count(t):
    if t is None:
        return None
    x = t.detach().cpu() if isinstance(t, torch.Tensor) else torch.as_tensor(t)
    if x.numel() == 0:
        return 0
    while x.ndim > 3:
        x = x[0]
    return int((x > 0).sum())


def batch_n_boxes(batch):
    box = batch["pred_box"]
    if box is None:
        return None
    if isinstance(box, torch.Tensor):
        return int(box.shape[0])
    return len(box)


def batch_pred_shape(batch):
    pred = batch["pred"] if "pred" in batch else None
    if pred is None:
        for k in batch:
            if isinstance(k, str) and ("LIDCA" in k or k.endswith("-QUARK")):
                pred = batch[k]
                break
    if pred is None:
        return None
    t = pred.detach().cpu() if isinstance(pred, torch.Tensor) else pred
    while isinstance(t, (list, tuple)):
        t = t[0]
    return tuple(int(v) for v in t.shape)


def first_box_xyzxyz(batch):
    box = batch["pred_box"]
    if box is None or (isinstance(box, torch.Tensor) and box.numel() == 0):
        return None
    row = box[0].detach().cpu().numpy() if isinstance(box, torch.Tensor) else np.asarray(box[0])
    return [float(v) for v in row]


def fg_mask_equal(a, b):
    ta = a.detach().cpu() if isinstance(a, torch.Tensor) else torch.as_tensor(a)
    tb = b.detach().cpu() if isinstance(b, torch.Tensor) else torch.as_tensor(b)
    while ta.ndim > 3:
        ta = ta[0]
    while tb.ndim > 3:
        tb = tb[0]
    fa = ta > 0
    fb = tb > 0
    return bool(torch.equal(fa, fb)), int((fa != fb).sum()), int(fa.sum()), int(fb.sum())


def lmg_nbrhood_count(lm, ignore_labels):
    L = LabelMapGeometryPT(li=lm, ignore_labels=ignore_labels, compute_feret=False)
    return len(L.nbrhoods)


def lmg_centroid_dist(lm_a, lm_b, ignore_labels):
    La = LabelMapGeometryPT(li=lm_a, ignore_labels=ignore_labels, compute_feret=False)
    Lb = LabelMapGeometryPT(li=lm_b, ignore_labels=ignore_labels, compute_feret=False)
    if La.nbrhoods.empty or Lb.nbrhoods.empty:
        return None
    ca = La.nbrhoods.iloc[0][["centroid_x", "centroid_y", "centroid_z"]].astype(float).values
    cb = Lb.nbrhoods.iloc[0][["centroid_x", "centroid_y", "centroid_z"]].astype(float).values
    return float(np.linalg.norm(ca - cb))


def component_masks_from_lmg(lm, ignore_labels):
    #AI li_cc masks aligned to lm tensor (x,y,z)
    import SimpleITK as sitk

    L = LabelMapGeometryPT(li=lm, ignore_labels=ignore_labels, compute_feret=False)
    arr = sitk.GetArrayFromImage(L.li_cc_sitk)
    cc = torch.from_numpy(arr.transpose(2, 1, 0))
    masks = []
    for _, row in L.nbrhoods.iterrows():
        masks.append(cc == int(row["label_cc"]))
    return masks, L.nbrhoods


def _row_centroid(row):
    if "centroid_x" in row.index:
        return row[["centroid_x", "centroid_y", "centroid_z"]].astype(float).values
    cent = row["cent"]
    if isinstance(cent, str):
        from ast import literal_eval

        cent = literal_eval(cent)
    return np.array(cent, dtype=np.float64)


def match_nbrhoods_by_centroid(gt_df, pr_df):
    pairs = []
    used = set()
    for i, gt_row in gt_df.iterrows():
        gt_c = _row_centroid(gt_row)
        best_j, best_d = None, np.inf
        for j, pr_row in pr_df.iterrows():
            if j in used:
                continue
            pr_c = _row_centroid(pr_row)
            d = float(np.linalg.norm(gt_c - pr_c))
            if d < best_d:
                best_d, best_j = d, j
        used.add(best_j)
        pairs.append((i, best_j))
    return pairs


def assert_component_masks_equal(gt_masks, pr_masks, pairs):
    for gt_i, pr_i in pairs:
        if not torch.equal(gt_masks[gt_i], pr_masks[pr_i]):
            diff = int((gt_masks[gt_i] != pr_masks[pr_i]).sum())
            raise AssertionError(f"component mask mismatch pair ({gt_i},{pr_i}) on {diff} voxels")


def match_boxes_by_centroid(gt_boxes, pred_boxes):
    gt = gt_boxes if isinstance(gt_boxes, np.ndarray) else gt_boxes.detach().cpu().numpy()
    pr = pred_boxes if isinstance(pred_boxes, np.ndarray) else pred_boxes.detach().cpu().numpy()
    gt_c = (gt[:, :3] + gt[:, 3:]) / 2
    pr_c = (pr[:, :3] + pr[:, 3:]) / 2
    pairs = []
    used = set()
    for i, gc in enumerate(gt_c):
        best_j, best_d = None, np.inf
        for j, pc in enumerate(pr_c):
            if j in used:
                continue
            d = float(np.linalg.norm(gc - pc))
            if d < best_d:
                best_d, best_j = d, j
        used.add(best_j)
        pairs.append((i, best_j))
    return pairs


def boxes_max_delta(boxes_a, boxes_b, pairs):
    a = boxes_a if isinstance(boxes_a, np.ndarray) else boxes_a.detach().cpu().numpy()
    b = boxes_b if isinstance(boxes_b, np.ndarray) else boxes_b.detach().cpu().numpy()
    deltas = []
    for gt_i, pr_i in pairs:
        deltas.append(float(np.max(np.abs(a[gt_i] - b[pr_i]))))
    return deltas


def assert_plan_roundtrip(original, recovered, pred_boxes, gt_boxes, ignore_labels, n_lesions):
    eq, diff_n, fg_o, fg_r = fg_mask_equal(original, recovered)
    if fg_o != fg_r:
        raise AssertionError(f"fg voxel count mismatch orig={fg_o} recovered={fg_r}")
    if not eq:
        raise AssertionError(f"fg mask mismatch on {diff_n} voxels")

    gt_masks, gt_df = component_masks_from_lmg(original, ignore_labels)
    pr_masks, pr_df = component_masks_from_lmg(recovered, ignore_labels)
    if len(gt_df) != n_lesions or len(pr_df) != n_lesions:
        raise AssertionError(
            f"expected {n_lesions} lesions, got gt={len(gt_df)} pred={len(pr_df)}"
        )
    pairs = match_nbrhoods_by_centroid(gt_df, pr_df)
    assert_component_masks_equal(gt_masks, pr_masks, pairs)

    box_pairs = match_boxes_by_centroid(gt_boxes, pred_boxes)
    deltas = boxes_max_delta(gt_boxes, pred_boxes, box_pairs)
    for d in deltas:
        if d > 1.0:
            raise AssertionError(f"box roundtrip |d|={d} exceeds 1 voxel")
    return {"fg_equal": eq, "box_max_delta": deltas, "lesion_pairs": pairs}
