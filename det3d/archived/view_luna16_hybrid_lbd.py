"""ImageBBoxViewer scratch — hybrid LBD .pt volumes (luna16_training_dm_hybrid checkpoint)."""
# %%
import matplotlib.pyplot as plt
import torch
from det3d.archived.luna16_view_cases import N_VIEW, PRED_DIR, list_val_cases, print_val_cases
from det3d.inference.hybrid_lbd import build_hybrid_detector
from det3d.inference.hybrid_samples import eval_dataset_for_cases, infer_batch
from utilz.imageviewers import ImageBBoxViewer

MODEL_PATH = "/s/agent_rw/tmp/luna16_lidc_dm_hybrid/detector.pt"
DEVICE_ID = 1
INDEX = 2
# SHOW: gt | pred | both
# ImageBBoxViewer colors boxes by list index (utilz.imageviewers.BBOX_COLORS):
#   #0 #e41a1c red   #1 #377eb8 blue  #2 #4daf4a green  #3 #984ea3 purple
#   #4 #ff7f00 orange #5 #a65628 brown #6 #f781bf pink  #7 #999999 grey  (cycles)
# gt   — all GT boxes: 1st GT red, 2nd GT blue, 3rd green, ...
# pred — all pred boxes (score >= SCORE_MIN): 1st pred red, 2nd pred blue, ...
# both — GT first then pred: 1st GT red, 2nd GT blue, ...; 1st pred takes next color after last GT
SHOW = "both"
SCORE_MIN = 0.3
USE_SCORE030 = True


def viewer_boxes_from_infer(gt, pred, scores, detector, show=SHOW, score_min=SCORE_MIN):
    pred_boxes = pred[detector.target_box_key]
    pred_scores = pred[detector.pred_score_key]
    if pred_scores.numel() > 0:
        keep = pred_scores >= float(score_min)
        pred_boxes = pred_boxes[keep]
    gt_boxes = gt[detector.target_box_key]
    if show == "gt":
        return gt_boxes
    if show == "pred":
        return pred_boxes
    if gt_boxes.numel() == 0:
        return pred_boxes
    if pred_boxes.numel() == 0:
        return gt_boxes
    return torch.cat([gt_boxes, pred_boxes], dim=0)


def view_hybrid_case(case_row, dm, plan, index, show=SHOW, score_min=SCORE_MIN, model_path=MODEL_PATH, device_id=DEVICE_ID):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    detector = build_hybrid_detector(plan, model_path, device)
    ds = eval_dataset_for_cases(dm.valid_manager, [case_row["case_id"]])
    batch = dm.valid_manager.collate_fn([ds[0]])
    val_input, val_target, val_output = infer_batch(detector, batch, plan, device, amp=True)
    boxes = viewer_boxes_from_infer(val_target, val_output, None, detector, show=show, score_min=score_min)
    vol = val_input[0] if val_input.dim() == 4 else val_input
    n_pred = int(val_output[detector.target_box_key].shape[0])
    n_gt = int(val_target[detector.target_box_key].shape[0])
    print(
        f"view {case_row['case_id']} idx={index} lbd_pt={case_row['lbd_pt'].name} "
        f"gt={n_gt} pred_raw={n_pred} boxes_drawn={boxes.shape[0]} score_min={score_min}"
    )
    viewer = ImageBBoxViewer(vol, boxes, orientation="axial")
    if boxes.numel() > 0:
        center_z = (boxes[:, 2] + boxes[:, 5]) / 2.0
        viewer.slider.set_val(int(round(float(center_z.mean()))))
    plt.show(block=True)
    return val_output


# %%
cases, dm = list_val_cases(n=N_VIEW, pred_dir=PRED_DIR, use_score030=USE_SCORE030)
plan = dm.valid_manager.plan
print_val_cases(cases)

view_hybrid_case(cases[INDEX], dm, plan, index=INDEX, show=SHOW, score_min=SCORE_MIN)
# %%
