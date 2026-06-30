import json
from pathlib import Path

import torch
from det3d.configs.parser import ConfigMakerDet
from det3d.inference.hybrid_lbd import build_hybrid_detector, load_lbd_pt
from det3d.managers.data import DataManagerDualDet, DataManagerDualDetBTfms
from fran.managers import Project
from monai.data import Dataset
from utilz.fileio import save_json

BOX_KEY = "bbox"
LABEL_KEY = "label"


def normalize_plan_modes_for_det_pipeline(configs):
    for key in ("plan_train", "plan_valid", "plan_test"):
        plan = configs[key]
        if plan["mode"] in {"det", "lbd"}:
            plan["mode"] = "lbd"


def setup_det_dataloaders(project_title, configs, batch_size=1, batch_tfms=True, debug=False):
    normalize_plan_modes_for_det_pipeline(configs)
    plan = configs["plan_train"]
    configs["dataset_params"]["batch_size"] = int(batch_size)
    plan["batch_size"] = int(batch_size)
    dm_class = DataManagerDualDetBTfms if batch_tfms else DataManagerDualDet
    dm = dm_class(
        project_title=project_title,
        configs=configs,
        batch_size=int(batch_size),
        cache_rate=configs["dataset_params"].get("cache_rate", 0.0),
        device=configs["dataset_params"].get("device", "cuda"),
        ds_type=configs["dataset_params"].get("ds_type"),
        debug=debug,
        batch_tfms=batch_tfms,
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    return dm


def dm_val_inputs(batch, device):
    images = batch["image"].to(device)
    return [images[i].contiguous() for i in range(images.shape[0])]


def dm_val_targets(batch, device):
    return [
        {
            LABEL_KEY: batch[LABEL_KEY][i].to(device),
            BOX_KEY: batch[BOX_KEY][i].to(device),
        }
        for i in range(batch["image"].shape[0])
    ]


def setup_hybrid_dm(project_title, plan_id, fold=0, batch_tfms=True, debug=False):
    project = Project(project_title)
    config_maker = ConfigMakerDet(project)
    config_maker.setup(int(plan_id))
    configs = config_maker.configs
    configs["dataset_params"]["fold"] = int(fold)
    dm = setup_det_dataloaders(
        project_title=project_title,
        configs=configs,
        batch_size=1,
        batch_tfms=batch_tfms,
        debug=debug,
    )
    plan = configs["plan_train"]
    return dm, plan, configs


def first_n_case_ids(data, n):
    case_ids = []
    seen = set()
    for row in data:
        case_id = str(row["case_id"])
        if case_id in seen:
            continue
        seen.add(case_id)
        case_ids.append(case_id)
        if len(case_ids) >= n:
            break
    return case_ids


def eval_dataset_for_cases(valid_manager, case_ids):
    rows = valid_manager.create_data_dicts(case_ids)
    return Dataset(data=rows, transform=valid_manager.transforms)


def boxes_to_lists(boxes):
    box_t = torch.as_tensor(boxes, dtype=torch.float32)
    if box_t.numel() == 0:
        return []
    if box_t.ndim == 1:
        box_t = box_t.unsqueeze(0)
    return box_t.cpu().tolist()


def scores_to_lists(scores):
    score_t = torch.as_tensor(scores, dtype=torch.float32)
    if score_t.numel() == 0:
        return []
    return score_t.reshape(-1).cpu().tolist()


def infer_batch(detector, batch, plan, device, amp=True):
    val_inputs = dm_val_inputs(batch, device)
    val_targets = dm_val_targets(batch, device)
    use_inferer = True
    detector.eval()
    with torch.no_grad():
        if amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                val_outputs = detector(val_inputs, use_inferer=use_inferer)
        else:
            val_outputs = detector(val_inputs, use_inferer=use_inferer)
    return val_inputs[0], val_targets[0], val_outputs[0]


def save_viewer_sidecar(
    out_fn,
    case_id,
    split,
    index,
    image_path,
    gt_boxes,
    pred_boxes,
    pred_scores,
    score_min=0.0,
):
    pred_boxes = torch.as_tensor(pred_boxes, dtype=torch.float32)
    pred_scores = torch.as_tensor(pred_scores, dtype=torch.float32)
    if pred_scores.numel() > 0:
        keep = pred_scores >= float(score_min)
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
    payload = {
        "case_id": str(case_id),
        "split": str(split),
        "index": int(index),
        "image": str(image_path),
        "gt_bbox": boxes_to_lists(gt_boxes),
        "pred_bbox": boxes_to_lists(pred_boxes),
        "pred_score": scores_to_lists(pred_scores),
    }
    save_json(payload, out_fn)
    return payload


def infer_split_samples(
    detector,
    valid_manager,
    case_ids,
    split,
    out_dir,
    device,
    amp=True,
    score_min=0.0,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = eval_dataset_for_cases(valid_manager, case_ids)
    manifest = []
    for index, row in enumerate(ds):
        batch = valid_manager.collate_fn([row])
        _val_input, val_target, val_output = infer_batch(detector, batch, valid_manager.plan, device, amp=amp)
        case_id = row["case_id"]
        image_path = row["image"]
        sidecar_fn = out_dir / f"{split}_{index:02d}_{case_id}.json"
        payload = save_viewer_sidecar(
            sidecar_fn,
            case_id=case_id,
            split=split,
            index=index,
            image_path=image_path,
            gt_boxes=val_target[detector.target_box_key],
            pred_boxes=val_output[detector.target_box_key],
            pred_scores=val_output[detector.pred_score_key],
            score_min=score_min,
        )
        n_pred = len(payload["pred_bbox"])
        n_gt = len(payload["gt_bbox"])
        print(f"{split}[{index:02d}] {case_id}\tgt={n_gt}\tpred={n_pred}\t{sidecar_fn.name}")
        manifest.append(
            {
                "split": split,
                "index": index,
                "case_id": case_id,
                "sidecar": str(sidecar_fn),
                "image": str(image_path),
                "n_gt": n_gt,
                "n_pred": n_pred,
            }
        )
    return manifest


def run_hybrid_sample_infer(
    project_title,
    plan_id,
    model_path,
    out_dir,
    n_train=20,
    n_val=20,
    fold=0,
    batch_tfms=True,
    device=None,
    amp=True,
    score_min=0.0,
    debug=False,
):
    dm, plan, _configs = setup_hybrid_dm(
        project_title=project_title,
        plan_id=plan_id,
        fold=fold,
        batch_tfms=batch_tfms,
        debug=debug,
    )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    detector = build_hybrid_detector(plan, model_path, device)

    train_case_ids = first_n_case_ids(dm.train_manager.data, n_train)
    val_case_ids = first_n_case_ids(dm.valid_manager.data, n_val)
    print(f"train_cases={len(train_case_ids)} val_cases={len(val_case_ids)}")

    manifest = []
    manifest.extend(
        infer_split_samples(
            detector,
            dm.valid_manager,
            train_case_ids,
            "train",
            out_dir,
            device,
            amp=amp,
            score_min=score_min,
        )
    )
    manifest.extend(
        infer_split_samples(
            detector,
            dm.valid_manager,
            val_case_ids,
            "val",
            out_dir,
            device,
            amp=amp,
            score_min=score_min,
        )
    )
    manifest_fn = Path(out_dir) / "manifest.json"
    manifest_fn.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} sidecars -> {out_dir}")
    return manifest


def list_viewer_sidecars(out_dir):
    return sorted(Path(out_dir).glob("*.json"), key=lambda p: p.name)


def load_viewer_sidecar(sidecar_fn):
    payload = json.loads(Path(sidecar_fn).read_text())
    if "image" not in payload:
        raise KeyError(f"not a hybrid viewer sidecar: {sidecar_fn}")
    return payload


def viewer_boxes(sidecar, show="both"):
    gt = torch.as_tensor(sidecar["gt_bbox"], dtype=torch.float32)
    pred = torch.as_tensor(sidecar["pred_bbox"], dtype=torch.float32)
    if show == "gt":
        return gt
    if show == "pred":
        return pred
    if gt.numel() == 0:
        return pred
    if pred.numel() == 0:
        return gt
    return torch.cat([gt, pred], dim=0)


def view_hybrid_sidecar(sidecar_fn, show="both", orientation="axial", score_min=0.0):
    import matplotlib.pyplot as plt
    from utilz.imageviewers import ImageBBoxViewer

    sidecar = load_viewer_sidecar(sidecar_fn)
    if show in {"pred", "both"} and score_min > 0.0:
        scores = torch.as_tensor(sidecar["pred_score"], dtype=torch.float32)
        pred = torch.as_tensor(sidecar["pred_bbox"], dtype=torch.float32)
        if scores.numel() > 0:
            keep = scores >= float(score_min)
            pred = pred[keep]
        if show == "pred":
            boxes = pred
        else:
            gt = torch.as_tensor(sidecar["gt_bbox"], dtype=torch.float32)
            boxes = pred if gt.numel() == 0 else torch.cat([gt, pred], dim=0)
    else:
        boxes = viewer_boxes(sidecar, show=show)

    img = load_lbd_pt(sidecar["image"])
    vol = img[0] if img.dim() == 4 else img
    viewer = ImageBBoxViewer(vol, boxes, orientation=orientation)
    if boxes.numel() > 0 and orientation == "axial":
        center_z = (boxes[:, 2] + boxes[:, 5]) / 2.0
        viewer.slider.set_val(int(round(float(center_z.mean()))))
    plt.show(block=True)
    return sidecar
