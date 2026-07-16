"""RetinaUNet volume inference on LIDC nifti cases; export seg NIfTI for Slicer."""
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from det3d.inference.lbd_pt import intensity_clip_range, load_lbd_pt, normalize_lbd_image
from det3d.managers.helpers.nndet_retinaunet import ensure_nndet_importable
from det3d.utils.tensor import plain_tensor, to_numpy
from fran.data.dataregistry import DS
from fran.managers import Project
from fran.utils.folder_names import FolderNames
from utilz.fileio import save_json


def case_id_from_nifti(nifti_path):
  #AI
    stem = Path(nifti_path).name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.endswith(".nii"):
        stem = stem[:-4]
    return stem


def lbd_pt_for_case(project, plan, case_id):
  #AI
    lbd_folder = Path(FolderNames(project, plan).lbd_folder)
    pt_path = lbd_folder / "images" / f"{case_id}.pt"
    if not pt_path.is_file():
        raise FileNotFoundError(f"missing LBD pt for {case_id}: {pt_path}")
    return pt_path


def select_lidc2_nifti(n=10):
  #AI
    return sorted((DS.lidc2.folder / "images").glob("*.nii.gz"))[: int(n)]


def load_retinaunet_manager(ckpt_path, device):
  #AI
    from det3d.managers.retinaunet import RetinaUNetManager

    manager = RetinaUNetManager.load_from_checkpoint(
        str(ckpt_path),
        map_location=device,
    )
    manager.eval()
    manager.to(device)
    return manager


def spacing_from_plan(plan):
  #AI
    spacing = plan["spacing"]
    return [float(v) for v in spacing]


def tensor_to_nifti_array(vol):
  #AI
    arr = to_numpy(vol) if torch.is_tensor(vol) else np.asarray(vol)
    if arr.ndim == 4:
        arr = arr[0]
    return arr


def save_nifti_volume(arr, out_path, spacing):
  #AI
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aff = np.diag([float(spacing[0]), float(spacing[1]), float(spacing[2]), 1.0])
    img = nib.Nifti1Image(arr, aff)
    nib.save(img, str(out_path))
    return out_path


def predict_plan_for_inference(nndet_plan, plan_train):
    #AI
    arch = nndet_plan["architecture"]
    plan = deepcopy(nndet_plan)
    plan["network_dim"] = 3
    plan["batch_size"] = 1
    plan["transpose_backward"] = [0, 1, 2]
    plan["inference_plan"] = {
        "model_score_thresh": float(plan_train["score_thresh"]),
        "model_detections_per_image": int(arch["detections_per_img"]),
        "model_topk": int(arch["topk_candidates"]),
        "remove_small_boxes": float(arch["remove_small_boxes"]),
    }
    return plan


def identity_properties(spatial_shape, spacing):
    #AI
    shape = tuple(int(v) for v in spatial_shape)
    spacing = [float(v) for v in spacing]
    crop_bbox = [[0, s] for s in shape]
    out = {
        "transpose_backward": [0, 1, 2],
        "original_spacing": spacing,
        "spacing_after_resampling": spacing,
        "crop_bbox": crop_bbox,
        "size_after_cropping": list(shape),
        "original_size_of_raw_data": list(shape),
        "itk_origin": [0.0, 0.0, 0.0],
        "itk_spacing": spacing,
        "itk_direction": np.eye(3).reshape(-1).tolist(),
    }
    return out


@torch.no_grad()
def nndet_predict_case(
    net, nndet_plan, plan_train, image, device, overlap=0.25, do_seg=True, num_tta=0
):
    #AI
    ensure_nndet_importable()
    from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001

    image = plain_tensor(image)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    case_np = to_numpy(image).astype(np.float32)
    spacing = plan_train["spacing"]
    properties = identity_properties(case_np.shape[1:], spacing)
    plan = predict_plan_for_inference(nndet_plan, plan_train)
    predictor = RetinaUNetV001.get_predictor(
        plan=plan,
        models=[net],
        num_tta_transforms=int(num_tta),
        do_seg=do_seg,
        device=str(device),
        overlap=float(overlap),
    )
    result = predictor.predict_case(
        {"data": case_np},
        properties,
        save_dir=None,
        case_id=None,
        restore=False,
    )
    boxes = result["boxes"]
    out = {
        "pred_boxes": [torch.as_tensor(boxes["pred_boxes"], device=device)],
        "pred_scores": [torch.as_tensor(boxes["pred_scores"], device=device)],
        "pred_labels": [torch.as_tensor(boxes["pred_labels"], device=device)],
    }
    if do_seg:
        seg = result["seg"]["pred_seg"]
        if torch.is_tensor(seg):
            seg_label = seg.to(dtype=torch.uint8)
        else:
            seg_label = torch.as_tensor(seg, dtype=torch.uint8)
        out["pred_seg_label"] = seg_label
    return out


def infer_case_seg(manager, pt_path, device, clip_range, overlap=0.25, num_tta=0):
  #AI
    img = normalize_lbd_image(load_lbd_pt(pt_path), clip_range)
    pred = nndet_predict_case(
        manager.net,
        manager.nndet_plan,
        manager.plan,
        img,
        device=device,
        overlap=overlap,
        do_seg=True,
        num_tta=num_tta,
    )
    return img, pred


def run_lidc2_seg_infer(
    ckpt_path,
    out_dir,
    n_cases=10,
    project_title="lidca",
    device=None,
    overlap=0.25,
    num_tta=0,
    open_slicer=False,
    slicer_bin="/home/ub/programs/Slicer/Slicer-SuperBuild-Debug/Slicer-build/Slicer",
):
  #AI
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    project = Project(project_title)
    manager = load_retinaunet_manager(ckpt_path, device)
    plan = manager.plan
    clip_range = intensity_clip_range(project=project, plan=plan)
    spacing = spacing_from_plan(plan)

    manifest = []
    nifti_paths = select_lidc2_nifti(n=n_cases)
    for index, nifti_path in enumerate(nifti_paths):
        case_id = case_id_from_nifti(nifti_path)
        pt_path = lbd_pt_for_case(project, plan, case_id)
        img, pred = infer_case_seg(
            manager,
            pt_path,
            device,
            clip_range,
            overlap=overlap,
            num_tta=num_tta,
        )
        seg_label = pred["pred_seg_label"]
        n_fg = int((seg_label > 0).sum().item())
        image_nii = out_dir / f"{case_id}_image.nii.gz"
        seg_nii = out_dir / f"{case_id}_pred_seg.nii.gz"
        save_nifti_volume(tensor_to_nifti_array(img), image_nii, spacing)
        save_nifti_volume(tensor_to_nifti_array(seg_label), seg_nii, spacing)
        n_boxes = int(pred["pred_boxes"][0].shape[0])
        row = {
            "index": index,
            "case_id": case_id,
            "source_nifti": str(nifti_path),
            "lbd_pt": str(pt_path),
            "image_nii": str(image_nii),
            "pred_seg_nii": str(seg_nii),
            "n_pred_boxes": n_boxes,
            "n_fg_voxels": n_fg,
        }
        manifest.append(row)
        print(
            f"[{index:02d}] {case_id}\tboxes={n_boxes}\tfg_vox={n_fg}\t"
            f"seg={seg_nii.name}\tsource={nifti_path.name}"
        )

    manifest_fn = out_dir / "manifest.json"
    save_json(manifest, manifest_fn)
    print(f"wrote {len(manifest)} cases -> {out_dir}")

    if open_slicer and manifest:
        first = manifest[0]
        launch_slicer(first["image_nii"], first["pred_seg_nii"], slicer_bin=slicer_bin)
    return manifest


def launch_slicer(image_nii, seg_nii, slicer_bin):
  #AI
    cmd = [str(slicer_bin), str(image_nii), str(seg_nii)]
    subprocess.Popen(cmd)
    print("slicer:", " ".join(cmd))
    return cmd


def load_manifest(out_dir):
  #AI
    manifest_fn = Path(out_dir) / "manifest.json"
    return json.loads(manifest_fn.read_text())


def open_slicer_case(out_dir, index=0, slicer_bin="/home/ub/programs/Slicer/Slicer-SuperBuild-Debug/Slicer-build/Slicer"):
  #AI
    manifest = load_manifest(out_dir)
    row = manifest[int(index)]
    return launch_slicer(row["image_nii"], row["pred_seg_nii"], slicer_bin=slicer_bin)
