import json
from copy import deepcopy
from pathlib import Path

import label_analysis
import numpy as np
import SimpleITK as sitk
from utilz.fileio import save_json

SCHEMA_URL = (
    "https://raw.githubusercontent.com/slicer/slicer/master/"
    "Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#"
)
# Matches Slicer-exported ROI Box .mrk.json (see vtkMRMLMarkupsROIJsonStorageNode).
LPS_ORIENTATION = [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0]
_TEMPLATE_PATH = Path(__file__).parent / "templates" / "roi_box_markup.json"
_DETECTION_COLORS_PATH = (
    Path(label_analysis.__file__).parent / "markup" / "templates" / "detection.json"
)


_ROI_SHELL = None


def _load_roi_shell():
    global _ROI_SHELL
    if _ROI_SHELL is None:
        _ROI_SHELL = json.loads(_TEMPLATE_PATH.read_text())
    return _ROI_SHELL


def _label_colors():
    rows = json.loads(_DETECTION_COLORS_PATH.read_text())
    return {int(row["label"]): row["color"] for row in rows}


def _voxel_box_physical_envelope(image, box_xyzxyz):
    """#AI Exclusive xyzxyz voxel box -> physical AABB center, size, world xyzxyz."""
    img = sitk.ReadImage(str(image)) if isinstance(image, (str, Path)) else image
    x0, y0, z0, x1, y1, z1 = [int(v) for v in box_xyzxyz]
    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
    spacing = np.array(img.GetSpacing(), dtype=float)
    axis_step = direction @ np.diag(spacing)
    c000 = np.array(img.TransformIndexToPhysicalPoint((x0, y0, z0)))
    c111 = np.array(img.TransformIndexToPhysicalPoint((x1 - 1, y1 - 1, z1 - 1)))
    half = 0.5 * (axis_step[:, 0] + axis_step[:, 1] + axis_step[:, 2])
    lo = c000 - half
    hi = c111 + half
    center = 0.5 * (lo + hi)
    size = hi - lo
    world = [float(lo[0]), float(lo[1]), float(lo[2]), float(hi[0]), float(hi[1]), float(hi[2])]
    return center, size, world


def roi_center_size_from_world_xyzxyz(box_world_xyzxyz):
    """#AI xyzxyz world/LPS mm → Slicer ROI center + size (matches hand-drawn .mrk.json)."""
    x0, y0, z0, x1, y1, z1 = [float(v) for v in box_world_xyzxyz]
    x_lo, x_hi = min(x0, x1), max(x0, x1)
    y_lo, y_hi = min(y0, y1), max(y0, y1)
    z_lo, z_hi = min(z0, z1), max(z0, z1)
    center = [0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi), 0.5 * (z_lo + z_hi)]
    size = [x_hi - x_lo, y_hi - y_lo, z_hi - z_lo]
    return center, size


def roi_center_size_from_voxel_box(image, box_xyzxyz):
    """#AI Voxel xyzxyz exclusive box + sitk image → Slicer ROI center + size mm."""
    center, size, _world = _voxel_box_physical_envelope(image, box_xyzxyz)
    return [float(v) for v in center], [float(v) for v in size]


def voxel_box_to_world_xyzxyz(image, box_xyzxyz):
    """#AI Voxel xyzxyz exclusive → world/LPS mm xyzxyz envelope."""
    _center, _size, world = _voxel_box_physical_envelope(image, box_xyzxyz)
    return world


def bbox_world_ras_to_roi_lps(bbox_world):
    """#AI xyzxyz world mm corners → Slicer ROI LPS .mrk.json center, size, orientation."""
    center, size = roi_center_size_from_world_xyzxyz(bbox_world)
    orient = list(LPS_ORIENTATION)
    return center, size, orient


def roi_markup_from_center_size(center, size, label, description="", color=None):
    """#AI Build one Slicer ROI Box markup dict from center/size mm."""
    shell = deepcopy(_load_roi_shell())
    orient = list(LPS_ORIENTATION)
    shell["center"] = [float(v) for v in center]
    shell["size"] = [float(v) for v in size]
    shell["orientation"] = orient
    shell["description"] = description
    cp = shell["controlPoints"][0]
    cp["label"] = label
    cp["position"] = list(center)
    cp["orientation"] = list(orient)
    if color is not None:
        shell["display"]["color"] = color
    return shell


def roi_markup_from_voxel_box(image, box_xyzxyz, label, description="", color=None):
    """#AI sitk image + voxel xyzxyz box → Slicer ROI Box markup dict."""
    center, size = roi_center_size_from_voxel_box(image, box_xyzxyz)
    markup = roi_markup_from_center_size(center, size, label, description=description, color=color)
    return markup


def _prediction_to_roi(pred, case_id, idx, colors, source_image=None):
    shell = deepcopy(_load_roi_shell())
    orient = list(LPS_ORIENTATION)
    center, size = roi_center_size_from_voxel_box(source_image, pred["bbox_voxel_full"])
    label = f"{case_id}-{idx + 1}"
    score = float(pred["score"])
    shell["center"] = center
    shell["size"] = size
    shell["orientation"] = orient
    shell["description"] = f"{score:.3f}"
    shell["score"] = score
    shell["class"] = int(pred["label"])
    cp = shell["controlPoints"][0]
    cp["label"] = label
    cp["position"] = list(center)
    cp["orientation"] = list(orient)
    if pred["label"] in colors:
        shell["display"]["color"] = colors[pred["label"]]
    return shell


def inference_sidecar_to_mrk_payload(sidecar, score_min=0.0):
    """#AI Build Slicer ROI Box markups dict from inference sidecar."""
    case_id = sidecar["case_id"]
    source_image = sidecar["source_image"]
    img = (
        sitk.ReadImage(str(source_image))
        if isinstance(source_image, (str, Path))
        else source_image
    )
    colors = _label_colors()
    markups = []
    idx = 0
    for pred in sidecar["predictions"]:
        if pred["score"] < score_min:
            continue
        markups.append(_prediction_to_roi(pred, case_id, idx, colors, img))
        idx += 1
    payload = {"@schema": SCHEMA_URL, "markups": markups}
    return payload


def save_inference_markups(out_fn, sidecar, score_min=0.0):
    """#AI Write `{case}.mrk.json` beside inference sidecar."""
    payload = inference_sidecar_to_mrk_payload(sidecar, score_min=score_min)
    save_json(payload, out_fn)
    return out_fn


def save_voxel_box_markups(out_fn, image, boxes, labels, descriptions=None, colors=None):
    """#AI Write Slicer ROI Box .mrk.json from voxel xyzxyz boxes."""
    img = sitk.ReadImage(str(image)) if isinstance(image, (str, Path)) else image
    markups = []
    for i, box in enumerate(boxes):
        label = labels[i]
        desc = descriptions[i] if descriptions is not None else ""
        color = colors[i] if colors is not None else None
        markups.append(roi_markup_from_voxel_box(img, box, label, description=desc, color=color))
    payload = {"@schema": SCHEMA_URL, "markups": markups}
    save_json(payload, out_fn)
    return out_fn
