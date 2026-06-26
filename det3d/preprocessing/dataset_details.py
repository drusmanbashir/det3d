from pathlib import Path

import pandas as pd
import torch
from det3d.utils.bbox_sidecar import bbox_sidecar_path, load_detection_sidecar, sidecar_bbox_empty
from utilz.stringz import info_from_filename


def dataset_details_from_image_file(image_fn, bbox_fn):
    image_fn = Path(image_fn)
    image = torch.load(image_fn, map_location="cpu", weights_only=False)
    shape = tuple(int(v) for v in torch.as_tensor(image).shape)
    n_voxels = int(torch.as_tensor(image).numel())
    boxes, _labels, _instances = load_detection_sidecar(bbox_fn)
    n_fg = int(len(boxes))
    return {
        "case_id": info_from_filename(image_fn.name, full_caseid=True)["case_id"],
        "fn_name": image_fn.name,
        "shape": shape,
        "n_fg": n_fg,
        "n_bg": n_voxels - n_fg,
        "has_fg": bool(n_fg > 0),
        "bbox_empty": sidecar_bbox_empty(bbox_fn),
    }


def create_results_df_from_det_folder(output_folder: Path) -> pd.DataFrame:
    output_folder = Path(output_folder)
    images_dir = output_folder / "images"
    bboxes_dir = output_folder / "bboxes"
    rows = []
    for img_fn in sorted(images_dir.glob("*.pt")):
        bbox_fn = bbox_sidecar_path(bboxes_dir, img_fn.stem)
        if not bbox_fn.is_file():
            continue
        rows.append(dataset_details_from_image_file(img_fn, bbox_fn))
    assert len(rows) > 0, f"No det cases under {output_folder}"
    return pd.DataFrame(rows)


def write_dataset_details_csv(output_folder: Path, overwrite=False) -> Path:
    output_folder = Path(output_folder)
    csv_fn = output_folder / "dataset_details.csv"
    if not overwrite and csv_fn.is_file():
        return csv_fn
    df = create_results_df_from_det_folder(output_folder)
    df.to_csv(csv_fn, index=False)
    return csv_fn
