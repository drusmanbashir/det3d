from pathlib import Path

import pandas as pd
import torch
from det3d.transforms.crop_indices import mask_fg_bg_flat_indices
from det3d.utils.bbox_sidecar import bbox_sidecar_path, sidecar_bbox_empty
from utilz.stringz import info_from_filename


def dataset_details_from_mask_file(mask_fn, bbox_fn):
    mask_fn = Path(mask_fn)
    mask = torch.load(mask_fn, map_location="cpu", weights_only=False)
    fg, bg = mask_fg_bg_flat_indices(mask)
    n_fg = int(len(fg))
    n_bg = int(len(bg))
    return {
        "case_id": info_from_filename(mask_fn.name, full_caseid=True)["case_id"],
        "fn_name": mask_fn.name,
        "shape": tuple(int(v) for v in torch.as_tensor(mask).shape),
        "n_fg": n_fg,
        "n_bg": n_bg,
        "has_fg": bool(n_fg > 0),
        "bbox_empty": sidecar_bbox_empty(bbox_fn),
    }


def create_results_df_from_det_folder(output_folder: Path) -> pd.DataFrame:
    output_folder = Path(output_folder)
    images_dir = output_folder / "images"
    masks_dir = output_folder / "masks"
    bboxes_dir = output_folder / "bboxes"
    rows = []
    for img_fn in sorted(images_dir.glob("*.pt")):
        mask_fn = masks_dir / img_fn.name
        bbox_fn = bbox_sidecar_path(bboxes_dir, img_fn.stem)
        if not mask_fn.is_file() or not bbox_fn.is_file():
            continue
        rows.append(dataset_details_from_mask_file(mask_fn, bbox_fn))
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
