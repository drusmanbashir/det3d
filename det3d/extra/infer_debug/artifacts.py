"""Write LM NIfTI artifacts aligned to a reference volume."""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch


def write_lm_nifti_artifacts(original, recovered, ref_image_path, out_dir, case_id):
    ref = sitk.ReadImage(str(ref_image_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig = np.transpose(original.detach().cpu().numpy(), (2, 1, 0)).astype(np.uint8)
    rec = np.transpose(recovered.detach().cpu().numpy(), (2, 1, 0)).astype(np.uint8)
    diff = ((orig > 0) != (rec > 0)).astype(np.uint8)

    for suffix, arr in [
        ("lm_gt", orig),
        ("lm_recovered", rec),
        ("diff", diff),
    ]:
        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(ref)
        sitk.WriteImage(img, str(out_dir / f"{case_id}_{suffix}.nii.gz"))
