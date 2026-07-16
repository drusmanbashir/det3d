#!/usr/bin/env python3
"""Build lms_3as2: copy labels into a new folder with voxel value 3 remapped to 2."""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk


def load_label(path: Path) -> tuple[np.ndarray, nib.Nifti1Image | None, sitk.Image | None]:
    if path.suffixes[-2:] == [".nii", ".gz"] or path.suffix == ".nii":
        img = nib.load(str(path))
        return img.get_fdata().astype(np.int32), img, None
    if path.suffix == ".nrrd":
        sitk_img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(sitk_img).astype(np.int32)
        return arr, None, sitk_img
    raise ValueError(f"unsupported label format: {path}")


def save_nifti_gz(out: Path, data: np.ndarray, ref_nib: nib.Nifti1Image | None, ref_sitk: sitk.Image | None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if ref_nib is not None:
        hdr = ref_nib.header.copy()
        hdr.set_data_dtype(np.int16)
        nib.save(nib.Nifti1Image(data.astype(np.int16), ref_nib.affine, hdr), str(out))
        return
    if ref_sitk is not None:
        itk = sitk.GetImageFromArray(data.astype(np.int16))
        itk.SetOrigin(ref_sitk.GetOrigin())
        itk.SetDirection(ref_sitk.GetDirection())
        itk.SetSpacing(ref_sitk.GetSpacing())
        sitk.WriteImage(itk, str(out))
        return
    raise RuntimeError("no reference image for save")


def resolve_source(stem: str, sources: list[Path]) -> Path | None:
    exts = (".nii.gz", ".nii", ".nrrd")
    for src_dir in sources:
        for ext in exts:
            p = src_dir / f"{stem}{ext}"
            if p.is_file():
                return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap label 3 -> 2 into a new LITS label folder.")
    parser.add_argument("--lits-root", default="/s/datasets/lits_segs_improved")
    parser.add_argument("--out-subdir", default="lms_3as2")
    parser.add_argument(
        "--source-subdirs",
        nargs="+",
        default=["lms_singlelesionclass", "lms"],
        help="Search order for source labels (first hit wins)",
    )
    args = parser.parse_args()

    root = Path(args.lits_root)
    out_dir = root / args.out_subdir
    sources = [root / s for s in args.source_subdirs]
    images = sorted((root / "images").glob("*.nii.gz"))

    written = 0
    skipped = []
    for img in images:
        stem = img.name.removesuffix(".nii.gz")
        src = resolve_source(stem, sources)
        if src is None:
            skipped.append(stem)
            continue
        data, ref_nib, ref_sitk = load_label(src)
        before = set(np.unique(data).tolist())
        data = data.copy()
        data[data == 3] = 2
        after = set(np.unique(data).tolist())
        out = out_dir / f"{stem}.nii.gz"
        save_nifti_gz(out, data, ref_nib, ref_sitk)
        written += 1
        if 3 in before:
            print(f"{stem}: {sorted(before)} -> {sorted(after)} from {src.parent.name}/{src.name}")

    print(f"wrote {written} labels -> {out_dir}")
    if skipped:
        print(f"skipped {len(skipped)} missing sources: {skipped}")


if __name__ == "__main__":
    main()
