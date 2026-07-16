#!/usr/bin/env python3
"""Stage local LITS NIfTI pairs into nnDet Task003_Liver raw layout."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Symlink local LITS image/lm pairs into Task003_Liver/raw."
    )
    parser.add_argument(
        "--lits-root",
        default="/s/datasets/lits_segs_improved",
        help="FRAN lits root with images/ and label subfolder",
    )
    parser.add_argument(
        "--task-raw",
        required=True,
        help="Task003_Liver/raw directory under det_data silo",
    )
    parser.add_argument(
        "--labels-subdir",
        default="lms_singlelesionclass",
        help="Label folder (0=bg, 1=liver, 2=tumor)",
    )
    args = parser.parse_args()

    root = Path(args.lits_root)
    images_dir = root / "images"
    labels_dir = root / args.labels_subdir
    task_raw = Path(args.task_raw)
    images_tr = task_raw / "imagesTr"
    labels_tr = task_raw / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*.nii.gz"))
    if not image_files:
        raise FileNotFoundError(f"no images under {images_dir}")

    def resolve_label(stem: str) -> Path:
        for suffix in (".nii.gz", ".nii"):
            candidate = labels_dir / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"missing label for {stem} under {labels_dir}")

    case_ids = []
    skipped = []
    for img in image_files:
        stem = img.name.removesuffix(".nii.gz")
        try:
            lab = resolve_label(stem)
        except FileNotFoundError:
            skipped.append(stem)
            continue
        img_link = images_tr / img.name
        lab_link = labels_tr / img.name
        if img_link.exists() or img_link.is_symlink():
            img_link.unlink()
        if lab_link.exists() or lab_link.is_symlink():
            lab_link.unlink()
        img_link.symlink_to(img.resolve())
        lab_link.symlink_to(lab.resolve())
        case_ids.append(stem)

    dataset_json = {
        "name": "Liver",
        "description": "Local LITS segs (MSD-compatible label map)",
        "reference": "lits_segs_improved",
        "licence": "local",
        "release": "local",
        "tensorImageSize": "3D",
        "modality": {"0": "CT"},
        "labels": {
            "0": "background",
            "1": "liver",
            "2": "cancer",
        },
        "numTraining": len(case_ids),
        "numTest": 0,
    }
    (task_raw / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n")
    import sys
    print(f"staged {len(case_ids)} cases from {root} → {task_raw} (skipped {len(skipped)}: {skipped})", file=sys.stderr)


if __name__ == "__main__":
    main()
