"""Binarize LBD label maps: background 0, all lesion voxels 1.

Optionally sets ``label_org`` to 1 on matching bbox sidecars.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from monai.data import MetaTensor

from fran.managers import Project
from fran.utils.bbox_sidecar import load_nbrhood_sidecar, save_nbrhood_sidecar
from fran.utils.folder_names import FolderNames


def binarize_lm_tensor(lm: MetaTensor | torch.Tensor) -> MetaTensor | torch.Tensor:
    # AI
    meta = lm.meta if isinstance(lm, MetaTensor) else None
    arr = lm.as_tensor() if isinstance(lm, MetaTensor) else lm
    out = (arr > 0).to(torch.uint8)
    if meta is not None:
        result = MetaTensor(out.contiguous(), meta=meta)
        return result
    result = out.contiguous()
    return result


def binarize_lbd_folder(
    lbd_folder: Path,
    *,
    dry_run: bool = False,
) -> dict:
    # AI
    lbd_folder = Path(lbd_folder)
    lms_dir = lbd_folder / "lms"
    lm_files = sorted(lms_dir.glob("*.pt"))
    if not lm_files:
        raise FileNotFoundError(f"no label maps under {lms_dir}")

    lm_updated = 0
    lm_skipped = 0
    sidecar_updated = 0
    sidecar_skipped = 0
    rows = []

    for lm_fn in lm_files:
        lm = torch.load(lm_fn, weights_only=False)
        arr = lm.as_tensor() if isinstance(lm, MetaTensor) else lm
        uniq = sorted(int(v) for v in arr.unique().tolist())
        already_binary = set(uniq).issubset({0, 1})
        if already_binary and max(uniq, default=0) <= 1:
            lm_skipped += 1
        else:
            if dry_run:
                print(f"would binarize {lm_fn.name} labels {uniq}")
            else:
                torch.save(binarize_lm_tensor(lm), lm_fn)
            lm_updated += 1

        sidecar_fn = lbd_folder / "bboxes" / f"{lm_fn.stem}.csv"
        if sidecar_fn.is_file():
            nh = load_nbrhood_sidecar(sidecar_fn)
            if nh.empty:
                sidecar_skipped += 1
            elif (nh["label_org"].astype(int) == 1).all():
                sidecar_skipped += 1
            else:
                if dry_run:
                    print(f"would set label_org=1 on {sidecar_fn.name} ({len(nh)} rows)")
                else:
                    nh = nh.copy()
                    nh["label_org"] = 1
                    save_nbrhood_sidecar(sidecar_fn, nh)
                sidecar_updated += 1

        rows.append({"case_id": lm_fn.stem, "labels_before": uniq})

    report_fn = lbd_folder / "_logs" / "binarize_lbd_lms_report.tsv"
    if not dry_run:
        report_fn.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(report_fn, sep="\t", index=False)

    return {
        "lbd_folder": str(lbd_folder),
        "lm_updated": lm_updated,
        "lm_skipped": lm_skipped,
        "sidecar_updated": sidecar_updated,
        "sidecar_skipped": sidecar_skipped,
        "report": str(report_fn) if not dry_run else None,
    }


def resolve_lbd_folder(args) -> Path:
    # AI
    if args.lbd_folder is not None:
        return Path(args.lbd_folder)
    project = Project(args.project)
    try:
        from det3d.configs.parser import ConfigMakerDet

        maker = ConfigMakerDet(project)
    except ImportError:
        from fran.configs.parser import ConfigMaker

        maker = ConfigMaker(project)
    maker.setup(args.plan_id)
    plan = maker.configs["plan_train"]
    return Path(FolderNames(project, plan).lbd_folder)


def main(args) -> None:
    lbd_folder = resolve_lbd_folder(args)
    report = binarize_lbd_folder(
        lbd_folder,
        dry_run=args.dry_run,
    )
    print(report)
    if not args.dry_run and report["lm_updated"]:
        print("re-run LBD HDF5 shard build if shards already exist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="bones")
    parser.add_argument("--plan-id", type=int, default=1)
    parser.add_argument("--lbd-folder", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_known_args()[0]
    main(args)
