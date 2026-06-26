"""Backfill ``instances`` on detection sidecar JSON.

Uses identity mapping ``{str(x): x for x in labels}`` (same as
``save_detection_sidecar`` when ``instances`` is omitted).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from det3d.utils.bbox_sidecar import load_detection_sidecar, save_detection_sidecar
from fran.configs.parser import ConfigMaker
from fran.managers import Project


def instances_mapping_from_labels(labels):
    # AI
    label_vals = [int(torch.as_tensor(l).item()) for l in labels]
    mapping = {str(x): x for x in label_vals}
    return mapping


def backfill_lbd_folder(
    lbd_folder: Path,
    *,
    dry_run: bool = False,
) -> dict:
    lbd_folder = Path(lbd_folder)
    bboxes_dir = lbd_folder / "bboxes"
    updated = 0
    skipped = 0
    for sidecar_fn in sorted(bboxes_dir.glob("*.json")):
        _boxes, labels, existing = load_detection_sidecar(sidecar_fn)
        mapping = instances_mapping_from_labels(labels)
        if existing == mapping:
            skipped += 1
            continue
        if dry_run:
            print(f"would update {sidecar_fn.name} instances={mapping}")
            updated += 1
            continue
        payload = json.loads(sidecar_fn.read_text())
        boxes = payload["bbox"]
        labels = payload["label"]
        ignore = payload["ignore_labels"] if "ignore_labels" in payload else None
        save_detection_sidecar(
            sidecar_fn,
            boxes,
            labels,
            ignore_labels=ignore,
            instances=mapping,
        )
        updated += 1
    return {
        "lbd_folder": str(lbd_folder),
        "updated": updated,
        "skipped_unchanged": skipped,
        "total_sidecars": len(list(bboxes_dir.glob("*.json"))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="lidca")
    parser.add_argument("--plan-id", type=int, default=4)
    parser.add_argument("--lbd-folder", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    P = Project(args.project)
    if args.lbd_folder is None:
        try:
            from det3d.configs.parser import ConfigMakerDet

            C = ConfigMakerDet(P)
        except ImportError:
            C = ConfigMaker(P)
        C.setup(args.plan_id)
        plan = C.configs["plan_train"]
        from fran.utils.folder_names import FolderNames

        lbd_folder = FolderNames(P, plan).lbd_folder
    else:
        lbd_folder = args.lbd_folder

    report = backfill_lbd_folder(
        lbd_folder,
        dry_run=args.dry_run,
    )
    print(report)


if __name__ == "__main__":
    main()
