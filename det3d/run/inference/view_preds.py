#!/usr/bin/env python3
"""View stored detection sidecars with ImageBBoxViewer. No inference."""
import argparse
from pathlib import Path

from det3d.inference.visualize import (
    list_sidecar_files,
    save_sidecar_png,
    sidecar_file_at_index,
    view_inference_sidecar,
)
from det3d.utils.bbox_sidecar import load_inference_sidecar

DEFAULT_PREDICTIONS_DIR = Path("/s/fran_storage/predictions/lidc/LIDC-TAINT")


def predictions_dir(args):
    if args.dir is not None:
        return Path(args.dir)
    if args.run_p is not None:
        return Path("/s/fran_storage/predictions/lidc") / args.run_p
    return DEFAULT_PREDICTIONS_DIR


def main(args):
    pred_dir = predictions_dir(args)
    sidecars = list_sidecar_files(pred_dir)

    if args.list:
        for i, fn in enumerate(sidecars):
            print(f"{i}\t{fn.name}")
        return

    if args.index is None:
        raise ValueError("Pass --index N (0 .. {}) or --list".format(max(len(sidecars) - 1, 0)))

    sidecar_fn = sidecar_file_at_index(pred_dir, args.index)
    print(f"[{args.index}/{len(sidecars) - 1}] {sidecar_fn.name}")

    if args.save_png is not None:
        sidecar = load_inference_sidecar(sidecar_fn)
        save_sidecar_png(
            sidecar,
            args.save_png,
            score_min=args.score_min,
            top_k=args.top_k,
            crop_lbd=args.crop_lbd,
        )

    print("Opening ImageBBoxViewer (close window to exit)...")
    view_inference_sidecar(
        sidecar_fn,
        score_min=args.score_min,
        top_k=args.top_k,
        crop_lbd=args.crop_lbd,
        orientation=args.orientation,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ImageBBoxViewer for stored detection sidecars. No inference."
    )
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dir", default=None)
    parser.add_argument("--run-p", default=None)
    parser.add_argument("--score-min", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--orientation",
        default="axial",
        choices=["axial", "sagittal", "coronal"],
    )
    parser.add_argument("--crop-lbd", action="store_true")
    parser.add_argument("--save-png", default=None, help="Optional PNG export before viewer.")
    args = parser.parse_known_args()[0]
    main(args)
