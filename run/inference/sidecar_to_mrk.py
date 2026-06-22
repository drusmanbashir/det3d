#!/usr/bin/env python3
"""Batch-convert inference sidecar JSON to Slicer ROI Box `.mrk.json`."""
import argparse
from pathlib import Path

from det3d.inference.markups import save_inference_markups
from det3d.inference.visualize import list_sidecar_files
from det3d.utils.bbox_sidecar import load_inference_sidecar
from fran.inference.helpers import infer_project, load_params


def predictions_dir(args):
    if args.dir is not None:
        return Path(args.dir)
    params = load_params(args.run_p)
    project = infer_project(params)
    return project.predictions_folder / args.run_p


def main(args):
    pred_dir = predictions_dir(args)
    sidecars = list_sidecar_files(pred_dir)
    for sidecar_fn in sidecars:
        out_fn = sidecar_fn.with_suffix(".mrk.json")
        if out_fn.exists() and not args.overwrite:
            continue
        sidecar = load_inference_sidecar(sidecar_fn)
        save_inference_markups(out_fn, sidecar, score_min=args.score_min)
        print(out_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write Slicer ROI Box markups from inference sidecar JSON."
    )
    parser.add_argument("--dir", default=None, help="Predictions folder with sidecar JSON.")
    parser.add_argument("--run-p", default=None, help="Run id under project predictions folder.")
    parser.add_argument("--project", default="lidca")
    parser.add_argument("--score-min", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.dir is None and args.run_p is None:
        raise SystemExit("Pass --dir or --run-p")
    main(args)
