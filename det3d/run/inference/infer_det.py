#!/usr/bin/env python3
import argparse
from pathlib import Path

from det3d.inference.cascade import DetBBoxCascadeInferer, DetSegBBoxCascadeInferer
from det3d.inference.lbd import DetLBDRunner

from _infer_common import (
    default_run_w,
    resolve_input_images,
    resolve_localiser_labels,
)


def resolve_lbd_pt_paths(lbd_folder, case_ids=None):
    folder = Path(lbd_folder)
    if case_ids is not None:
        return [folder / f"{case_id}.pt" for case_id in case_ids]
    return sorted(folder.glob("*.pt"))


def inferer_cls(arch):
    if arch == "retinaunet":
        return DetSegBBoxCascadeInferer
    if arch == "retinanet":
        return DetBBoxCascadeInferer
    raise ValueError(f"unsupported --arch {arch!r}; use retinanet or retinaunet")


def main(args):
    if args.lbd_folder is not None:
        pt_paths = resolve_lbd_pt_paths(args.lbd_folder, args.case_ids)
        runner = DetLBDRunner(
            run_p=args.run_p,
            project_title=args.project,
            devices=args.gpus,
            patch_overlap=args.patch_overlap,
            safe_mode=args.safe_mode,
            debug=args.debug,
            pred_run_p=args.pred_run_p if args.pred_run_p is not None else f"{args.run_p}-lbd",
            save=args.save,
        )
        runner.run(pt_paths, overwrite=args.overwrite, chunksize=args.chunksize)
        return

    cls = inferer_cls(args.arch)
    localiser_labels = resolve_localiser_labels(args)
    run_w = args.run_w if args.run_w is not None else default_run_w(args.run_p)
    input_images = resolve_input_images(args.folder, args.dataset)
    inferer = cls(
        run_w=run_w,
        run_p=args.run_p,
        project_title=args.project,
        localiser_labels=localiser_labels,
        devices=args.gpus,
        patch_overlap=args.patch_overlap,
        save=args.save,
        save_localiser=False,
        debug=args.debug,
    )
    inferer.run(input_images, overwrite=args.overwrite, chunksize=args.chunksize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cascade detection inference: TotalSeg localiser + RetinaNet or RetinaUNet on localiser crop."
    )
    parser.add_argument("--run-p", required=True, help="Detection run id (checkpoint).")
    parser.add_argument(
        "--arch",
        choices=("retinanet", "retinaunet"),
        default="retinanet",
        help="retinanet: bbox sidecar only; retinaunet: bbox sidecar + pred_seg NIfTI.",
    )
    parser.add_argument("--project", default=None, help="FRAN project title (required for retinaunet).")
    parser.add_argument("--run-w", default=None, help="TotalSeg whole-image localiser run id.")
    parser.add_argument(
        "--localiser-labels",
        nargs="+",
        default=None,
        help="TotalSeg label ids for fg bbox; default label_region, or lung labels with --lung-localiser.",
    )
    parser.add_argument(
        "--lung-localiser",
        action="store_true",
        help="Use lung TotalSeg labels for run-w (same as LIDC / chest runs).",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--folder", nargs="+")
    source.add_argument("--dataset", nargs="+")
    source.add_argument(
        "--lbd-folder",
        help="Pre-cropped LBD .pt folder; runs DetPatchLBD directly (no localiser).",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Optional case stems with --lbd-folder (default: all *.pt).",
    )
    parser.add_argument(
        "--pred-run-p",
        default=None,
        help="Predictions subfolder (default: {run-p}-lbd with --lbd-folder).",
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=[1])
    parser.add_argument("--chunksize", type=int, default=4)
    parser.add_argument("--patch-overlap", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--no-save", dest="save", action="store_false")
    args = parser.parse_known_args()[0]
    if args.arch == "retinaunet" and args.project is None:
        args.project = "lidca"
    main(args)
