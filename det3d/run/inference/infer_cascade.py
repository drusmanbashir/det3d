#!/usr/bin/env python3
"""FRAN localiser + RetinaUNet cascade inference."""
import argparse

from det3d.inference.cascade import DetSegBBoxCascadeInferer as CascadeInferer

from _infer_common import (
    default_run_w,
    resolve_input_images,
    resolve_localiser_labels,
)


def main(args):
    run_w = args.run_w if args.run_w is not None else default_run_w(args.run_p)
    imgs = resolve_input_images(args.folder, args.dataset)
    labels = resolve_localiser_labels(args)
    inferer = CascadeInferer(
        run_w=run_w,
        run_p=args.run_p,
        project_title=args.project,
        localiser_labels=labels,
        devices=args.gpus,
        patch_overlap=args.patch_overlap,
        save=True,
    )
    inferer.run(imgs, chunksize=args.chunksize, overwrite=args.overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RetinaUNet cascade inference")
    parser.add_argument("--run-w", default=None, help="FRAN localiser run name")
    parser.add_argument("--run-p", required=True, help="RetinaUNet det run name")
    parser.add_argument("--project", default="lidca")
    parser.add_argument("--gpus", nargs="+", type=int, default=[1])
    parser.add_argument("--patch-overlap", type=float, default=0.25)
    parser.add_argument("--localiser-labels", nargs="*", type=int, default=None)
    parser.add_argument("--lung-localiser", action="store_true")
    parser.add_argument("--folder", nargs="*", default=None)
    parser.add_argument("--dataset", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunksize", type=int, default=12)
    args = parser.parse_known_args()[0]
    main(args)
