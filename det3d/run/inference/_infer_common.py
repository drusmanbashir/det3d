#!/usr/bin/env python3
"""Resolve runs, input images and localiser labels for infer_det.py and infer_cascade.py.
Stock TotalSegmentator is not integrated for detection; default_run_w raises,
so cascade inference must supply --run-w with a FRAN localiser run.
"""
from pathlib import Path

from label_analysis.totalseg import TotalSegmenterLabels

from fran.data.dataregistry import DS
from fran.inference.helpers import load_params
from fran.run.inference.infer import get_run_row, tsl_label_loc



def default_run_w(run_p=None):
    raise ValueError("det3d does not yet support the default TotalSegmentator localiser; pass --run-w explicitly to use a FRAN localiser")


def resolve_input_images(folder, datasets):
    if (folder is None) == (datasets is None):
        raise ValueError("Pass exactly one of --folder or --dataset")

    def supported_image_files(image_dir):
        image_files = [fn for fn in image_dir.glob("*") if fn.is_file()]
        return sorted(
            [fn for fn in image_files if str(fn).endswith((".nii.gz", ".nii", ".nrrd"))]
        )

    img_fns = []
    if folder is not None:
        for fldr in folder:
            img_fns.extend(supported_image_files(Path(fldr)))
        return img_fns

    for item in datasets:
        ds = DS[item].folder / "images"
        img_fns.extend(supported_image_files(ds))
    return img_fns


def resolve_localiser_labels(args):
    run_w = args.run_w if args.run_w is not None else default_run_w(args.run_p)
    if args.localiser_labels is not None:
        return list(args.localiser_labels)
    if args.lung_localiser:
        return list(tsl_label_loc("lung", get_run_row(run_w)))
    mnemonic = load_params(args.run_p)["configs"]["mnemonic"]
    if mnemonic == "bones":
        return list(tsl_label_loc("bones", get_run_row(run_w)))
    return list(TotalSegmenterLabels().label_region)
