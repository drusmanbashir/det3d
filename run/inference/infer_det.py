#!/usr/bin/env python3
import argparse
from pathlib import Path

from label_analysis.totalseg import TotalSegmenterLabels
from utilz.fileio import load_yaml

from det3d.inference.cascade import DetCascadeInferer
from fran.data.dataregistry import DS
from fran.utils.common import COMMON_PATHS


def resolve_input_images(folder: list[str] | None, datasets: list[str] | None) -> list[Path]:
    if (folder is None) == (datasets is None):
        raise ValueError("Pass exactly one of --folder or --dataset")

    def supported_image_files(image_dir: Path) -> list[Path]:
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


def default_run_w():
    cs = COMMON_PATHS["cold_storage_folder"]
    best_runs = load_yaml(Path(cs) / "conf" / "best_runs.yaml")
    return best_runs["totalseg"]["whole"]["runs"][0]


def main(args):
    localiser_labels = args.localiser_labels
    if localiser_labels is None:
        localiser_labels = list(TotalSegmenterLabels().label_region)
    run_w = args.run_w if args.run_w is not None else default_run_w()
    input_images = resolve_input_images(args.folder, args.dataset)
    inferer = DetCascadeInferer(
        run_w=run_w,
        run_p=args.run_p,
        localiser_labels=localiser_labels,
        devices=args.gpus,
        patch_overlap=args.patch_overlap,
        save=args.save,
        save_localiser=False,
        safe_mode=args.safe_mode,
        debug=args.debug,
    )
    inferer.run(input_images, overwrite=args.overwrite, chunksize=args.chunksize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cascade detection inference: TotalSeg localiser + RetinaNet on LBD crop."
    )
    parser.add_argument("--run-p", required=True, help="Detection run id (RetinaNet checkpoint).")
    parser.add_argument("--run-w", default=None, help="TotalSeg whole-image localiser run id.")
    parser.add_argument(
        "--localiser-labels",
        nargs="+",
        default=None,
        help="TotalSeg label ids for fg bbox; default TotalSegmenterLabels().label_region.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--folder", nargs="+")
    source.add_argument("--dataset", nargs="+")
    parser.add_argument("--gpus", nargs="+", type=int, default=[1])
    parser.add_argument("--chunksize", type=int, default=4)
    parser.add_argument("--patch-overlap", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--no-save", dest="save", action="store_false")
    args = parser.parse_known_args()[0]
    main(args)
