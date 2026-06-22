#!/usr/bin/env python3
"""Open RetinaUNet seg NIfTI outputs in 3D Slicer."""
import argparse

from det3d.inference.retinaunet_nifti import load_manifest, open_slicer_case


def main(args):
    if args.list:
        manifest = load_manifest(args.out_dir)
        for row in manifest:
            print(
                f"{row['index']:>2}\t{row['case_id']}\t"
                f"boxes={row['n_pred_boxes']}\t{row['pred_seg_nii']}"
            )
        return
    open_slicer_case(args.out_dir, index=args.index, slicer_bin=args.slicer_bin)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open infer_retinaunet_lidc2_seg outputs in 3D Slicer.")
    parser.add_argument(
        "--out-dir",
        default="/s/agent_rw/tmp/lidca_sill_lidc2_seg",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--slicer-bin",
        default="/home/ub/programs/Slicer/Slicer-SuperBuild-Debug/Slicer-build/Slicer",
    )
    args = parser.parse_known_args()[0]
    main(args)
