#!/usr/bin/env python3
"""RetinaUNet seg inference on N LIDC2 nifti cases; export NIfTI pairs for 3D Slicer."""
import argparse

from det3d.inference.retinaunet_nifti import run_lidc2_seg_infer


def main(args):
    run_lidc2_seg_infer(
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        n_cases=args.n,
        project_title=args.project,
        device=args.device,
        overlap=args.overlap,
        num_tta=args.num_tta,
        open_slicer=args.open_slicer,
        slicer_bin=args.slicer_bin,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RetinaUNet tiled inference on LIDC2 cases; writes image+pred_seg NIfTI for Slicer."
    )
    parser.add_argument(
        "--ckpt",
        default="/s/fran_storage/checkpoints/lidca/LIDC/LIDCA-SILL/checkpoints/last.ckpt",
        help="Lightning RetinaUNetManager checkpoint.",
    )
    parser.add_argument(
        "--out-dir",
        default="/s/agent_rw/tmp/lidca_sill_lidc2_seg",
        help="Output folder for image/seg NIfTI + manifest.json",
    )
    parser.add_argument("--project", default="lidca")
    parser.add_argument("-n", type=int, default=10, help="Number of LIDC2 nifti cases (sorted).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--num-tta", type=int, default=0, help="nnDetection mirror TTA count (0=off, 8=full 3D)")
    parser.add_argument(
        "--open-slicer",
        action="store_true",
        help="Launch 3D Slicer on first case after inference.",
    )
    parser.add_argument(
        "--slicer-bin",
        default="/home/ub/programs/Slicer/Slicer-SuperBuild-Debug/Slicer-build/Slicer",
    )
    args = parser.parse_known_args()[0]
    main(args)
