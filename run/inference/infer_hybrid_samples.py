#!/usr/bin/env python3
"""Infer hybrid RetinaNet on N train + N val LBD volumes; save ImageBBoxViewer sidecars."""
import argparse
from pathlib import Path

from det3d.inference.hybrid_samples import run_hybrid_sample_infer


def main(args):
    run_hybrid_sample_infer(
        project_title=args.project,
        plan_id=args.plan_id,
        model_path=args.model,
        out_dir=args.out_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        fold=args.fold,
        batch_tfms=not args.no_batch_tfms,
        device=args.device,
        amp=not args.no_amp,
        score_min=args.score_min,
        debug=args.debug,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid DM inference on train/val samples; writes viewer JSON sidecars."
    )
    parser.add_argument("--project", default="lidca")
    parser.add_argument("--plan-id", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--model",
        default="/s/agent_rw/tmp/luna16_lidc_dm_hybrid/detector.pt",
        help="TorchScript checkpoint (best detector.pt).",
    )
    parser.add_argument(
        "--out-dir",
        default="/s/agent_rw/tmp/hybrid_sample_preds",
        help="Output folder for sidecar JSON + manifest.json",
    )
    parser.add_argument("--n-train", type=int, default=20)
    parser.add_argument("--n-val", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-min", type=float, default=0.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-batch-tfms", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
