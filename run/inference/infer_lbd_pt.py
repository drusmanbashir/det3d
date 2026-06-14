#!/usr/bin/env python3
"""Run Luna16-hybrid RetinaNet on label-bounded torch .pt volumes; write pred PNG overlays."""
import argparse
from pathlib import Path

import torch

from det3d.inference.hybrid_lbd import (
    build_hybrid_detector,
    collect_lbd_pt_paths,
    infer_lbd_volume,
    load_lbd_pt,
    load_plan_from_project,
    load_plan_json,
    save_lbd_pred_png,
)


def resolve_plan(args):
    if args.plan_json is not None:
        return load_plan_json(args.plan_json)
    if args.project is not None and args.plan_id is not None:
        return load_plan_from_project(args.project, args.plan_id)
    raise ValueError("Pass --plan-json or both --project and --plan-id")


def resolve_out_png(out_dir, pt_path):
    out_dir = Path(out_dir)
    return out_dir / f"{pt_path.stem}_pred.png"


def main(args):
    plan = resolve_plan(args)
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    detector = build_hybrid_detector(plan, args.model, device)
    pt_paths = collect_lbd_pt_paths(input_path=args.input, folder=args.folder)
    out_dir = Path(args.out_dir) if args.out_dir is not None else None

    for pt_path in pt_paths:
        img = load_lbd_pt(pt_path)
        pred = infer_lbd_volume(detector, img, plan, device)
        if out_dir is None:
            out_png = pt_path.with_name(f"{pt_path.stem}_pred.png")
        else:
            out_png = resolve_out_png(out_dir, pt_path)
        saved = save_lbd_pred_png(
            img,
            pred,
            detector,
            out_png,
            score_min=args.score_min,
        )
        n = int(pred[detector.target_box_key].shape[0])
        print(f"{pt_path.name}\tboxes={n}\t{saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid Luna16 RetinaNet inference on LBD torch .pt image(s); saves PNG overlays."
    )
    parser.add_argument("--model", required=True, help="TorchScript network checkpoint (detector.pt).")
    plan_src = parser.add_mutually_exclusive_group(required=True)
    plan_src.add_argument("--plan-json", help="Plan JSON (e.g. detection/config/config_train_luna16_16g.json).")
    plan_src.add_argument("--project", help="Project title for ConfigMakerDet plan lookup.")
    parser.add_argument("--plan-id", type=int, help="Plan id with --project.")
    input_src = parser.add_mutually_exclusive_group(required=True)
    input_src.add_argument("--input", help="Single LBD torch image .pt")
    input_src.add_argument("--folder", help="Folder of LBD torch image .pt files")
    parser.add_argument("--out-dir", default=None, help="PNG output dir; default next to each input .pt")
    parser.add_argument("--device", default=None, help="cuda or cpu; default cuda if available")
    parser.add_argument("--score-min", type=float, default=0.0, help="Min score for boxes drawn on PNG")
    args = parser.parse_args()
    if args.project is not None and args.plan_id is None:
        parser.error("--plan-id required with --project")
    main(args)
