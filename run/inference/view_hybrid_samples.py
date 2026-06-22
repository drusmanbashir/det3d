#!/usr/bin/env python3
"""Open hybrid sample sidecars in ImageBBoxViewer."""
import argparse
import json
from pathlib import Path

from det3d.inference.hybrid_samples import list_viewer_sidecars, view_hybrid_sidecar


def load_manifest(out_dir):
    manifest_fn = Path(out_dir) / "manifest.json"
    return json.loads(manifest_fn.read_text())


def sidecar_at_index(out_dir, index):
    manifest = load_manifest(out_dir)
    if index < 0 or index >= len(manifest):
        raise IndexError(f"index {index} out of range for {len(manifest)} samples in {out_dir}")
    return Path(manifest[index]["sidecar"])


def main(args):
    out_dir = Path(args.out_dir)

    if args.list:
        manifest = load_manifest(out_dir)
        for i, row in enumerate(manifest):
            print(
                f"{i}\t{row['split']}\t{row['case_id']}\t"
                f"gt={row['n_gt']}\tpred={row['n_pred']}\t{Path(row['sidecar']).name}"
            )
        return

    if args.index is None:
        raise ValueError("Pass --index N or --list")

    if args.sidecar is not None:
        sidecar_fn = Path(args.sidecar)
    else:
        sidecar_fn = sidecar_at_index(out_dir, args.index)

    print(f"viewing {sidecar_fn}")
    view_hybrid_sidecar(
        sidecar_fn,
        show=args.show,
        orientation=args.orientation,
        score_min=args.score_min,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ImageBBoxViewer for hybrid sample sidecars (gt + pred boxes)."
    )
    parser.add_argument(
        "--out-dir",
        default="/s/agent_rw/tmp/hybrid_sample_preds",
        help="Folder written by infer_hybrid_samples.py",
    )
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--sidecar", default=None, help="Direct path to one sidecar JSON")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--show",
        default="both",
        choices=["gt", "pred", "both"],
        help="gt=green-ish first boxes; both=gt then pred (auto colors)",
    )
    parser.add_argument(
        "--orientation",
        default="axial",
        choices=["axial", "sagittal", "coronal"],
    )
    parser.add_argument("--score-min", type=float, default=0.0)
    args = parser.parse_args()
    main(args)
# %%
