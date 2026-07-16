#!/usr/bin/env python
"""CLI for det3d inference debug suite."""

import argparse
from pathlib import Path

from det3d.extra.infer_debug.fixtures import FIXTURES
from det3d.extra.infer_debug.streams.cascade_lm_roundtrip import run_cascade_lm_roundtrip


def main():
    p = argparse.ArgumentParser(description="det3d inference alignment debug suite")
    p.add_argument(
        "--fixture",
        choices=sorted(FIXTURES),
        default="pseudo_cuboid",
        help="fixture case (pseudo_cuboid | lidc_0001)",
    )
    p.add_argument("--run-p", default="LIDCA-QUARK", help="plan run name for spacing/norm")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/s/agent_rw/tmp/infer_debug"),
        help="output root for stage reports",
    )
    p.add_argument(
        "--stream",
        choices=["cascade_lm_roundtrip"],
        default="cascade_lm_roundtrip",
    )
    p.add_argument(
        "--strict-full-fg",
        action="store_true",
        help="fail if full-volume fg mask differs (off by default for lidc resampling)",
    )
    args = p.parse_args()

    if args.stream == "cascade_lm_roundtrip":
        result = run_cascade_lm_roundtrip(
            args.fixture,
            run_p=args.run_p,
            out_dir=args.out,
            strict_full_fg=args.strict_full_fg,
        )
    else:
        raise SystemExit(f"unknown stream {args.stream}")

    status = "PASS" if result["pass"] else "CHECK"
    print(status, result["out_dir"])
    print(
        f"fg_equal={result['fg_equal']} fg_diff={result['fg_diff_voxels']} "
        f"lesion_masks_exact={result.get('lesion_masks_exact')}"
    )
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
