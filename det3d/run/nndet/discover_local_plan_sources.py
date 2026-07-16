#!/usr/bin/env python3
"""Report which local nnDet plan sources are available on disk."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from stage_local_msd import load_spec, resolve_source

PLACEHOLDER_SIZE = 789_064


def pkl_status(out_pkl: Path) -> str:
    if not out_pkl.is_file():
        return "missing"
    size = out_pkl.stat().st_size
    if size == PLACEHOLDER_SIZE:
        return "placeholder"
    return f"ready ({size} B)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources-yaml",
        default=str(Path(__file__).resolve().parent / "local_plan_sources.yaml"),
    )
    parser.add_argument("--nndet-conf", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    sources_yaml = Path(args.sources_yaml)
    specs = yaml.safe_load(sources_yaml.read_text())

    nndet_conf = args.nndet_conf
    if nndet_conf is None:
        import os

        fran_conf = os.environ.get("FRAN_CONF", "/s/fran_storage/conf")
        nndet_conf = yaml.safe_load(Path(fran_conf, "config.yaml").read_text())["nndet_conf"]

    rows = []
    for mnemonic, spec in specs.items():
        row = {
            "mnemonic": mnemonic,
            "task": spec["task"],
            "layout": spec["layout"],
            "available": False,
            "cases": 0,
            "source_root": None,
            "pkl": pkl_status(Path(nndet_conf) / "plans" / mnemonic / "D3V001_3d.pkl"),
        }
        try:
            root, _i, _l, pairs = resolve_source(spec)
            row["available"] = True
            row["cases"] = len(pairs)
            row["source_root"] = str(root)
        except FileNotFoundError:
            pass
        rows.append(row)

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    print(f"{'mnemonic':10} {'task':18} {'cases':>6}  {'pkl':16}  source")
    print("-" * 80)
    for row in rows:
        src = row["source_root"] or "-"
        print(
            f"{row['mnemonic']:10} {row['task']:18} {row['cases']:6}  "
            f"{row['pkl']:16}  {src}"
        )


if __name__ == "__main__":
    main()
