#!/usr/bin/env python3
import argparse
import json

from det3d.preprocessing.run_build import build_from_plan


def main():
    parser = argparse.ArgumentParser(description="Build MONAI detection JSON from lesion_stats.csv")
    parser.add_argument("--project", required=True, dest="project_title")
    parser.add_argument("--plan", required=True, type=int)
    args = parser.parse_args()
    summary, _ = build_from_plan(args.project_title, args.plan)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
