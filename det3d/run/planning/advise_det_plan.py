#!/usr/bin/env python3
"""Compare manual detection plan with PlanAdvisorDet suggestions."""
import argparse

from det3d.configs.parser import ConfigMakerDet
from fran.managers import Project


def main():
    parser = argparse.ArgumentParser(description="Advise detection plan columns for Excel paste")
    parser.add_argument("--project", required=True)
    parser.add_argument("--plan-id", type=int, default=1)
    args = parser.parse_args()

    project = Project(args.project)
    cm = ConfigMakerDet(project)
    cm.setup(args.plan_id)
    table = cm.compare_plan_with_advisor(args.plan_id)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
