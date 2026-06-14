#!/usr/bin/env python3
import argparse

from det3d.configs.parser import ConfigMakerDet
from det3d.preprocessing.labelbounded import LabelBoundedDetDataGenerator
from det3d.preprocessing.object_bounded import resolve_input_folder
from fran.managers import Project


def main():
    parser = argparse.ArgumentParser(
        description="Label-bounded detection preprocessing to images/masks/bboxes/*.json per case"
    )
    parser.add_argument("--project", required=True, dest="project_title")
    parser.add_argument("--plan", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--case-ids", default="", help="comma-separated case filter for smoke")
    parser.add_argument("--input-folder", default="", help="optional input data folder override")
    args = parser.parse_args()

    project = Project(project_title=args.project_title)
    config_maker = ConfigMakerDet(project)
    config_maker.setup(args.plan)
    plan = config_maker.configs["plan_train"]
    input_folder = resolve_input_folder(
        project,
        plan,
        input_folder=args.input_folder or None,
    )

    generator = LabelBoundedDetDataGenerator(
        project=project,
        plan=plan,
        data_folder=input_folder,
    )
    generator.setup(debug=args.debug)
    if args.case_ids.strip():
        case_ids = {c.strip() for c in args.case_ids.split(",") if c.strip()}
        generator.df = generator.df[generator.df["case_id"].isin(case_ids)].reset_index(
            drop=True
        )
    generator.run(overwrite=args.overwrite, num_processes=args.num_processes)


if __name__ == "__main__":
    main()
