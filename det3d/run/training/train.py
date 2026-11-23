#!/usr/bin/env python3
"""Shim: delegates to fran/run/training/train.py with --pipeline det."""
import sys

from fran.run.training.train import build_arg_parser, main


if __name__ == "__main__":
    parser = build_arg_parser()
    argv = sys.argv[1:]
    pipelines = []
    for index, arg in enumerate(argv):
        if arg in ("--pipeline", "-pipeline"):
            if index + 1 == len(argv):
                parser.error(f"{arg} requires a value")
            pipelines.append(argv[index + 1])
        elif arg.startswith("--pipeline="):
            pipelines.append(arg.split("=", 1)[1])
    if any(value != "det" for value in pipelines):
        parser.error("det3d training requires --pipeline det")
    if len(set(pipelines)) > 1:
        parser.error("conflicting --pipeline selectors")
    if not pipelines:
        argv = ["--pipeline", "det"] + argv
    args = parser.parse_args(argv)
    main(args)
