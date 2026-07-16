#!/usr/bin/env python3
"""Shim: delegates to fran/run/training/train.py with --pipeline det."""
import sys

from fran.run.training.train import build_arg_parser, main


if __name__ == "__main__":
    parser = build_arg_parser()
    argv = sys.argv[1:]
    if not any(a in ("--pipeline", "-pipeline") for a in argv):
        argv = ["--pipeline", "det"] + argv
    args = parser.parse_known_args(argv)[0]
    main(args)
