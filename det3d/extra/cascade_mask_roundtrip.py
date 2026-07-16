"""Cascade LM mask roundtrip: lidc_0001 real LM (2 lesions) through patch+cascade post.

Plan entry point. Generic suite: det3d/extra/infer_debug/
"""

from pathlib import Path

from det3d.extra.infer_debug.fixtures import load_fixture
from det3d.extra.infer_debug.streams.cascade_lm_roundtrip import run_cascade_lm_roundtrip

CASE_ID = "lidc_0001"
RUN_P = "LIDCA-QUARK"
OUT_DIR = Path("/s/agent_rw/tmp/cascade_lm_roundtrip")


def run_roundtrip():
    case = load_fixture(CASE_ID)
    if case.n_lesions != 2:
        raise ValueError(f"{CASE_ID} LM: expected 2 lesions, got {case.n_lesions}")
    print(f"preflight: {case.n_lesions} lesions, ignore_labels={case.ignore_labels}")
    print("bounding_box slices", case.bounding_box[1:])

    result = run_cascade_lm_roundtrip(
        CASE_ID,
        run_p=RUN_P,
        out_dir=OUT_DIR,
        assert_plan=True,
        write_nifti=True,
        flat_output=True,
    )
    print(f"artifacts: {OUT_DIR}")
    print("PASS", result["out_dir"])


if __name__ == "__main__":
    run_roundtrip()
