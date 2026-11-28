"""fran specs/run_refactor/best_runs.md Acceptance 3 (det3d staging: `kidney` and `kidneys` find one plan dir) and
paths.md Acceptance 6 (det3d staging part: real file lookups through `resolve_path`, each with a second configured key).

Defect: `stage_local_msd.load_spec` / `generate_nndet_plan_silo.sh:30` matched the raw token, so `kidney` and
`kidneys` named different plan dirs; `local_plan_sources.yaml` roots were Sinclair literals read straight into `Path`
(`stage_local_msd.py:64`). Unit rows: the shipped `load_spec` / `resolve_source` on the shipped sources YAML, each
config in its own interpreter (`BEST_RUNS`, `COMMON_PATHS` load at import).

MUTATION: `resolve_source` reads `Path(src["root"])` again -> test_roots_resolve_per_config reds (no source found).
CONTROL: an unknown organ token still raises (test_one_plan_dir_per_organ).
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml

NNDET = Path(__file__).resolve().parents[3] / "det3d" / "run" / "nndet"
BEST_RUNS = {"liver": {"mnemonics": ["lits"], "runs": []}, "kidney": {"mnemonics": ["kidneys", "kits23"], "runs": []},
             "pancreas": {"runs": []}, "colon": {"runs": []}, "yolo": {}, "pretrained": {}}


def make_conf(root: Path, datasets: dict) -> Path:
    conf = root / "conf"
    conf.mkdir(parents=True)
    store = root / "store"
    config = {"cold_storage_folder": str(store), "checkpoints_parent_folder": str(store / "checkpoints"),
              "cache_folder": str(root / "cache"), "rapid_access_folder": str(root / "rapid"), "training_num_workers": 2}
    (conf / "config.yaml").write_text(yaml.safe_dump(config))
    (conf / "datasets.yaml").write_text(yaml.safe_dump({"datasets": datasets}))
    (conf / "best_runs.yaml").write_text(yaml.safe_dump(BEST_RUNS))
    return conf


def pairs(root: Path, n=2) -> Path:
    for sub in ("images", "lms"):
        (root / sub).mkdir(parents=True)
        for i in range(n):
            (root / sub / f"kits23_{i:05d}.nii.gz").write_bytes(b"")
    return root


def run(conf: Path, code: str) -> dict:
    env = dict(os.environ, FRAN_CONF=str(conf), CUDA_VISIBLE_DEVICES="")
    env["PYTHONPATH"] = os.pathsep.join([str(NNDET), env.get("PYTHONPATH", "")])
    prelude = "import json\nfrom pathlib import Path\nfrom stage_local_msd import DEFAULT_SOURCES, load_spec, resolve_source\n"
    result = subprocess.run([sys.executable, "-c", prelude + textwrap.dedent(code) + "\nprint('JSON=' + json.dumps(out))"],
                            env=env, cwd=conf, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads([ln for ln in result.stdout.splitlines() if ln.startswith("JSON=")][-1][5:])


def test_one_plan_dir_per_organ(tmp_path):
    """best_runs.md 3: every mnemonic of the kidney organ selects the one sources entry (its key names the plan dir)."""
    conf = make_conf(tmp_path, {})
    out = run(conf, """
keys = {t: load_spec(t, DEFAULT_SOURCES)[0] for t in ("kidney", "kidneys", "KITS23", "liver", "lits")}
try:
    load_spec("foo", DEFAULT_SOURCES)
    unknown = None
except ValueError as e:
    unknown = str(e)
out = dict(keys=keys, unknown=unknown)
""")
    assert out["keys"] == {"kidney": "kidneys", "kidneys": "kidneys", "KITS23": "kidneys", "liver": "liver", "lits": "liver"}
    assert out["unknown"].startswith("Unknown mnemonic foo")


def test_roots_resolve_per_config(tmp_path):
    """paths.md 6 (det3d staging): `$cold_storage_folder/...` roots resolve under each config's own root; a `DS.<name>`
    root resolves through that config's datasets.yaml and wins when it has pairs."""
    seen = {}
    for name in ("a", "b"):
        root = tmp_path / name
        pairs(root / "store" / "datasets" / "raw_data" / "kits23")
        conf = make_conf(root, {"kits23": {"ds": "kits23", "folder": str(root / "absent"), "alias": None, "organ": "kidney"}})
        seen[name] = run(conf, """
root, images, labels, found = resolve_source(load_spec("kidney", DEFAULT_SOURCES)[1])
out = dict(root=str(root), cases=len(found))
""")
        assert seen[name] == {"root": str(root / "store" / "datasets" / "raw_data" / "kits23"), "cases": 2}
    root = tmp_path / "c"
    ds_root = pairs(root / "kits23_ds", n=3)
    pairs(root / "store" / "datasets" / "raw_data" / "kits23")
    conf = make_conf(root, {"kits23": {"ds": "kits23", "folder": str(ds_root), "alias": None, "organ": "kidney"}})
    out = run(conf, """
root, images, labels, found = resolve_source(load_spec("kidneys", DEFAULT_SOURCES)[1])
out = dict(root=str(root), cases=len(found))
""")
    assert out == {"root": str(ds_root), "cases": 3}
