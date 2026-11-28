"""fran specs/run_refactor/best_runs.md "Mnemonics" (a run's organ comes from best_runs membership or its saved
checkpoint mnemonic) and "Localiser resolution" (an unsupported localiser pair raises at start-up, before a model
loads) for the det3d CLIs' shared helpers and cascades.

Defects (audit 2026-11-28): `det3d/run/inference/_infer_common.py:50` read `configs["mnemonic"]`, absent from most
pinned checkpoints (KeyError); the det cascades checked the localiser pair only through fran's `infer()` resolver, so
`infer_cascade.py` / `infer_det.py` built with a TSL localiser loaded checkpoints first. Unit rows: the real
`resolve_localiser_labels` on synthetic pinned checkpoints (the TSL region table is a stub), and the real `DetSegBBoxCascadeInferer` /
`DetBBoxCascadeInferer` constructors with counting stubs at fran's checkpoint loaders.

MUTATION: `_infer_common` back to `canonical_mnemonic(...["configs"]["mnemonic"])` -> test_labels_from_membership reds
(KeyError 'mnemonic').
CONTROL: a FRAN localiser passes the pair check and reaches the first checkpoint load.
"""
from test_organ_keep_det import run_case


def test_labels_from_membership(tmp_path):
    """RUN-D is listed under lung and its checkpoint has no mnemonic: the non-bones branch gives the TSL regions."""
    out = run_case(tmp_path, """
from types import SimpleNamespace
import det3d.run.inference._infer_common as IC
IC.TotalSegmenterLabels = lambda: SimpleNamespace(label_region=[7, 8])  # its table reads the totalseg_meta dataset
write_run("RUN-N", plan("pn", "nodes", mode="source"))
write_run("RUN-D", dict(plan("pd", "lidc"), fg_labels=[1, 2]), arch="retinaunet")
path = Path(COMMON_PATHS["checkpoints_parent_folder"]) / "tproj" / "sub" / "RUN-D" / "checkpoints" / "last.ckpt"
state = torch.load(path, weights_only=False)
del state["datamodule_hyper_parameters"]["configs"]["mnemonic"]  # one hp dict shared with hyper_parameters
torch.save(state, path)
args = SimpleNamespace(run_w="RUN-N", run_p="RUN-D", localiser_labels=None, lung_localiser=False)
out = dict(labels=IC.resolve_localiser_labels(args))
""")
    assert out == {"labels": [7, 8]}


def test_unsupported_pair_raises_before_any_load(tmp_path):
    out = run_case(tmp_path, """
import fran.inference.cascade as C
from det3d.inference.cascade import DetBBoxCascadeInferer, DetSegBBoxCascadeInferer
LOADS = []


def stop(*args, **kwargs):
    LOADS.append(args)
    raise RuntimeError("checkpoint touched")


C.pinned_checkpoint = C.load_params = stop


def attempt(cls, run_w):
    before = len(LOADS)
    try:
        cls(run_w, "RUN-D", [1], PREDS.parent / "out", devices=[0], project_title="tproj")
    except Exception as e:
        return [type(e).__name__, str(e), len(LOADS) - before]


out = dict(seg=attempt(DetSegBBoxCascadeInferer, "TSL-3MM"), box=attempt(DetBBoxCascadeInferer, "TSL-3MM"),
           fran=attempt(DetSegBBoxCascadeInferer, "RUN-N"))
""")
    assert out == {
        "seg": ["ValueError", "localiser TSL:TSL-3MM unsupported for DetSegBBoxCascadeInferer; supported: ['FRAN']", 0],
        "box": ["ValueError", "localiser TSL:TSL-3MM unsupported for DetBBoxCascadeInferer; supported: ['FRAN']", 0],
        "fran": ["RuntimeError", "checkpoint touched", 1],
    }
