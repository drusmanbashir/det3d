"""fran specs/run_refactor/inference_run.md "Entry point" (`output` reaches every inferer; intermediates under
`output/_stages/<stage>/`; completion against the expected final file; `run` returns the final path per input, in
input order; all-complete inputs return their paths without predicting) for the det3d cascades.

Defect: DetBBoxCascadeInferer / DetSegBBoxCascadeInferer took no `output`; their final folder was the removed
`self.predictions_folder / run_p` (`det3d/inference/cascade.py:78`) and `run` returned nothing usable. Unit rows: the
real cascade `run` on two real NIfTI inputs (given out of sorted order) through the real localiser bbox, crop, decollate
and postprocess chain including the real `SaveDetOutputd`; stubs sit at the model boundary only: the localiser's and
the patch inferer's `predict_postprocess` (stub predictions; each also writes one intermediate file where its own
inferer says its output goes). The project stub's `predictions_folder` lies outside `output`.

MUTATION: the patch inferer built with `output=self.output` instead of its stage folder -> both rows red on the
`written` check (`out/_stages/RUN-D/...` missing). MUTATION: drop `DetBBoxCascadeInferer.maybe_filter_images` (fran's
filter checks a `.nii.gz` the boxes-only cascade never writes) -> test_boxes_only_* reds on `recalls`.
CONTROL: test_organ_keep_det rows (the same chain, no save) stay green.
"""
from test_organ_keep_det import run_case

SCENE = """
import SimpleITK as sitk
from det3d.inference.post import PackRetinaUNetPredsd

ROOT = Path(COMMON_PATHS["cold_storage_folder"]).parent
OUT = ROOT / "out"
IMAGES = ROOT / "images"
IMAGES.mkdir()
for case in ("lidc_001", "lidc_002"):
    sitk.WriteImage(sitk.GetImageFromArray(np.random.rand(40, 40, 40).astype(np.float32)), str(IMAGES / f"{case}.nii.gz"))
INPUTS = [IMAGES / "lidc_002.nii.gz", IMAGES / "lidc_001.nii.gz"]


def localiser_predict(self, data):
    preds = []
    for dat in data:
        img = dat["image"]
        mask = torch.zeros(1, *img.shape[-3:], dtype=torch.uint8)
        mask[0, 10:30, 10:30, 10:30] = 1
        preds.append({"pred": MetaTensor(mask, meta=dict(img.meta))})
        self.final_path(img.meta["filename_or_obj"]).write_text("localiser")
    return preds


CALLS = []


def patch_predict(self, data, collate_fn=None):
    CALLS.append(len(data))
    batches = []
    for dat in data:
        crop = tuple(dat["image"].shape[-3:])
        seg = torch.zeros(1, 3, *crop)
        seg[:, 0] = 1
        seg[:, 0, 2:8, 2:8, 2:8], seg[:, 2, 2:8, 2:8, 2:8] = 0, 1
        batch = PackRetinaUNetPredsd()({"image": image(crop), "stitched_seg": seg,
                                        "merged_boxes": BOXES.clone(), "merged_labels": torch.zeros(1),
                                        "merged_scores": torch.ones(1)})
        batch["pred_box"], batch["pred_label"], batch["pred_score"] = BOXES.clone(), torch.zeros(1), torch.ones(1)
        batches.append(batch)
        self.final_path(dat["image"].meta["filename_or_obj"]).write_text("patch")
    return batches


B.BaseInferer.predict_postprocess = patch_predict
write_run("RUN-N", plan("pn", "nodes", mode="source"))
write_run("RUN-D", dict(plan("pd", "lidc", remapping_source=NODULES_TO_2), fg_labels=[1, 2]), arch=ARCH)
En = CLS("RUN-N", "RUN-D", [1], OUT, devices=[0], project_title="tproj", save_localiser=True)
En.W.predict_postprocess = localiser_predict.__get__(En.W)
before = {p for p in ROOT.rglob("*") if p.is_file()}
paths = En.run(INPUTS, chunksize=2)
written = sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and p not in before)
calls = len(CALLS)
again = En.run(INPUTS, chunksize=2)
out = dict(paths=[str(p) for p in paths], exist=[p.exists() for p in paths], written=written, out=str(OUT),
           again=[str(p) for p in again], recalls=len(CALLS) - calls)
"""


def expected_written(final_ext):
    stages = [f"out/_stages/{run}/lidc/{case}.nii.gz" for run in ("RUN-D", "RUN-N") for case in ("lidc_001", "lidc_002")]
    finals = [f"out/lidc/{case}{ext}" for case in ("lidc_001", "lidc_002") for ext in final_ext]
    return sorted(stages + finals)


def test_retinaunet_cascade_writes_under_output(tmp_path):
    """Retina U-Net cascade: every write under `output`, stages under `_stages/<run>`, finals `<proj>/<case>.nii.gz`
    plus sidecar and markups; `run` returns one final NIfTI per input, in input order."""
    out = run_case(tmp_path, "from det3d.inference.cascade import DetSegBBoxCascadeInferer as CLS\nARCH = 'retinaunet'\n" + SCENE)
    assert out["written"] == expected_written((".json", ".mrk.json", ".nii.gz"))
    assert out["paths"] == [f"{out['out']}/lidc/lidc_002.nii.gz", f"{out['out']}/lidc/lidc_001.nii.gz"]
    assert out["exist"] == [True, True]
    assert (out["again"], out["recalls"]) == (out["paths"], 0)


def test_boxes_only_cascade_writes_under_output(tmp_path):
    """Boxes-only RetinaNet cascade: same layout without a seg NIfTI; its final file per input is the sidecar JSON,
    which also marks the input complete."""
    out = run_case(tmp_path, "from det3d.inference.cascade import DetBBoxCascadeInferer as CLS\nARCH = 'retinanet'\n" + SCENE)
    assert out["written"] == expected_written((".json", ".mrk.json"))
    assert out["paths"] == [f"{out['out']}/lidc/lidc_002.json", f"{out['out']}/lidc/lidc_001.json"]
    assert out["exist"] == [True, True]
    assert (out["again"], out["recalls"]) == (out["paths"], 0)
