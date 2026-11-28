"""fran specs/run_refactor/inference_run.md "Run settings precedence" acceptance 4 (NNDetInferer and the RetinaNet window).

Defect: NNDetInferer overwrote the checkpoint's tile filter, overlap, TTA and merge settings with constructor
defaults (0.0 / 0.6 / 100 / 10000 / 0.01, overlap 0.5, 8 views, model_iou 0.1; `patch.py:373-400`), rejected
TTA 0, and its no-TTA merge skipped the score filter (`:172`). Unit rows: the real NNDetInferer constructor and
setup() on synthetic checkpoints carrying the saved settings of LIDC-SOTA-F0-20261123B (`postprocess` sheet,
whole-scan merge plan) and LIDCA-OILS (`plan_train`); the model is a stub whose box filter is nnDetection's own
`BaseRetinaNet.postprocess_detections_single_image`.

MUTATION: restore `tile_nms_thresh=0.6` as the constructor default -> the d=6 box pair survives on LIDC-SOTA and test_lidc_sota_saved_settings reds.
CONTROL: an explicit `tile_nms_thresh=0.5` changes only NMS (test_override_changes_one_field).
"""
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml

FRAN_ROOT = Path(importlib.util.find_spec("fran").submodule_search_locations[0]).parent
SOTA_POST = {"topk_candidates_per_level": 1000, "remove_small_boxes": 0, "score_thresh": 0.02, "nms_thresh": 0.22, "detections_per_img": 100}
OILS_PLAN = {"score_thresh": 0.02, "nms_thresh": 0.22, "detections_per_img": 25, "topk_candidates_per_level": 1000, "remove_small_boxes": 0}
SOTA_MERGE = {"model_score_thresh": 0.0, "model_topk": 1000, "model_detections_per_image": 100, "remove_small_boxes": 2.0,
              "model_iou": 1e-05, "model_nms_fn": "batched_weighted_nms_model", "ensemble_iou": 0.30000000000000004,
              "ensemble_nms_fn": "batched_wbc_ensemble", "ensemble_topk": 1000, "ensemble_score_thresh": 0.0}

PRELUDE = '''
import hashlib, json, torch
from pathlib import Path
from types import SimpleNamespace
import fran.inference.base as B
from fran.utils.common import COMMON_PATHS
import nndet.compat  # noqa: F401
from nndet.core.retina import BaseRetinaNet
from det3d.inference import patch as P

B.Project = lambda project_title: SimpleNamespace(project_title=project_title)
torch.cuda.is_available = lambda: True  # BaseInferer's construction guard; every device below is the CPU
TILE = ("score_thresh", "nms_thresh", "detections_per_img", "topk_candidates", "remove_small_boxes")


class Net:
    """Model-boundary stub: unset filter attributes; the filter itself is nnDetection's."""
    num_foreground_classes = 1
    score_thresh = nms_thresh = detections_per_img = topk_candidates = remove_small_boxes = "unset"
    tile_filter = BaseRetinaNet.postprocess_detections_single_image

    def parameters(self):
        return iter([torch.zeros(1)])


def stub_model(self):
    self.model = SimpleNamespace(net=Net(), detector=SimpleNamespace(calls=[], set_sliding_window_inferer=lambda **kw: self.model.detector.calls.append(kw)),
                                 val_patch_size=[16, 16, 8])
    self.fabric = "stub"


for cls in (P.NNDetInferer, P.DetPatchRetinaNet):
    cls.prepare_model = stub_model
    cls.create_and_set_preprocess_transforms = lambda self: None  # image transforms play no part in the settings
root = Path(COMMON_PATHS["checkpoints_parent_folder"]) / "lidc" / "det3d"


def write_run(run, configs, merge=None):
    path = root / run / "checkpoints" / "last.ckpt"
    path.parent.mkdir(parents=True)
    hp = {"project_title": "lidc", "configs": configs, "lr": None}
    torch.save({"epoch": 60, "state_dict": {}, "hyper_parameters": hp, "datamodule_hyper_parameters": hp}, path)
    if merge is not None:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        record = dict(checkpoint_sha256=sha, mode="whole-scan", parameters=merge)
        (path.parent / f"inference_plan_{sha}_whole-scan.json").write_text(json.dumps(record))
    return path


def configs(plan_extra=None, dataset_params=None, postprocess=None):
    out = {"plan_train": {"mode": "lbd", "spacing": [0.7, 0.7, 1.8], "patch_size": [16, 16, 8], "fg_labels": [1], **(plan_extra or {})},
           "dataset_params": dataset_params or {}, "model_params": {"arch": "retinaunet", "compiled": 0}}
    if postprocess is not None:
        out["postprocess"] = postprocess
    return out


def cube(z, size=10.0):
    return [0.0, 0.0, size, size, z, z + size]


def kept(net, boxes, scores, shape=(4000, 4000, 4000)):
    boxes, scores, _ = net.tile_filter(torch.tensor(boxes), torch.tensor(scores).reshape(-1, 1), shape)
    return sorted(round(float(s), 3) for s in scores)


def effective(inf):
    net = inf.model.net
    merge = inf.inference_params()
    return dict(tile={k: getattr(net, k) for k in TILE}, num_tta=inf.num_tta, overlap=inf.inferer.overlap,
                tiles=inf.inferer.sw_batch_size, merge={k: merge[k] for k in ("model_score_thresh", "model_iou", "model_topk")})


def emit(out):
    print("JSON=" + json.dumps(out))
'''


def run_case(tmp_path, code):
    store = tmp_path / "store"
    conf = tmp_path / "conf"
    conf.mkdir()
    config = {"cold_storage_folder": str(store), "checkpoints_parent_folder": str(store / "checkpoints"),
              "configurations_folder": str(FRAN_ROOT / "configurations"), "cache_folder": str(tmp_path / "cache"),
              "rapid_access_folder": str(tmp_path / "rapid"), "training_num_workers": 2}
    (conf / "config.yaml").write_text(yaml.safe_dump(config))
    (conf / "datasets.yaml").write_text(yaml.safe_dump({"datasets": {}}))
    (conf / "best_runs.yaml").write_text(yaml.safe_dump({"yolo": {}, "pretrained": {}}))
    (conf / "organs_registry.yaml").write_text("{}\n")
    env = dict(os.environ, FRAN_CONF=str(conf), CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-c", PRELUDE + textwrap.dedent(code)], env=env, cwd=conf,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads([ln for ln in result.stdout.splitlines() if ln.startswith("JSON=")][-1][5:])


def test_lidc_sota_saved_settings(tmp_path):
    code = f'''
write_run("SOTA", configs(dataset_params={{"inference_overlap": 0.5, "inference_tta": 8}}, postprocess={SOTA_POST!r}), merge={SOTA_MERGE!r})
inf = P.NNDetInferer("SOTA", devices="cpu", project_title="lidc")
inf.setup()
net = inf.model.net
emit(dict(effective(inf), score=kept(net, [cube(0), cube(100)], [0.019, 0.021]),
          nms_below=kept(net, [cube(0), cube(6.8)], [0.9, 0.8]), nms_above=kept(net, [cube(0), cube(6)], [0.9, 0.8]),
          cap=kept(net, [cube(20 * i) for i in range(120)], [0.5] * 120)))
'''
    out = run_case(tmp_path, code)
    assert out["tile"] == {"score_thresh": 0.02, "nms_thresh": 0.22, "detections_per_img": 100, "topk_candidates": 1000, "remove_small_boxes": 0}
    assert out["merge"] == {"model_score_thresh": 0.0, "model_iou": 1e-05, "model_topk": 1000}
    assert (out["num_tta"], out["overlap"]) == (8, 0.5)
    assert out["score"] == [0.021]
    assert out["nms_below"] == [0.8, 0.9]
    assert out["nms_above"] == [0.9]
    assert len(out["cap"]) == 100


def test_saved_cap_of_1000(tmp_path):
    code = f'''
write_run("CAP", configs(postprocess={{**{SOTA_POST!r}, "detections_per_img": 1000, "topk_candidates_per_level": 2000}}))
inf = P.NNDetInferer("CAP", devices="cpu", project_title="lidc")
inf.setup()
emit(dict(effective(inf), cap=kept(inf.model.net, [cube(20 * i) for i in range(1100)], [0.5] * 1100, shape=(40000, 40000, 40000))))
'''
    out = run_case(tmp_path, code)
    assert (out["tile"]["detections_per_img"], out["tile"]["topk_candidates"]) == (1000, 2000)
    assert len(out["cap"]) == 1000


def test_lidca_oils_plan_train_settings(tmp_path):
    code = f'''
write_run("OILS", configs(plan_extra={OILS_PLAN!r}))
inf = P.NNDetInferer("OILS", devices="cpu", project_title="lidc")
inf.setup()
emit(dict(effective(inf), cap=kept(inf.model.net, [cube(20 * i) for i in range(30)], [0.5] * 30)))
'''
    out = run_case(tmp_path, code)
    assert out["tile"] == {"score_thresh": 0.02, "nms_thresh": 0.22, "detections_per_img": 25, "topk_candidates": 1000, "remove_small_boxes": 0}
    assert len(out["cap"]) == 25
    assert out["merge"] == {"model_score_thresh": 0.0, "model_iou": 0.1, "model_topk": 1000}
    assert out["num_tta"] == 0


def test_override_changes_one_field(tmp_path):
    code = f'''
write_run("SOTA", configs(dataset_params={{"inference_overlap": 0.5, "inference_tta": 8}}, postprocess={SOTA_POST!r}), merge={SOTA_MERGE!r})
base = P.NNDetInferer("SOTA", devices="cpu", project_title="lidc")
base.setup()
inf = P.NNDetInferer("SOTA", devices="cpu", project_title="lidc", tile_nms_thresh=0.5)
inf.setup()
emit(dict(base=effective(base), given=effective(inf), nms_above=kept(inf.model.net, [cube(0), cube(6)], [0.9, 0.8])))
'''
    out = run_case(tmp_path, code)
    expected = dict(out["base"], tile=dict(out["base"]["tile"], nms_thresh=0.5))
    assert out["given"] == expected
    assert out["nms_above"] == [0.8, 0.9]


def test_tta_zero_overlap_zero_and_no_tta_score_filter(tmp_path):
    code = f'''
write_run("SOTA", configs(dataset_params={{"inference_overlap": 0.5, "inference_tta": 8}}, postprocess={SOTA_POST!r}), merge={SOTA_MERGE!r})
inf = P.NNDetInferer("SOTA", devices="cpu", project_title="lidc", mirror_views=0, patch_overlap=0, model_score_thresh=0.3)
inf.setup()
boxes = torch.tensor([cube(0), cube(100)])
_, scores, _ = inf.filter_boxes(boxes, torch.tensor([0.2, 0.4]), torch.zeros(2), torch.ones(2), (400, 400, 400))
emit(dict(effective(inf), merged=[round(float(s), 3) for s in scores]))
'''
    out = run_case(tmp_path, code)
    assert (out["num_tta"], out["overlap"]) == (0, 0)
    assert out["merge"]["model_score_thresh"] == 0.3
    assert out["merged"] == [0.4]


def test_retinanet_window_uses_inferer_overlap_and_batch(tmp_path):
    code = '''
write_run("RNET", configs(plan_extra={"infer_sw_batch_size": 3}, dataset_params={"inference_overlap": 0.4}))
inf = P.DetPatchRetinaNet("RNET", devices="cpu", project_title="lidc")
inf.setup()
call = inf.model.detector.calls[-1]
emit(dict(overlap=call["overlap"], tiles=call["sw_batch_size"], roi=list(call["roi_size"]), inferer=[inf.inferer.overlap, inf.inferer.sw_batch_size]))
'''
    out = run_case(tmp_path, code)
    assert [out["overlap"], out["tiles"]] == out["inferer"]
    assert (out["overlap"], out["roi"]) == (0.4, [16, 16, 8])
