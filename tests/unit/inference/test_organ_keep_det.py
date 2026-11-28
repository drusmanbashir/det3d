"""fran specs/run_refactor/organ_label.md Acceptance 11 and 12 (Retina U-Net cascade, boxes-only RetinaNet) and the
Label chain "det seg target" row (`det3d/managers/data/labels.py` re-derives `organ_label` after `fg_labels`).

Defect: the Retina U-Net cascade argmaxed the packed one-channel seg map (`det3d/inference/cascade.py:153`), which
zeroes it, and its K came from a `k_largest` argument instead of the run's plan. Unit rows: the real
DetSegBBoxCascadeInferer / DetBBoxCascadeInferer built from synthetic pinned checkpoints on CPU; the patch output is
the real `PackRetinaUNetPredsd` on stub network seg logits and boxes; the cascade's own assembled chain runs on it.

Scene: det seg target lung -> 1, nodules -> 2 (fg_labels [1, 2]); two lung blobs, a third smaller lung blob, five
detached nodules. Reference (spec): only the third blob is removed, boxes are offset into the full volume.

MUTATION: restore "A" in DetSegBBoxCascadeInferer.keys_postproc -> test_retinaunet_cascade_keeps_seg reds (seg zeroed).
CONTROL: the boxes-only cascade returns its offset boxes with no K (test_boxes_only_cascade_no_k).
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

LUNG = {1: {"name": "lung", "type": "organ"}, **{i: {"name": f"nodule{i}", "type": "lesion"} for i in range(2, 7)}}
BEST_RUNS = {"lung": {"mnemonics": ["lungs", "lidc"], "organ": [{"number": 2}], "runs": ["RUN-D"]},
             "nodes": {"runs": ["RUN-N"]}, "yolo": {}, "pretrained": {}}

PRELUDE = '''
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from monai.data.meta_tensor import MetaTensor
from fran.configs.plan_parse import parse_plan_row
from fran.utils.common import COMMON_PATHS
import fran.inference.base as B
import fran.inference.helpers as H

BLANK = dict(remapping_source=None, remapping_lbd_rbd=None, remapping_whole=None, remapping_imported=None,
             imported_folder=None, merge_imported_labels=None, expand_by=0, fg_indices_exclude=None, spacing="auto")
PREDS = Path(COMMON_PATHS["cold_storage_folder"]) / "predictions"
B.Project = H.Project = lambda project_title: SimpleNamespace(project_title=project_title, predictions_folder=PREDS)
torch.cuda.is_available = lambda: True  # BaseInferer's construction guard; every device below is the CPU
NODULES_TO_2 = "{1:1,2:2,3:2,4:2,5:2,6:2}"


def plan(name, datasources, mode="lbd", **cells):
    row = dict(BLANK, datasources=datasources, mode=mode)
    row.update(cells)
    parsed = parse_plan_row(row)
    parsed["plan_id"] = parsed["plan_name"] = name
    parsed["spacing"], parsed["patch_size"] = [1.0, 1.0, 1.0], [8, 8, 8]
    return parsed


def write_run(run, plan_train, arch="unet"):
    path = Path(COMMON_PATHS["checkpoints_parent_folder"]) / "tproj" / "sub" / run / "checkpoints" / "last.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    configs = {"plan_train": plan_train, "dataset_params": {}, "mnemonic": "lungs",
               "model_params": {"arch": arch, "out_channels": 3, "compiled": 0}}
    hp = {"project_title": "tproj", "configs": configs, "lr": None}
    torch.save({"epoch": 1, "state_dict": {}, "hyper_parameters": hp, "datamodule_hyper_parameters": hp}, path)


CROP, FULL = (30, 30, 30), (40, 40, 40)
BBOX = [slice(0, 1), slice(5, 35), slice(5, 35), slice(5, 35)]
BOXES = torch.tensor([[2.0, 3.0, 4.0, 10.0, 11.0, 12.0]])


def scene():
    lm = torch.zeros(CROP, dtype=torch.uint8)
    lm[2:12, 2:12, 2:12] = 1
    lm[2:12, 18:28, 2:12] = 1
    lm[18:23, 2:7, 2:7] = 1
    for x, y, z in [(18, 18, 18), (24, 18, 18), (18, 24, 18), (24, 24, 24), (18, 18, 26)]:
        lm[x:x + 2, y:y + 2, z:z + 2] = 2
    expected = torch.zeros(FULL, dtype=torch.uint8)
    expected[5:35, 5:35, 5:35] = lm
    expected[23:28, 7:12, 7:12] = 0  # the third lung blob, in full-volume voxels
    return lm, expected


def image(shape):
    meta = {"filename_or_obj": "/x/lidc_001.nii.gz", "spatial_shape": np.array(shape), "original_affine": torch.eye(4)}
    return MetaTensor(torch.rand(1, 1, *shape), affine=torch.eye(4), meta=meta)


def cascade(cls, safe):
    write_run("RUN-N", plan("pn", "nodes", mode="source"))
    En = cls("RUN-N", "RUN-D", localiser_labels=[1], devices=[0], project_title="tproj", safe_mode=safe, save=False)
    En.create_and_set_postprocess_transforms()
    return En


def run_chain(En, batch):
    batch.update(source_image="/x/lidc_001.nii.gz", crop_spatial_shape=CROP, bounding_box=[BBOX])
    full_meta = H.full_meta_from_image(image(FULL)[0])
    cases = En.decollate_patches({En.P.run_name: [batch]}, [BBOX], [full_meta])
    out = En.postprocess(cases)[0]
    return out
'''


def run_case(tmp_path, code):
    store = tmp_path / "store"
    conf = tmp_path / "conf"
    conf.mkdir()
    datasets = {}
    for name, organ, labels in (("lidc", "lung", LUNG), ("nodes", "nodes", {1: {"name": "node", "type": "lesion"}})):
        folder = tmp_path / "datasets" / name
        folder.mkdir(parents=True)
        (folder / "labels.yaml").write_text(yaml.safe_dump({"labels": labels}))
        datasets[name] = {"ds": name, "folder": str(folder), "alias": None, "organ": organ}
    config = {"cold_storage_folder": str(store), "checkpoints_parent_folder": str(store / "checkpoints"),
              "cache_folder": str(tmp_path / "cache"), "rapid_access_folder": str(tmp_path / "rapid"),
              "configurations_folder": str(FRAN_ROOT / "configurations"), "training_num_workers": 2}
    (conf / "config.yaml").write_text(yaml.safe_dump(config))
    (conf / "datasets.yaml").write_text(yaml.safe_dump({"datasets": datasets}))
    (conf / "best_runs.yaml").write_text(yaml.safe_dump(BEST_RUNS))
    (conf / "organs_registry.yaml").write_text("{}\n")
    env = dict(os.environ, FRAN_CONF=str(conf), CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-c", PRELUDE + textwrap.dedent(code) + "\nprint('JSON=' + json.dumps(out))"],
                            env=env, cwd=conf, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads([ln for ln in result.stdout.splitlines() if ln.startswith("JSON=")][-1][5:])


def test_retinaunet_cascade_keeps_seg(tmp_path):
    """Acceptance 11 + 12: Retina U-Net cascade, safe and ordinary: the nonzero packed seg map survives (no argmax),
    K on lung (seg value 1) removes only the third blob, nodules and boxes kept."""
    out = run_case(tmp_path, """
from det3d.inference.cascade import DetSegBBoxCascadeInferer
from det3d.inference.post import PackRetinaUNetPredsd
write_run("RUN-D", dict(plan("pd", "lidc", remapping_source=NODULES_TO_2), fg_labels=[1, 2]), arch="retinaunet")
lm, expected = scene()
out = dict()
for safe in (False, True):
    En = cascade(DetSegBBoxCascadeInferer, safe)
    seg = torch.nn.functional.one_hot(lm.long(), 3).permute(3, 0, 1, 2).float().unsqueeze(0)  # stub network seg logits
    batch = PackRetinaUNetPredsd()({"image": image(CROP), "stitched_seg": seg, "merged_boxes": BOXES.clone(),
                                    "merged_labels": torch.zeros(1), "merged_scores": torch.ones(1)})
    res = run_chain(En, batch)
    pred = res["pred"].squeeze(0)
    out[str(safe)] = dict(keys=En.postprocess_tfms_keys, nonzero=int((pred > 0).sum()),
                          equal=bool(torch.equal(pred.to(torch.uint8), expected)), boxes=res["pred_box"].tolist())
""")
    for safe in ("False", "True"):
        assert out[safe]["nonzero"] > 0, out[safe]
        assert out[safe]["equal"], out[safe]
        assert out[safe]["boxes"] == [[7.0, 8.0, 9.0, 15.0, 16.0, 17.0]]
    assert out["False"]["keys"] == "SqL,MR,Int,W,K,F,R,Off,BoxR"
    assert out["True"]["keys"] == "SqL,MR,W,K,F,R,Off,BoxR"


def test_boxes_only_cascade_no_k(tmp_path):
    """Acceptance 12: boxes-only RetinaNet cascade: no mask, no K although the plan declares lung; boxes still returned."""
    out = run_case(tmp_path, """
from det3d.inference.cascade import DetBBoxCascadeInferer
write_run("RUN-D", dict(plan("pd", "lidc", remapping_source=NODULES_TO_2), fg_labels=[1, 2]), arch="retinanet")
En = cascade(DetBBoxCascadeInferer, False)
batch = {"image": image(CROP), "pred_box": BOXES.clone(), "pred_label": torch.zeros(1), "pred_score": torch.ones(1)}
res = run_chain(En, batch)
out = dict(keys=En.postprocess_tfms_keys, organ=En.organ_keep is not None, boxes=res["pred_box"].tolist(), pred="pred" in res)
""")
    assert out == {"keys": "SqL,Off,BoxR", "organ": True, "boxes": [[7.0, 8.0, 9.0, 15.0, 16.0, 17.0]], "pred": False}


def test_det_labels_rederive_organ_label(tmp_path):
    """Label chain "det seg target": fg_labels set from the data folder re-derive organ_label (lung dropped -> None)."""
    out = run_case(tmp_path, """
import copy
from det3d.managers.data.labels import infer_det_labels_from_data_folder
from fran.configs.plan_keys import checked_organ_label
folder = Path(COMMON_PATHS["cold_storage_folder"]) / "lbd"
folder.mkdir(parents=True)
(folder / "manifest.json").write_text(json.dumps({"labels_all": [0, 2]}))
p = dict(plan("pd", "lidc", remapping_source=NODULES_TO_2), organ_label=1)  # active_plan's value, before fg_labels
configs = {"plan_train": p, "plan_valid": copy.deepcopy(p), "plan_test": copy.deepcopy(p), "model_params": {}}
infer_det_labels_from_data_folder(SimpleNamespace(prepare_data=lambda: None, data_folder=folder), configs)
out = dict(labels=[configs[k]["organ_label"] for k in ("plan_train", "plan_valid", "plan_test")],
           checked=checked_organ_label(configs["plan_train"], "RUN-D"))
""")
    assert out == {"labels": [None, None, None], "checked": None}
