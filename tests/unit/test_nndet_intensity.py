import pickle
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from monai.transforms import ScaleIntensityRanged

from det3d.configs import parser
from fran.utils.folder_names import FolderNames

MODEL_PARAMS = {"spatial_dims": 3, "in_channels": 1, "encoder_start_channels": 32, "encoder_max_channels": 320}
from fran.preprocessing.fixed_spacing import NiftiResamplerLocal


def test_plan_histogram_uses_selected_cases_and_composed_foreground(tmp_path):  #AI
    """Match NumPy over kept voxels after both remaps and CC exclusion."""
    from fran.preprocessing.plan_intensity import plan_intensity_properties

    h5_path = tmp_path / "fg_voxels.h5"
    with h5py.File(h5_path, "w") as archive:
        a = archive.create_dataset("a", data=[[25, 0, 1, 0], [0, 0, 1, 2], [0, 9, 0, 0]])
        a.attrs["hist_labels"] = [1, 2, 3]
        a.attrs["hu_min"] = -3
        b = archive.create_dataset("b", data=[[0, 1, 0, 2], [0, 8, 0, 0]])
        b.attrs["hist_labels"] = [2, 3]
        b.attrs["hu_min"] = 1
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE datasources (ds TEXT, case_id TEXT, test INTEGER)")
    db.executemany("INSERT INTO datasources VALUES (?, ?, ?)",
                   [("lidc", "a", 0), ("lidc", "b", 0), ("lidc", "heldout", 1),
                    ("other", "different_source", 0)])
    project = SimpleNamespace(
        sql_query=lambda query: db.execute(query).fetchall(),
        global_properties={"datasources": [{"ds": "lidc", "h5_fname": str(h5_path)}]},
    )
    plan = {
        "plan_name": 9, "datasources": "lidc", "remapping_source": {2: 4},
        "remapping_lbd_rbd": {4: 5, 3: 0}, "ignore_labels": [1],
    }
    try:
        stats, labels = plan_intensity_properties(project, plan)
        voxels = np.array([-1, 0, 0, 2, 4, 4])
        assert labels == [5]
        assert stats == pytest.approx({
            "mean": voxels.mean(), "sd": voxels.std(),
            "percentile_00_5": -1.0,  # cases normalised to equal mass: 0.5% of 6 voxels interpolates to the minimum
            "percentile_99_5": np.percentile(voxels, 99.5),
        })
        with pytest.raises(ValueError, match="empty foreground"):
            plan_intensity_properties(project, dict(plan, ignore_labels=[1, 5]))
    finally:
        db.close()


def test_zscore_clips_before_standardizing():  #AI
    """Apply the computed clip, mean, and SD in nnDetection order."""
    from fran.transforms.intensitytransforms import NormaliseClipd

    plan = {"intensity_properties": {
        "mean": 1.1, "sd": 1.7, "percentile_00_5": -3.4, "percentile_99_5": 4.8,
    }}
    image = torch.tensor([-15, -3, 1, 3, 18], dtype=torch.float16)
    transform = NormaliseClipd.from_plan(
        ["image"], plan, detection_params={"intensity_norm": "zscore_fg"},
    )
    expected = (image.float().clamp(-3.4, 4.8) - 1.1) / 1.7
    assert torch.equal(transform({"image": image})["image"], expected)


def test_scale01_keeps_main_arithmetic_for_identical_bounds():  #AI
    """Scale with the same MONAI transform as existing detectors."""
    from fran.transforms.intensitytransforms import NormaliseClipd

    bounds = [-973.0, 429.0]
    image = torch.linspace(-1024.0, 450.0, 51)
    plan = {"intensity_properties": {
        "mean": -525.0, "sd": 197.0,
        "percentile_00_5": bounds[0], "percentile_99_5": bounds[1],
    }}
    new = NormaliseClipd.from_plan(["image"], plan, detection_params={"intensity_norm": "scale01"})
    old = ScaleIntensityRanged(keys=["image"], a_min=bounds[0], a_max=bounds[1],
                               b_min=0.0, b_max=1.0, clip=True)
    assert torch.equal(new({"image": image})["image"], old({"image": image})["image"])


MODEL = {
    "conv_kernels": [[3, 3, 3]] * 3,
    "strides": [[2, 2, 1], [2, 2, 2]],
    "decoder_levels": (1, 2),
    "anchors": {"width": [(4,), (8,)], "height": [(4,), (8,)], "depth": [(2,), (4,)], "stride": 1},
}


def _det_plan_maker(tmp_path, row, monkeypatch):  #AI
    """Supply one plans row, nested parameter sheets and a private plan-file location to setup.
    The row names one datasource of an undeclared organ (no labels.yaml), so the organ_label setup reads gives None."""
    monkeypatch.setattr("fran.configs.plan_keys.DS", {"tds": SimpleNamespace(organ="misc", folder=tmp_path, name="tds")})
    maker = object.__new__(parser.ConfigMakerDet)
    maker.project = SimpleNamespace(project_folder=tmp_path, project_title="test")
    full_row = {"batch_size": "auto", "model": "auto", "val_patch_size": "auto", "encoder": None, "pretrained": None,
                "datasources": "tds", "remapping_source": None, "remapping_lbd_rbd": None, "remapping_train": None,
                "remapping_imported": None, "merge_imported_labels": False} | row
    maker.plans = pd.DataFrame([full_row], index=[1])
    maker.configs = {"dataset_params": {}, "model_params": dict(MODEL_PARAMS)}
    maker._sheet_configs = {sheet: dict(values) for sheet, values in maker.configs.items()}
    return maker


def _write_resolved(maker, resolved, fits=None):  #AI
    """Save a resolved plan (and its per-GPU-key fits) made from the maker's current sheet values."""
    maker.setup(1, load_computed=False)
    pkl = FolderNames(maker.project, {"plan_name": 1}, maker.PIPELINE).plan_file
    pkl.parent.mkdir(exist_ok=True)
    fit_row = maker.gpu_fit_row(maker.read_plan(1))
    for fit in (fits or {}).values():
        fit["fit"]["row"] = fit_row
    pkl.write_bytes(pickle.dumps({"plan_train": resolved, "fits": fits or {}}))
    return pkl


def test_detection_config_loads_without_plan_file(tmp_path, monkeypatch):  #AI
    """Load the sheet plan while its auto values are still unresolved."""
    maker = _det_plan_maker(tmp_path, {
        "mode": "lbd", "spacing": "auto", "patch_dim0": 192, "patch_dim1": 96, "batch_size": 4,
    }, monkeypatch)
    maker.setup(1)
    plan = maker.configs["plan_train"]
    assert plan["spacing"] == "auto"
    assert plan["patch_size"] == [192, 192, 96]
    assert plan["batch_size"] == 4
    assert plan["model"] == "auto"


def test_detection_resolved_plan_keeps_fixed_workbook_patches(tmp_path, monkeypatch):  #AI
    """A saved fit cannot replace fixed training or validation patch dimensions."""
    maker = _det_plan_maker(tmp_path, {"mode": "lbd", "spacing": [1, 1, 2], "patch_dim0": 192, "patch_dim1": 96}, monkeypatch)
    intensity = {"mean": 1, "sd": 2, "percentile_00_5": -3, "percentile_99_5": 5}
    gpu_fit = {"patch_size": [160, 160, 96], "val_patch_size": [160, 160, 96], "batch_size": 4, "model": MODEL}
    _write_resolved(maker, {
        "spacing": [1, 1, 2], "fg_labels": [2], "intensity_properties": intensity, "optimizer": "unused",
    }, {"m32": {"plan_train": gpu_fit, "fit": {}}, "m24": {"plan_train": dict(gpu_fit, batch_size=2), "fit": {}}})
    maker.setup(1)
    assert maker.configs["plan_train"]["patch_size"] == [192, 192, 96]  # no GPU key: GPU-independent values only
    assert maker.configs["plan_train"]["fg_labels"] == [2]
    maker.setup(1, gpu_key="m24")
    assert maker.configs["plan_train"]["batch_size"] == 2
    maker.setup(1, gpu_key="m32")
    assert maker.configs["plan_train"] == maker.configs["plan_valid"] == maker.configs["plan_test"]
    assert maker.configs["plan_train"] is not maker.configs["plan_valid"]
    for split in ("train", "valid", "test"):
        plan = maker.configs[f"plan_{split}"]
        assert plan["patch_size"] == [192, 192, 96]
        assert plan["val_patch_size"] == [192, 192, 96]
        assert (plan["patch_dim0"], plan["patch_dim1"]) == (192, 96)
        assert plan["batch_size"] == 4
        assert plan["model"] == MODEL
        assert plan["fg_labels"] == [2]
        assert plan["intensity_properties"] == intensity
        assert "optimizer" not in plan
    assert maker.configs["model_params"] == MODEL_PARAMS


def test_detection_reuses_saved_values_by_plan_id(tmp_path, monkeypatch):
    row = {"mode": "lbd", "spacing": "auto", "patch_dim0": 192, "patch_dim1": 96}
    resolved = {"spacing": [1, 1, 2], "val_patch_size": [192, 192, 96]}
    path = _write_resolved(_det_plan_maker(tmp_path, row, monkeypatch), resolved)
    path.write_bytes(pickle.dumps({"requested": {"legacy": True}, "plan_train": resolved}))
    edited = _det_plan_maker(tmp_path, row | {"spacing": [2, 2, 2], "val_patch_size": [64, 64, 64]}, monkeypatch)
    edited.setup(1)
    assert edited.configs["plan_train"]["spacing"] == resolved["spacing"]
    assert edited.configs["plan_train"]["val_patch_size"] == [64, 64, 64]


def test_fixed_spacing_worker_accepts_missing_legacy_stats(tmp_path, monkeypatch):  #AI
    """Skip legacy intensity keys when resampling raw HU values."""
    monkeypatch.setattr("fran.preprocessing.rayworker_base.RayWorkerBase.__init__", lambda *a, **kw: None)
    worker = NiftiResamplerLocal(
        SimpleNamespace(global_properties={}), {"spacing": [1, 1, 1]}, tmp_path, tmp_path,
    )
    assert not worker.clip_center


def test_plan_rejects_datasources_sharing_one_preprocessing_alias(monkeypatch):  #AI
    """Refuse to pool two remaps that would share one preprocessing transform."""
    from fran.preprocessing import plan_intensity

    specs = {
        "first": SimpleNamespace(ds="first", name="shared"),
        "second": SimpleNamespace(ds="second", name="shared"),
    }
    monkeypatch.setattr(plan_intensity, "DS", specs)
    with pytest.raises(ValueError, match="ambiguous datasources"):
        plan_intensity.plan_cases(object(), {"plan_name": 1, "datasources": "first,second"})


@pytest.mark.parametrize("workbook_patch", ["auto", [24, 24, 8]])
def test_detection_planning_stores_intensity_and_labels_and_leaves_the_fits(tmp_path, monkeypatch, workbook_patch):  #AI
    """Spacing is stored with its row value; plan_detection adds intensity, box labels and data key, never a fit."""
    from fran.run.preproc import analyze_resample
    from fran.run.preproc.analyze_resample import configure_plan

    plan = {"mode": "lbd", "plan_name": 1, "spacing": [1, 1, 2], "prezoom_scale": 1.0, "patch_size": workbook_patch,
            "batch_size": 2 if workbook_patch != "auto" else "auto", "model": MODEL if workbook_patch != "auto" else "auto",
            "val_patch_size": "auto"}
    project = SimpleNamespace(project_folder=tmp_path, project_title="t")
    maker = SimpleNamespace(
        PIPELINE=parser.ConfigMakerDet.PIPELINE,
        setup=lambda plan_id, load_computed=True, gpu_key=None, plan_overrides=None: None,
        configs={f"plan_{key}": dict(plan) for key in ("train", "valid", "test")} | {"model_params": dict(MODEL_PARAMS)},
    )
    configs = configure_plan(project, maker, 1, False, False, pipeline="det")
    pkl = tmp_path / "plans" / "det3d_plan1.pkl"
    spacing_only = {"plan_train": {"spacing": [1, 1, 2]}, "spacing_fit": {"row": {"spacing": [1, 1, 2]}, "input_key": None}}
    assert pickle.loads(pkl.read_bytes()) == spacing_only
    fits = {"m32": {"plan_train": {"patch_size": [16, 16, 8]}, "fit": {"input_key": "measured boxes", "row": {}}}}
    pkl.write_bytes(pickle.dumps(dict(spacing_only, fits=fits)))
    intensity = {"mean": 1.0, "sd": 0.5, "percentile_00_5": 0.1, "percentile_99_5": 1.9}
    monkeypatch.setattr(analyze_resample, "plan_intensity_properties", lambda p, s: (intensity, [2]))
    monkeypatch.setattr(analyze_resample, "plan_cases", lambda p, plan: [])
    monkeypatch.setattr(analyze_resample, "selected_lbd_details", lambda folder, case_ids: pd.DataFrame({"case_id": ["a", "b"]}))
    monkeypatch.setattr(analyze_resample, "lbd_input_key", lambda folder, details: "measured boxes")
    monkeypatch.setattr(analyze_resample, "lbd_box_records", lambda folder, row: (None, [2, 2]))
    monkeypatch.setattr(parser.ConfigMakerDet, "__init__", lambda self, project: None)
    monkeypatch.setattr(parser.ConfigMakerDet, "read_plan", lambda self, plan_id: dict(plan))
    manager = object.__new__(analyze_resample.PreprocessingManager)
    manager.project, manager.configs, manager.L = project, configs, SimpleNamespace(output_folder=tmp_path)
    output = manager.plan_detection(configs["plan_train"])
    assert output == pickle.loads(pkl.read_bytes())
    plan_train = {"spacing": [1, 1, 2], "intensity_properties": intensity, "fg_labels": [2]}
    if workbook_patch != "auto":
        plan_train["val_patch_size"] = [24, 24, 8]
    assert output == dict(spacing_only, fits=fits, plan_train=plan_train,
                          fit={"input_key": "measured boxes", "histogram_labels": [2]})
    calls = Mock(side_effect=lambda p, s: (intensity, [2]))
    monkeypatch.setattr(analyze_resample, "plan_intensity_properties", calls)
    assert manager.plan_detection(configs["plan_train"]) == output
    calls.assert_not_called()
    pkl.write_bytes(pickle.dumps(dict(spacing_only, fits=fits)))
    before = pkl.read_bytes()
    monkeypatch.setattr(analyze_resample, "plan_intensity_properties", lambda p, s: (intensity, [3]))
    with pytest.raises(ValueError, match="ambiguous detection foreground"):
        manager.plan_detection(configs["plan_train"])
    assert pkl.read_bytes() == before


def test_detection_base_builds_transforms_without_legacy_intensity_keys():  #AI
    """Build the inherited transform stack used by detection training."""
    from det3d.managers.data.main import DataManagerDet

    manager = object.__new__(DataManagerDet)
    manager.device = "cpu"
    manager.plan = {"patch_size": [16, 16, 16], "samples_per_file": 1, "intensity_properties": {
        "mean": 1.0, "sd": 0.5, "percentile_00_5": 0.0, "percentile_99_5": 2.0,
    }}
    manager.dataset_params = {"intensity_norm": "zscore_fg", "prezoom_scale": 1.0, "fgbg_ratio": 1.0, "augment": "monai"}
    manager.configs = {"affine3d": {"p": 0.0, "rotate_range": 0.0, "scale_range": 0.0}}
    from det3d.transforms.box_tracker import CornerTracker

    manager.configs["model_params"] = {"arch": "retinanet"}
    manager.box_tracker_name = "corner"
    manager.box_tracker = CornerTracker()
    manager.mirror = dict(prob=0.0, value=(0, 1, 2))
    manager.scale = dict(prob=0.0, value=0.1)
    manager.noise = dict(prob=0.0, value=[0.0, 0.1])
    manager.shift = dict(prob=0.0, value=0.1)
    manager.contrast = dict(prob=0.0, value=[0.8, 1.2])
    manager.create_transforms()
    image = torch.tensor([-3.0, 1.0, 10.0])
    output = manager.transforms_dict["N"]({"image": image})["image"]
    assert torch.equal(output, torch.tensor([-2.0, 0.0, 2.0]))
