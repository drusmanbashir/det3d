import pickle
import sqlite3
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from monai.transforms import ScaleIntensityRanged

from det3d.configs import parser
from det3d.configs.nndet_planning import nndet_plan_path
from fran.preprocessing.fixed_spacing import NiftiResamplerLocal


def test_plan_histogram_uses_selected_cases_and_composed_foreground(tmp_path):  #AI
    """Match NumPy over kept voxels after both remaps and CC exclusion."""
    from det3d.configs.nndet_intensity import plan_intensity_properties

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
        "remapping_lbd_rbd": {4: 5, 3: 0}, "ignore_labels_cc": 1,
    }
    try:
        stats, labels = plan_intensity_properties(project, plan)
        voxels = np.array([-1, 0, 0, 2, 4, 4])
        assert labels == [5]
        assert stats == pytest.approx({
            "mean": voxels.mean(), "sd": voxels.std(),
            "percentile_00_5": np.percentile(voxels, 0.5),
            "percentile_99_5": np.percentile(voxels, 99.5),
        })
        with pytest.raises(ValueError, match="empty detection foreground"):
            plan_intensity_properties(project, dict(plan, ignore_labels_cc=[1, 5]))
    finally:
        db.close()


def test_zscore_clips_before_standardizing():  #AI
    """Apply the computed clip, mean, and SD in nnDetection order."""
    from det3d.transforms.intensity import det_intensity_transform

    plan = {"intensity_properties": {
        "mean": 1.1, "sd": 1.7, "percentile_00_5": -3.4, "percentile_99_5": 4.8,
    }}
    image = torch.tensor([-15, -3, 1, 3, 18], dtype=torch.float16)
    transform = det_intensity_transform({"intensity_norm": "zscore_fg"}, plan)
    expected = (image.float().clamp(-3.4, 4.8) - 1.1) / 1.7
    assert torch.equal(transform({"image": image})["image"], expected)


def test_scale01_keeps_main_arithmetic_for_identical_bounds():  #AI
    """Scale with the same MONAI transform as existing detectors."""
    from det3d.transforms.intensity import det_intensity_transform

    bounds = [-973.0, 429.0]
    image = torch.linspace(-1024.0, 450.0, 51)
    plan = {"intensity_properties": {
        "mean": -525.0, "sd": 197.0,
        "percentile_00_5": bounds[0], "percentile_99_5": bounds[1],
    }}
    new = det_intensity_transform({"intensity_norm": "scale01"}, plan)
    old = ScaleIntensityRanged(keys=["image"], a_min=bounds[0], a_max=bounds[1],
                               b_min=0.0, b_max=1.0, clip=True)
    assert torch.equal(new({"image": image})["image"], old({"image": image})["image"])
    legacy = det_intensity_transform({"intensity_clip_range": bounds}, {}, legacy_checkpoint=True)
    assert torch.equal(legacy({"image": image})["image"], old({"image": image})["image"])


def test_fitted_plan_missing_intensity_requires_refit(tmp_path, monkeypatch):  #AI
    """Fail explicitly for pre-intensity fit files before applying them."""
    project = SimpleNamespace(project_folder=tmp_path)
    maker = object.__new__(parser.ConfigMakerDet)
    maker.project = project
    plan = {"mode": "lbd", "plan_name": 1, "spacing": "auto"}
    maker.configs = {"plan_train": plan}
    pkl = nndet_plan_path(project, plan)
    pkl.parent.mkdir()
    pkl.write_bytes(pickle.dumps({"plan": plan, "input_key": "old"}))
    monkeypatch.setattr(parser, "apply_nndet_plan", lambda configs, computed: configs)
    with pytest.raises(ValueError, match="intensity_properties.*[Rr]erun"):
        maker._load_computed_plan(1)


def test_fixed_spacing_worker_accepts_missing_legacy_stats(tmp_path, monkeypatch):  #AI
    """Skip legacy intensity keys when resampling raw HU values."""
    monkeypatch.setattr("fran.preprocessing.rayworker_base.RayWorkerBase.__init__", lambda *a, **kw: None)
    worker = NiftiResamplerLocal(
        SimpleNamespace(global_properties={}), {"spacing": [1, 1, 1]}, tmp_path, tmp_path,
    )
    assert not worker.clip_center


def test_plan_rejects_datasources_sharing_one_preprocessing_alias(monkeypatch):  #AI
    """Refuse to pool two remaps that would share one preprocessing transform."""
    from det3d.configs import nndet_intensity

    specs = {
        "first": SimpleNamespace(ds="first", name="shared"),
        "second": SimpleNamespace(ds="second", name="shared"),
    }
    monkeypatch.setattr(nndet_intensity, "DS", specs)
    with pytest.raises(ValueError, match="ambiguous detection datasources"):
        nndet_intensity.selected_detection_cases(object(), {"plan_name": 1, "datasources": "first,second"})


def test_detection_planning_updates_intensity_and_preserves_the_fit(tmp_path, monkeypatch):  #AI
    """Keep the early spacing record and add intensity to a compatible BoxC002 fit."""
    from det3d.configs import nndet_planning
    from fran.run.preproc.analyze_resample import configure_plan

    plan = {"mode": "lbd", "plan_name": 1, "spacing": [1, 1, 2], "prezoom_scale": 1.0}
    project = SimpleNamespace(project_folder=tmp_path)
    maker = SimpleNamespace(
        setup=lambda plan_id, load_computed: None,
        configs={f"plan_{key}": dict(plan) for key in ("train", "valid", "test")},
    )
    configs = configure_plan(project, maker, 1, "det")
    pkl = configs["nndet_plan_file"]
    spacing_only = pickle.loads(pkl.read_bytes())
    assert spacing_only == {"plan": configs["nndet_plan_snapshot"], "target_spacing": [1, 1, 2]}
    fitted = dict(spacing_only, input_key="measured boxes", patch_size=[16, 16, 8], fg_labels=[2])
    pkl.write_bytes(pickle.dumps(fitted))
    intensity = {"mean": 1.0, "sd": 0.5, "percentile_00_5": 0.1, "percentile_99_5": 1.9}
    monkeypatch.setattr(nndet_planning, "plan_intensity_properties", lambda p, s: (intensity, [2]))
    monkeypatch.setattr(nndet_planning, "nndet_plan_input_key", lambda p, s, folder: "measured boxes")
    monkeypatch.setattr(nndet_planning, "append_lbd_center_ranges", lambda folder, patch, scale: None)
    output = nndet_planning.build_nndet_plan_from_lbd(
        project, configs["plan_train"], tmp_path, pkl,
        plan_snapshot=configs["nndet_plan_snapshot"],
    )
    assert output == pickle.loads(pkl.read_bytes())
    assert output == dict(fitted, intensity_properties=intensity)
    before = pkl.read_bytes()
    monkeypatch.setattr(nndet_planning, "plan_intensity_properties", lambda p, s: (intensity, [3]))
    with pytest.raises(ValueError, match="ambiguous detection foreground"):
        nndet_planning.build_nndet_plan_from_lbd(
            project, configs["plan_train"], tmp_path, pkl,
            plan_snapshot=configs["nndet_plan_snapshot"],
        )
    assert pkl.read_bytes() == before
    concurrent = dict(output, plan=dict(plan, prezoom_scale=1.2))
    pkl.write_bytes(pickle.dumps(concurrent))
    with pytest.raises(parser.StalePlanError, match="changed during preprocessing"):
        nndet_planning.build_nndet_plan_from_lbd(
            project, configs["plan_train"], tmp_path, pkl,
            plan_snapshot=configs["nndet_plan_snapshot"],
        )
    assert pickle.loads(pkl.read_bytes()) == concurrent


def test_detection_base_builds_transforms_without_legacy_intensity_keys():  #AI
    """Build the inherited transform stack used by detection training."""
    from det3d.managers.data.main import DataManagerDet

    manager = object.__new__(DataManagerDet)
    manager.device = "cpu"
    manager.plan = {"patch_size": [16, 16, 16], "samples_per_file": 1, "intensity_properties": {
        "mean": 1.0, "sd": 0.5, "percentile_00_5": 0.0, "percentile_99_5": 2.0,
    }}
    manager.dataset_params = {"intensity_norm": "zscore_fg", "prezoom_scale": 1.0, "fgbg_ratio": 1.0}
    manager.configs = {"affine3d": {"p": 0.0, "rotate_range": 0.0, "scale_range": 0.0}}
    manager.flip = dict(prob=0.0)
    manager.scale = dict(prob=0.0, value=0.1)
    manager.noise = dict(prob=0.0, value=[0.0, 0.1])
    manager.shift = dict(prob=0.0, value=0.1)
    manager.contrast = dict(prob=0.0, value=[0.8, 1.2])
    manager.create_transforms()
    image = torch.tensor([-3.0, 1.0, 10.0])
    output = manager.transforms_dict["N"]({"image": image})["image"]
    assert torch.equal(output, torch.tensor([-2.0, 0.0, 2.0]))
