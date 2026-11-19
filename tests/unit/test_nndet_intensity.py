import pickle
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
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
        with pytest.raises(ValueError, match="empty foreground"):
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


POSTPROCESS = {"score_thresh": 0.02, "nms_thresh": 0.22, "detections_per_img": 25}
MODEL = {
    "conv_kernels": [[3, 3, 3]] * 3,
    "strides": [[2, 2, 1], [2, 2, 2]],
    "decoder_levels": (1, 2),
    "anchors": {"width": [(4,), (8,)], "height": [(4,), (8,)], "depth": [(2,), (4,)], "stride": 1},
}


def _det_plan_maker(tmp_path, row):  #AI
    """Supply one plans row, nested parameter sheets and a private plan-file location to setup."""
    maker = object.__new__(parser.ConfigMakerDet)
    maker.project = SimpleNamespace(project_folder=tmp_path)
    full_row = {"batch_size": "auto", "model": "auto", "postprocess": POSTPROCESS, "val_patch_size": "auto"} | row
    maker.plans = parser.pd.DataFrame([full_row], index=[1])
    maker.configs = {"dataset_params": {}, "model_params": dict(MODEL_PARAMS)}
    return maker


def _write_resolved(maker, resolved):  #AI
    """Save a resolved plan made from the maker's current sheet values."""
    maker.setup(1, load_computed=False)
    pkl = FolderNames(maker.project, {"plan_name": 1}).plan_file
    pkl.parent.mkdir(exist_ok=True)
    pkl.write_bytes(pickle.dumps({"requested": parser.ConfigMakerDet.plan_request(maker.configs), "plan_train": resolved}))
    return pkl


def test_detection_config_loads_without_plan_file(tmp_path):  #AI
    """Load the sheet plan while its auto values are still unresolved."""
    maker = _det_plan_maker(tmp_path, {
        "mode": "lbd", "spacing": "auto", "patch_dim0": 192, "patch_dim1": 96, "batch_size": 4,
    })
    maker.setup(1)
    plan = maker.configs["plan_train"]
    assert plan["spacing"] == "auto"
    assert plan["patch_size"] == [192, 192, 96]
    assert plan["batch_size"] == 4
    assert plan["model"] == "auto"
    assert plan["postprocess"] == POSTPROCESS


def test_detection_resolved_plan_wins_including_an_adjusted_patch(tmp_path):  #AI
    """Resolved values replace the sheet's, the crop identity keeps the sheet patch, postprocess stays the sheet's."""
    maker = _det_plan_maker(tmp_path, {"mode": "lbd", "spacing": [1, 1, 2], "patch_dim0": 192, "patch_dim1": 96})
    intensity = {"mean": 1, "sd": 2, "percentile_00_5": -3, "percentile_99_5": 5}
    _write_resolved(maker, {
        "spacing": [1, 1, 2], "patch_size": [160, 160, 96], "val_patch_size": [160, 160, 96], "batch_size": 4,
        "model": MODEL, "fg_labels": [2], "intensity_properties": intensity, "optimizer": "unused",
    })
    maker.setup(1)
    for split in ("train", "valid", "test"):
        plan = maker.configs[f"plan_{split}"]
        assert plan["patch_size"] == [160, 160, 96]
        assert plan["val_patch_size"] == [160, 160, 96]
        assert plan["preprocessing_patch_size"] == [192, 192, 96]
        assert plan["batch_size"] == 4
        assert plan["model"] == MODEL
        assert plan["fg_labels"] == [2]
        assert plan["intensity_properties"] == intensity
        assert plan["postprocess"] == POSTPROCESS
        assert "optimizer" not in plan
    assert maker.configs["model_params"] == MODEL_PARAMS


def test_detection_stale_resolved_plan_is_refused(tmp_path):  #AI
    """A planning-relevant sheet edit after det preprocessing is refused; a postprocess edit is not."""
    row = {"mode": "lbd", "spacing": "auto", "patch_dim0": 192, "patch_dim1": 96}
    _write_resolved(_det_plan_maker(tmp_path, row), {"spacing": [1, 1, 2], "val_patch_size": [192, 192, 96]})
    edited = _det_plan_maker(tmp_path, row | {"postprocess": dict(POSTPROCESS, score_thresh=0.5)})
    edited.setup(1)
    assert edited.configs["plan_train"]["spacing"] == [1, 1, 2]
    val_edit = _det_plan_maker(tmp_path, row | {"val_patch_size": [64, 64, 64]})
    with pytest.raises(parser.StalePlanError, match="val_patch_size differs"):
        val_edit.setup(1)
    edited.configs["model_params"]["encoder_start_channels"] = 16
    with pytest.raises(parser.StalePlanError, match="model_params.encoder_start_channels: 32 -> 16"):
        edited.setup(1)


def test_detection_postprocess_auto_is_refused(tmp_path):  #AI
    """postprocess = auto has no resolver until the postprocess sweep exists."""
    maker = _det_plan_maker(tmp_path, {
        "mode": "lbd", "spacing": [1, 1, 2], "patch_dim0": 192, "patch_dim1": 96, "postprocess": "auto",
    })
    with pytest.raises(ValueError, match="postprocess sweep"):
        maker.setup(1)


def test_boxc002_preset_model_strides_round_a_patch_down():  #AI
    """A preset model's total stride rounds the patch down to a multiple of it."""
    from det3d.planning.c002 import LBDBoxC002

    planner = object.__new__(LBDBoxC002)
    planner.preset_model = MODEL
    _, strides, kernels, patch, divisor = planner.plan_pool_and_conv_pool_late([102, 99, 51], [1, 1, 2])
    assert list(patch) == [100, 96, 50]
    assert list(divisor) == [4, 4, 2]
    assert strides == MODEL["strides"] and kernels == MODEL["conv_kernels"]


def test_boxc002_fixed_patch_keeps_the_patch_and_fits_only_an_auto_batch():  #AI
    """Fixed patch: the patch is untouched; an auto batch is lowered until it fits; a fixed batch stays."""
    from det3d.planning.c002 import LBDBoxC002

    planner = object.__new__(LBDBoxC002)
    planner.preset_patch = [64, 64, 32]
    planner.preset_batch = None
    planner.batch_size = 4
    planner.num_decoder_level, planner.min_decoder_level = 4, 2
    planner.architecture_kwargs = {}
    planner.estimator = SimpleNamespace(batch_size=4)
    planner.cases = [(None, [1], (64, 64, 32))] * 10
    planner.plan_pool_and_conv_pool_late = lambda p, s: (None, [[2, 2, 2]] * 3, [[3, 3, 3]] * 4, np.asarray(p), None)
    planner._fits_at_batch = lambda batch, patch: batch <= 2
    patch = planner._plan_architecture([1, 1, 1], [64, 64, 32], [0, 1, 2])
    assert list(patch) == [64, 64, 32] and planner.batch_size == 2
    assert planner.architecture_kwargs["decoder_levels"] == (1, 2, 3)
    planner.preset_batch, planner.batch_size = 8, 8
    assert list(planner._plan_architecture([1, 1, 1], [64, 64, 32], [0, 1, 2])) == [64, 64, 32]
    assert planner.batch_size == 8


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
def test_detection_planning_updates_intensity_and_preserves_the_fit(tmp_path, monkeypatch, workbook_patch):  #AI
    """Keep the early spacing record and add intensity to a compatible BoxC002 fit."""
    from det3d.planning.c002 import LBDBoxC002
    from fran.run.preproc import analyze_resample
    from fran.run.preproc.analyze_resample import configure_plan

    plan = {"mode": "lbd", "plan_name": 1, "spacing": [1, 1, 2], "prezoom_scale": 1.0, "patch_size": workbook_patch,
            "batch_size": "auto", "model": "auto", "postprocess": POSTPROCESS, "val_patch_size": "auto"}
    project = SimpleNamespace(project_folder=tmp_path)
    maker = SimpleNamespace(
        setup=lambda plan_id, load_computed: None,
        configs={f"plan_{key}": dict(plan) for key in ("train", "valid", "test")} | {"model_params": dict(MODEL_PARAMS)},
    )
    configs = configure_plan(project, maker, 1, "det")
    pkl = configs["nndet_plan_file"]
    spacing_only = pickle.loads(pkl.read_bytes())
    assert spacing_only == {"requested": configs["plan_request"], "plan_train": {"spacing": [1, 1, 2]}}
    resolved = {
        "requested": configs["plan_request"],
        "plan_train": {"spacing": [1, 1, 2], "patch_size": [16, 16, 8], "fg_labels": [2]},
        "fit": {"input_key": "measured boxes", "gpu": {"name": "card", "total_memory": 1, "buffer_mb": 1}},
    }
    pkl.write_bytes(pickle.dumps(resolved))
    intensity = {"mean": 1.0, "sd": 0.5, "percentile_00_5": 0.1, "percentile_99_5": 1.9}
    monkeypatch.setattr(analyze_resample, "plan_intensity_properties", lambda p, s: (intensity, [2]))
    monkeypatch.setattr(analyze_resample, "plan_cases", lambda p, plan: [])
    monkeypatch.setattr(analyze_resample, "selected_lbd_details", lambda folder, case_ids: None)
    monkeypatch.setattr(analyze_resample, "lbd_input_key", lambda folder, details: "measured boxes")
    monkeypatch.setattr(
        LBDBoxC002, "planning_gpu", staticmethod(lambda: {"name": "card", "total_memory": 1, "buffer_mb": 1, "budget": 1})
    )
    append_ranges = Mock()
    monkeypatch.setattr(analyze_resample, "append_center_ranges", append_ranges)
    manager = object.__new__(analyze_resample.PreprocessingManager)
    manager.project, manager.configs, manager.L = project, configs, SimpleNamespace(output_folder=tmp_path)
    output = manager.plan_detection()
    assert output == pickle.loads(pkl.read_bytes())
    assert output == dict(
        resolved,
        plan_train=dict(resolved["plan_train"], intensity_properties=intensity, val_patch_size=[16, 16, 8]),
        fit=dict(resolved["fit"], histogram_labels=[2]),
    )
    append_ranges.assert_called_once_with(tmp_path, [16, 16, 8], 1.0, False)
    calls = Mock(side_effect=lambda p, s: (intensity, [2]))
    monkeypatch.setattr(analyze_resample, "plan_intensity_properties", calls)
    assert manager.plan_detection() == output
    calls.assert_not_called()
    pkl.write_bytes(pickle.dumps(resolved))
    before = pkl.read_bytes()
    monkeypatch.setattr(analyze_resample, "plan_intensity_properties", lambda p, s: (intensity, [3]))
    with pytest.raises(ValueError, match="ambiguous detection foreground"):
        manager.plan_detection()
    assert pkl.read_bytes() == before
    concurrent = dict(output, requested=dict(output["requested"], plan_train=dict(plan, prezoom_scale=1.2)))
    pkl.write_bytes(pickle.dumps(concurrent))
    with pytest.raises(analyze_resample.StalePlanError, match="changed during preprocessing"):
        manager.plan_detection()
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


def test_boxc002_grows_the_batch_at_the_patch_cap_up_to_dataset_coverage():  #AI
    """The batch rises while the patch still fits, and stops at 5% dataset coverage."""
    from det3d.planning.c002 import LBDBoxC002

    planner = object.__new__(LBDBoxC002)
    planner.estimator = SimpleNamespace(batch_size=4)
    planner.batch_size = 4
    planner.cases = [(None, [1], (100, 100, 100))] * 20
    planner._fits_at_batch = lambda batch, patch: batch <= 6
    planner._grow_batch([20, 20, 20])
    assert planner.batch_size == 6 and planner.estimator.batch_size == 6
    planner.batch_size = 4
    planner._fits_at_batch = lambda batch, patch: True
    planner._grow_batch([50, 50, 50])
    assert planner.batch_size == 8
