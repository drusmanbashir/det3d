"""Compare tracker output with nearest-neighbor voxels and the corner baseline."""

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from monai.data import MetaTensor
from monai.transforms import RandAffined, RandAxisFlipd, ResizeWithPadOrCropd

import det3d.managers.data  # noqa: F401
from det3d.transforms.gpu_det import (
    RandAffineBoxSyncd,
    RandFlipBoxSyncd,
    ResizeWithPadOrCropBoxSyncd,
)

ROTATION = [(0.35, 0.35)] * 3
BASELINE = {
    0: [[1077238372, 1091566849, 1090739246, 1098891584, 1100809108, 1101599634],
        [1071387828, 1101551592, 0, 1096350944, 1104674816, 1086961180]],
    1: [[1092905222, 1086851948, 1091032102, 1100519730, 1100189757, 1101752172],
        [1100590831, 1051511496, 0, 1103626240, 1093464340, 1086011610]],
    5: [[1082630176, 1087438406, 1091865429, 1096796738, 1101190550, 1101626892],
        [0, 1097299770, 0, 1071652196, 1104401652, 1088473330]],
    11: [[1091472196, 1083491317, 1089653004, 1100327894, 1099910135, 1102613893],
         [1098814906, 0, 0, 1103626240, 1091696790, 1083379934]],
}


def voxel_box(mask):  #AI
    """Measure the tight exclusive box directly from the occupied voxels."""
    coords = torch.nonzero(mask)
    return torch.cat((coords.amin(0), coords.amax(0) + 1)).float()


def label_map():  #AI
    """Return an interior object, a cut object and a corner object."""
    lm = torch.zeros((1, 32, 32, 32), dtype=torch.int16)
    lm[0, 12:20, 13:15, 13:16] = 2
    lm[0, 12:14, 13:21, 13:16] = 2
    lm[0, 4:10, 12:20, 12:19] = 3
    lm[0, 0:3, 0:3, 15:18] = 4
    return lm


def rows_for(lm, labels):  #AI
    """Build sidecar rows in the requested order from known objects."""
    boxes = torch.stack([voxel_box(lm[0] == lab) for lab in labels])
    rows = pd.DataFrame({"label_org": labels, "bbox_xyzxyz": boxes.int().tolist()})
    return boxes, rows


def run_tracker(name, lm, seed, shape):  #AI
    """Exercise real transforms with independently chosen fixed seeds."""
    from det3d.transforms.box_tracker import BOX_TRACKERS

    tracker = BOX_TRACKERS[name]()
    boxes, rows = rows_for(lm, [2, 3, 4])
    item = {"image": MetaTensor(lm.float()), "bbox": boxes, "label": torch.tensor([2, 3, 4])}
    key = "inst" if name == "remeasure" else "lm"
    if name == "remeasure":
        item[key] = MetaTensor(torch.as_tensor(tracker.compute(lm, rows))[None])
        item["inst_labels"] = item["label"]
    else:
        item[key] = MetaTensor(lm)
        item["hull_xyz"] = tracker.compute(lm, rows)
    keys = ["image", key]
    flip = RandFlipBoxSyncd(keys, "bbox", 1, tracker=tracker)
    affine = RandAffineBoxSyncd(keys, "bbox", ["bilinear", "nearest"], 1, ROTATION, None, tracker=tracker)
    resize = ResizeWithPadOrCropBoxSyncd(keys, "bbox", "label", shape, tracker=tracker)
    flip.rand_flip.set_random_state(seed)
    affine.rand_affine.set_random_state(seed)
    return tracker.finalize(resize(affine(flip(item))))


def reference(lm, seed, shape):  #AI
    """Warp the semantic voxels using MONAI alone."""
    f = RandAxisFlipd("lm", 1)
    a = RandAffined("lm", prob=1, rotate_range=ROTATION, mode="nearest", padding_mode="zeros")
    f.set_random_state(seed)
    a.set_random_state(seed)
    warped = a(f({"lm": MetaTensor(lm)}))
    cropped = ResizeWithPadOrCropd("lm", shape)(warped)
    return cropped["lm"][0]


def test_remeasure_matches_surviving_voxels():  #AI
    """Remeasure exactly after flip, affine and crop, including clipped and lost objects."""
    lm = label_map()
    ref = reference(lm, 9, (24, 24, 24))
    uncut = reference(lm, 9, (32, 32, 32))
    assert 0 < (ref == 3).sum() < (uncut == 3).sum()
    assert not (uncut == 4).any()
    out = run_tracker("remeasure", lm, 9, (24, 24, 24))
    assert out["label"].tolist() == [2, 3]
    assert torch.equal(out["bbox"], torch.stack([voxel_box(ref == 2), voxel_box(ref == 3)]))
    assert torch.equal(out["lm"][0], ref)


def test_hull_interior_within_one_voxel():  #AI
    """The transformed hull bounds an interior object to one voxel per face."""
    lm = label_map()
    ref = reference(lm, 9, (32, 32, 32))
    out = run_tracker("hull", lm, 9, (32, 32, 32))
    assert torch.all((out["bbox"][0] - voxel_box(ref == 2)).abs() <= 1)


def test_hull_drops_rotated_out_object():  #AI
    """Discard a hull whose clipped box has no voxels of its label."""
    lm = label_map()
    lm[lm == 4] = 0
    lm[0, 0:5, 2:7, 29:32] = 4
    ref = reference(lm, 9, (24, 24, 24))
    assert not (ref == 4).any()
    out = run_tracker("hull", lm, 9, (24, 24, 24))
    assert out["label"].tolist() == [2, 3]


@pytest.mark.parametrize("seed", BASELINE)
def test_corner_same_float_bits_as_main(seed):  #AI
    """Compare output with float32 bits captured on main at 33c5d3b."""
    from det3d.transforms.box_tracker import BOX_TRACKERS

    tracker = BOX_TRACKERS["corner"]()
    item = {"image": MetaTensor(torch.zeros(1, 32, 32, 32)),
            "bbox": torch.tensor([[10., 11., 12., 17., 20., 23.], [0., 1., 2., 7., 9., 10.]]),
            "label": torch.tensor([2, 4])}
    f = RandFlipBoxSyncd(["image"], "bbox", 1, tracker=tracker)
    a = RandAffineBoxSyncd(["image"], "bbox", ["bilinear"], 1, [0.4] * 3, [0.15] * 3, tracker=tracker)
    r = ResizeWithPadOrCropBoxSyncd(["image"], "bbox", "label", (25, 27, 29), tracker=tracker)
    f.rand_flip.set_random_state(seed)
    a.rand_affine.set_random_state(seed)
    out = tracker.finalize(r(a(f(item))))
    assert out["bbox"].view(torch.int32).tolist() == BASELINE[seed]
    assert out["label"].tolist() == [2, 4]


def test_preprocessing_writes_hulls_and_shard_instances(tmp_path):  #AI
    """Write the real sidecar and shard, then check both against reversed rows."""
    from ast import literal_eval

    from det3d.transforms.box_tracker import HullStatsd
    from fran.preprocessing.hdf5_shards_det import DetHDF5ShardWorker
    from fran.preprocessing.labelbounded import _LBDSamplerWorkerBase

    lm = label_map()
    boxes, rows = rows_for(lm, [4, 2])
    item = HullStatsd()({"lm": lm, "bbox": boxes, "nbrhoods": rows})
    worker = object.__new__(_LBDSamplerWorkerBase)
    worker.output_folder = tmp_path
    worker.defer_extended_boxes = True
    (tmp_path / "bboxes").mkdir()
    worker.save_bbox_sidecar(item, "case.pt", "case")
    sidecar = tmp_path / "bboxes/case.csv"
    out = pd.read_csv(sidecar)
    for raw, box in zip(out["hull_xyz"], boxes):
        points = np.array(literal_eval(raw))
        np.testing.assert_array_equal(np.r_[points.min(0), points.max(0)], box.numpy())
    pt = tmp_path / "case.pt"
    torch.save(lm[0], pt)
    with h5py.File(tmp_path / "shard.h5", "w") as f:
        DetHDF5ShardWorker()._write_case(f, "case", pt, pt, sidecar, (32, 32, 32), None, None)
        np.testing.assert_array_equal(f["cases/case/bbox"][:], boxes.numpy())
        np.testing.assert_array_equal(f["cases/case/label"][:], [4, 2])
        expected = (lm[0] == 4).int() + 2 * (lm[0] == 2).int()
        np.testing.assert_array_equal(f["cases/case/inst"][:], expected.numpy())


@pytest.mark.parametrize("name", ["hull", "remeasure"])
def test_patch_stream_finalizes_each_crop(name):  #AI
    """A tile inside the case box can contain no voxels from the object."""
    from det3d.managers.data.valid_patch_stream import PatchStreamDatasetDet, patch_stream_det_collated
    from det3d.managers.helpers.nndet_retinaunet import det3d_batch_to_nndet
    from det3d.transforms.box_tracker import BOX_TRACKERS

    tracker = BOX_TRACKERS[name]()
    lm = torch.zeros(1, 16, 16, 8, dtype=torch.int32)
    lm[0, 1:15, 1:4, 1:4] = 2
    lm[0, 1:4, 1:15, 1:4] = 2
    boxes, rows = rows_for(lm, [2])
    item = {
        "case_id": "case", "image": MetaTensor(lm.float()), "lm": MetaTensor(lm),
        "bbox": boxes, "label": torch.tensor([2]),
    }
    if name == "hull":
        item["hull_xyz"] = tracker.compute(lm, rows)
    else:
        item["inst"] = MetaTensor(torch.as_tensor(tracker.compute(lm, rows))[None])
        item["inst_labels"] = item["label"]
    tiles = list(PatchStreamDatasetDet(
        [item], (8, 8, 8), tracker=tracker, instance_key="inst" if name == "remeasure" else None
    ))
    assert len(tiles) == 4
    batch = patch_stream_det_collated(tiles)
    assert det3d_batch_to_nndet(batch, [2])["target_seg"].shape == (4, 8, 8, 8)
    for tile in tiles:
        mask = tile["lm"][0] == 2
        assert len(tile["bbox"]) == int(mask.any())
        if name == "remeasure" and mask.any():
            assert torch.equal(tile["bbox"][0], voxel_box(mask))


def test_hull_keeps_visible_subvoxel_extent():  #AI
    """A subvoxel hull remains a target when its voxel survives resampling."""
    from det3d.transforms.box_tracker import HullTracker

    tracker = HullTracker()
    lm = torch.zeros(1, 9, 9, 9)
    lm[0, 4, 3:6, 2:7] = 2
    boxes, rows = rows_for(lm, [2])
    item = {"image": MetaTensor(lm), "lm": MetaTensor(lm), "bbox": boxes,
            "label": torch.tensor([2]), "hull_xyz": tracker.compute(lm, rows)}
    affine = RandAffineBoxSyncd(
        ["image", "lm"], "bbox", ["bilinear", "nearest"], 1, None,
        [(0.1, 0.1)] * 3, tracker=tracker
    )
    affine.rand_affine.set_random_state(0)
    out = tracker.finalize(affine(item))
    assert (out["lm"] == 2).any()
    assert out["label"].tolist() == [2]
    assert 0 < out["bbox"][0, 3] - out["bbox"][0, 0] < 1
