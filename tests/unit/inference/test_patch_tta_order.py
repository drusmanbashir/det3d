"""TTA must follow nnDetection's order: reduce each mirror view (topk/clip/NMS), then WBC across views.

MUTATION: pool the views into one weighted NMS (the old merge_tile_boxes path) and the TTA-on test fails.
CONTROL: TTA off still equals merge_tile_boxes on the single view.
"""

import pytest
import torch

import nndet.compat  # noqa: F401
from nndet.inference.ensembler.detection import BoxEnsemblerSelective
from nndet.inference.transforms import get_tta_transforms

from det3d.inference.patch import DetPatchRetinaUNet

SHAPE = (24, 28, 20)
OBJECT = (slice(8, 14), slice(10, 19), slice(5, 9))


class ToyNet(torch.nn.Module):  #AI
    """Detects the nonzero block plus one weaker off-centre box; boxes are nnDetection xyxyzz, tile-local."""

    topk_candidates = 1000
    detections_per_img = 25
    remove_small_boxes = 0.0

    def inference_step(self, data, **_):
        mask = data[0, 0] > 0.5
        idx = mask.nonzero().float()
        lo, hi = idx.min(0).values, idx.max(0).values
        box = torch.stack([lo[0], lo[1], hi[0], hi[1], lo[2], hi[2]])
        shifted = box + torch.tensor([1.0, 0, 1, 0, 0, 0])
        boxes = torch.stack([box, shifted])
        return {
            "pred_boxes": [boxes],
            "pred_scores": [torch.tensor([0.9, 0.4])],
            "pred_labels": [torch.zeros(2)],
            "pred_seg": data.clone(),
        }


def make_inferer(num_tta):  #AI
    inf = DetPatchRetinaUNet.__new__(DetPatchRetinaUNet)
    inf.model = type("M", (), {"net": ToyNet()})()
    inf.num_tta = num_tta
    spatial = tuple(slice(0, s) for s in SHAPE)
    inf.inferer = lambda inputs, network: network(inputs, [(slice(None), slice(None)) + spatial])
    return inf


def volume():  #AI
    vol = torch.zeros(1, 1, *SHAPE)
    vol[(0, 0) + OBJECT] = 1.0
    return vol


def reference(vol, params):  #AI
    """nnDetection's own ensembler fed per view exactly as Predictor.predict_tiles does."""
    parameters = BoxEnsemblerSelective.get_default_parameters()
    parameters.update(params)
    ens = BoxEnsemblerSelective(properties={"shape": SHAPE}, parameters=parameters)
    net = ToyNet()
    transforms, inverses = get_tta_transforms(8, True)
    for t, (tf, inv) in enumerate(zip(transforms, inverses)):
        ens.add_model(name=f"model0_t{t}", model_weight=1.0)
        result = inv(**net.inference_step(tf(data=vol)["data"]))
        ens.process_batch(result=result, batch={"tile_origin": [[0], [0], [0]], "data": vol})
    parts = [ens.process_model(n) for n in ens.model_results]
    return ens.process_ensemble(*[[p[i] for p in parts] for i in range(4)])


def test_tta_on_matches_nndet_ensembler():  #AI
    from det3d.geometry.nndet_boxes import from_nndet

    vol = volume()
    inf = make_inferer(8)
    _, boxes, scores, labels = inf.run_swi(vol)
    ref_b, ref_s, ref_l = reference(vol, inf.inference_params())
    assert torch.equal(boxes, from_nndet(ref_b))
    assert torch.equal(scores, ref_s)
    assert torch.equal(labels, ref_l.long())
    assert boxes.shape[0] >= 1


def test_tta_off_unchanged():  #AI
    vol = volume()
    inf = make_inferer(0)
    _, boxes, scores, labels = inf.run_swi(vol)
    pred = ToyNet().inference_step(vol)
    acc = [{
        "boxes": pred["pred_boxes"][0],
        "scores": pred["pred_scores"][0],
        "labels": pred["pred_labels"][0],
        "weights": inf.box_tile_weight(pred["pred_boxes"][0], SHAPE),
    }]
    inf.model.net.parameters = lambda: iter([torch.zeros(1)])
    exp_b, exp_s, exp_l = inf.merge_tile_boxes(acc, SHAPE)
    assert torch.equal(boxes, exp_b) and torch.equal(scores, exp_s) and torch.equal(labels, exp_l)
