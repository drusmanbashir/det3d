"""Cascade RetinaUNet LM-injection roundtrip with stepwise stage report."""

from pathlib import Path

import torch

from det3d.extra.infer_debug.artifacts import write_lm_nifti_artifacts
from det3d.extra.infer_debug.fixtures import InferFixtureCase, load_fixture
from det3d.extra.infer_debug.fixtures.lidc_0001 import IMAGE_FN, crop_fixture as crop_lidc
from det3d.extra.infer_debug.fixtures.pseudo_cuboid import crop_fixture as crop_pseudo
from det3d.extra.infer_debug.metrics import (
    assert_component_masks_equal,
    assert_plan_roundtrip,
    component_masks_from_lmg,
    fg_mask_equal,
    match_boxes_by_centroid,
    match_nbrhoods_by_centroid,
    boxes_max_delta,
)
from det3d.extra.infer_debug.step_runner import run_transform_chain, write_stage_report
from det3d.extra.infer_debug.streams.cascade_retinaunet import (
    CASCADE_POST_KEYS_SAFE,
    PATCH_POST_KEYS,
    build_cascade_post_dict,
    build_fake_patch_batch,
    build_patch_post_dict,
    build_patch_preprocess,
    load_run_params,
    patch_batch_to_cascade_item,
)

DEFAULT_RUN_P = "LIDCA-QUARK"
DEFAULT_OUT = Path("/s/agent_rw/tmp/infer_debug")


def _crop(case: InferFixtureCase):
    if case.name == "pseudo_cuboid":
        return crop_pseudo(case)
    return crop_lidc(case)


def run_cascade_lm_roundtrip(
    fixture_name: str,
    *,
    run_p: str = DEFAULT_RUN_P,
    out_dir: Path = DEFAULT_OUT,
    strict_full_fg: bool = False,
    assert_plan: bool = False,
    write_nifti: bool = False,
    flat_output: bool = False,
) -> dict:
    case = load_fixture(fixture_name)
    if flat_output:
        artifact_dir = Path(out_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
    else:
        artifact_dir = Path(out_dir) / fixture_name / run_p
        artifact_dir.mkdir(parents=True, exist_ok=True)

    params = load_run_params(run_p)
    preprocess = build_patch_preprocess(params)
    patch_post = build_patch_post_dict(preprocess)
    cascade_post = build_cascade_post_dict(run_p)

    img_crop, lm_crop = _crop(case)
    batch = build_fake_patch_batch(
        case, img_crop, lm_crop, run_p=run_p, params=params, preprocess_compose=preprocess
    )

    batch, patch_records = run_transform_chain(
        batch, patch_post, PATCH_POST_KEYS, chain_name="patch"
    )
    print(
        "after patch post: pred shape",
        tuple(batch["pred"].shape),
        "n_boxes",
        batch["pred_box"].shape[0],
    )

    item = patch_batch_to_cascade_item(batch, case, run_p)
    item, cascade_records = run_transform_chain(
        item, cascade_post, CASCADE_POST_KEYS_SAFE, chain_name="cascade"
    )

    write_stage_report(patch_records, artifact_dir / "patch_stages.tsv")
    write_stage_report(cascade_records, artifact_dir / "cascade_stages.tsv")

    recovered = item["pred"][0].detach().cpu()
    original = case.lm_full.detach().cpu()

    eq, diff_n, fg_o, fg_r = fg_mask_equal(original, recovered)
    result = {
        "fixture": fixture_name,
        "run_p": run_p,
        "out_dir": str(artifact_dir),
        "fg_orig": fg_o,
        "fg_rec": fg_r,
        "fg_diff_voxels": diff_n,
        "fg_equal": eq,
        "patch_stages": patch_records,
        "cascade_stages": cascade_records,
    }

    gt_masks, gt_df = component_masks_from_lmg(original, case.ignore_labels)
    pr_masks, pr_df = component_masks_from_lmg(recovered, case.ignore_labels)
    result["n_lesions_gt"] = len(gt_masks)
    result["n_lesions_rec"] = len(pr_masks)

    pred_boxes = item["pred_box"].detach().cpu().numpy()
    gt_boxes = case.lesion_boxes_full
    box_pairs = match_boxes_by_centroid(gt_boxes, pred_boxes)
    result["box_max_delta"] = boxes_max_delta(gt_boxes, pred_boxes, box_pairs)

    if len(gt_masks) == len(pr_masks) == case.n_lesions:
        pairs = match_nbrhoods_by_centroid(gt_df, pr_df)
        assert_component_masks_equal(gt_masks, pr_masks, pairs)
        result["lesion_masks_exact"] = True
    else:
        result["lesion_masks_exact"] = False

    if write_nifti and fixture_name == "lidc_0001":
        write_lm_nifti_artifacts(original, recovered, IMAGE_FN, artifact_dir, "lidc_0001")

    if assert_plan:
        assert_plan_roundtrip(
            original,
            recovered,
            pred_boxes,
            gt_boxes,
            case.ignore_labels,
            case.n_lesions,
        )
        result["pass"] = True
    else:
        if strict_full_fg and not eq:
            raise AssertionError(f"full fg mismatch on {diff_n} voxels (orig={fg_o} rec={fg_r})")
        if fixture_name == "pseudo_cuboid" and not eq:
            raise AssertionError(f"pseudo_cuboid must roundtrip exactly; diff={diff_n}")
        result["pass"] = result.get("lesion_masks_exact", False) and (
            eq or fixture_name != "pseudo_cuboid"
        )

    report_path = artifact_dir / "summary.txt"
    lines = [
        f"fixture={fixture_name} run_p={run_p}",
        f"fg_orig={fg_o} fg_rec={fg_r} fg_diff={diff_n} fg_equal={eq}",
        f"lesions_gt={result['n_lesions_gt']} lesions_rec={result['n_lesions_rec']}",
        f"lesion_masks_exact={result.get('lesion_masks_exact')}",
        f"box_max_delta={result.get('box_max_delta')}",
        "",
        "=== patch stages ===",
        write_stage_report(patch_records),
        "",
        "=== cascade stages ===",
        write_stage_report(cascade_records),
    ]
    report_path.write_text("\n".join(lines) + "\n")
    return result
