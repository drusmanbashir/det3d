#!/usr/bin/env python3
"""convert_seg2det with runtime label_remap from dataset.json (det3d plan silo)."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import SimpleITK as sitk
from hydra import initialize_config_module
from loguru import logger
from tqdm import tqdm

from nndet.core.boxes import box_size_np
from nndet.io import save_json
from nndet.io.transforms.instances import get_bbox_np
from nndet.io.itk import load_sitk, load_sitk_as_array
from nndet.utils.check import env_guard
from nndet.utils.clustering import reorder_classes, seg_to_instances
from nndet.utils.config import compose


def _normalize_label_remap(label_remap: Mapping | None) -> Dict[int, int] | None:
    if not label_remap:
        return None
    return {int(k): int(v) for k, v in label_remap.items()}


def prepare_detection_label(
    case_id: str,
    label_dir: Path,
    things_classes: Sequence[int],
    stuff_classes: Sequence[int],
    min_size: float = 0,
    min_vol: float = 0,
    label_remap: Dict[int, int] | None = None,
) -> None:
    if (label_dir / f"{case_id}.json").is_file():
        logger.info(f"Found existing case {case_id} -> skipping")
        return
    logger.info(f"Processing {case_id}")
    seg_itk = load_sitk(label_dir / f"{case_id}.nii.gz")
    spacing = np.asarray(seg_itk.GetSpacing())[::-1]
    seg = sitk.GetArrayFromImage(seg_itk)
    if label_remap:
        seg = np.array(seg, copy=True)
        reorder_classes(seg, label_remap)

    stuff_seg = np.zeros_like(seg)
    if stuff_classes:
        for new_class, old_class in enumerate(stuff_classes, start=1):
            stuff_seg[seg == old_class] = new_class
        stuff_seg_itk = sitk.GetImageFromArray(stuff_seg)
        stuff_seg_itk.SetOrigin(seg_itk.GetOrigin())
        stuff_seg_itk.SetDirection(seg_itk.GetDirection())
        stuff_seg_itk.SetSpacing(seg_itk.GetSpacing())
        sitk.WriteImage(stuff_seg_itk, str(label_dir / f"{case_id}_stuff.nii.gz"))

    things_seg = np.copy(seg)
    things_seg[stuff_seg > 0] = 0

    instances_not_filtered, instances_not_filtered_classes = seg_to_instances(things_seg)
    final_mapping = {}
    if instances_not_filtered.max() > 0:
        boxes = get_bbox_np(instances_not_filtered[None])["boxes"]
        box_sizes = box_size_np(boxes)
        instance_ids = np.unique(instances_not_filtered)
        instance_ids = instance_ids[instance_ids > 0]
        assert len(instance_ids) == len(boxes)
        isotopic_axis = list(range(seg.ndim))
        isotopic_axis.pop(np.argmax(spacing))
        instances = np.zeros_like(instances_not_filtered)
        start_id = 1
        for iid, bsize in zip(instance_ids, box_sizes):
            bsize_world = bsize * spacing
            instance_mask = instances_not_filtered == iid
            instance_vol = instance_mask.sum()
            if all(bsize_world[isotopic_axis] > min_size) and (instance_vol > min_vol):
                instances[instance_mask] = start_id
                semantic_class = instances_not_filtered_classes[int(iid)]
                final_mapping[start_id] = things_classes.index(semantic_class)
                start_id += 1
    else:
        instances = np.zeros_like(instances_not_filtered)

    final_instances_itk = sitk.GetImageFromArray(instances)
    final_instances_itk.SetOrigin(seg_itk.GetOrigin())
    final_instances_itk.SetDirection(seg_itk.GetDirection())
    final_instances_itk.SetSpacing(seg_itk.GetSpacing())
    sitk.WriteImage(final_instances_itk, str(label_dir / f"{case_id}.nii.gz"))
    save_json({"instances": final_mapping}, label_dir / f"{case_id}.json")
    sitk.WriteImage(seg_itk, str(label_dir / f"{case_id}_orig.nii.gz"))


@env_guard
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", type=str, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-o", "--overwrites", type=str, nargs="+", required=False)
    parser.add_argument("--volume_ranking", action="store_true")
    parser.add_argument("--num_processes", type=int, default=4, required=False)
    args = parser.parse_args()

    initialize_config_module(config_module="nndet.conf")
    for task in args.tasks:
        cfg = compose(task, "config.yaml", overrides=args.overwrites or [])
        splitted_dir = Path(cfg["host"]["splitted_4d_output_dir"])
        label_remap = _normalize_label_remap(cfg["data"].get("label_remap"))

        logger.remove()
        logger.add(sys.stdout, level="INFO")
        logger.add(splitted_dir / "convert_seg2det.log", level="DEBUG")
        logger.info(f"+++++ Running conversion: {datetime.now()} +++++")
        if label_remap:
            logger.info(f"runtime label_remap {label_remap}")

        for postfix in ["Tr", "Ts"]:
            label_dir = splitted_dir / f"labels{postfix}"
            case_ids = [f.name[:-7] for f in label_dir.glob("*.nii.gz")]
            logger.info(f"Found {len(case_ids)} cases for conversion with postfix {postfix}.")
            with Pool(processes=args.num_processes) as pool:
                pool.starmap(
                    prepare_detection_label,
                    zip(
                        case_ids,
                        repeat(label_dir),
                        repeat(cfg["data"]["seg2det_things"]),
                        repeat(cfg["data"]["seg2det_stuff"]),
                        repeat(cfg["data"].get("min_size", 0)),
                        repeat(cfg["data"].get("min_vol", 0)),
                        repeat(label_remap),
                    ),
                )

        if args.volume_ranking:
            for postfix in ["Tr", "Ts"]:
                label_dir = splitted_dir / f"labels{postfix}"
                if not label_dir.is_dir():
                    continue
                ranking = []
                for case_id in tqdm([f.stem for f in label_dir.glob("*.json")]):
                    instances = load_sitk_as_array(label_dir / f"{case_id}.nii.gz")[0]
                    instance_ids, instance_counts = np.unique(instances, return_counts=True)
                    cps = [np.argwhere(instances == iid)[0].tolist() for iid in instance_ids[1:]]
                    ranking.extend(
                        {
                            "case_id": str(case_id),
                            "instance_id": int(iid),
                            "vol": int(vol),
                            "cp": list(cp)[::-1],
                        }
                        for iid, vol, cp in zip(instance_ids[1:], instance_counts[1:], cps)
                    )
                save_json(sorted(ranking, key=lambda x: x["vol"]), label_dir / f"volume_ranking_{postfix}.json")


if __name__ == "__main__":
    main()
