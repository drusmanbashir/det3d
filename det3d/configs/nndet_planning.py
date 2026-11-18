"""Plan RetinaUNet architecture from FRAN's label-bounded outputs."""

import csv
import json
import pickle
import sys
from ast import literal_eval
from collections import Counter
from contextlib import contextmanager
from fcntl import LOCK_EX, flock
from pathlib import Path
from subprocess import check_output
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from utilz.cprint import cprint

from fran.configs.defaults import lbd_crop_minimum, patch_minimum, plan_gpu_max_process_mb
from fran.configs.helpers import normalize_tree
from fran.data.dataregistry import DS
from fran.transforms.detection import GenerateExtendedBoxMask, patch_size_manifest_key
from fran.utils.bbox_sidecar import (
    extended_bbox_df_column,
    nbrhood_df_to_detection_records,
)
from fran.utils.preprocessing_identity import identity_hash

from det3d.configs.nndet_bridge import load_nndet_train_cfgs
from det3d.managers.helpers.nndet_retinaunet import (
    apply_det3d_plan_to_nndet_model_cfg,
    ensure_nndet_importable,
    to_nndet,
)

ensure_nndet_importable()
from nndet.planning.architecture.boxes.c002 import BoxC002
from nndet.planning.estimator import MemoryEstimatorDetection
from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001


TARGET_MEMORY = "RTX2080TI"
BOX_COLUMNS = ["label_org", "bbox_xyzxyz"]
BYTES_PER_MB = 1024**2


def _model_cfg(plan):  #AI
    model_cfg, _ = load_nndet_train_cfgs()
    model_cfg = apply_det3d_plan_to_nndet_model_cfg(model_cfg, plan)
    start_channels = int(plan["encoder_start_channels"])
    model_cfg["plan_arch_overwrites"].update(
        dim=int(plan["spatial_dims"]),
        in_channels=int(plan["n_input_channels"]),
        start_channels=start_channels,
        max_channels=int(plan["encoder_max_channels"]),
        fpn_channels=start_channels * 4,
        head_channels=start_channels * 4,
    )
    return model_cfg


def nndet_plan_path(project, plan):  #AI
    return Path(project.project_folder) / "plans" / f"{int(plan['plan_name'])}.pkl"


@contextmanager
def _write_lock(path):  #AI
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        flock(stream, LOCK_EX)
        yield


def _atomic_write(path, content):  #AI
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def load_nndet_plan(path):  #AI
    with path.open("rb") as stream:
        return pickle.load(stream)


def resolve_nndet_spacing(project, plan, path):  #AI
    """Persist the resolved spacing before any stage creates derived data."""
    snapshot = normalize_tree(plan, tuple_mode="list")
    with _write_lock(path.with_suffix(".lock")):
        if path.is_file():
            stored = load_nndet_plan(path)
            if stored.get("plan") == snapshot:
                return stored["target_spacing"]
        spacing = plan["spacing"]
        if spacing == "auto":
            from fran.preprocessing.target_spacing import TargetSpacingPlanner

            spacing = TargetSpacingPlanner(project, plan).run()["target_spacing"]
        spacing = list(spacing)
        _atomic_write(path, pickle.dumps({"plan": snapshot, "target_spacing": spacing}))
    return spacing


def _selected_lbd_details(project, plan, lbd_folder):  #AI
    details = pd.read_csv(lbd_folder / "dataset_details.csv")
    datasets = {DS[name.strip()].ds for name in plan["datasources"].split(",")}
    rows = project.sql_query("SELECT ds, case_id FROM datasources WHERE test=0 ORDER BY ds, case_id")
    selected = [str(case_id) for ds, case_id in rows if ds in datasets]
    if len(set(selected)) != len(selected):
        raise ValueError("Selected datasources must have unique LBD case IDs")
    details = details[details["case_id"].astype(str).isin(selected)]
    if details["case_id"].duplicated().any() or set(details["case_id"].astype(str)) != set(selected):
        raise ValueError(f"Expected one LBD dataset_details row for each selected case in {lbd_folder}")
    return details


def _sidecar(lbd_folder, row):  #AI
    return lbd_folder / "bboxes" / f"{Path(str(row.fn_name)).stem}.csv"


def _shape(row):  #AI
    shape = literal_eval(row.shape) if isinstance(row.shape, str) else row.shape
    return tuple(int(axis) for axis in shape[-3:])


def nndet_plan_input_key(project, plan, lbd_folder):  #AI
    """Hash measured shapes and base boxes without derived sampling columns."""
    details = _selected_lbd_details(project, plan, lbd_folder)
    boxes = [
        pd.read_csv(_sidecar(lbd_folder, row), usecols=BOX_COLUMNS)[BOX_COLUMNS].to_csv(index=False)
        for row in details.itertuples(index=False)
    ]
    return identity_hash({"details": details.to_csv(index=False), "boxes": boxes})


class LBDBoxC002(BoxC002):
    """Run nnDetection's architecture and anchor algorithms on FRAN objects."""

    def __init__(self, cases, spacing, model_cfg, estimator, in_channels):
        self.cases = cases
        self.spacing = np.asarray(spacing)
        self.model_cfg = model_cfg
        self.estimator = estimator
        self.in_channels = in_channels
        self.network_cls = RetinaUNetV001
        self.dim = 3
        self.transpose_forward = [0, 1, 2]
        self.architecture_kwargs = {}

    def process_properties(self):
        counts = Counter(label for _, labels, _ in self.cases for label in labels)
        self.labels = sorted(counts)
        if not self.labels:
            raise ValueError("BoxC002 requires foreground boxes in the selected LBD cases")
        self.all_boxes = [boxes if len(boxes) else [] for boxes, _, _ in self.cases]
        self.all_spacings = [self.spacing] * len(self.cases)
        self.dataset_properties = {
            "num_instances": {index: counts[label] for index, label in enumerate(self.labels)}
        }
        self.architecture_kwargs.update(
            dim=self.dim, in_channels=self.in_channels,
            classifier_classes=len(self.labels), seg_classes=len(self.labels),
        )

    def _check_planning_gpu(self):  #AI
        index = self.estimator.gpu_id
        device = torch.cuda.get_device_properties(index)
        gpu_uuid = f"GPU-{device.uuid}"
        max_process_mb = plan_gpu_max_process_mb()
        processes = check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        offenders = [
            f"PID {pid} ({name}): {memory} MB"
            for uuid, pid, name, memory in csv.reader(processes.splitlines(), skipinitialspace=True)
            if uuid == gpu_uuid and int(memory) > max_process_mb
        ]
        if offenders:
            raise RuntimeError(
                f"BoxC002 on GPU {index} ({device.name}) found compute processes "
                f"above {max_process_mb} MB: " + ", ".join(offenders)
            )
        free, total = torch.cuda.mem_get_info(index)
        free_mb = free // BYTES_PER_MB
        target_mem = self.estimator.target_mem
        cprint(
            f"BoxC002 planning on GPU {index} ({device.name}) with {free_mb} MB free "
            f"of {total // BYTES_PER_MB} MB and an nnDetection budget of "
            f"{target_mem / BYTES_PER_MB:g} MB",
            bold=True,
        )
        sys.stdout.flush()
        return {
            "target_memory": TARGET_MEMORY,
            "target_memory_bytes": target_mem,
            "gpu_name": device.name,
            "gpu_free_memory_mb": free_mb,
            "torch_version": str(torch.__version__),
        }

    def fit_lbd(self):  #AI
        self.create_default_settings()
        self.process_properties()
        class_weight = self.compute_class_weights()
        self.architecture_kwargs["class_weight"] = class_weight
        median_shape = np.median([shape for _, _, shape in self.cases], axis=0)
        gpu_metadata = self._check_planning_gpu()
        patch = self._plan_architecture(
            self.spacing, median_shape, self.transpose_forward
        )
        anchors = self._plan_anchors(
            target_spacing_transposed=self.spacing, transpose_forward=self.transpose_forward
        )
        return {
            "patch_size": [int(value) for value in patch],
            "batch_size": int(self.batch_size),
            "conv_kernels": self.architecture_kwargs["conv_kernels"],
            "strides": self.architecture_kwargs["strides"],
            "decoder_levels": self.architecture_kwargs["decoder_levels"],
            "anchors": anchors,
            "fg_labels": self.labels,
            "class_weight": class_weight,
            "estimator": gpu_metadata,
        }


def append_lbd_center_ranges(lbd_folder, patch_size, prezoom_scale):  #AI
    """Populate missing centre columns, then publish their available sizes."""
    scales = (1.0, prezoom_scale, 1.1, 1.2, 1.4)
    sizes = sorted({tuple(int(axis * scale) for axis in patch_size) for scale in scales})
    manifest_path = lbd_folder / "manifest.json"
    with _write_lock(lbd_folder / ".center_ranges.lock"):
        manifest = json.loads(manifest_path.read_text())
        current = {tuple(size) for size in manifest.get("extended_bboxes_patch_sizes", [])}
        details = pd.read_csv(lbd_folder / "dataset_details.csv")
        for row in details.itertuples(index=False):
            path = _sidecar(lbd_folder, row)
            frame = pd.read_csv(path)
            missing = {
                size: extended_bbox_df_column(patch_size_manifest_key(size))
                for size in sizes
                if extended_bbox_df_column(patch_size_manifest_key(size)) not in frame
            }
            if not missing:
                continue
            boxes, _, _ = nbrhood_df_to_detection_records(frame)
            box_array = torch.stack(boxes).numpy() if boxes else np.empty((0, 6))
            for size, column in missing.items():
                generator = GenerateExtendedBoxMask(
                    keys="bbox", image_key="image", spatial_size=size, whole_box=False
                )
                ranges = generator.generate_fg_center_boxes_np(
                    box_array, _shape(row), whole_box=False
                )
                frame[column] = ranges.tolist()
            _atomic_write(path, frame.to_csv(index=False).encode())
        updated = current.union(sizes)
        if updated != current:
            manifest["extended_bboxes_patch_sizes"] = sorted(updated)
            _atomic_write(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode())


def build_nndet_plan_from_lbd(project, plan, lbd_folder, path, *, plan_snapshot):  #AI
    """Reuse a current fit and restore missing sampling columns after LBD rewrites."""
    with _write_lock(path.with_suffix(".lock")):
        stored = load_nndet_plan(path)
        if stored["plan"] != plan_snapshot:
            raise RuntimeError(
                f"Plan {plan['plan_name']} changed during preprocessing. "
                "Rerun preprocessing for the current spreadsheet plan."
            )
        input_key = nndet_plan_input_key(project, plan, lbd_folder)
        if stored.get("input_key") != input_key:
            details = _selected_lbd_details(project, plan, lbd_folder)
            cases = []
            for row in details.itertuples(index=False):
                frame = pd.read_csv(_sidecar(lbd_folder, row), usecols=BOX_COLUMNS)
                boxes, labels, _ = nbrhood_df_to_detection_records(frame)
                tensor = torch.stack(boxes) if boxes else torch.empty((0, 6))
                cases.append((to_nndet(tensor).numpy(), [int(label) for label in labels], _shape(row)))
            estimator = MemoryEstimatorDetection(target_mem=TARGET_MEMORY, gpu_id=0)
            planner = LBDBoxC002(
                cases, stored["target_spacing"], _model_cfg(plan), estimator, plan["n_input_channels"]
            )
            stored.update(planner.fit_lbd())
            stored.update(
                input_key=input_key,
                PATCH_MIN=patch_minimum(),
            )
            _atomic_write(path, pickle.dumps(stored))
        append_lbd_center_ranges(lbd_folder, stored["patch_size"], plan["prezoom_scale"])
    return stored


def apply_nndet_plan(configs, computed):  #AI
    """Overlay fitted values while retaining the preprocessing crop identity."""
    for key in ("plan_train", "plan_valid", "plan_test"):
        plan = configs[key]
        plan["preprocessing_patch_size"] = plan["patch_size"]
        plan["patch_size"] = list(computed["patch_size"])
        plan["encoder_conv_kernels"] = computed["conv_kernels"]
        plan["encoder_strides"] = computed["strides"]
        plan["decoder_levels"] = tuple(computed["decoder_levels"])
        plan["nndet_anchors"] = computed["anchors"]
        plan["fg_labels"] = list(computed["fg_labels"])
        plan["spacing"] = list(computed["target_spacing"])
        plan["src_dims"] = list(lbd_crop_minimum(plan))
    configs["dataset_params"]["batch_size"] = int(computed["batch_size"])
    return configs
