"""nnDetection's BoxC002 architecture planner on FRAN label-bounded crops, and the det3d fit at training start."""

import sys
from collections import Counter

import numpy as np
import torch
from utilz.cprint import cprint

from fran.configs.defaults import plan_gpu_buffer_mb
from fran.nn_compat.fit_store import FitStore
from fran.nn_compat.gpu_memory import check_fixed_memory, check_planning_gpu, planning_budget
from fran.nn_compat.linked_config_keys import LinkedConfigKeys
from fran.preprocessing.dataset_details_lbd import (
    append_center_ranges, details_shape, lbd_box_records, lbd_input_key, selected_lbd_details,
)
from fran.preprocessing.plan_intensity import plan_cases
from fran.utils.folder_names import FolderNames

from det3d.managers.retinaunet import RetinaUNetManager
from det3d.geometry.nndet_boxes import to_nndet

import nndet.compat  # noqa: F401
from nndet.planning.architecture.boxes.c002 import BoxC002
from nndet.planning.estimator import MemoryEstimatorDetection
from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001


BYTES_PER_MB = 1024**2


class LBDBoxC002(BoxC002):
    """Run nnDetection's architecture and anchor algorithms on FRAN label-bounded crops: batch 4, the patch adapts."""

    @staticmethod
    def trial_model_cfg(configs):  #AI
        """nnDetection model config for BoxC002's trial networks, from the det sheets."""
        model = configs["model_params"]
        model_cfg, _ = RetinaUNetManager.nndet_cfgs(configs)
        start_channels = int(model["encoder_start_channels"])
        model_cfg["plan_arch_overwrites"].update(
            dim=int(model["spatial_dims"]),
            in_channels=int(model["in_channels"]),
            start_channels=start_channels,
            max_channels=int(model["encoder_max_channels"]),
            fpn_channels=start_channels * 4,
            head_channels=start_channels * 4,
        )
        return model_cfg

    @classmethod
    def from_lbd(cls, configs, lbd_folder, details, spacing, gpu, index):  #AI
        """Planner for the selected LBD cases: boxes in nnDetection layout, model config from the det sheets, memory budget from `gpu` (GPU `index`)."""
        cases = []
        for row in details.itertuples(index=False):
            boxes, labels = lbd_box_records(lbd_folder, row)
            cases.append((to_nndet(boxes).numpy(), labels, details_shape(row)))
        estimator = MemoryEstimatorDetection(target_mem=gpu["budget"], gpu_id=index)
        planner = cls(cases, spacing, cls.trial_model_cfg(configs), estimator, configs["model_params"]["in_channels"])
        planner.network_cls = RetinaUNetManager.network_class(
            configs["plan_train"], foundation="nnfoundation" in configs
        )
        return planner

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
        device = check_planning_gpu("BoxC002", index)
        free, total = torch.cuda.mem_get_info(index)
        free_mb = free // BYTES_PER_MB
        target_mem = self.estimator.target_mem
        cprint(
            f"BoxC002 planning on GPU {index} ({device.name}) with {free_mb} MB free "
            f"of {total // BYTES_PER_MB} MB; budget {target_mem / BYTES_PER_MB:g} MB (free memory minus {plan_gpu_buffer_mb()} MB buffer)",
            bold=True,
        )
        sys.stdout.flush()
        return {
            "target_memory_bytes": target_mem,
            "gpu_name": device.name,
            "gpu_free_memory_mb": free_mb,
            "torch_version": str(torch.__version__),
        }

    def _prepare(self):  #AI
        """BoxC002's default settings (batch 4), the box properties and the class weights."""
        self.create_default_settings()
        self.process_properties()
        self.architecture_kwargs["class_weight"] = self.compute_class_weights()

    def fit_lbd(self):  #AI
        """nnDetection's patch, batch (4) and model, with anchors, for the LBD crops."""
        self._prepare()
        median_shape = np.median([shape for _, _, shape in self.cases], axis=0)
        gpu_metadata = self._check_planning_gpu()
        patch = [int(value) for value in self._plan_architecture(self.spacing, median_shape, self.transpose_forward)]
        anchors = self._plan_anchors(target_spacing_transposed=self.spacing, transpose_forward=self.transpose_forward)
        model = {
            "conv_kernels": self.architecture_kwargs["conv_kernels"],
            "strides": self.architecture_kwargs["strides"],
            "decoder_levels": self.architecture_kwargs["decoder_levels"],
            "anchors": anchors,
        }
        fit = {
            "plan_train": {"patch_size": patch, "batch_size": int(self.batch_size), "model": model},
            "fit": {"class_weight": self.architecture_kwargs["class_weight"], "estimator": gpu_metadata},
        }
        return fit

    def memory_estimate(self, patch, batch, model):  #AI
        """Bytes nnDetection's estimator measures for training the fixed `model` (kernels, strides, decoder levels, anchors) on `patch` at `batch`."""
        self._prepare()
        self.architecture_kwargs.update(
            conv_kernels=model["conv_kernels"], strides=model["strides"], decoder_levels=tuple(model["decoder_levels"])
        )
        self.estimator.batch_size = batch
        estimate, _ = self.estimator.estimate(
            min_shape=np.prod(model["strides"], axis=0),
            target_shape=patch,
            in_channels=self.architecture_kwargs["in_channels"],
            network=self.network_cls.from_config_plan(
                model_cfg=self.model_cfg, plan_arch=self.architecture_kwargs, plan_anchors=model["anchors"]
            ),
            optimizer_cls=torch.optim.Adam,
            num_instances=self._estimte_num_instances_per_patch(
                patch_size=patch, target_spacing_transposed=self.spacing, transpose_forward=self.transpose_forward
            ),
        )
        return estimate


def fit_plan(project, maker, plan, index, refit):  #AI
    """det3d, at training start on GPU `index`: BoxC002 fit of the row's linked `auto` keys into the plan file; all-fixed rows get a memory estimate.

    lbd plans only. One fit is kept per GPU memory class (`FitStore`): a changed row or changed LBD crops and boxes stop
    the run until `refit`. A fit refuses a busy GPU.
    """
    row = maker.read_plan(plan["plan_name"])  # the sheet row, never a loaded fit
    linked = LinkedConfigKeys(row)
    if plan["mode"] != "lbd":
        if linked.to_fit:
            raise ValueError(f"Plan {plan['plan_name']}: {linked.to_fit} are fitted by BoxC002 in mode lbd only, not {plan['mode']}")
        cprint(f"plan {plan['plan_name']} ({maker.PIPELINE}): no memory estimate in mode {plan['mode']} (BoxC002 reads lbd crops)", color="yellow")
        return
    lbd_folder = FolderNames(project, plan).lbd_folder
    details = selected_lbd_details(lbd_folder, [case_id for _, case_id in plan_cases(project, plan)])
    gpu = planning_budget("det", index=index)
    planner = LBDBoxC002.from_lbd(maker.configs, lbd_folder, details, plan["spacing"], gpu, index)
    if not linked.to_fit:
        check_fixed_memory(
            plan["plan_name"], maker.PIPELINE, row["patch_size"], row["batch_size"], gpu,
            lambda patch, batch: planner.memory_estimate(patch, batch, row["model"]),
        )
        return

    def fit():
        fitted = planner.fit_lbd()
        plan_train = {key: fitted["plan_train"][key] for key in linked.to_fit}
        plan_train["val_patch_size"] = maker.resolved_val_patch_size(row, plan_train["patch_size"])
        return plan_train, fitted["fit"]

    store = FitStore(project, plan["plan_name"], maker.PIPELINE)
    fit_row = {key: row[key] for key in linked.to_fit}
    if planner.network_cls is not RetinaUNetV001:
        fit_row["planner"] = "resenc"
    entry, fitted_now = store.fit(fit_row, gpu["gpu_key"], lbd_input_key(lbd_folder, details), fit, refit, index)
    linked.report(plan["plan_name"], maker.PIPELINE, entry["plan_train"])
    append_center_ranges(lbd_folder, entry["plan_train"]["patch_size"], plan["prezoom_scale"], fitted_now)
