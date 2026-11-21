"""nnDetection's BoxC002 architecture planner on FRAN label-bounded crops, with the plan's presets."""

import sys
from collections import Counter

import numpy as np
import torch
from utilz.cprint import cprint

from fran.configs.parser import ConfigMaker
from fran.configs.defaults import plan_gpu_buffer_mb
from fran.configs.helpers import is_unspecified
from fran.preprocessing.dataset_details_lbd import details_shape, lbd_box_records
from fran.preprocessing.plan_budget import check_planning_gpu, planning_budget

from det3d.managers.retinaunet import RetinaUNetManager
from det3d.geometry.nndet_boxes import to_nndet

import nndet.compat  # noqa: F401
from nndet.planning.architecture.boxes.c002 import BoxC002
from nndet.planning.estimator import MemoryEstimatorDetection
from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001


MAX_DATASET_COVERED = 0.05  # nnU-Net ExperimentPlanner: one batch sees at most 5% of the dataset's voxels
BYTES_PER_MB = 1024**2


class LBDBoxC002(BoxC002):
    """Run nnDetection's architecture and anchor algorithms on FRAN label-bounded crops, honouring the plan's presets."""

    @staticmethod
    def planning_gpu():  #AI
        """The card BoxC002 plans on: fran's det budget (free memory of GPU 0 minus the buffer)."""
        gpu = planning_budget("det")
        return gpu

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
    def from_lbd(cls, configs, lbd_folder, details, spacing, gpu):  #AI
        """Planner for the selected LBD cases: boxes in nnDetection layout, model config from the det sheets, memory budget from `gpu`."""
        cases = []
        for row in details.itertuples(index=False):
            boxes, labels = lbd_box_records(lbd_folder, row)
            cases.append((to_nndet(boxes).numpy(), labels, details_shape(row)))
        estimator = MemoryEstimatorDetection(target_mem=gpu["budget"], gpu_id=0)
        planner = cls(
            cases, spacing, cls.trial_model_cfg(configs), estimator, configs["model_params"]["in_channels"], configs["plan_train"]
        )
        return planner

    def __init__(self, cases, spacing, model_cfg, estimator, in_channels, plan):
        self.cases = cases
        self.spacing = np.asarray(spacing)
        self.model_cfg = model_cfg
        self.estimator = estimator
        self.in_channels = in_channels
        self.network_cls = RetinaUNetV001
        self.dim = 3
        self.transpose_forward = [0, 1, 2]
        self.architecture_kwargs = {}
        self.preset_patch = None if is_unspecified(plan["patch_size"]) else [int(v) for v in plan["patch_size"]]
        self.preset_batch = None if is_unspecified(plan["batch_size"]) else int(plan["batch_size"])
        self.preset_model = None if is_unspecified(plan["model"]) else plan["model"]

    def create_default_settings(self):  #AI
        super().create_default_settings()
        self.patch_was_shrunk = False
        if self.preset_batch is not None:
            self.batch_size = self.preset_batch

    def _decrease_patch_size(self, *args, **kwargs):  #AI
        """Record that memory forced the patch below its cap, then shrink as nnDetection does."""
        self.patch_was_shrunk = True
        patch = super()._decrease_patch_size(*args, **kwargs)
        return patch

    def _fits_at_batch(self, batch_size, patch):  #AI
        """Measure on the GPU whether the planned network and patch fit the memory target at this batch."""
        _, _, _, _, divisor = self.plan_pool_and_conv_pool_late(patch, self.spacing)
        self.estimator.batch_size = batch_size
        _, fits = self.estimator.estimate(
            min_shape=divisor,
            target_shape=patch,
            in_channels=self.architecture_kwargs["in_channels"],
            network=self.network_cls.from_config_plan(
                model_cfg=self.model_cfg,
                plan_arch=self.architecture_kwargs,
                plan_anchors=self.get_anchors_for_estimation(),
            ),
            optimizer_cls=torch.optim.Adam,
            num_instances=self._estimte_num_instances_per_patch(
                patch_size=patch,
                target_spacing_transposed=self.spacing,
                transpose_forward=self.transpose_forward,
            ),
        )
        return fits

    def _grow_batch(self, patch):  #AI
        """nnU-Net rule: once the patch is at its cap, raise the batch while it still fits, up to 5% dataset coverage."""
        total_voxels = sum(int(np.prod(shape)) for _, _, shape in self.cases)
        cap = max(self.batch_size, int(MAX_DATASET_COVERED * total_voxels / np.prod(patch)))
        start = self.batch_size
        while self.batch_size < cap and self._fits_at_batch(self.batch_size + 1, patch):
            self.batch_size += 1
        self.estimator.batch_size = self.batch_size
        if self.batch_size != start:
            cprint(f"BoxC002: patch {list(patch)} is at its cap; batch raised {start} -> {self.batch_size} (cap {cap})", color="green", bold=True)

    def _plan_architecture(self, target_spacing_transposed, target_median_shape_transposed, transpose_forward, **kwargs):  #AI
        """nnDetection's patch fit, extended to the plan's presets.

        Patch auto: nnDetection shrinks the patch until it fits; if it did not shrink and the batch is auto, the batch
        grows (nnU-Net rule). Patch fixed: the patch is kept and only an auto batch is fitted to it. Both fixed: no change.
        """
        if self.preset_patch is None:
            patch = super()._plan_architecture(
                target_spacing_transposed, target_median_shape_transposed, transpose_forward, **kwargs
            )
            if self.preset_batch is None and not self.patch_was_shrunk:
                self._grow_batch(patch)
            return patch
        _, pooling, convs, patch, _ = self.plan_pool_and_conv_pool_late(
            np.asarray(self.preset_patch), target_spacing_transposed
        )
        ConfigMaker.patch_size_stride_compat(self.preset_patch, pooling)
        patch = np.asarray(self.preset_patch)
        num_resolutions = len(convs)
        decoder_levels_start = min(max(1, num_resolutions - self.num_decoder_level), self.min_decoder_level)
        self.architecture_kwargs.update(
            conv_kernels=convs, strides=pooling, decoder_levels=tuple(range(decoder_levels_start, num_resolutions))
        )
        if self.preset_batch is None:
            while self.batch_size > 1 and not self._fits_at_batch(self.batch_size, patch):
                self.batch_size -= 1
            self._grow_batch(patch)
        return patch

    def plan_pool_and_conv_pool_late(self, patch_size, spacing):  #AI
        """Use a preset model's strides and kernels; round the patch down to their total stride."""
        if self.preset_model is None:
            return super().plan_pool_and_conv_pool_late(patch_size, spacing)
        strides = np.asarray(self.preset_model["strides"])
        divisor = np.prod(strides, axis=0)
        if self.preset_patch is not None:
            ConfigMaker.patch_size_stride_compat(patch_size, strides)
        patch = (np.asarray(patch_size) // divisor) * divisor
        if (patch == 0).any():
            raise ValueError(f"patch_size {list(patch_size)} is smaller than the preset model's total stride {list(divisor)}")
        num_pool = (strides > 1).sum(axis=0)
        return list(num_pool), self.preset_model["strides"], self.preset_model["conv_kernels"], patch, divisor

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

    def fit_lbd(self):  #AI
        """Resolve patch, batch and model for the plan; presets are kept unless GPU memory or strides force a change."""
        self.create_default_settings()
        self.process_properties()
        class_weight = self.compute_class_weights()
        self.architecture_kwargs["class_weight"] = class_weight
        median_shape = np.median([shape for _, _, shape in self.cases], axis=0)
        gpu_metadata = self._check_planning_gpu()
        patch = [int(value) for value in self._plan_architecture(self.spacing, median_shape, self.transpose_forward)]
        if self.preset_model is None:
            anchors = self._plan_anchors(
                target_spacing_transposed=self.spacing, transpose_forward=self.transpose_forward
            )
            model = {
                "conv_kernels": self.architecture_kwargs["conv_kernels"],
                "strides": self.architecture_kwargs["strides"],
                "decoder_levels": self.architecture_kwargs["decoder_levels"],
                "anchors": anchors,
            }
        else:
            model = self.preset_model
        fit = {
            "plan_train": {"patch_size": patch, "batch_size": int(self.batch_size), "model": model, "fg_labels": self.labels},
            "fit": {"class_weight": class_weight, "estimator": gpu_metadata},
        }
        return fit
