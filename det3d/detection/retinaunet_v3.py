"""RetinaUNet v3 — network + detector build (single file; TrainerDet / RetinaUNetManagerV3)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from utilz.stringz import ast_literal_eval

from det3d.detection.arch.nndet.blocks.basic import StackedConvBlock2
from det3d.detection.arch.nndet.conv import ConvGroupRelu, ConvInstanceRelu, Generator
from det3d.detection.arch.nndet.decoder.base import UFPNModular
from det3d.detection.arch.nndet.encoder.modular import Encoder
from det3d.detection.loss_config import apply_detector_loss_plan, apply_detector_sampler_plan
from det3d.detection.retinanet_detector2 import RetinaNetDetector2
from fran.configs.helpers import is_excel_None
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape

INFER_OVERLAP = 0.25
DEFAULT_CONV_KERNELS_3D = [[3, 3, 3]] * 5
DEFAULT_STRIDES_3D = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]


# --- segmenter ---


class RetinaUNetSegmenterHead(nn.Module):
    def __init__(self, conv, in_channels: int, seg_classes: int = 2):
        super().__init__()
        self.seg_classes = int(seg_classes)
        self.conv_out = conv(
            in_channels,
            self.seg_classes,
            kernel_size=1,
            padding=0,
            add_norm=None,
            add_act=None,
            bias=True,
        )

    def forward(self, decoder_maps):
        x = decoder_maps[0]
        seg_logits = self.conv_out(x)
        return {"seg_logits": seg_logits}

    def postprocess_for_inference(self, pred_seg):
        logits = pred_seg["seg_logits"]
        pred_seg_softmax = torch.softmax(logits, dim=1)
        return {"pred_seg": pred_seg_softmax}


# --- backbone ---


def _plan_arch(plan: dict) -> dict:
    conv_kernels = plan.get("encoder_conv_kernels", "auto")
    strides = plan.get("encoder_strides", "auto")
    if conv_kernels in (None, "", "auto"):
        conv_kernels = DEFAULT_CONV_KERNELS_3D
    elif isinstance(conv_kernels, str):
        conv_kernels = ast_literal_eval(conv_kernels)
    if strides in (None, "", "auto"):
        strides = DEFAULT_STRIDES_3D
    elif isinstance(strides, str):
        strides = ast_literal_eval(strides)
    start_channels = int(plan.get("encoder_start_channels", 32))
    max_channels = int(plan.get("encoder_max_channels", 320))
    decoder_levels = plan.get("decoder_levels", (1, 2, 3, 4))
    if isinstance(decoder_levels, str):
        decoder_levels = ast_literal_eval(decoder_levels)
    fpn_channels = start_channels * 4
    return {
        "dim": int(plan.get("spatial_dims", 3)),
        "in_channels": int(plan.get("n_input_channels", 1)),
        "start_channels": start_channels,
        "max_channels": max_channels,
        "conv_kernels": conv_kernels,
        "strides": strides,
        "decoder_levels": tuple(decoder_levels),
        "fpn_channels": fpn_channels,
        "head_channels": fpn_channels,
    }


def encoder_abs_strides_from_plan(plan):
    arch = _plan_arch(plan)
    dim = int(plan.get("spatial_dims", 3))
    strides = arch["strides"]
    out_strides = [[1] * dim]
    for stage_id in range(1, len(strides)):
        pool = strides[stage_id - 1]
        if not isinstance(pool, (list, tuple)):
            pool = [pool] * dim
        out_strides.append(
            [int(prev * p) for prev, p in zip(out_strides[stage_id - 1], pool)]
        )
    return out_strides


class RetinaUNetFeatureExtractor(nn.Module):
    def __init__(self, encoder, decoder, decoder_levels):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_levels = list(decoder_levels)
        self.out_channels = decoder.fixed_out_channels

    def _encode(self, images: torch.Tensor):
        return self.encoder(images)

    def _decode(self, feats):
        return self.decoder(feats)

    def forward(self, images: torch.Tensor):
        if self.training:
            feats = checkpoint(self._encode, images, use_reentrant=False)
            fpn_feats = checkpoint(self._decode, feats, use_reentrant=False)
        else:
            feats = self.encoder(images)
            fpn_feats = self.decoder(feats)
        head_maps = [fpn_feats[i] for i in self.decoder_levels]
        return head_maps, fpn_feats


def build_retinaunet_feature_extractor(plan: dict) -> RetinaUNetFeatureExtractor:
    arch = _plan_arch(plan)
    conv = Generator(ConvInstanceRelu, arch["dim"])
    encoder = Encoder(
        conv=conv,
        conv_kernels=arch["conv_kernels"],
        strides=arch["strides"],
        block_cls=StackedConvBlock2,
        in_channels=arch["in_channels"],
        start_channels=arch["start_channels"],
        stage_kwargs=None,
        max_channels=arch["max_channels"],
    )
    decoder = UFPNModular(
        conv=conv,
        conv_kernels=arch["conv_kernels"],
        strides=encoder.get_strides(),
        in_channels=encoder.get_channels(),
        decoder_levels=arch["decoder_levels"],
        fixed_out_channels=arch["fpn_channels"],
        upsampling_mode="transpose",
    )
    return RetinaUNetFeatureExtractor(encoder, decoder, arch["decoder_levels"])


# --- det heads + RetinaUNetV3 ---


def _head_conv_stack(conv, in_channels, internal_channels, num_convs):
    layers = nn.Sequential()
    layers.add_module(
        "c_in",
        conv(in_channels, internal_channels, kernel_size=3, stride=1, padding=1),
    )
    for i in range(num_convs):
        layers.add_module(
            f"c_internal{i}",
            conv(internal_channels, internal_channels, kernel_size=3, stride=1, padding=1),
        )
    return layers


class RetinaUNetClassificationHead(nn.Module):
    def __init__(
        self,
        conv,
        in_channels,
        internal_channels,
        num_anchors,
        num_classes,
        num_convs=1,
        prior_probability=0.01,
    ):
        super().__init__()
        self.conv_internal = _head_conv_stack(
            conv, in_channels, internal_channels, num_convs
        )
        self.conv_out = conv(
            internal_channels,
            num_anchors * num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
            add_norm=False,
            add_act=False,
            bias=True,
        )
        bias_value = -math.log((1 - prior_probability) / prior_probability)
        for layer in self.conv_out.modules():
            if isinstance(layer, (nn.Conv2d, nn.Conv3d)) and layer.bias is not None:
                nn.init.constant_(layer.bias, bias_value)

    def forward(self, feature_maps):
        return [self.conv_out(self.conv_internal(fmap)) for fmap in feature_maps]


class RetinaUNetRegressionHead(nn.Module):
    def __init__(
        self,
        conv,
        in_channels,
        internal_channels,
        num_anchors,
        spatial_dims,
        num_convs=1,
    ):
        super().__init__()
        self.conv_internal = _head_conv_stack(
            conv, in_channels, internal_channels, num_convs
        )
        self.conv_out = conv(
            internal_channels,
            num_anchors * spatial_dims * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            add_norm=False,
            add_act=False,
            bias=True,
        )

    def forward(self, feature_maps):
        return [self.conv_out(self.conv_internal(fmap)) for fmap in feature_maps]


class RetinaUNetV3(nn.Module):
    cls_key = "classification"
    box_reg_key = "box_regression"
    seg_key = "seg_logits"

    def __init__(
        self,
        spatial_dims,
        num_classes,
        num_anchors,
        feature_extractor,
        head_channels,
        size_divisible,
        decoder_channels,
        head_num_convs=1,
    ):
        super().__init__()
        self.spatial_dims = int(spatial_dims)
        self.num_classes = int(num_classes)
        self.num_anchors = int(num_anchors)
        self.feature_extractor = feature_extractor
        self.size_divisible = size_divisible
        conv = Generator(ConvGroupRelu, self.spatial_dims)
        self.classification_head = RetinaUNetClassificationHead(
            conv,
            head_channels,
            head_channels,
            self.num_anchors,
            self.num_classes,
            num_convs=head_num_convs,
        )
        self.regression_head = RetinaUNetRegressionHead(
            conv,
            head_channels,
            head_channels,
            self.num_anchors,
            self.spatial_dims,
            num_convs=head_num_convs,
        )
        self.segmenter = RetinaUNetSegmenterHead(
            conv,
            in_channels=int(decoder_channels[0]),
            seg_classes=2,
        )

    def forward(self, images: torch.Tensor):
        head_maps, all_maps = self.feature_extractor(images)
        pred_seg = self.segmenter(all_maps)
        classification = self.classification_head(head_maps)
        box_regression = self.regression_head(head_maps)
        return {
            self.cls_key: classification,
            self.box_reg_key: box_regression,
            self.seg_key: pred_seg["seg_logits"],
        }


def build_retinaunet_v3(plan: dict, num_anchors: int) -> RetinaUNetV3:
    feature_extractor = build_retinaunet_feature_extractor(plan)
    size_divisible = encoder_abs_strides_from_plan(plan)[-1]
    head_channels = int(plan.get("encoder_start_channels", 32)) * 4
    spatial = int(plan["spatial_dims"])
    probe = torch.zeros(1, int(plan["n_input_channels"]), *([64] * spatial))
    feature_extractor.eval()
    with torch.no_grad():
        _, all_maps = feature_extractor(probe)
    decoder_channels = [int(t.shape[1]) for t in all_maps]
    return RetinaUNetV3(
        spatial_dims=int(plan["spatial_dims"]),
        num_classes=len(plan["fg_labels"]),
        num_anchors=int(num_anchors),
        feature_extractor=feature_extractor,
        head_channels=head_channels,
        size_divisible=size_divisible,
        decoder_channels=decoder_channels,
        head_num_convs=1,
    )


# --- detector factory (TrainerDet entry) ---


def _anchor_generator(plan, feature_levels):
    if plan.get("base_anchor_shapes") is not None:
        shapes = plan["base_anchor_shapes"]
    else:
        shapes = [[6, 8, 4], [8, 6, 5], [10, 10, 6]]
    return AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**level for level in feature_levels],
        base_anchor_shapes=shapes,
    )


def val_patch_size_from_conf(configs):
    val_patch_size = configs["model_params"]["val_patch_size"]
    if isinstance(val_patch_size, str):
        val_patch_size = ast_literal_eval(val_patch_size)
    return [int(v) for v in val_patch_size]


def create_retinaunet_v3_from_conf(plan, script=False, debug=False):
    decoder_levels = plan.get("decoder_levels", (1, 2, 3, 4))
    if isinstance(decoder_levels, str):
        decoder_levels = ast_literal_eval(decoder_levels)
    anchor_generator = _anchor_generator(plan, decoder_levels)
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    net = build_retinaunet_v3(plan, num_anchors)
    if script:
        net = torch.jit.script(net)
    return RetinaNetDetector2(
        network=net, anchor_generator=anchor_generator, debug=debug
    )


def wire_retinaunet_v3_detector(detector, configs, val_patch_size):
    apply_detector_loss_plan(detector, configs)
    apply_detector_sampler_plan(detector, configs)
    plan = configs["plan_train"]
    detector.set_target_keys(box_key="bbox", label_key="label")
    detector.set_box_selector_parameters(
        score_thresh=float(plan.get("score_thresh", 0.02)),
        topk_candidates_per_level=int(plan.get("topk_candidates_per_level", 1000)),
        nms_thresh=float(plan.get("nms_thresh", 0.22)),
        detections_per_img=int(plan["detections_per_img"]),
    )
    detector.set_sliding_window_inferer(
        roi_size=val_patch_size,
        overlap=float(plan.get("infer_overlap", INFER_OVERLAP)),
        sw_batch_size=int(plan.get("infer_sw_batch_size", 1)),
        mode="constant",
        device="cpu",
    )
    return detector


def create_retinaunet_v3_detector(configs, script=False, debug=False):
    """Build RetinaNetDetector2 + val patch size from experiment configs."""
    plan = configs["plan_train"]
    detector = create_retinaunet_v3_from_conf(plan, script=script, debug=debug)
    val_patch_size = val_patch_size_from_conf(configs)
    wire_retinaunet_v3_detector(detector, configs, val_patch_size)
    return detector, val_patch_size


def arch_from_conf(configs) -> str:
    arch = configs["model_params"]["arch"]
    if is_excel_None(arch):
        arch = "retinanet"
    return str(arch).lower()


# %%
if __name__ == '__main__':
#SECTION:-------------------- setup --------------------------------------------------------------------------------------
    from fran.managers import Project
    from torch import Tensor
    from utilz.imageviewers import ImageBBoxViewer

    from det3d.configs.parser import ConfigMakerDet

    project_title = "lidca"
    plan_id = 4

    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = C.configs
    conf["dataset_params"]["fold"] = 0
    plan = conf["plan_train"]
    plan["fg_labels"]= [1,2,3]
    val_patch_size=[128,128,64]
    detector = create_retinaunet_v3_from_conf(plan, script=False, debug=True)

    wire_retinaunet_v3_detector(detector, conf, val_patch_size)
    R = detector
# %%
    head_maps, all_maps = R.feature_extractor(images)
    pred_seg = R.segmenter(all_maps)
    classification = R.classification_head(head_maps)
    box_regression = R.regression_head(head_maps)


