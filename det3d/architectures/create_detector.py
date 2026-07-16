"""Build detection network + RetinaNetDetector2 from configs (fran create_network pattern)."""

import torch
from fran.configs.helpers import is_excel_None
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.networks.nets import resnet
from utilz.stringz import ast_literal_eval

from det3d.detection.loss_config import (
    apply_detector_loss_plan,
    apply_detector_sampler_plan,
)
from det3d.detection.retinanet_detector2 import RetinaNetDetector2
from det3d.detection.retinaunet import build_retinaunet

VAL_PATCH_SIZE = [512, 512, 208]
INFER_OVERLAP = 0.25


def arch_from_conf(configs) -> str:
    arch = configs["model_params"]["arch"]
    if is_excel_None(arch):
        arch = "retinanet"
    return str(arch).lower()


#


def size_divisible_from_conf(configs):
    plan = configs["plan_train"]
    if arch_from_conf(configs) == "retinaunet":
        from det3d.detection.retinaunet_network import encoder_abs_strides_from_plan as _easfp

        return _easfp(plan)[-1]
    return [
        step * 2 * 2 ** max(plan["returned_layers"]) for step in plan["conv1_t_stride"]
    ]


def val_patch_size_from_conf(configs):
    val_patch_size = configs["plan_valid"].get("val_patch_size")
    if is_excel_None(val_patch_size):
        val_patch_size = configs["plan_valid"]["patch_size"]
    if isinstance(val_patch_size, str):
        val_patch_size = ast_literal_eval(val_patch_size)
    return [int(v) for v in val_patch_size]


def _anchor_generator(plan, feature_levels):
    if plan.get("base_anchor_shapes") is not None:
        shapes = plan["base_anchor_shapes"]
    else:
        shapes = [[6, 8, 4], [8, 6, 5], [10, 10, 6]]
    return AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**level for level in feature_levels],
        base_anchor_shapes=shapes,
    )


def create_retinanet_from_conf(plan, script=True, debug=False):
    anchor_generator = _anchor_generator(plan, range(len(plan["returned_layers"]) + 1))
    conv1_t_size = [max(7, 2 * stride + 1) for stride in plan["conv1_t_stride"]]
    backbone = resnet.ResNet(
        block=resnet.ResNetBottleneck,
        layers=[3, 4, 6, 3],
        block_inplanes=resnet.get_inplanes(),
        n_input_channels=int(plan["n_input_channels"]),
        conv1_t_stride=plan["conv1_t_stride"],
        conv1_t_size=conv1_t_size,
    )
    feature_extractor = resnet_fpn_feature_extractor(
        backbone=backbone,
        spatial_dims=int(plan["spatial_dims"]),
        pretrained_backbone=False,
        trainable_backbone_layers=None,
        returned_layers=plan["returned_layers"],
    )
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    size_divisible = [
        step * 2 * 2 ** max(plan["returned_layers"])
        for step in feature_extractor.body.conv1.stride
    ]
    net = RetinaNet(
        spatial_dims=int(plan["spatial_dims"]),
        num_classes=len(plan["fg_labels"]),
        num_anchors=num_anchors,
        feature_extractor=feature_extractor,
        size_divisible=size_divisible,
    )
    if script:
        net = torch.jit.script(net)
    return RetinaNetDetector2(
        network=net, anchor_generator=anchor_generator, debug=debug
    )



def create_retinaunet_from_conf(plan, script=False, debug=False):
    decoder_levels = plan.get("decoder_levels", (1, 2, 3, 4))
    if isinstance(decoder_levels, str):
        decoder_levels = ast_literal_eval(decoder_levels)
    anchor_generator = _anchor_generator(plan, decoder_levels)
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    net = build_retinaunet(plan, num_anchors)
    if script:
        net = torch.jit.script(net)
    return RetinaNetDetector2(
        network=net, anchor_generator=anchor_generator, debug=debug
    )


def wire_detector_from_conf(detector, configs, val_patch_size):
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


def create_detector_from_conf(configs, script=None, debug=False):
    plan = configs["plan_train"]
    arch = arch_from_conf(configs)
    if script is None:
        script = arch == "retinanet"
    if arch == "retinaunet":
        raise ValueError(
            "arch=retinaunet uses RetinaUNetManager via resolve_detector_manager, "
            "not create_detector_from_conf"
        )
    else:
        detector = create_retinanet_from_conf(plan, script=script, debug=debug)
    val_patch_size = val_patch_size_from_conf(configs)
    wire_detector_from_conf(detector, configs, val_patch_size)
    return detector, val_patch_size

