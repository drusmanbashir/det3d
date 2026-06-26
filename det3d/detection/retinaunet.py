"""RetinaUNet: vendored nnDetection encoder/UFPN + nnDetection-style 1-conv heads."""

import math

import torch
import torch.nn as nn

from det3d.detection.arch.nndet.conv import ConvGroupRelu, Generator
from det3d.detection.retinaunet_network import (
    build_retinaunet_feature_extractor,
    encoder_abs_strides_from_plan,
)


def _head_conv_stack(conv, in_channels, internal_channels, num_convs):
    layers = nn.Sequential()
    layers.add_module(
        "c_in",
        conv(
            in_channels,
            internal_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
    )
    for i in range(num_convs):
        layers.add_module(
            f"c_internal{i}",
            conv(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
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


class RetinaUNet(nn.Module):
    cls_key = "classification"
    box_reg_key = "box_regression"

    def __init__(
        self,
        spatial_dims,
        num_classes,
        num_anchors,
        feature_extractor,
        head_channels,
        size_divisible,
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

    def forward(self, images: torch.Tensor):
        head_maps, _ = self.feature_extractor(images)
        classification = self.classification_head(head_maps)
        box_regression = self.regression_head(head_maps)
        out = {
            self.cls_key: classification,
            self.box_reg_key: box_regression,
        }
        return out


def build_retinaunet(plan: dict, num_anchors: int) -> RetinaUNet:
    feature_extractor = build_retinaunet_feature_extractor(plan)
    size_divisible = encoder_abs_strides_from_plan(plan)[-1]
    head_channels = int(plan.get("encoder_start_channels", 32)) * 4
    return RetinaUNet(
        spatial_dims=int(plan["spatial_dims"]),
        num_classes=len(plan["fg_labels"]),
        num_anchors=int(num_anchors),
        feature_extractor=feature_extractor,
        head_channels=head_channels,
        size_divisible=size_divisible,
        head_num_convs=1,
    )


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

# %%
# SECTION:-------------------- TRAINING --------------------------------------------------------------------------------------
    net = build_retinaunet(conf["plan_train"], num_anchors=3)
    feature_extractor = build_retinaunet_feature_extractor(plan)
    size_divisible = encoder_abs_strides_from_plan(plan)[-1]
    head_channels = int(plan.get("encoder_start_channels", 32)) * 4
    plan["fg_labels"] = [1, 2, 3]
# %%
    R = RetinaUNet(
        spatial_dims=int(plan["spatial_dims"]),
        num_classes=len(plan["fg_labels"]),
        num_anchors=3,
        feature_extractor=feature_extractor,
        head_channels=head_channels,
        size_divisible=size_divisible,
        head_num_convs=1,
    )
# %%
