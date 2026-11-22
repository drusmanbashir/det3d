from pathlib import Path

import torch.nn as nn
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from fran.architectures.nnfoundation import load_nnfoundation_encoder
from fran.architectures.nnunet import _RESENC_L_BLOCKS_PER_STAGE_ENCODER
from fran.managers.project import COMMON_PATHS
from nndet.arch.encoder.abstract import AbstractEncoder
from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001


class ResEncEncoder(AbstractEncoder):
    """nnU-Net residual encoder on the plan's kernels and strides, in nnDetection's encoder interface."""

    def __init__(self, in_channels, conv_kernels, strides):  #AI
        super().__init__()
        n_stages = len(conv_kernels)
        self.strides = [[1] * len(conv_kernels[0])] + [list(s) for s in strides]
        self.out_channels = [min(320, 32 * 2**i) for i in range(n_stages)]
        self.encoder = ResidualEncoder(
            input_channels=in_channels,
            n_stages=n_stages,
            features_per_stage=self.out_channels,
            conv_op=nn.Conv3d,
            kernel_sizes=conv_kernels,
            strides=self.strides,
            n_blocks_per_stage=_RESENC_L_BLOCKS_PER_STAGE_ENCODER[:n_stages],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            return_skips=True,
        )
        self.arch_kwargs = {"n_stages": n_stages, "kernel_sizes": conv_kernels}

    def forward(self, x):
        return self.encoder(x)

    def get_channels(self):
        return list(self.out_channels)

    def get_strides(self):  #AI
        out = [list(self.strides[0])]
        for stride in self.strides[1:]:
            out.append([prev * s for prev, s in zip(out[-1], stride)])
        return out


class RetinaUNetV001NNF(RetinaUNetV001):
    """RetinaUNetV001 with the residual encoder; decoder, heads, anchors and losses are nnDetection's."""

    @classmethod
    def _build_encoder(cls, plan_arch, model_cfg):
        encoder = ResEncEncoder(plan_arch["in_channels"], plan_arch["conv_kernels"], plan_arch["strides"])
        return encoder


def load_nnfoundation_weights(module, configs):  #AI
    """Copy nnFoundationCNN stem and encoder weights into the module's residual encoder."""
    checkpoint = Path(COMMON_PATHS["cold_storage_folder"]) / configs["nnfoundation"]["checkpoint"]
    load_nnfoundation_encoder(module.model.encoder, checkpoint)
