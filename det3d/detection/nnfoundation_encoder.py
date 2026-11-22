import torch.nn as nn
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from fran.architectures.nnunet import resenc_encoder_kwargs
from nndet.arch.encoder.abstract import AbstractEncoder
from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001


class ResEncEncoder(AbstractEncoder):
    """FRAN's ResEnc-L encoder on the plan's kernels and strides, in nnDetection's encoder interface."""

    def __init__(self, in_channels, conv_kernels, strides):  #AI
        super().__init__()
        self.strides = [[1] * len(conv_kernels[0])] + [list(s) for s in strides]
        layers = resenc_encoder_kwargs(nn.Conv3d, len(conv_kernels))
        self.out_channels = list(layers["features_per_stage"])
        self.encoder = ResidualEncoder(
            input_channels=in_channels, kernel_sizes=conv_kernels, strides=self.strides, return_skips=True, **layers
        )
        self.arch_kwargs = {"n_stages": layers["n_stages"], "kernel_sizes": conv_kernels}

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
