import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from utilz.stringz import ast_literal_eval

from det3d.detection.arch.nndet.blocks.basic import StackedConvBlock2
from det3d.detection.arch.nndet.conv import ConvInstanceRelu, Generator
from det3d.detection.arch.nndet.decoder.base import UFPNModular
from det3d.detection.arch.nndet.encoder.modular import Encoder


DEFAULT_CONV_KERNELS_3D = [[3, 3, 3]] * 5
DEFAULT_STRIDES_3D = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]


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
