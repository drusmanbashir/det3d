import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from fran.architectures.nnunet import encoder_widths
from fran.configs.helpers import plan_encoder
from det3d.detection.arch.nndet.blocks.basic import StackedConvBlock2
from det3d.detection.arch.nndet.conv import ConvInstanceRelu, Generator
from det3d.detection.arch.nndet.decoder.base import UFPNModular
from det3d.detection.arch.nndet.encoder.modular import Encoder


def plan_model(configs):  #AI
    """The plan's resolved model (conv_kernels, strides, decoder_levels, anchors)."""
    model = configs["plan_train"]["model"]
    if model == "auto":
        raise ValueError("The plan's model is auto and no resolved plan is loaded; run det preprocessing (BoxC002) first.")
    return model


def encoder_abs_strides_from_plan(configs):
    dim = int(configs["model_params"]["spatial_dims"])
    strides = plan_model(configs)["strides"]
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


def build_retinaunet_feature_extractor(configs: dict) -> RetinaUNetFeatureExtractor:
    model_params = configs["model_params"]
    plan = configs["plan_train"]
    model = plan_model(configs)
    start_channels, max_channels = encoder_widths(model_params, plan)
    decoder_levels = tuple(model["decoder_levels"])
    conv = Generator(ConvInstanceRelu, int(model_params["spatial_dims"]))
    if plan_encoder(plan) == "resenc":
        from det3d.detection.nnfoundation_encoder import ResEncEncoder

        encoder = ResEncEncoder(int(model_params["in_channels"]), model["conv_kernels"], model["strides"])
    else:
        encoder = Encoder(
            conv=conv,
            conv_kernels=model["conv_kernels"],
            strides=model["strides"],
            block_cls=StackedConvBlock2,
            in_channels=int(model_params["in_channels"]),
            start_channels=start_channels,
            stage_kwargs=None,
            max_channels=max_channels,
        )
    decoder = UFPNModular(
        conv=conv,
        conv_kernels=model["conv_kernels"],
        strides=encoder.get_strides(),
        in_channels=encoder.get_channels(),
        decoder_levels=decoder_levels,
        fixed_out_channels=start_channels * 4,
        upsampling_mode="transpose",
    )
    return RetinaUNetFeatureExtractor(encoder, decoder, decoder_levels)
