"""nnFoundation ResEncL as a det3d RetinaUNet encoder (Step 2, fran/run/nnfoundation/STEP2_RETINAUNET_PLAN.md).

Same interface as `modular.Encoder` (list of stage features, get_channels, get_strides) so the UFPN decoder,
heads, anchors and losses stay unchanged. Architecture = nnU-Net ResEncL encoder: stride-1 stem + 6 residual
stages (features 32,64,128,256,320,320; strides 1,2,4,8,16,32), InstanceNorm3d + LeakyReLU, all kernels 3x3x3
and strides isotropic, so axis order is irrelevant to the weights. Returned levels = [stem, stage0..stage5]
(7 maps, strides 1,1,2,4,8,16,32), mirroring GYRO's encoder whose stack is [1,1,2,4,8]: nnDetection's decoder needs
a level below the first head level, and decoder_levels (1,2,3,4) then give head strides 1,2,4,8 as in GYRO. Only encoder tensors of the pretrained
checkpoint are loaded (the decoder is random, as in upstream `PretrainedTrainer`).
"""

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder

from det3d.detection.arch.nndet.encoder.abstract import AbstractEncoder

__all__ = ["ResEncLEncoder", "load_nnfoundation_encoder"]

FEATURES = (32, 64, 128, 256, 320, 320)  # stage widths (weights); returned maps prepend the stem (32)
N_BLOCKS = (1, 3, 4, 6, 6, 6)
STRIDES = ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2))  # per stage
OUT_CHANNELS = (FEATURES[0],) + FEATURES


class ResEncLEncoder(AbstractEncoder):
    def __init__(self, in_channels: int = 1):  # AI
        super().__init__()
        self.dim = 3
        self.net = ResidualEncoder(
            input_channels=in_channels,
            n_stages=len(FEATURES),
            features_per_stage=list(FEATURES),
            conv_op=nn.Conv3d,
            kernel_sizes=[3] * len(FEATURES),
            strides=[list(s) for s in STRIDES],
            n_blocks_per_stage=list(N_BLOCKS),
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            return_skips=False,
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # AI
        x = self.net.stem(x)
        outs = [x]
        for stage in self.net.stages:
            x = stage(x)
            outs.append(x)
        return outs

    def get_channels(self) -> List[int]:  # AI
        return list(OUT_CHANNELS)

    def get_strides(self) -> List[List[int]]:  # AI
        out, cur = [[1, 1, 1]], [1, 1, 1]
        for s in STRIDES:
            cur = [c * k for c, k in zip(cur, s)]
            out.append(list(cur))
        return out


def load_nnfoundation_encoder(encoder: ResEncLEncoder, ckpt_fn: str | Path) -> dict:  # AI
    """Load `encoder.*` tensors of the nnFoundation checkpoint; strict on the encoder (fail-fast)."""
    ck = torch.load(ckpt_fn, map_location="cpu", weights_only=False)
    sd = ck["network_weights"]
    enc_sd = {k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = encoder.net.load_state_dict(enc_sd, strict=True), None
    return {"n_tensors": len(enc_sd), "result": str(missing)}
