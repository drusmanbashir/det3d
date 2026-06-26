"""nnDetection DiCE segmenter head (forward only; losses in det3d/evaluation/losses.py)."""

import torch
import torch.nn as nn


class RetinaUNetSegmenterHead(nn.Module):
    """DiCESegmenterFgBg topology: finest decoder map -> 1x1 logits."""

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
        out = {"seg_logits": seg_logits}
        return out

    def postprocess_for_inference(self, pred_seg):
        logits = pred_seg["seg_logits"]
        pred_seg_softmax = torch.softmax(logits, dim=1)
        out = {"pred_seg": pred_seg_softmax}
        return out
