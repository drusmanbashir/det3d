"""RetinaUNetV001 with the nnFoundation ResEncL encoder (Step 2, fran/run/nnfoundation/STEP2_RETINAUNET_PLAN.md).

Only `_build_encoder` differs: decoder (UFPN), detection heads, segmenter, anchors, matcher, sampler and losses are
inherited unchanged from nnDetection's RetinaUNetV001 as configured by det3d for LIDCA-GYRO.
"""

from pathlib import Path

from loguru import logger

from det3d.detection.arch.nndet.encoder.resenc import ResEncLEncoder, load_nnfoundation_encoder
from det3d.managers.helpers.nndet_retinaunet import ensure_nndet_importable

ensure_nndet_importable()
from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001  # noqa: E402


class RetinaUNetV001ResEnc(RetinaUNetV001):  # AI
    @classmethod
    def _build_encoder(cls, plan_arch: dict, model_cfg: dict):  # AI
        encoder = ResEncLEncoder(plan_arch["in_channels"])
        fn = plan_arch.get("encoder_pretrained")
        if fn and Path(fn).is_file():
            info = load_nnfoundation_encoder(encoder, fn)
            logger.info(f"nnFoundation encoder weights loaded from {fn}: {info}")
        elif fn:
            logger.warning(f"encoder_pretrained {fn} not found: encoder left at random init")
        return encoder
