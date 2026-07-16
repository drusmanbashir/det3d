"""Trace LIDCA-GYRO ckpt load without Lightning — pin build vs state_dict mismatch."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from det3d.detection.retinaunet_network import _plan_arch
from det3d.managers.helpers.nndet_retinaunet import (
    build_nndet_retinaunet_module,
    plan_architecture_from_det3d,
    plan_anchors_from_det3d,
    plan_from_det3d,
)


RUN_NAME = "LIDCA-GYRO"
CKPT = Path("/s/fran_storage/checkpoints/lidca/lidc/LIDCA-GYRO/checkpoints/last.ckpt")
CKPT_BKP = Path(str(CKPT).replace(".ckpt", ".ckpt_bkp"))


def net_state_dict_from_ckpt(state_dict: dict) -> dict:
    for prefix in (
        "nndet_module.model._orig_mod.",
        "nndet_module.model.",
    ):
        out = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if out:
            return out
    raise KeyError("no nndet_module.model.* keys in checkpoint state_dict")


def resolve_ckpt() -> Path:
    if CKPT.exists():
        return CKPT
    if CKPT_BKP.exists():
        return CKPT_BKP
    raise FileNotFoundError(f"no ckpt at {CKPT} or {CKPT_BKP}")


def summarize_plan_arch(label: str, plan: dict) -> None:
    arch = plan["architecture"]
    anchors = plan["anchors"]
    print(f"\n--- {label} ---")
    for k in (
        "conv_kernels",
        "strides",
        "decoder_levels",
        "start_channels",
        "fpn_channels",
        "head_channels",
        "classifier_classes",
        "seg_classes",
    ):
        print(f"  architecture.{k}: {arch.get(k)}")
    print(f"  anchors.sizes: {anchors.get('sizes')}")
    print(f"  anchors.zsizes: {anchors.get('zsizes')}")
    print(f"  patch_size: {plan.get('patch_size')}")


class CkptLoadTracer:
    """Minimal self.* stand-in for TrainerDet.init_dm_unet → load_trainer path."""

    def __init__(self, ckpt_path: Path | None = None):
        self.ckpt_path = ckpt_path or resolve_ckpt()
        self.checkpoint = None
        self.configs = None
        self.state_dict = None
        self.N = None
        self.nndet_module = None
        self.plan = None
        self.plan_pickle = None
        self.plan_det3d_only = None

    def step1_load_checkpoint(self):
        print(f"[1] torch.load({self.ckpt_path})")
        self.checkpoint = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        hp = self.checkpoint["hyper_parameters"]
        self.configs = deepcopy(hp["configs"])
        self.state_dict = self.checkpoint["state_dict"]
        pt = self.configs["plan_train"]
        print(f"    plan_id={pt['plan_id']} fg_labels={pt['fg_labels']}")
        print(f"    nndet_plan_path={self.configs['nndet_plan_path']}")
        print(f"    state_dict keys={len(self.state_dict)}")
        return self

    def step2_ckpt_shapes(self):
        sd = self.state_dict
        probes = [
            "nndet_module.model.encoder.stages.0.convs.0.0.conv.weight",
            "nndet_module.model.decoder.lateral.P0.0.conv.weight",
            "nndet_module.model.head.classifier.conv_out.conv.weight",
        ]
        print("\n[2] checkpoint tensor shapes (weights on disk)")
        for k in probes:
            print(f"    {k}: {tuple(sd[k].shape)}")
        stages = sorted(
            {int(k.split("stages.")[1].split(".")[0]) for k in sd if "encoder.stages." in k}
        )
        levels = sorted(
            {k.split("lateral.")[1].split(".")[0] for k in sd if "decoder.lateral." in k}
        )
        print(f"    encoder stages: {stages}")
        print(f"    decoder levels: {levels}")
        return self

    def step3_plan_sources(self):
        print("\n[3] plan sources — topology input to RetinaUNetV001")
        plan_train = self.configs["plan_train"]
        plan_path = self.configs["nndet_plan_path"]

        from det3d.managers.helpers.nndet_retinaunet import ensure_nndet_importable
        from nndet.io.load import load_pickle

        ensure_nndet_importable()
        self.plan_pickle = load_pickle(plan_path)
        self.plan_det3d_only = plan_from_det3d(plan_train, plan_path=None)
        self.plan = self.plan_det3d_only

        summarize_plan_arch("pickle raw (not used at build)", self.plan_pickle)
        summarize_plan_arch("plan_from_det3d plan_train only (build)", self.plan)

        print("\n    plan_train drives build (pickle ignored):")
        for k in ("encoder_conv_kernels", "encoder_strides", "decoder_levels"):
            print(f"      {k}: {plan_train[k]!r}")
        print(f"      _plan_arch conv_kernels: {_plan_arch(plan_train)['conv_kernels']}")
        return self

    def step4_build_model(self):
        print("\n[4] build_nndet_retinaunet_module(self.configs)")
        self.nndet_module, self.plan = build_nndet_retinaunet_module(self.configs)

        class _DummyManager(nn.Module):
            pass

        self.N = _DummyManager()
        self.N.configs = self.configs
        self.N.nndet_module = self.nndet_module

        model = self.nndet_module.model
        for suffix in (
            "encoder.stages.0.convs.0.0.conv.weight",
            "head.classifier.conv_out.conv.weight",
        ):
            for n, p in model.named_parameters():
                if n.endswith(suffix):
                    print(f"    built {n}: {tuple(p.shape)}")
                    break
        stages = sorted(
            {int(n.split("stages.")[1].split(".")[0]) for n, _ in model.named_parameters() if "encoder.stages." in n}
        )
        print(f"    encoder stages: {stages}")
        return self

    def step5_load_state_dict(self, strict: bool = True):
        print(f"\n[5] self.N.nndet_module.model.load_state_dict(net_sd, strict={strict})")
        print("    same tensors as Lightning; trace loads inner net only")
        net_sd = net_state_dict_from_ckpt(self.state_dict)
        model = self.nndet_module.model
        keys = model.load_state_dict(net_sd, strict=False)
        model_sd = model.state_dict()

        print(f"    missing (model wants, ckpt lacks): {len(keys.missing_keys)}")
        for k in keys.missing_keys[:8]:
            print(f"      MISSING {k} model={tuple(model_sd[k].shape)}")

        mismatches = []
        for k, v in model_sd.items():
            if k not in net_sd:
                continue
            if tuple(v.shape) != tuple(net_sd[k].shape):
                mismatches.append((k, tuple(net_sd[k].shape), tuple(v.shape)))
        print(f"    shape mismatches: {len(mismatches)}")
        for k, cs, ms in mismatches[:10]:
            print(f"      MISMATCH {k}")
            print(f"        ckpt  {cs}")
            print(f"        model {ms}")

        if strict:
            print("\n[5b] strict=True (TrainerDet.load_trainer failure site)")
            try:
                self.N.load_state_dict(ckpt_sd, strict=True)
                print("    PASS")
            except RuntimeError as e:
                print("    FAIL:")
                for line in str(e).split("\n")[:12]:
                    print(f"      {line}")
        return self

    def step6_levers(self):
        sd = self.state_dict
        n_fg = len(self.configs["plan_train"]["fg_labels"])
        ckpt_cls = sd["nndet_module.model.head.classifier.conv_out.conv.weight"].shape[0]
        model_cls = self.nndet_module.model.head.classifier.conv_out.conv.weight.shape[0]
        print("\n[6] mismatch summary")
        print(f"    ckpt classifier={ckpt_cls} (~{ckpt_cls // n_fg} anchors/class)")
        print(f"    built classifier={model_cls} (~{model_cls // n_fg} anchors/class)")
        if ckpt_cls == model_cls:
            print("    classifier channels match — strict load should pass")
        else:
            print("    classifier still mismatched — check anchors / decoder_levels")
        print("    build path: plan_architecture_from_det3d → _plan_arch(plan_train)")
        print("    verify: nndet_retinaunet.py __main__ BEFORE vs AFTER strict load")
        return self

    def run(self, strict: bool = True):
        return (
            self.step1_load_checkpoint()
            .step2_ckpt_shapes()
            .step3_plan_sources()
            .step4_build_model()
            .step5_load_state_dict(strict=strict)
            .step6_levers()
        )


if __name__ == "__main__":
# SECTION:--- config ---
    ckpt_path = None  # default LIDCA-GYRO last.ckpt
    strict = True

# SECTION:--- run trace ---
    self = CkptLoadTracer(ckpt_path=ckpt_path)
    self.run(strict=strict)
