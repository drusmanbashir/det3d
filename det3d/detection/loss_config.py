import torch
from monai.losses import FocalLoss
from monai.losses.giou_loss import BoxGIoULoss


def apply_detector_loss_plan(detector, configs):
    plan = configs["plan_train"]
    arch = str(configs["model_params"]["arch"]).lower()
    cls_loss = str(plan.get("cls_loss", "bce")).lower()
    reg_loss = str(plan.get("reg_loss", "smooth_l1")).lower()
    if arch in ("retinaunet", "retinaunet_v3"):
        reg_loss = "giou"

    if cls_loss == "focal":
        detector.set_cls_loss(FocalLoss(reduction="mean", gamma=2.0))
    else:
        detector.set_cls_loss(torch.nn.BCEWithLogitsLoss(reduction="mean"))

    if reg_loss == "giou":
        detector.set_box_regression_loss(BoxGIoULoss(reduction="mean"), encode_gt=False, decode_pred=True)
    else:
        detector.set_box_regression_loss(
            torch.nn.SmoothL1Loss(beta=1.0 / 9, reduction="mean"),
            encode_gt=True,
            decode_pred=False,
        )


def apply_detector_sampler_plan(detector, configs):
    plan = configs["plan_train"]
    arch = str(configs["model_params"]["arch"]).lower()
    if arch in ("retinaunet", "retinaunet_v3"):
        batch_size_per_image = 32
        positive_fraction = 0.33
        min_neg = 1
    else:
        batch_size_per_image = int(plan.get("sampler_batch_size_per_image", 64))
        positive_fraction = float(plan.get("balanced_sampler_pos_fraction", 0.3))
        min_neg = int(plan.get("sampler_min_neg", 16))
    detector.set_atss_matcher(
        num_candidates=int(plan.get("matcher_num_candidates", 4)),
        center_in_gt=bool(plan.get("matcher_center_in_gt", False)),
    )
    detector.set_hard_negative_sampler(
        batch_size_per_image=batch_size_per_image,
        positive_fraction=positive_fraction,
        pool_size=int(plan.get("sampler_pool_size", 20)),
        min_neg=min_neg,
    )
