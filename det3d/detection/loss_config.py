import torch
from monai.losses import FocalLoss
from monai.losses.giou_loss import BoxGIoULoss


def apply_detector_loss_plan(detector, configs):
    retinanet = configs["retinanet"]
    arch = str(configs["model_params"]["arch"]).lower()
    cls_loss = str(retinanet["cls_loss"]).lower()
    reg_loss = str(retinanet["reg_loss"]).lower()
    if arch == "retinaunet":
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
    loss_params = configs["loss_params"]
    detector.set_atss_matcher(
        num_candidates=int(loss_params["matcher_num_candidates"]),
        center_in_gt=bool(loss_params["matcher_center_in_gt"]),
    )
    detector.set_hard_negative_sampler(
        batch_size_per_image=int(loss_params["sampler_batch_size_per_image"]),
        positive_fraction=float(loss_params["balanced_sampler_pos_fraction"]),
        pool_size=int(loss_params["sampler_pool_size"]),
        min_neg=int(loss_params["sampler_min_neg"]),
    )
