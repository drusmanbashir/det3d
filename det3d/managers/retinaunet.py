from copy import deepcopy
from pathlib import Path

import nndet
import nndet.compat  # noqa: F401  nnDetection's runtime shims, as its own scripts/train.py imports them
import numpy as np
import torch
import yaml
from fran.managers.project import Project
from lightning.pytorch import LightningModule
from nndet.evaluator.seg import SegmentationEvaluator

from det3d.detection.postprocess import postprocess_values
from det3d.managers.det_schedule import configure_detection_optimizers
from det3d.geometry.nndet_boxes import nndet_pred_to_vis, to_nndet
from det3d.utils.tensor import to_numpy


# %%
class RetinaUNetManager(LightningModule):
    """nnDetection RetinaUNetV001 on det3d DataManager batches."""

    NNDET_LOSS_LOG_NAMES = {
        "cls": "cls_loss",
        "reg": "box_reg_loss",
        "seg_ce": "loss_ce",
        "seg_dice": "loss_dice",
    }

    def __init__(self, project_title, configs, lr=None, sync_dist=False):
        super().__init__()
        self.sync_dist = sync_dist
        self.project = Project(project_title)
        configs = deepcopy(configs)
        if configs.get("mnemonic") is None:
            configs["mnemonic"] = self.project.global_properties["mnemonic"]
        self.save_hyperparameters("project_title", "configs", "lr")
        self.configs = configs
        self.plan = configs["plan_train"]
        self.model_params = configs["model_params"]
        self.lr = float(lr if lr is not None else self.model_params["lr"])
        self.class_names = [self.model_params["class_name"]]
        self.nndet_module, self.nndet_plan = self.build_nndet_module(self.configs)
        self.val_loss_sums = {}
        self.val_loss_count = 0
        self._val_patch_stream = False

    @property
    def net(self):
        return self.nndet_module.model

    @staticmethod
    def nndet_cfgs(configs):  #AI
        """nnDetection's V001 model and trainer config, with the det loss sheet applied to the matcher and sampler."""
        train_cfg = yaml.safe_load((Path(nndet.__file__).parent / "conf/train/v001.yaml").read_text())
        model_cfg, trainer_cfg = train_cfg["model_cfg"], train_cfg["trainer_cfg"]
        loss = configs["loss_params"]
        model_cfg["matcher_kwargs"].update(
            num_candidates=int(loss["matcher_num_candidates"]), center_in_gt=bool(loss["matcher_center_in_gt"])
        )
        model_cfg["head_sampler_kwargs"].update(
            batch_size_per_image=int(loss["sampler_batch_size_per_image"]),
            positive_fraction=float(loss["balanced_sampler_pos_fraction"]),
            pool_size=int(loss["sampler_pool_size"]),
            min_neg=int(loss["sampler_min_neg"]),
        )
        return model_cfg, trainer_cfg

    @staticmethod
    def network_class(plan, *, foundation=False):
        if plan.get("planner") == "resenc" or foundation:
            from det3d.detection.nnfoundation_encoder import RetinaUNetV001NNF

            return RetinaUNetV001NNF
        from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001

        return RetinaUNetV001

    @staticmethod
    def build_nndet_module(configs):  #AI
        """nnDetection RetinaUNetV001 and the plan it is built from; an `nnfoundation` config swaps in the pretrained residual encoder."""
        model_cfg, trainer_cfg = RetinaUNetManager.nndet_cfgs(configs)
        plan = RetinaUNetManager.plan_from_det3d(configs)
        foundation = "nnfoundation" in configs
        network_cls = RetinaUNetManager.network_class(configs["plan_train"], foundation=foundation)
        module = network_cls(model_cfg=model_cfg, trainer_cfg=trainer_cfg, plan=plan)
        if foundation:
            from fran.architectures.nnfoundation import load_nnfoundation_encoder, nnfoundation_checkpoint

            load_nnfoundation_encoder(module.model.encoder, nnfoundation_checkpoint(configs))
        module.seg_evaluator = SegmentationEvaluatorTorch.create()
        return module, plan

    @staticmethod
    def plan_from_det3d(configs):
        # AI
        plan_train = configs["plan_train"]
        plan = {
            "architecture": RetinaUNetManager.plan_architecture_from_det3d(configs),
            "anchors": deepcopy(configs["plan_train"]["model"]["anchors"]),
            "patch_size": [int(v) for v in plan_train["patch_size"]],
        }
        return plan

    @staticmethod
    def plan_architecture_from_det3d(configs):
        # AI
        """nnDetection's flat architecture dict, built from the nested configs; the only place these keys are flattened."""
        from det3d.detection.retinaunet_network import plan_model

        plan_train = configs["plan_train"]
        model_params = configs["model_params"]
        model = plan_model(configs)
        postprocess = postprocess_values(configs)
        n_fg = len(plan_train["fg_labels"])
        start_channels = int(model_params["encoder_start_channels"])
        arch = {
            "dim": int(model_params["spatial_dims"]),
            "in_channels": int(model_params["in_channels"]),
            "start_channels": start_channels,
            "max_channels": int(model_params["encoder_max_channels"]),
            "conv_kernels": model["conv_kernels"],
            "strides": model["strides"],
            "decoder_levels": tuple(model["decoder_levels"]),
            "fpn_channels": start_channels * 4,
            "head_channels": start_channels * 4,
            "classifier_classes": n_fg,
            "seg_classes": n_fg,
            "score_thresh": float(postprocess["score_thresh"]),
            "nms_thresh": float(postprocess["nms_thresh"]),
            "detections_per_img": int(postprocess["detections_per_img"]),
            "topk_candidates": int(configs["postprocess"]["topk_candidates_per_level"]),
            "remove_small_boxes": float(configs["postprocess"]["remove_small_boxes"]),
        }
        return arch

    @staticmethod
    def nndet_target_classes(target_boxes, target_classes_raw, fg_labels):
        # AI
        label_to_idx = {int(v): i for i, v in enumerate(fg_labels)}
        out = []
        for boxes, cls in zip(target_boxes, target_classes_raw):
            cls = torch.as_tensor(cls, dtype=torch.long).reshape(-1)
            n = int(boxes.shape[0])
            cls = cls[:n]
            mapped = torch.tensor(
                [label_to_idx[int(v.item())] for v in cls],
                dtype=torch.long,
                device=boxes.device,
            )
            out.append(mapped)
        return out

    @staticmethod
    def det3d_batch_to_nndet(
        batch,
        fg_labels,
        *,
        seg_key="lm",
    ):
        # AI
        data = batch["image"]
        n = int(data.shape[0])
        target_boxes = []
        target_classes_raw = []
        for i in range(n):
            target_boxes.append(to_nndet(batch["bbox"][i]))
            label = torch.as_tensor(batch["label"][i], dtype=torch.long).reshape(-1)
            target_classes_raw.append(label)
        target_classes = RetinaUNetManager.nndet_target_classes(target_boxes, target_classes_raw, fg_labels)
        lm = batch[seg_key]
        if isinstance(lm, list):
            lm = torch.stack(lm, dim=0)
        lm = lm.squeeze(1).long()
        label_to_seg = {int(label): index + 1 for index, label in enumerate(fg_labels)}
        target_seg = torch.zeros_like(lm)
        for label, mapped in label_to_seg.items():
            target_seg[lm == label] = mapped
        out = {
            "data": data,
            "target_boxes": target_boxes,
            "target_classes": target_classes,
            "target_seg": target_seg,
        }
        return out

    @staticmethod
    def fast_nndet_batch_to_device(nb, device):
        # AI
        nb["data"] = nb["data"].to(device)
        nb["target_boxes"] = [b.to(device) for b in nb["target_boxes"]]
        nb["target_classes"] = [c.to(device) for c in nb["target_classes"]]
        nb["target_seg"] = nb["target_seg"].to(device)
        return nb

    def log_step_losses(self, losses, prefix):  #AI
        """Log the summed and per-term nnDetection losses; return the sum."""
        total = sum(losses.values())
        self.log(f"{prefix}_loss", total, prog_bar=(prefix == "train0"), on_step=True, on_epoch=True, sync_dist=self.sync_dist)
        for key, val in losses.items():
            self.log(
                f"{prefix}_{self.NNDET_LOSS_LOG_NAMES.get(key, key)}",
                val,
                on_step=(prefix == "train0"),
                on_epoch=True,
                sync_dist=self.sync_dist,
            )
        return total

    def step_losses(self, batch, batch_idx, evaluation=False):
        nb = self.det3d_batch_to_nndet(batch, self.plan["fg_labels"])
        nb = self.fast_nndet_batch_to_device(nb, batch["image"].device)
        losses, prediction = self.net.train_step(
            images=nb["data"],
            targets={
                "target_boxes": nb["target_boxes"],
                "target_classes": nb["target_classes"],
                "target_seg": nb["target_seg"],
            },
            evaluation=evaluation,
            batch_num=batch_idx,
        )
        return losses, prediction, nb

    def training_step(self, batch, batch_idx):
        store_preds = getattr(self.trainer, "store_preds", False)
        losses, prediction, _ = self.step_losses(batch, batch_idx, evaluation=store_preds)
        if store_preds:
            batch["pred"] = batch_pred_to_vis_list(prediction)
        total = self.log_step_losses(losses, "train0")
        return total

    def on_validation_epoch_start(self):
        self.val_loss_sums = {}
        self.val_loss_count = 0
        self._val_patch_stream = False
        self.nndet_module.box_evaluator.reset()
        self.nndet_module.seg_evaluator.reset()

    def validate_full_volume(self, batch):
        from det3d.inference.retinaunet_nifti import nndet_predict_case

        image = batch["image"][0]
        prediction = nndet_predict_case(
            self.net, self.nndet_plan, self.configs, image, image.device, do_seg=False, native_boxes=True
        )
        limit = postprocess_values(self.configs)["detections_per_img"]
        keep = prediction["pred_scores"][0].argsort(descending=True)[:limit]
        boxes = prediction["pred_boxes"][0][keep]
        scores = prediction["pred_scores"][0][keep]
        labels = prediction["pred_labels"][0][keep]
        targets = self.nndet_target_classes(batch["bbox"], batch["label"], self.plan["fg_labels"])
        self.nndet_module.box_evaluator.run_online_evaluation(
            pred_boxes=to_numpy([boxes]),
            pred_classes=to_numpy([labels]),
            pred_scores=to_numpy([scores]),
            gt_boxes=to_numpy([to_nndet(box) for box in batch["bbox"]]),
            gt_classes=to_numpy(targets),
            gt_ignore=None,
        )
        batch["pred"] = [nndet_pred_to_vis(
            {"pred_boxes": [boxes], "pred_labels": [labels], "pred_scores": [scores]}
        )]

    def validation_step(self, batch, batch_idx):
        if self.configs["dataset_params"].get("valid_impl") == "full_volume":
            self.validate_full_volume(batch)
            return
        patch_stream = batch["validation_impl"] == "patch_stream"
        self._val_patch_stream = self._val_patch_stream or patch_stream
        losses, prediction, nb = self.step_losses(
            batch, batch_idx, evaluation=not patch_stream
        )
        if patch_stream:
            _, _, pred_seg = self.net(nb["data"])
            prediction = {
                "pred_seg": self.net.segmenter.postprocess_for_inference(pred_seg)[
                    "pred_seg"
                ]
            }
        for key, val in losses.items():
            log_key = self.NNDET_LOSS_LOG_NAMES.get(key, key)
            self.val_loss_sums[log_key] = self.val_loss_sums.get(log_key, 0.0) + float(
                val.detach()
            )
        self.val_loss_sums["loss"] = self.val_loss_sums.get("loss", 0.0) + float(
            sum(losses.values()).detach()
        )
        self.val_loss_count += 1
        if not patch_stream:
            self.nndet_module.box_evaluator.run_online_evaluation(
                pred_boxes=to_numpy(prediction["pred_boxes"]),
                pred_classes=to_numpy(prediction["pred_labels"]),
                pred_scores=to_numpy(prediction["pred_scores"]),
                gt_boxes=to_numpy(nb["target_boxes"]),
                gt_classes=to_numpy(nb["target_classes"]),
                gt_ignore=None,
            )
            batch["pred"] = batch_pred_to_vis_list(prediction)
        self.nndet_module.seg_evaluator.run_online_evaluation(
            seg_probs=prediction["pred_seg"],
            target=nb["target_seg"],
        )

    def on_validation_epoch_end(self):
        n = self.val_loss_count
        for log_key, total in self.val_loss_sums.items():
            self.log(f"val0_{log_key}", total / n, sync_dist=self.sync_dist)
        metric_scores, _ = self.nndet_module.box_evaluator.finish_online_evaluation()
        self.nndet_module.box_evaluator.reset()
        for key, value in metric_scores.items():
            self.log(f"val0_{key}", float(value), sync_dist=self.sync_dist)

        val_metric = float(metric_scores[self.nndet_module.eval_score_key])
        if n:
            seg_scores, _ = self.nndet_module.seg_evaluator.finish_online_evaluation()
            self.nndet_module.seg_evaluator.reset()
            for key, value in seg_scores.items():
                self.log(f"val0_{key}", float(value), sync_dist=self.sync_dist)
            if self._val_patch_stream:
                val_metric = float(seg_scores["seg_dice"])
        self.log("val0_metric", val_metric, prog_bar=True, sync_dist=self.sync_dist)

    def configure_optimizers(self):
        holder = {}
        return configure_detection_optimizers(
            self.net,
            self.configs,
            self.lr,
            holder,
            self.trainer,
            monitor=self.configs["scheduler_params"]["scheduler_monitor"],
        )


def batch_pred_to_vis_list(pred):
    #AI
    """Per-case grid predictions as detached views on the device; the grid copies only what it draws."""
    vis_preds = []
    for b in range(len(pred["pred_boxes"])):
        item = {
            "pred_boxes": [pred["pred_boxes"][b].detach()],
            "pred_labels": [pred["pred_labels"][b].detach()],
            "pred_scores": [pred["pred_scores"][b].detach()],
            "pred_seg": pred["pred_seg"][b : b + 1].detach(),
        }
        vis_preds.append(nndet_pred_to_vis(item))
    return vis_preds


class SegmentationEvaluatorTorch(SegmentationEvaluator):
    """nnDetection's global seg dice with tp/fp/fn counted on the device; finish_online_evaluation is nnDetection's."""

    def run_online_evaluation(self, seg_probs, target):  #AI
        output_seg = seg_probs.argmax(dim=1)
        tp, fp, fn = [], [], []
        for c in range(1, seg_probs.shape[1]):
            pred_c = output_seg == c
            true_c = target == c
            tp.append((pred_c & true_c).sum().item())
            fp.append((pred_c & ~true_c).sum().item())
            fn.append((~pred_c & true_c).sum().item())
        self.results_list["tp"].append(np.array(tp, dtype=np.float64))
        self.results_list["fp"].append(np.array(fp, dtype=np.float64))
        self.results_list["fn"].append(np.array(fn, dtype=np.float64))
        return {}


# %%
if __name__ == "__main__":
# SECTION:-------------------- setup --------------------------------------------------------------------------------------
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers import Project

    project_title = "lidca"
    plan_id = 1

    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = C.configs
    conf["dataset_params"]["fold"] = 0
    conf["model_params"]["arch"] = "retinaunet"
    conf["plan_train"]["patch_size"] = [128, 128, 64]
    lr = None
    sync_dist = False

    N = RetinaUNetManager(
        project_title=P.project_title,
        configs=conf,
        lr=lr,
        sync_dist=sync_dist,
    )
# %%
    print(N.net)
    print([int(v) for v in N.plan["patch_size"]])
    print(N.class_names)
    print(N.nndet_plan)
# %%
