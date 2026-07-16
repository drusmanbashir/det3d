import torch
from copy import deepcopy
from fran.managers.project import Project
from lightning.pytorch import LightningModule

from det3d.managers.det_schedule import configure_detection_optimizers
from det3d.managers.helpers.nndet_retinaunet import (
    build_nndet_retinaunet_module,
    det3d_batch_to_nndet,
    fast_nndet_batch_to_device,
    log_nndet_det3d_step_losses,
    NNDET_LOSS_LOG_NAMES,
    nndet_pred_to_vis,
)
from det3d.utils.tensor import sanitize_tensor_for_numpy, to_numpy


# %%
class RetinaUNetManager(LightningModule):
    """nnDetection RetinaUNetV001 on det3d DataManager batches."""

    def __init__(self, project_title, configs, lr=None, sync_dist=False):
        super().__init__()
        self.sync_dist = sync_dist
        self.project = Project(project_title)
        configs = deepcopy(configs)
        if configs.get("mnemonic") is None:
            configs["mnemonic"] = self.project.global_properties["mnemonic"]
        model_params = dict(configs["model_params"])
        if "max_epochs" not in model_params:
            model_params["max_epochs"] = 500
        configs["model_params"] = model_params
        self.save_hyperparameters("project_title", "configs", "lr")
        self.configs = configs
        self.plan = configs["plan_train"]
        self.model_params = configs["model_params"]
        self.lr = float(lr if lr is not None else self.model_params["lr"])
        self.class_names = [self.plan["class_name"]]
        self.nndet_module, self.nndet_plan = build_nndet_retinaunet_module(self.configs)
        self.val_loss_sums = {}
        self.val_loss_count = 0
        self._val_patch_stream = False
        self._nndet_wandb_grid_val_batches = []

    @property
    def net(self):
        return self.nndet_module.model

    def step_losses(self, batch, batch_idx, evaluation=False):
        nb = det3d_batch_to_nndet(batch, self.plan["fg_labels"])
        nb = fast_nndet_batch_to_device(nb, batch["image"].device)
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

    def maybe_store_grid_preds(self, batch, batch_idx):
        if not (hasattr(self.trainer, "store_preds") and self.trainer.store_preds):
            return
        with torch.no_grad():
            _, prediction, _ = self.step_losses(batch, batch_idx, evaluation=True)
            store_batch_grid_preds(self, batch, prediction)

    def training_step(self, batch, batch_idx):
        losses, _, _ = self.step_losses(batch, batch_idx, evaluation=False)
        total = log_nndet_det3d_step_losses(self, losses, "train0", self.sync_dist)
        self.maybe_store_grid_preds(batch, batch_idx)
        return total

    def on_validation_epoch_start(self):
        self.val_loss_sums = {}
        self.val_loss_count = 0
        self._val_patch_stream = False
        self.nndet_module.box_evaluator.reset()
        self.nndet_module.seg_evaluator.reset()

    def validation_step(self, batch, batch_idx):
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
            log_key = NNDET_LOSS_LOG_NAMES.get(key, key)
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
        self.nndet_module.seg_evaluator.run_online_evaluation(
            seg_probs=to_numpy(prediction["pred_seg"]),
            target=to_numpy(nb["target_seg"]),
        )
        if not patch_stream:
            store_batch_grid_preds(self, batch, prediction)

    def on_validation_epoch_end(self):
        n = self.val_loss_count
        for log_key, total in self.val_loss_sums.items():
            self.log(f"val0_{log_key}", total / n, sync_dist=self.sync_dist)
        metric_scores, _ = self.nndet_module.box_evaluator.finish_online_evaluation()
        self.nndet_module.box_evaluator.reset()
        for key, value in metric_scores.items():
            self.log(f"val0_{key}", float(value), sync_dist=self.sync_dist)

        seg_scores, _ = self.nndet_module.seg_evaluator.finish_online_evaluation()
        self.nndet_module.seg_evaluator.reset()
        for key, value in seg_scores.items():
            self.log(f"val0_{key}", float(value), sync_dist=self.sync_dist)

        if self._val_patch_stream:
            val_metric = float(seg_scores["seg_dice"])
        else:
            val_metric = float(metric_scores[self.nndet_module.eval_score_key])
        self.log("val0_metric", val_metric, prog_bar=True, sync_dist=self.sync_dist)

    def configure_optimizers(self):
        holder = {}
        return configure_detection_optimizers(
            self.net.parameters(),
            self.plan,
            self.lr,
            holder,
            monitor=self.plan["scheduler_monitor"],
        )


def batch_pred_to_vis_list(pred):
    #AI
    n = len(pred["pred_boxes"])
    vis_preds = []
    for b in range(n):
        item = {
            "pred_boxes": [pred["pred_boxes"][b]],
            "pred_labels": [pred["pred_labels"][b]],
            "pred_scores": [pred["pred_scores"][b]],
        }
        if "pred_seg" in pred:
            item["pred_seg"] = pred["pred_seg"][b : b + 1]
        vis = nndet_pred_to_vis(item)
        vis_preds.append(
            {
                k: sanitize_tensor_for_numpy(v) if isinstance(v, torch.Tensor) else v
                for k, v in vis.items()
            }
        )
    return vis_preds


def store_batch_grid_preds(pl_module, batch, preds):
    #AI
    if isinstance(preds, list):
        batch["pred"] = [
            {
                k: sanitize_tensor_for_numpy(v) if isinstance(v, torch.Tensor) else v
                for k, v in p.items()
            }
            for p in preds
        ]
    else:
        batch["pred"] = batch_pred_to_vis_list(preds)
    pl_module._nndet_wandb_grid_val_batches.append(batch)


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
