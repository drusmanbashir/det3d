import torch
from det3d.detection.nndet_train import (
    build_nndet_retinaunet_module,
    ensure_nndet_importable,
    forward_patch_size_from_configs,
    maybe_store_batch_grid_preds,
    xyzxyz_exclusive_batch_to_nndet,
)
from det3d.utils.tensor import to_numpy
from det3d.extra.trainer_nndet import (
    apply_det3d_plan_to_nndet_model_cfg,
    load_nndet_train_cfgs,
    plan_from_det3d,
)
from fran.managers.project import Project
from lightning.pytorch import LightningModule


class RetinaUNetManager(LightningModule):
    """nnDetection RetinaUNetV001 on det3d DataManager batches (skip pre_trafo)."""

    def __init__(self, project_title, configs, lr=None, sync_dist=False):
        super().__init__()
        self.sync_dist = sync_dist
        self.project = Project(project_title)
        self.save_hyperparameters("project_title", "configs", "lr")
        self.configs = configs
        self.plan = configs["plan_train"]
        self.lr = float(lr if lr is not None else self.configs["model_params"]["lr"])
        self.class_names = [self.plan["class_name"]]
        self.forward_patch_size = forward_patch_size_from_configs(configs)
        self.val_patch_size = self.forward_patch_size
        self.nndet_module, self.nndet_plan = build_nndet_retinaunet_module(
            configs, num_train_batches=2500
        )
        self.val_loss_sum = 0.0
        self.val_loss_count = 0
        self._val_patch_stream = False

    @property
    def net(self):
        return self.nndet_module.model

    def _nndet_targets(self, batch):
        from monai.data.box_utils import clip_boxes_to_image
        data = batch["image"]
        forward_patch_size = self.plan["patch_size"]
        spatial = tuple(int(v) for v in data.shape[-3:])
        crop_sl = None
        crop_starts = None
        if any(s > p for s, p in zip(spatial, forward_patch_size)):
            crop_starts = tuple(
                max(0, (int(full) - int(ps)) // 2)
                for full, ps in zip(spatial, forward_patch_size)
            )
            crop_sl = tuple(
                slice(st, st + ps)
                for st, ps in zip(crop_starts, forward_patch_size)
            )
            data = data[(..., *crop_sl)]

        label_to_idx = {int(v): i for i, v in enumerate(self.plan["fg_labels"])}
        target_seg_list = []
        target_boxes = []
        target_classes = []
        for i in range(data.shape[0]):
            lm_vol = batch["lm"][i][0].long()
            if crop_sl is not None:
                lm_vol = lm_vol[crop_sl]
            target_seg_list.append(lm_vol)
            box = batch["bbox"][i]
            if crop_starts is not None:
                starts_t = torch.tensor(crop_starts, device=box.device, dtype=box.dtype)
                shifted = box.clone()
                for j in range(3):
                    shifted[:, j] -= starts_t[j]
                    shifted[:, j + 3] -= starts_t[j]
                box, _ = clip_boxes_to_image(
                    shifted, forward_patch_size, remove_empty=True
                )

            nndet_box = xyzxyz_exclusive_batch_to_nndet(box)
            target_boxes.append(nndet_box)

            cls = batch["label"][i][: box.shape[0]]
            mapped = torch.tensor(
                [label_to_idx[int(v.item())] for v in cls],
                dtype=torch.long,
                device=nndet_box.device,
            )
            target_classes.append(mapped)

        out = {
            "data": data,
            "target_boxes": target_boxes,
            "target_classes": target_classes,
            "target_seg": torch.stack(target_seg_list, 0),
        }
        return out

    def _step_losses(self, batch, batch_idx, evaluation=False):
        nb = self._nndet_targets(batch)
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

    def _log_losses(self, losses, prefix):
        cls_seg = losses["cls"] + losses["seg_ce"] + losses["seg_dice"]
        total = sum(losses.values())
        self.log(
            f"{prefix}_cls_seg_loss",
            cls_seg,
            prog_bar=(prefix == "train0"),
            sync_dist=self.sync_dist,
        )
        self.log(f"{prefix}_loss", total, sync_dist=self.sync_dist)
        for key, val in losses.items():
            self.log(f"{prefix}_{key}", val, sync_dist=self.sync_dist)
        return cls_seg

    def on_fit_start(self):
        n = len(self.trainer.datamodule.train_dataloader())
        self.nndet_module.trainer_cfg["num_train_batches_per_epoch"] = int(n)

    def training_step(self, batch, batch_idx):
        losses, _, _ = self._step_losses(batch, batch_idx, evaluation=False)
        return self._log_losses(losses, "train0")

    def on_validation_epoch_start(self):
        self.val_loss_sum = 0.0
        self.val_loss_count = 0
        self._val_patch_stream = False
        self.nndet_module.box_evaluator.reset()
        self.nndet_module.seg_evaluator.reset()

    def validation_step(self, batch, batch_idx):
        patch_stream = batch["validation_impl"] == "patch_stream"
        self._val_patch_stream = self._val_patch_stream or patch_stream
        losses, prediction, nb = self._step_losses(
            batch, batch_idx, evaluation=not patch_stream
        )
        if patch_stream:
            _, _, pred_seg = self.net(nb["data"])
            prediction = {
                "pred_seg": self.net.segmenter.postprocess_for_inference(pred_seg)[
                    "pred_seg"
                ]
            }
        cls_seg = float(
            (losses["cls"] + losses["seg_ce"] + losses["seg_dice"]).detach()
        )
        self.val_loss_sum += cls_seg
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
        maybe_store_batch_grid_preds(self, batch, prediction)

    def on_validation_epoch_end(self):
        val_loss = self.val_loss_sum / self.val_loss_count
        self.log("val0_cls_seg_loss", val_loss, sync_dist=self.sync_dist)
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
        self.nndet_module.trainer_cfg["initial_lr"] = float(self.lr)
        return self.nndet_module.configure_optimizers()


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
    print(N.forward_patch_size)
    print(N.class_names)
# %%

    ensure_nndet_importable()
    from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001

    num_train_batches = 2500

    plan_train = conf["plan_train"]
    plan_path = conf["model_params"].get("nndet_plan_path")
    model_cfg, trainer_cfg = load_nndet_train_cfgs()
    model_cfg = apply_det3d_plan_to_nndet_model_cfg(model_cfg, plan_train)
    trainer_cfg["num_train_batches_per_epoch"] = int(num_train_batches)
    trainer_cfg["max_num_epochs"] = int(conf["model_params"].get("max_epochs", 600))
    plan = plan_from_det3d(plan_train, plan_path=plan_path)
# %%
    module = RetinaUNetV001(
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        plan=plan,
    )

# %%

