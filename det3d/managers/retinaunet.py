import torch
from det3d.detection.nndet_train import (
    _batch_pre_for_semantic,
    _center_crop_spatial,
    _crop_boxes_to_patch,
    _nndet_target_classes,
    det3d_semantic_target_seg_from_batch,
    disk_bbox_to_nndet_xyxyzz,
    ensure_nndet_importable,
    nndet_batch_pred_to_vis_list,
)
from det3d.utils.tensor import to_numpy
from fran.managers.project import Project
from lightning.pytorch import LightningModule
from utilz.stringz import ast_literal_eval


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
        self.forward_patch_size = self._forward_patch_size_from_configs(configs)
        self.val_patch_size = self.forward_patch_size
        self.nndet_module, self.nndet_plan = self._build_nndet_module(
            configs, num_train_batches=2500
        )
        self.val_loss_sum = 0.0
        self.val_loss_count = 0
        self._val_patch_stream = False

    @property
    def net(self):
        return self.nndet_module.model

    def _forward_patch_size_from_configs(self, configs):
        fps = configs["model_params"].get("nndet_forward_patch_size")
        if fps is None:
            fps = configs["plan_train"]["patch_size"]
        if fps is None:
            return None
        if isinstance(fps, str):
            fps = ast_literal_eval(fps)
        return [int(v) for v in fps]

    def _build_nndet_module(self, configs, num_train_batches):
        from pathlib import Path
        ensure_nndet_importable()
        from nndet.ptmodule.retinaunet.v001 import RetinaUNetV001
        from det3d.configs.parser import resolve_nndet_plan_path
        from det3d.extra.trainer_nndet import (
            apply_det3d_plan_to_nndet_model_cfg,
            load_nndet_train_cfgs,
            plan_from_det3d,
        )

        plan_train = configs["plan_train"]
        plan_path = configs["model_params"].get("nndet_plan_path")
        if plan_path is None:
            plan_path = resolve_nndet_plan_path(
                configs["mnemonic"], Path(configs["configurations_dir"])
            )
        else:
            plan_path = Path(plan_path)
        if not plan_path.is_file():
            raise FileNotFoundError(plan_path)
        model_cfg, trainer_cfg = load_nndet_train_cfgs()
        model_cfg = apply_det3d_plan_to_nndet_model_cfg(model_cfg, plan_train)
        trainer_cfg["num_train_batches_per_epoch"] = int(num_train_batches)
        trainer_cfg["max_num_epochs"] = int(configs["model_params"].get("max_epochs", 600))
        plan = plan_from_det3d(plan_train, plan_path=str(plan_path))
        module = RetinaUNetV001(
            model_cfg=model_cfg,
            trainer_cfg=trainer_cfg,
            plan=plan,
        )
        return module, plan

    def _det3d_batch_to_nndet(self, batch, seg_key="lm", use_disk_box_plug=True):
        from det3d.detection.nndet_train import xyzxyz_exclusive_batch_to_nndet

        data = batch["image"]
        forward_patch_size = self.forward_patch_size
        crop_starts = None
        if forward_patch_size is not None:
            forward_patch_size = tuple(int(v) for v in forward_patch_size)
            spatial = tuple(int(v) for v in data.shape[-3:])
            if any(s > p for s, p in zip(spatial, forward_patch_size)):
                data, crop_starts = _center_crop_spatial(data, forward_patch_size)

        lm_src = batch[seg_key]
        n = int(data.shape[0])
        target_boxes = []
        target_classes_raw = []
        instances_batch = batch["instances"] if "instances" in batch else None
        if use_disk_box_plug:
            box_to_nndet = disk_bbox_to_nndet_xyxyzz
        else:
            box_to_nndet = xyzxyz_exclusive_batch_to_nndet

        for i in range(n):
            box = batch["bbox"][i]
            if crop_starts is not None:
                box = _crop_boxes_to_patch(box, crop_starts, forward_patch_size)
            target_boxes.append(box_to_nndet(box))

            label = torch.as_tensor(batch["label"][i], dtype=torch.long).reshape(-1)
            target_classes_raw.append(label)

        target_classes = _nndet_target_classes(
            target_boxes, target_classes_raw, self.plan["fg_labels"]
        )
        batch_pre = _batch_pre_for_semantic(
            lm_src,
            batch["label"],
            instances_batch,
            self.plan["fg_labels"],
            crop_starts,
            forward_patch_size,
            n,
        )
        target_seg = det3d_semantic_target_seg_from_batch(
            batch_pre, device=data.device if isinstance(data, torch.Tensor) else None
        )
        return {
            "data": data,
            "target_boxes": target_boxes,
            "target_classes": target_classes,
            "target_seg": target_seg,
        }

    def _step_losses(self, batch, batch_idx, evaluation=False):
        nb = self._det3d_batch_to_nndet(batch)
        nb.keys()
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
        batch["pred"] = nndet_batch_pred_to_vis_list(prediction)

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
    from det3d.extra.trainer_nndet import (
        apply_det3d_plan_to_nndet_model_cfg,
        load_nndet_train_cfgs,
        plan_from_det3d,
    )

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

