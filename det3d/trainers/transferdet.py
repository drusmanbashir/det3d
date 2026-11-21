from __future__ import annotations

from fran.trainers.transfer import TrainerTransfer
from utilz.stringz import headline

from det3d.architectures.create_detector import arch_from_conf
from det3d.trainers.trainerdet import TrainerDet
from det3d.trainers.trainerdet_rt import TrainerDetRunThrough


class TrainerDetTransfer(TrainerTransfer, TrainerDet):
    """Transfer-learning det trainer: fran ``TrainerTransfer`` source selection and weight copy on ``TrainerDet``."""

    def transfer_module(self, manager):
        """Det managers transfer their whole state dict."""
        return manager

    def report_head_mismatch(self, source):
        old_n = len(source.plan["fg_labels"])
        new_n = len(self.configs["plan_train"]["fg_labels"])
        if old_n != new_n:
            headline(
                "Source fg_labels count ({0}) != target ({1}); detection heads "
                "will be skipped by shape and remain target-initialized.".format(
                    old_n, new_n
                )
            )

    def freeze_encoder(self):
        arch = arch_from_conf(self.configs)
        if arch == "retinaunet":
            enc = self.N.net.encoder
        else:
            enc = self.N.detector.network.feature_extractor
        for param in enc.parameters():
            param.requires_grad = False


class TrainerDetTransferRT(TrainerDetTransfer, TrainerDetRunThrough):
    """Transfer det training in run-through mode."""



if __name__ == "__main__":
# %%
# SECTION:-------------------- setup<--------------------------------------------------------------------------------------
    from fran.managers import Project
    from torch import Tensor
    from utilz.imageviewers import ImageBBoxViewer

    from det3d.configs.parser import ConfigMakerDet

    project_title = "lidca"
    plan_id = 4

    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = C.configs
    conf["dataset_params"]["fold"] = 0

# SECTION:-------------------- TRAINING --------------------------------------------------------------------------------------
# %%
    conf["model_params"]["arch"] = "retinanet"
    conf["model_params"]["arch"] = "retinaunet"
    print(conf["dataset_params"]["prezoom_scale"])
    print(conf["plan_train"]["patch_size"])# = [128,128,64]

    bs = 10
    device_id = 0
    batch_tfms = True
    batch_tfms = False
    wandb = True
    batchsize_finder= False
    run_name = None
    run_name = "LIDCA-DIET"
    tags = []
    description = "TrainerDet lidca retinanet"
    lr = None
    debug_ = False
    profiler = False
    compiled = False
    cbs = []
    wandb_grid_epoch_freq = 40
    val_every_n_epochs = 5
    train_indices = None
    val_indices = None
    val_sampling = 1.0
    epochs = 500

# %%
#SECTION:-------------------- TRAIN--------------------------------------------------------------------------------------
    Tm = TrainerDetTransfer(P.project_title, conf, run_name=run_name,  source_ckpt="last", resume_lr=None)
# %%
    Tm.setup(
        compiled=compiled,
        train_indices=train_indices,
        batch_tfms=batch_tfms,
        cbs=cbs,
        debug=debug_,
        batch_size=bs,
        devices=[device_id],
        epochs=600,
        batchsize_finder=batchsize_finder,
        wandb=wandb,
        wandb_grid_epoch_freq=wandb_grid_epoch_freq,
        tags=tags,
        description=description,
        lr=lr,
    )
#
    Tm.fit()

# %%
