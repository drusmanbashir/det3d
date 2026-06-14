from det3d.trainers.trainerdet import TrainerDet


class TrainerDetTransfer(TrainerDet):
    """Transfer-learning det trainer (mirrors fran TrainerTransfer)."""

    def init_dm_unet(self, epochs, batch_size=None, override_dm_checkpoint=False):
        raise NotImplementedError("TrainerDetTransfer")
