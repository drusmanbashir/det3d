from det3d.trainers.trainerdet import TrainerDet


class TrainerDetRunThrough(TrainerDet):
    """Run-through det training on full train split (mirrors fran TrainerRT)."""

    case_id_recorder_cls = None

    def resolve_orchestrator_class(self, batch_tfms=None):
        raise NotImplementedError("TrainerDetRunThrough")


TrainerDetRT = TrainerDetRunThrough
