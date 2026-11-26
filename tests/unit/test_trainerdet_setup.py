from inspect import signature
from unittest.mock import Mock, patch

import pytest
from det3d.trainers.trainerdet import TrainerDet
from fran.trainers.trainer import Trainer


@pytest.mark.parametrize("interval", [None, 15, 0.2])
def test_permanent_checkpoint_interval_is_inherited(interval):
    trainer = object.__new__(TrainerDet)
    trainer.D = Mock()
    key = "permanent_checkpoint_every_n_epochs"
    kwargs = {} if interval is None else {key: interval}
    parent_signature = signature(Trainer.setup)
    with patch.object(Trainer, "setup", autospec=True) as setup:
        trainer.setup(**kwargs)

    forwarded = setup.call_args.kwargs
    if interval is None:
        assert key not in forwarded
        assert parent_signature.parameters[key].default == 0.25
    else:
        assert forwarded[key] is interval
