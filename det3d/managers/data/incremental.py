from dataclasses import replace
from typing import Optional

from fran.data.collate import grid_collated, patch_collated, source_collated, whole_collated
from fran.managers.data.incremental import (
    DataManagerDualI,
    DataManagerI,
    DataManagerLBDI,
    DataManagerModeSpec,
    DataManagerModes,
    DataManagerPatchI,
    DataManagerSourceI,
    DataManagerWholeI,
)

from det3d.managers.data.main import DataManagerDet


class DataManagerDetSourceI(DataManagerSourceI, DataManagerDetI):
    pass


class DataManagerDetWholeI(DataManagerWholeI, DataManagerDetI):
    pass


class DataManagerDetLBDI(DataManagerLBDI, DataManagerDetI):
    pass


class DataManagerDetPatchI(DataManagerPatchI, DataManagerDetI):
    pass


class DataManagerDetModeSpec(DataManagerModeSpec):
    pass


class DataManagerDetModes:
    SOURCE = DataManagerDetModeSpec(
        mode="source",
        manager_cls=DataManagerDetSourceI,
        collate_fn=source_collated,
    )
    sourcepbd = DataManagerDetModeSpec(
        mode="sourcepbd",
        manager_cls=DataManagerDetPatchI,
        collate_fn=patch_collated,
    )
    WHOLE = DataManagerDetModeSpec(
        mode="whole",
        manager_cls=DataManagerDetWholeI,
        collate_fn=whole_collated,
    )
    PATCH = DataManagerDetModeSpec(
        mode="patch",
        manager_cls=DataManagerDetPatchI,
        collate_fn=patch_collated,
    )
    LBD = DataManagerDetModeSpec(
        mode="lbd",
        manager_cls=DataManagerDetLBDI,
        collate_fn=source_collated,
    )

    _BY_MODE = {
        SOURCE.mode: SOURCE,
        sourcepbd.mode: sourcepbd,
        WHOLE.mode: WHOLE,
        PATCH.mode: PATCH,
        LBD.mode: LBD,
    }

    @classmethod
    def by_mode(cls, mode: str, split: Optional[str] = None) -> DataManagerDetModeSpec:
        try:
            spec = cls._BY_MODE[mode]
        except KeyError as exc:
            raise ValueError(
                f"Unrecognized mode: {mode}. Must be one of {list(cls._BY_MODE.keys())}"
            ) from exc
        if split in {"test", "train2"}:
            return replace(spec, collate_fn=grid_collated)
        return spec
