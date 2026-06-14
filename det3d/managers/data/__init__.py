from det3d.managers.data.batch_tfms import (
    DataManagerDetBTfms,
    DataManagerDetLBDBTfms,
    DataManagerDetPatchBTfms,
    DataManagerDetRBDBTfms,
    DataManagerDetSourceBTfms,
    DataManagerDetWholeBTfms,
    DataManagerDualDetBTfms,
    DataManagerMultiDetBTfms,
)
from det3d.managers.data.main import (
    DataManagerDet,
    DataManagerDetLBD,
    DataManagerDetPatch,
    DataManagerDetRBD,
    DataManagerDetShort,
    DataManagerDetSource,
    DataManagerDetWhole,
    DataManagerDualDet,
    DataManagerMultiDet,
    LoadHDF5DetCaseFulld,
    LoadHDF5DetCropd,
    LoadHDF5DetShardIndexd,
)
from det3d.managers.data.nifti import DataManagerNiftiDet
from det3d.managers.data.run_through import DataManagerRTDet, DataManagerRTDetBTfms
from det3d.managers.data.valid_patch_stream import PatchStreamDatasetDet

try:
    from det3d.managers.data.fromfolder import DataManagerTestFFDet
except Exception:
    DataManagerTestFFDet = None

try:
    from det3d.managers.data.incremental import (
        DataManagerDetI,
        DataManagerDetLBDI,
        DataManagerDetModeSpec,
        DataManagerDetModes,
        DataManagerDetPatchI,
        DataManagerDetSourceI,
        DataManagerDetWholeI,
        DataManagerDualDetI,
    )
except Exception:
    DataManagerDualDetI = None
    DataManagerDetI = None
    DataManagerDetSourceI = None
    DataManagerDetWholeI = None
    DataManagerDetLBDI = None
    DataManagerDetPatchI = None
    DataManagerDetModeSpec = None
    DataManagerDetModes = None

# Backward-compatible aliases for pre-mirror names.
DataManagerTrainDet = DataManagerDetSource
DataManagerTrainDetBTfms = DataManagerDetSourceBTfms
DataManagerTrainDetShard = DataManagerDetSource
DataManagerTrainDetShardBTfms = DataManagerDetSourceBTfms
DataManagerDetShard = DataManagerDetSource
DataManagerDetLBDShard = DataManagerDetLBD
DataManagerDualDetShardBTfms = DataManagerDualDetBTfms

__all__ = [
    "DataManagerDet",
    "DataManagerDetBTfms",
    "DataManagerDetI",
    "DataManagerDetLBD",
    "DataManagerDetLBDI",
    "DataManagerDetLBDBTfms",
    "DataManagerDetLBDShard",
    "DataManagerDetModeSpec",
    "DataManagerDetModes",
    "DataManagerDetPatch",
    "DataManagerDetPatchBTfms",
    "DataManagerDetPatchI",
    "DataManagerDetRBD",
    "DataManagerDetRBDBTfms",
    "DataManagerDetShard",
    "DataManagerDetShort",
    "DataManagerDetSource",
    "DataManagerDetSourceBTfms",
    "DataManagerDetSourceI",
    "DataManagerDetWhole",
    "DataManagerDetWholeBTfms",
    "DataManagerDetWholeI",
    "DataManagerDualDet",
    "DataManagerDualDetBTfms",
    "DataManagerDualDetShardBTfms",
    "DataManagerDualDetI",
    "DataManagerMultiDet",
    "DataManagerMultiDetBTfms",
    "DataManagerNiftiDet",
    "DataManagerRTDet",
    "DataManagerRTDetBTfms",
    "DataManagerTestFFDet",
    "DataManagerTrainDet",
    "DataManagerTrainDetBTfms",
    "DataManagerTrainDetShard",
    "DataManagerTrainDetShardBTfms",
    "LoadHDF5DetCaseFulld",
    "LoadHDF5DetCropd",
    "LoadHDF5DetShardIndexd",
    "PatchStreamDatasetDet",
]
