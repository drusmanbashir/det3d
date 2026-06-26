from det3d.managers.data.pre_trafo import (
    DataManagerDetPreTrafo,
    DataManagerDetLBDPreTrafo,
    DataManagerDetLBDPreTrafoBTfms,
    DataManagerDetPatchPreTrafo,
    DataManagerDetRBDPreTrafo,
    DataManagerDetRBDPreTrafoBTfms,
    DataManagerDetSourcePreTrafo,
    DataManagerDetSourcePreTrafoBTfms,
    DataManagerDetWholePreTrafo,
    DataManagerDualDetPreTrafo,
    DataManagerDualDetPreTrafoBTfms,
    DataManagerMultiDetPreTrafo,
)
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
    BboxCenterCropSlicesd,
    CropDetPatchd,
    LoadHDF5DetCaseFulld,
    LoadHDF5DetCropd,
    LoadHDF5DetShardExtendedBBoxd,
    RandCropExtendedBBoxd,
    PadDetPatchd,
)
from det3d.managers.data.nifti import DataManagerNiftiDet
from det3d.managers.data.tfm_debug import (
    DataManagerDualDetBTfmsTfmDebug,
    DataManagerDualDetTfmDebug,
    KEYS_CPU_NO_SPATIAL,
    KEYS_ITEM_NO_SPATIAL,
)
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
    "DataManagerDetLBDPreTrafo",
    "DataManagerDetLBDPreTrafoBTfms",
    "DataManagerDetLBDBTfms",
    "DataManagerDetLBDShard",
    "DataManagerDetModeSpec",
    "DataManagerDetModes",
    "DataManagerDetPatch",
    "DataManagerDetPatchBTfms",
    "DataManagerDetPatchI",
    "DataManagerDetPreTrafo",
    "DataManagerDetRBD",
    "DataManagerDetRBDBTfms",
    "DataManagerDetRBDPreTrafo",
    "DataManagerDetRBDPreTrafoBTfms",
    "DataManagerDetShard",
    "DataManagerDetShort",
    "DataManagerDetSource",
    "DataManagerDetSourceBTfms",
    "DataManagerDetSourceI",
    "DataManagerDetSourcePreTrafo",
    "DataManagerDetSourcePreTrafoBTfms",
    "DataManagerDetWhole",
    "DataManagerDetWholeBTfms",
    "DataManagerDetWholeI",
    "DataManagerDetWholePreTrafo",
    "DataManagerDualDet",
    "DataManagerDualDetBTfms",
    "DataManagerDualDetBTfmsTfmDebug",
    "DataManagerDualDetTfmDebug",
    "DataManagerDualDetPreTrafo",
    "DataManagerDualDetPreTrafoBTfms",
    "DataManagerDualDetShardBTfms",
    "DataManagerDualDetI",
    "DataManagerMultiDet",
    "DataManagerMultiDetBTfms",
    "DataManagerMultiDetPreTrafo",
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
    "LoadHDF5DetShardExtendedBBoxd",
    "RandCropExtendedBBoxd",
    "BboxCenterCropSlicesd",
    "CropDetPatchd",
    "PadDetPatchd",
    "PatchStreamDatasetDet",
]
