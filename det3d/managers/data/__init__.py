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
)
from det3d.managers.data.dual import DataManagerDualDet
from det3d.managers.data.transforms import (
    BboxCenterCropSlicesd,
    CropDetPatchd,
    LoadHDF5DetCaseFulld,
    LoadHDF5DetCropd,
    LoadHDF5DetShardExtendedBBoxd,
    PadDetPatchd,
    RandCropExtendedBBoxd,
)
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
    DataManagerDetModes = None

__all__ = [
    "DataManagerDet",
    "DataManagerDetBTfms",
    "DataManagerDetI",
    "DataManagerDetLBD",
    "DataManagerDetLBDI",
    "DataManagerDetLBDBTfms",
    "DataManagerDetModes",
    "DataManagerDetPatch",
    "DataManagerDetPatchBTfms",
    "DataManagerDetPatchI",
    "DataManagerDetRBD",
    "DataManagerDetRBDBTfms",
    "DataManagerDetShort",
    "DataManagerDetSource",
    "DataManagerDetSourceBTfms",
    "DataManagerDetSourceI",
    "DataManagerDetWhole",
    "DataManagerDetWholeBTfms",
    "DataManagerDetWholeI",
    "DataManagerDualDet",
    "DataManagerDualDetBTfms",
    "DataManagerDualDetBTfmsTfmDebug",
    "DataManagerDualDetTfmDebug",
    "DataManagerDualDetI",
    "DataManagerMultiDetBTfms",
    "DataManagerRTDet",
    "DataManagerRTDetBTfms",
    "DataManagerTestFFDet",
    "LoadHDF5DetCaseFulld",
    "LoadHDF5DetCropd",
    "LoadHDF5DetShardExtendedBBoxd",
    "RandCropExtendedBBoxd",
    "BboxCenterCropSlicesd",
    "CropDetPatchd",
    "PadDetPatchd",
    "PatchStreamDatasetDet",
]
