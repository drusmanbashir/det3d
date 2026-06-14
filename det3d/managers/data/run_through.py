from fran.managers.data.run_through import DataManagerRT, DataManagerRTBTfms

from det3d.managers.data.batch_tfms import DataManagerDualDetBTfms
from det3d.managers.data.main import DataManagerDualDet


class DataManagerRTDet(DataManagerRT, DataManagerDualDet):
    pass


class DataManagerRTDetBTfms(DataManagerRT, DataManagerDualDetBTfms):
    pass
