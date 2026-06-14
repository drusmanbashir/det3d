from fran.managers.data.fromfolder import DataManagerTestFF

from det3d.managers.data.main import DataManagerDet


class DataManagerTestFFDet(DataManagerTestFF, DataManagerDet):
    pass
