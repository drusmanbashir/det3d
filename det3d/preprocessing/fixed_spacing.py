# %%
import numpy as np
from fran.preprocessing.fixed_spacing import NiftiToTorchDataGenerator
from fran.configs.parser import ConfigMaker, FolderNames, parse_nested_remapping
from utilz.helpers import chunks
from fran.managers import Project
from utilz.fileio import load_dict
from fran.transforms.fg_indices import FgBgToIndicesd2
from fran.transforms.inferencetransforms import ToCPUd
from monai.transforms.utility.dictionary import (
    EnsureChannelFirstd,
    FgBgToIndicesd,
    ToDeviced,
)
from utilz.fileio import maybe_makedirs, save_json
from utilz.stringz import strip_extension
# %%
if __name__ == '__main__':
#SECTION:-------------------- setup --------------------------------------------------------------------------------------
    from det3d.configs.parser import ConfigMakerDet
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers import Project

    project_title = "lidca"
    plan_id = 3
    P = Project(project_title=project_title)
    config_maker = ConfigMakerDet(P)
    config_maker.setup(plan_id)
    conf = config_maker.configs
    plan = config_maker.configs["plan_train"]
    overwrite = False
    overwrite_hdf5_shards = False
    folders = FolderNames(project=P, plan=plan).folders
    folder_src = folders["data_folder_source"]

# %%

    plan = conf["plan_train"]
    plan["mode"]
    print(plan)

    FolderNames(P, plan).folders
    print(P.global_properties)
# %%
    # add_plan_to_db(plan,"/r/datasets/preprocessed/totalseg/lbd/spc_100_100_100_plan5",P.db)
    F = NiftiToTorchDataGenerator(P, plan, P.raw_data_folder)
    F.setup()
# %%
    F.run(overwrite=False, num_processes=16)
#
