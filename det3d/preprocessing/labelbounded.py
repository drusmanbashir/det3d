"""LBD preproc lives in fran.preprocessing.labelbounded — scratch entry only."""

from fran.preprocessing.labelbounded import LabelBoundedDataGenerator

# backward-compat alias
LabelBoundedDetDataGenerator = LabelBoundedDataGenerator

# %%
if __name__ == "__main__":
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers import Project
    from fran.utils.folder_names import FolderNames

    project_title = "lidca"
    plan_id = 4
    project = Project(project_title=project_title)
    config_maker = ConfigMakerDet(project)
    config_maker.setup(plan_id)
    plan = config_maker.configs["plan_train"]
    folders = FolderNames(project=project, plan=plan).folders
    folder_src = folders["data_folder_source"]
    G = LabelBoundedDataGenerator(
        project=project,
        plan=plan,
        data_folder=folder_src,
        hdf5_shard_mode="det",
    )
