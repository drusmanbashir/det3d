"""Scratch: DataManagerDualDet -> train/valid dataloaders."""

# %%
# SECTION:--- setup ---
from fran.managers import Project

from det3d.configs.parser import ConfigMakerDet
from det3d.managers.data import DataManagerDualDet, DataManagerDualDetBTfms

project_title = "lidca"
plan_id = 2
conf_fold = 0

P = Project(project_title)
C = ConfigMakerDet(P)
C.setup(plan_id)
conf = C.configs
conf["dataset_params"]["fold"] = conf_fold

# %%
# SECTION:--- dualdet datamanager ---
batch_size = 2
batch_tfms = False
debug_ = True
train_indices = None
val_indices = 10
val_sampling = 1.0
device = 0

for key in ("plan_train", "plan_valid", "plan_test"):
    plan = conf[key]
    if plan["mode"] in {"det", "lbd"}:
        plan["mode"] = "lbd"

DmCls = DataManagerDualDetBTfms if batch_tfms else DataManagerDualDet
D = DmCls(
    project_title=project_title,
    configs=conf,
    batch_size=batch_size,
    cache_rate=conf["dataset_params"]["cache_rate"],
    device=device,
    ds_type=conf["dataset_params"]["ds_type"],
    train_indices=train_indices,
    val_indices=val_indices,
    val_sampling=val_sampling,
    debug=debug_,
    batch_tfms=batch_tfms,
)

# %%
# SECTION:--- prepare_data ---
D.prepare_data()

# %%
# SECTION:--- setup fit ---
D.setup(stage="fit")
tmt = D.train_manager
tmv = D.valid_manager

# %%
# SECTION:--- train dataloader ---
tmt.setup()
train_dl = tmt.dl
print(f"train: {tmt}")
print(f"train keys: {tmt.keys}")
train_batch = next(iter(train_dl))
train_batch.keys()
train_batch["image"].shape

# %%
# SECTION:--- valid dataloader ---
tmv.setup()
val_dl = tmv.dl
print(f"valid: {tmv}")
print(f"valid keys: {tmv.keys}")
print(f"valid_impl: {tmv.dataset_params['valid_impl']}")
val_batch = next(iter(val_dl))
val_batch.keys()
val_batch["image"].shape
val_batch.get("validation_impl")
val_batch["case_id"]
