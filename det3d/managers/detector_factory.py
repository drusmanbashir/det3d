from typing import Protocol

from lightning.pytorch import LightningModule


class DetectionManager(Protocol):
    detector: object
    plan: dict
    configs: dict
    class_names: list

    def training_step(self, batch, batch_idx): ...
    def validation_step(self, batch, batch_idx): ...
    def configure_optimizers(self): ...


from det3d.architectures.create_detector import arch_from_conf


def resolve_detector_manager(configs: dict):
    detector = arch_from_conf(configs)
    if detector == "retinaunet":
        from det3d.managers.retinaunet import RetinaUNetManager

        return RetinaUNetManager
    from det3d.managers.retinanet import RetinaNetManager

    return RetinaNetManager


def build_detector_manager(project_title, configs, lr=None, sync_dist=False) -> LightningModule:
    detector = configs["model_params"]["arch"]
    if detector == "retinaunet":
        from det3d.managers.retinaunet import RetinaUNetManager
        manager_cls = RetinaUNetManager
    else:
        from det3d.managers.retinanet import RetinaNetManager
        manager_cls = RetinaNetManager
    N = manager_cls(
        project_title=project_title,
        configs=configs,
        lr=lr,
        sync_dist=sync_dist,
    )
    return N

# %%
if __name__ == '__main__':
#SECTION:-------------------- setup --------------------------------------------------------------------------------------
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers import Project

    project_title = "lidca"
    plan_id = 1

    P = Project(project_title)
    C = ConfigMakerDet(P)
    C.setup(plan_id)
    conf = C.configs
    conf["dataset_params"]["fold"] = 0
    conf["model_params"]["arch"] = "retinanet"

    manager = build_detector_manager("test", conf)
    print(manager)
# %%
