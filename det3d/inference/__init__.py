__all__ = ["DetCascadeInferer", "DetCascadeInfererRetinaUNet", "DetPatchInferer"]


def __getattr__(name):
    if name == "DetCascadeInferer":
        from det3d.inference.cascade import DetCascadeInferer

        return DetCascadeInferer
    if name == "DetCascadeInfererRetinaUNet":
        from det3d.inference.cascade import DetCascadeInfererRetinaUNet

        return DetCascadeInfererRetinaUNet
    if name == "DetPatchInferer":
        from det3d.inference.patch import DetPatchInferer

        return DetPatchInferer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
