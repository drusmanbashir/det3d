__all__ = [
    "CascadeInferer",
    "DetBBoxCascadeInferer",
    "DetSegBBoxCascadeInferer",
    "PatchInferer",
    "DetLBDRunner",
]


def __getattr__(name):
    if name == "CascadeInferer":
        from det3d.inference.cascade import CascadeInferer

        return CascadeInferer
    if name == "DetBBoxCascadeInferer":
        from det3d.inference.cascade import DetBBoxCascadeInferer

        return DetBBoxCascadeInferer
    if name == "DetSegBBoxCascadeInferer":
        from det3d.inference.cascade import DetSegBBoxCascadeInferer

        return DetSegBBoxCascadeInferer
    if name == "PatchInferer":
        from det3d.inference.patch import PatchInferer

        return PatchInferer
    if name == "DetLBDRunner":
        from det3d.inference.lbd import DetLBDRunner

        return DetLBDRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
