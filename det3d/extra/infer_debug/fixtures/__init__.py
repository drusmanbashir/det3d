"""Named inference debug fixtures (real + pseudo)."""

from det3d.extra.infer_debug.fixtures._case import InferFixtureCase
from det3d.extra.infer_debug.fixtures.lidc_0001 import build_lidc_0001
from det3d.extra.infer_debug.fixtures.pseudo_cuboid import build_pseudo_cuboid

FIXTURES = {
    "pseudo_cuboid": build_pseudo_cuboid,
    "lidc_0001": build_lidc_0001,
}


def load_fixture(name: str) -> InferFixtureCase:
    return FIXTURES[name]()
