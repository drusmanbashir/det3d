"""One-shot port of nnDetection encoder/decoder into det3d/detection/arch/nndet/."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NNDET = Path("/home/ub/code/nnDetection/nndet/arch")
OUT = ROOT / "det3d" / "detection" / "arch" / "nndet"

REPL = (
    ("from nndet.arch.", "from det3d.detection.arch.nndet."),
    ("from nndet.utils import to_dtype", ""),
    ("from nndet.utils.info import experimental", ""),
)


def _fix(text: str) -> str:
    for old, new in REPL:
        text = text.replace(old, new)
    return text


def _write(rel: Path, text: str):
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def port_basic():
    src = (NNDET / "blocks" / "basic.py").read_text()
    src = _fix(src)
    cut = src.index("\n\nclass StackedConvBlock3")
    src = src[:cut] + "\n"
    src = src.replace("from det3d.detection.arch.nndet.blocks.res import ResBasic\n", "")
    _write(Path("blocks/basic.py"), src)


def port_decoder():
    src = (NNDET / "decoder" / "base.py").read_text()
    src = _fix(src)
    src = src.replace("from loguru import logger\n\n", "")
    src = src.replace("from det3d.detection.arch.nndet.utils.info import experimental\n", "")
    inline = (
        "\n\ndef to_dtype(x, dtype):\n"
        "    if isinstance(x, torch.Tensor):\n"
        "        return x.to(dtype=dtype)\n"
        "    return dtype(x)\n"
    )
    src = src.replace(
        "from det3d.detection.arch.nndet.conv import conv_kwargs_helper\n",
        "from det3d.detection.arch.nndet.conv import conv_kwargs_helper\n" + inline,
    )
    cut = src.index("\n\nclass PAUFPN")
    src = src[:cut] + "\n"
    _write(Path("decoder/base.py"), src)


def port_file(rel_src: str, rel_dst: str | None = None):
    rel_dst = rel_dst or rel_src
    text = _fix((NNDET / rel_src).read_text())
    if rel_src == "conv.py":
        text = text.replace(
            "from det3d.detection.arch.nndet.initializer import InitWeights_He\n", ""
        )
    _write(Path(rel_dst), text)


def main():
    port_file("conv.py")
    port_file("encoder/abstract.py")
    port_file("encoder/modular.py")
    port_file("layers/norm.py")
    port_basic()
    port_decoder()
    for pkg in ("", "encoder", "decoder", "blocks", "layers"):
        init = OUT / pkg / "__init__.py"
        init.parent.mkdir(parents=True, exist_ok=True)
        init.write_text("")
    print(f"ported to {OUT}")


if __name__ == "__main__":
    main()
