#!/usr/bin/env python3
"""Stage local FRAN-style image/lm pairs into nnDetection task layouts."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCES = SCRIPT_DIR / "local_plan_sources.yaml"


def load_spec(mnemonic: str, sources_yaml: Path) -> dict:
    data = yaml.safe_load(sources_yaml.read_text())
    if mnemonic not in data:
        raise KeyError(f"unknown mnemonic {mnemonic!r}; known: {sorted(data)}")
    return data[mnemonic]


def label_candidates(folder: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for ext in (".nii.gz", ".nii"):
        for p in folder.glob(f"*{ext}"):
            if p.name.startswith("._"):
                continue
            stem = p.name[: -len(ext)]
            out.setdefault(stem, p)
    return out


def image_files(folder: Path) -> list[Path]:
    out: dict[str, Path] = {}
    for ext in (".nii.gz", ".nii"):
        for p in folder.glob(f"*{ext}"):
            if p.name.startswith("._"):
                continue
            stem = p.name[: -len(ext)]
            out.setdefault(stem, p)
    return [out[k] for k in sorted(out)]


def pair_images_labels(
    images_dir: Path, labels_dir: Path
) -> list[tuple[Path, Path, str]]:
    labels_by_stem = label_candidates(labels_dir)
    pairs: list[tuple[Path, Path, str]] = []
    for img in image_files(images_dir):
        stem = img.name
        for ext in (".nii.gz", ".nii"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        lab = labels_by_stem.get(stem)
        if lab is None:
            continue
        pairs.append((img, lab, stem))
    return pairs


def resolve_source(spec: dict) -> tuple[Path, Path, Path, list[tuple[Path, Path, str]]]:
    for src in spec["sources"]:
        root = Path(src["root"])
        images_dir = root / src.get("images_subdir", "images")
        labels_dir = root / src.get("labels_subdir", "lms")
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        pairs = pair_images_labels(images_dir, labels_dir)
        if pairs:
            return root, images_dir, labels_dir, pairs
    raise FileNotFoundError(f"no usable source for {spec.get('task')}")


def dataset_payload(dataset_meta: dict, label_remap: dict[int, int] | None) -> dict:
    payload = dict(dataset_meta)
    if label_remap:
        payload["label_remap"] = {str(k): v for k, v in label_remap.items()}
    return payload


def symlink_replace(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target.resolve())


def stage_decathlon(
    task_dir: Path,
    pairs: list[tuple[Path, Path, str]],
    dataset_meta: dict,
) -> int:
    raw = task_dir / "raw"
    images_tr = raw / "imagesTr"
    labels_tr = raw / "labelsTr"
    skipped = []
    for img, lab, stem in pairs:
        try:
            symlink_replace(images_tr / f"{stem}.nii.gz", img)
            symlink_replace(labels_tr / f"{stem}.nii.gz", lab)
        except OSError as exc:
            skipped.append((stem, str(exc)))
    payload = {
        **dataset_meta,
        "licence": dataset_meta.get("licence", "local"),
        "release": dataset_meta.get("release", "local"),
        "tensorImageSize": "3D",
        "numTraining": len(pairs) - len(skipped),
        "numTest": 0,
    }
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "dataset.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"staged decathlon raw {len(pairs) - len(skipped)} cases → {raw} "
        f"(skipped {len(skipped)})",
        file=sys.stderr,
    )
    return len(pairs) - len(skipped)


def kits_case_id(stem: str, case_prefix: str = "case_") -> str:
    m = re.search(r"(\d+)$", stem)
    if not m:
        raise ValueError(f"cannot derive KiTS case id from {stem!r}")
    return f"{case_prefix}{int(m.group(1)):05d}"


def stage_kits_splitted(
    task_dir: Path,
    pairs: list[tuple[Path, Path, str]],
    dataset_meta: dict,
    case_prefix: str = "case_",
) -> int:
    from nndet.io.prepare import create_test_split

    splitted = task_dir / "raw_splitted"
    images_tr = splitted / "imagesTr"
    labels_tr = splitted / "labelsTr"
    staged = 0
    skipped = []
    for img, lab, stem in pairs:
        try:
            case_id = kits_case_id(stem, case_prefix=case_prefix)
        except ValueError as exc:
            skipped.append((stem, str(exc)))
            continue
        symlink_replace(images_tr / f"{case_id}_0000.nii.gz", img)
        symlink_replace(labels_tr / f"{case_id}.nii.gz", lab)
        staged += 1
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "dataset.json").write_text(json.dumps(dataset_meta, indent=2) + "\n")
    create_test_split(
        splitted_dir=splitted,
        num_modalities=1,
        test_size=0.3,
        random_state=0,
        shuffle=True,
    )
    print(f"staged kits splitted {staged} cases → {splitted} (skipped {len(skipped)})", file=sys.stderr)
    return staged


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage local MSD/nnDet datasets.")
    parser.add_argument("--mnemonic", required=True, help="liver|kidneys|pancreas|colon")
    parser.add_argument("--task-dir", required=True, help="det_data/{Task}/ directory")
    parser.add_argument(
        "--sources-yaml",
        default=str(DEFAULT_SOURCES),
        help="local_plan_sources.yaml path",
    )
    args = parser.parse_args()

    spec = load_spec(args.mnemonic, Path(args.sources_yaml))
    root, _images_dir, _labels_dir, pairs = resolve_source(spec)
    task_dir = Path(args.task_dir).resolve()
    source_root = root.resolve()
    if source_root in task_dir.parents or task_dir == source_root:
        raise RuntimeError(
            f"task-dir {task_dir} must not be inside source dataset {source_root}"
        )

    layout = spec["layout"]
    label_remap = {int(k): int(v) for k, v in (spec.get("label_remap") or {}).items()}
    label_remap = label_remap or None
    dataset_meta = dataset_payload(spec["dataset_json"], label_remap)
    print(f"using source root {root} ({len(pairs)} pairs)", file=sys.stderr)
    if label_remap:
        print(f"label_remap {label_remap} (runtime in convert_seg2det_local)", file=sys.stderr)

    if layout == "decathlon":
        n = stage_decathlon(task_dir, pairs, dataset_meta)
    elif layout == "kits_splitted":
        n = stage_kits_splitted(
            task_dir,
            pairs,
            dataset_meta,
            case_prefix=spec.get("case_prefix", "case_"),
        )
    else:
        raise ValueError(f"unsupported layout {layout!r}")

    if n == 0:
        raise RuntimeError("no cases staged")
    print(n)


if __name__ == "__main__":
    main()
