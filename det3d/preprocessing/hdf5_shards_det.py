import json
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fran.preprocessing.helpers import import_h5py
from fran.preprocessing.hdf5_shards import (
    HDF5ShardGenerator,
    HDF5ShardWorker,
    _read_shard_case_ids,
)
from utilz.fileio import maybe_makedirs
from utilz.stringz import info_from_filename

from det3d.utils.bbox_sidecar import bbox_sidecar_path, load_detection_sidecar


def _src_dims_tag(src_dims) -> str:
    return "_".join(str(int(v)) for v in src_dims)


def _lbd_case_ids(lbd_folder: Path) -> frozenset[str]:
    images_dir = lbd_folder / "images"
    bboxes_dir = lbd_folder / "bboxes"
    case_ids = set()
    for img_fn in sorted(images_dir.glob("*.pt")):
        case_id = info_from_filename(img_fn.name, full_caseid=True)["case_id"]
        if bbox_sidecar_path(bboxes_dir, img_fn.stem).is_file():
            case_ids.add(case_id)
    return frozenset(case_ids)


def _manifest_case_ids(manifest_fn: Path) -> frozenset[str]:
    manifest = json.loads(manifest_fn.read_text(encoding="utf-8"))
    case_ids = set()
    for shard_info in manifest["shards"]:
        case_ids.update(str(case_id) for case_id in shard_info["case_ids"])
    return frozenset(case_ids)


def ensure_hdf5_shards_for_plan(lbd_folder: Path, plan_src_dims) -> tuple[Path, bool]:
    """Symlink plan shard dir to an existing src_* build when case_ids match LBD.

    Returns (shards_folder, linked). Path tag follows active plan src_dims.
    """
    lbd_folder = Path(lbd_folder)
    shards_root = lbd_folder / "hdf5_shards"
    target = shards_root / f"src_{_src_dims_tag(plan_src_dims)}"
    manifest_fn = target / "manifest.json"
    if manifest_fn.is_file():
        return target, False

    expected = _lbd_case_ids(lbd_folder)
    if not expected or not shards_root.is_dir():
        return target, False

    for candidate_manifest in sorted(shards_root.glob("src_*/manifest.json")):
        candidate = candidate_manifest.parent
        if candidate == target:
            continue
        if _manifest_case_ids(candidate_manifest) != expected:
            continue
        maybe_makedirs(shards_root)
        # Path tag follows active plan; manifest inside target dir is shard metadata only.
        target.symlink_to(candidate.resolve(), target_is_directory=True)
        return target, True

    return target, False


def shard_path_for_case(manifest_fn: Path, case_id: str) -> Path:
    #AI
    """Resolve HDF5 shard file path for case_id from det manifest.json."""
    manifest_fn = Path(manifest_fn)
    with open(manifest_fn, encoding="utf-8") as handle:
        manifest = json.load(handle)
    for shard_info in manifest["shards"]:
        shard_name = shard_info["shard"]
        if str(case_id) not in shard_info["case_ids"]:
            continue
        shard_path = Path(shard_name)
        if not shard_path.is_absolute():
            shard_path = manifest_fn.parent / shard_path
        return shard_path
    raise KeyError(f"case_id {case_id!r} not in manifest {manifest_fn}")


def read_case_from_shard(shard_path: Path, case_id: str) -> dict:
    #AI
    """Read one case group from det HDF5 shard -> numpy arrays."""
    h5py = import_h5py()
    shard_path = Path(shard_path)
    case_path = f"cases/{case_id}"
    with h5py.File(shard_path, "r") as h5f:
        case_grp = h5f[case_path]
        out = {
            "image": np.asarray(case_grp["image"][:]),
            "lm": np.asarray(case_grp["lm"][:]),
            "bbox": np.asarray(case_grp["bbox"][:], dtype=np.float32),
            "label": np.asarray(case_grp["label"][:], dtype=np.int64),
            "image_shape": list(case_grp.attrs["image_shape"]),
            "lm_shape": list(case_grp.attrs["lm_shape"]),
        }
    return out


class DetHDF5ShardWorker(HDF5ShardWorker):
    def _hdf5_chunks_for(self, shape, key, src_dims):
        if key == "lm":
            return super()._hdf5_chunks_for(shape, "lm", src_dims)
        if key == "bbox":
            shape = tuple(int(v) for v in shape)
            if shape[0] == 0:
                return None
            return (min(shape[0], 64), shape[1])
        if key == "label":
            shape = tuple(int(v) for v in shape)
            if shape[0] == 0:
                return None
            return (min(shape[0], 64),)
        return super()._hdf5_chunks_for(shape, key, src_dims)

    def _write_case(
        self,
        h5f,
        case_id,
        image,
        lm,
        bbox_fn,
        src_dims,
        compression,
        compression_opts,
    ):
        image = self._to_numpy_cpu(self._load_torch(image))
        lm = self._to_numpy_cpu(self._load_torch(lm))
        boxes, labels, _instances = load_detection_sidecar(bbox_fn)
        if len(boxes) == 0:
            bbox_arr = np.zeros((0, 6), dtype=np.float32)
            label_arr = np.zeros((0,), dtype=np.int64)
        else:
            bbox_arr = np.stack(
                [self._to_numpy_cpu(box).reshape(-1) for box in boxes]
            ).astype(np.float32)
            label_arr = np.array(
                [int(torch.as_tensor(label).reshape(-1)[0].item()) for label in labels],
                dtype=np.int64,
            )

        ds_kwargs = {}
        if compression is not None:
            ds_kwargs["compression"] = compression
            if compression_opts is not None:
                ds_kwargs["compression_opts"] = compression_opts
            ds_kwargs["shuffle"] = True

        cases_grp = h5f.require_group("cases")
        case_grp = cases_grp.create_group(case_id)
        case_grp.create_dataset(
            "image",
            data=image,
            chunks=self._hdf5_chunks_for(image.shape, "image", src_dims),
            **ds_kwargs,
        )
        case_grp.create_dataset(
            "lm",
            data=lm,
            chunks=self._hdf5_chunks_for(lm.shape, "lm", src_dims),
            **ds_kwargs,
        )
        bbox_chunks = self._hdf5_chunks_for(bbox_arr.shape, "bbox", src_dims)
        if bbox_chunks is None:
            case_grp.create_dataset("bbox", data=bbox_arr, **ds_kwargs)
        else:
            case_grp.create_dataset(
                "bbox", data=bbox_arr, chunks=bbox_chunks, **ds_kwargs
            )
        label_chunks = self._hdf5_chunks_for(label_arr.shape, "label", src_dims)
        if label_chunks is None:
            case_grp.create_dataset("label", data=label_arr, **ds_kwargs)
        else:
            case_grp.create_dataset(
                "label", data=label_arr, chunks=label_chunks, **ds_kwargs
            )
        case_grp.attrs["image_shape"] = list(image.shape)
        case_grp.attrs["lm_shape"] = list(lm.shape)

    def process_shard(
        self,
        shard_fn,
        shard_cases,
        src_dims,
        cases_per_shard,
        compression,
        compression_opts,
    ):
        shard_fn = Path(shard_fn)
        shard_tmp = shard_fn.with_suffix(".h5.tmp")
        src_dims = tuple(int(v) for v in src_dims)
        h5py = import_h5py()
        case_ids_shard = [rec["case_id"] for rec in shard_cases]
        try:
            if shard_tmp.exists():
                shard_tmp.unlink()
            with h5py.File(shard_tmp, "w") as h5f:
                h5f.attrs["format"] = "det_hdf5_shards_v2"
                h5f.attrs["src_dims"] = list(src_dims)
                h5f.attrs["cases_per_shard"] = int(cases_per_shard)
                h5f.attrs["case_ids_json"] = json.dumps(case_ids_shard)
                h5f.attrs["compression"] = (
                    "" if compression is None else str(compression)
                )
                h5f.attrs["compression_opts"] = (
                    -1 if compression_opts is None else int(compression_opts)
                )
                for rec in shard_cases:
                    image = self._load_torch(rec["image"])
                    image_np = self._to_numpy_cpu(image)
                    case_src_dims = tuple(int(v) for v in image_np.shape)
                    self._write_case(
                        h5f=h5f,
                        case_id=rec["case_id"],
                        image=rec["image"],
                        lm=rec["lm"],
                        bbox_fn=rec["bbox"],
                        src_dims=case_src_dims,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
            shard_tmp.replace(shard_fn)
        except Exception:
            if shard_tmp.exists():
                shard_tmp.unlink()
            raise
        return {
            "shard": shard_fn.name,
            "case_ids": case_ids_shard,
        }


def _process_det_hdf5_shard_worker(kwargs):
    return DetHDF5ShardWorker().process_shard(**kwargs)


class DetHDF5ShardGenerator(HDF5ShardGenerator):
    def _df_from_folder(self):
        images_dir = self.data_folder / "images"
        lms_dir = self.data_folder / "lms"
        bboxes_dir = self.data_folder / "bboxes"
        records = []
        for img_fn in sorted(images_dir.glob("*.pt")):
            case_id = info_from_filename(img_fn.name, full_caseid=True)["case_id"]
            lm_fn = lms_dir / img_fn.name
            bbox_fn = bbox_sidecar_path(bboxes_dir, img_fn.stem)
            if not lm_fn.is_file() or not bbox_fn.is_file():
                continue
            records.append(
                {
                    "case_id": case_id,
                    "image": str(img_fn),
                    "lm": str(lm_fn),
                    "bbox": str(bbox_fn),
                }
            )
        df = pd.DataFrame(records)
        assert len(df) > 0, f"No valid det cases found under {self.data_folder}"
        return df

    def create_data_df(self):
        self.df = self._df_from_folder()
        assert len(self.df) > 0, "No valid case files found in {}".format(
            self.data_folder
        )
        self.case_ids = self.df["case_id"].tolist()
        self.df = self.df.map(lambda x: x.lower() if isinstance(x, str) else x)
        self.df["pt_processed"] = None
        self.df["hdf5_processed"] = None
        print("Total number of cases: ", len(self.df))
        self.df.drop(columns=["pt_processed"], inplace=True)

    def setup(self, overwrite=False):
        if overwrite and self.shards_folder.exists():
            shutil.rmtree(self.shards_folder)
        super().setup()

    def register_existing_cases(self):
        self.shard_inds = []
        shards = sorted(self.shards_folder.glob("shard_*.h5"))
        case_ids_done = []
        bad_names = []
        for shard_fn in shards:
            self._store_shard_ind(shard_fn)
            shard_info = _read_shard_case_ids(shard_fn)
            if shard_info["error"] is not None:
                bad_names.append(shard_info["shard"])
                continue
            case_ids_done.extend(shard_info["case_ids"])

        case_ids_done_unique = set(case_ids_done)
        if len(case_ids_done) != len(case_ids_done_unique):
            dupes = {
                case_id for case_id in case_ids_done if case_ids_done.count(case_id) > 1
            }
            raise ValueError(
                "Duplicate case IDs found across shards: "
                f"{dupes}. Re-run with overwrite_hdf5_shards=True."
            )

        self.df.loc[self.df["case_id"].isin(case_ids_done), "hdf5_processed"] = True
        if len(bad_names) > 0:
            raise RuntimeError(f"Failed to read the following shard files: {bad_names}")

    def _manifest_payload(self, shard_manifest):
        payload = super()._manifest_payload(shard_manifest)
        payload["format"] = "det_hdf5_shards_v2"
        return payload

    def run(self, overwrite=False, num_processes=8):
        return super().run(overwrite=overwrite, num_processes=num_processes)

    def process(self, num_processes=8):
        worker_fn = _process_det_hdf5_shard_worker
        maybe_makedirs(self.shards_folder)
        completed_shards = []
        failed_shards = []
        max_workers = (
            max(1, min(int(num_processes), len(self.shard_jobs)))
            if self.shard_jobs
            else 0
        )
        if max_workers == 0:
            return sorted(self.shards_folder.glob("shard_*.h5"))
        if max_workers == 1:
            for job in self.shard_jobs:
                try:
                    shard_info = worker_fn(job)
                    completed_shards.append(shard_info)
                except Exception as e:
                    failed_shards.append(str(e))
                    continue
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
                futures = [ex.submit(worker_fn, job) for job in self.shard_jobs]
                for fut in as_completed(futures):
                    try:
                        shard_info = fut.result()
                    except Exception as e:
                        failed_shards.append(str(e))
                        continue
                    completed_shards.append(shard_info)
        self.shard_jobs = []
        if failed_shards:
            raise RuntimeError("HDF5 shard failures:\n" + "\n".join(failed_shards))
        print(f"Wrote {len(self.shard_paths)} HDF5 shards in {self.shards_folder}")
        return sorted(self.shards_folder.glob("shard_*.h5"))
