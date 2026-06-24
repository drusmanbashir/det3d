import json
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fran.preprocessing.helpers import import_h5py, sanitize_meta_for_monai
from fran.preprocessing.hdf5_shards import (
    HDF5ShardGenerator,
    HDF5ShardWorker,
    _read_shard_case_ids,
)
from utilz.fileio import maybe_makedirs
from utilz.stringz import info_from_filename

from det3d.utils.bbox_sidecar import bbox_sidecar_path, load_detection_sidecar


class DetHDF5ShardWorker(HDF5ShardWorker):
    def _hdf5_chunks_for(self, shape, key, src_dims):
        if key == "mask":
            return super()._hdf5_chunks_for(shape, "lm", src_dims)
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
        if key == "fg_indices":
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
        mask,
        lm,
        indices,
        bbox_fn,
        src_dims,
        compression,
        compression_opts,
    ):
        image = self._to_numpy_cpu(self._load_torch(image))
        mask = self._to_numpy_cpu(self._load_torch(mask))
        lm = self._to_numpy_cpu(self._load_torch(lm))
        indices = self._load_torch(indices)
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

        if not isinstance(indices, dict):
            raise ValueError(f"indices file must be a dict: {indices}")
        fg = self._to_numpy_cpu(indices["fg_indices"]).reshape(-1)

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
            "mask",
            data=mask,
            chunks=self._hdf5_chunks_for(mask.shape, "mask", src_dims),
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
        self._create_index_dataset(case_grp, "fg_indices", fg, ds_kwargs, src_dims)
        case_grp.attrs["mask_shape"] = list(mask.shape)
        case_grp.attrs["lm_shape"] = list(lm.shape)
        if "meta" not in indices:
            return
        meta = indices["meta"]
        if isinstance(meta, dict):
            meta = sanitize_meta_for_monai(dict(meta))
            case_grp.attrs["meta_json"] = json.dumps(meta, default=str)
            if "filename_or_obj" in meta and meta["filename_or_obj"] is not None:
                case_grp.attrs["source_meta_filename_or_obj"] = str(
                    meta["filename_or_obj"]
                )

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
                    self._write_case(
                        h5f=h5f,
                        case_id=rec["case_id"],
                        image=rec["image"],
                        mask=rec["mask"],
                        lm=rec["lm"],
                        indices=rec["indices"],
                        bbox_fn=rec["bbox"],
                        src_dims=src_dims,
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
    def _df_from_folder(self, indices_folder=None):
        indices_folder = indices_folder or self.indices_subfolder
        images_dir = self.data_folder / "images"
        masks_dir = self.data_folder / "masks"
        lms_dir = self.data_folder / "lms"
        bboxes_dir = self.data_folder / "bboxes"
        records = []
        for img_fn in sorted(images_dir.glob("*.pt")):
            case_id = info_from_filename(img_fn.name, full_caseid=True)["case_id"]
            mask_fn = masks_dir / img_fn.name
            lm_fn = lms_dir / img_fn.name
            ind_fn = indices_folder / img_fn.name
            bbox_fn = bbox_sidecar_path(bboxes_dir, img_fn.stem)
            if (
                not mask_fn.is_file()
                or not lm_fn.is_file()
                or not ind_fn.is_file()
                or not bbox_fn.is_file()
            ):
                continue
            records.append(
                {
                    "case_id": case_id,
                    "image": str(img_fn),
                    "mask": str(mask_fn),
                    "lm": str(lm_fn),
                    "indices": str(ind_fn),
                    "bbox": str(bbox_fn),
                }
            )
        df = pd.DataFrame(records)
        assert len(df) > 0, f"No valid det cases found under {self.data_folder}"
        return df

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
