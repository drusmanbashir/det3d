from pathlib import Path

import json
import shutil
import pandas as pd
import ray
import torch
import numpy as np
from fran.preprocessing.helpers import import_h5py, infer_indices_folder, sanitize_meta_for_monai
from fran.preprocessing.hdf5_shards import HDF5ShardGenerator, HDF5ShardWorker, _read_shard_case_ids
from fran.preprocessing.labelbounded import LabelBoundedDataGenerator
from fran.preprocessing.preprocessor import CPUS_PER_ACTOR
from fran.preprocessing.rayworker_base import MIN_SIZE, RayWorkerBase
from utilz.stringz import info_from_filename
from monai.apps.detection.transforms.dictionary import StandardizeEmptyBoxd
from monai.data import MetaTensor
from monai.transforms import Compose, EnsureTyped
from utilz.fileio import maybe_makedirs, save_json
from utilz.stringz import strip_extension

from det3d.preprocessing.object_bounded import _dusting_threshold
from det3d.transforms.bbox_stats import DetectionBBoxStatsd
from det3d.transforms.crop_indices import mask_fg_bg_flat_indices
from det3d.transforms.detection import GenerateExtendedBoxMask
from det3d.preprocessing.dataset_details import write_dataset_details_csv
from det3d.utils.bbox_sidecar import bbox_sidecar_path, load_detection_sidecar, save_detection_sidecar
from det3d.utils.folder_names import lbd_det_folder_from_plan


class DetHDF5ShardWorker(HDF5ShardWorker):
    def _hdf5_chunks_for(self, shape, key, src_dims):
        if key == "mask":
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
        if key in ("fg_indices", "bg_indices"):
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
        indices,
        bbox_fn,
        src_dims,
        compression,
        compression_opts,
    ):
        image = self._to_numpy_cpu(self._load_torch(image))
        mask = self._to_numpy_cpu(self._load_torch(mask))
        indices = self._load_torch(indices)
        boxes, labels = load_detection_sidecar(bbox_fn)
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
        if "fg_indices" in indices and "bg_indices" in indices:
            fg = self._to_numpy_cpu(indices["fg_indices"]).reshape(-1)
            bg = self._to_numpy_cpu(indices["bg_indices"]).reshape(-1)
        elif "lm_fg_indices" in indices and "lm_bg_indices" in indices:
            fg = self._to_numpy_cpu(indices["lm_fg_indices"]).reshape(-1)
            bg = self._to_numpy_cpu(indices["lm_bg_indices"]).reshape(-1)
        else:
            fg, bg = mask_fg_bg_flat_indices(mask)
        fg_m, bg_m = mask_fg_bg_flat_indices(mask)
        if fg.shape != fg_m.shape or not np.array_equal(fg, fg_m):
            fg, bg = fg_m, bg_m

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
        self._create_index_dataset(case_grp, "fg_indices", fg, ds_kwargs, src_dims)
        self._create_index_dataset(case_grp, "bg_indices", bg, ds_kwargs, src_dims)

        case_grp.attrs["image_shape"] = list(image.shape)
        case_grp.attrs["mask_shape"] = list(mask.shape)
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
                h5f.attrs["compression"] = "" if compression is None else str(compression)
                h5f.attrs["compression_opts"] = (
                    -1 if compression_opts is None else int(compression_opts)
                )
                for rec in shard_cases:
                    self._write_case(
                        h5f=h5f,
                        case_id=rec["case_id"],
                        image=rec["image"],
                        mask=rec["mask"],
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
        bboxes_dir = self.data_folder / "bboxes"
        records = []
        for img_fn in sorted(images_dir.glob("*.pt")):
            case_id = info_from_filename(img_fn.name, full_caseid=True)["case_id"]
            mask_fn = masks_dir / img_fn.name
            ind_fn = indices_folder / img_fn.name
            bbox_fn = bbox_sidecar_path(bboxes_dir, img_fn.stem)
            if not mask_fn.is_file() or not ind_fn.is_file() or not bbox_fn.is_file():
                continue
            records.append(
                {
                    "case_id": case_id,
                    "image": str(img_fn),
                    "mask": str(mask_fn),
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
                case_id
                for case_id in case_ids_done
                if case_ids_done.count(case_id) > 1
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
        import fran.preprocessing.hdf5_shards as hs

        original = hs._process_hdf5_shard_worker
        hs._process_hdf5_shard_worker = _process_det_hdf5_shard_worker
        try:
            return super().process(num_processes=num_processes)
        finally:
            hs._process_hdf5_shard_worker = original


class _LBDDetWorker(RayWorkerBase):
    """Label-bounded detection worker: fixed_spacing PT in → cropped volume + mask + bbox sidecar."""

    remapping_key = "remapping_lbd_rbd"
    box_key = "bbox"
    label_key = "label"

    def __init__(
        self,
        project,
        plan,
        data_folder,
        output_folder,
        crop_to_label,
        debug=False,
        dusting_threshold=3.0,
        ignore_labels=None,
        foreground_class_id=0,
        remapping_train=None,
    ):
        self.dusting_threshold = dusting_threshold
        self.ignore_labels = ignore_labels or []
        self.foreground_class_id = int(foreground_class_id)
        self.remapping_train = remapping_train
        RayWorkerBase.__init__(
            self,
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
            crop_to_label=crop_to_label,
            debug=debug,
            tfms_keys="LoadT,Chan,Dev,Crop,Remap,Labels,Stats,E,L,H",
            remapping_key=self.remapping_key,
        )

    @property
    def indices_subfolder(self):
        return infer_indices_folder(self.output_folder, self.plan)

    def _create_data_dict(self, row):
        return {
            "image": row["image"],
            "lm": row["lm"],
            "ds": row["ds"],
            "remapping": row["remapping"],
        }

    def create_transforms(self):
        super().create_transforms()
        plan = self.plan
        ignore_labels = plan["ignore_labels"]
        patch_size = tuple(int(v) for v in plan["patch_size"])
        self.Stats = DetectionBBoxStatsd(
            image_key=self.image_key,
            lm_key=self.lm_key,
            dusting_threshold=_dusting_threshold(plan),
            ignore_labels=ignore_labels,
            foreground_class_id=self.foreground_class_id,
            remapping_train=self.remapping_train,
            gt_box_mode=plan["gt_box_mode"],
        )
        self.E = StandardizeEmptyBoxd(
            box_keys=[self.box_key],
            box_ref_image_keys=self.image_key,
        )
        self.L = GenerateExtendedBoxMask(
            keys=self.box_key,
            image_key=self.image_key,
            spatial_size=patch_size,
            whole_box=True,
        )
        self.H = Compose(
            [
                EnsureTyped(keys=[self.image_key], dtype=torch.float16),
                EnsureTyped(keys=[self.box_key], dtype=torch.float32),
                EnsureTyped(keys=[self.label_key], dtype=torch.long),
            ]
        )
        self.transforms_dict["Stats"] = self.Stats
        self.transforms_dict["E"] = self.E
        self.transforms_dict["L"] = self.L
        self.transforms_dict["H"] = self.H

    def save_bbox_sidecar(self, data, fn_name):
        stem = strip_extension(fn_name)
        out_fn = bbox_sidecar_path(self.output_folder / "bboxes", stem)
        box = data[self.box_key]
        label = data[self.label_key]
        if box.shape[0] == 0:
            boxes = []
            labels = []
        else:
            boxes = [box[i] for i in range(box.shape[0])]
            labels = [label[i] for i in range(label.shape[0])]
        save_detection_sidecar(
            out_fn,
            boxes,
            labels,
            ignore_labels=list(self.ignore_labels),
        )

    def save_mask_pt(self, data, image):
        mask = torch.as_tensor(data["mask_image"])
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)
        mask = MetaTensor(mask, meta=dict(image.meta))
        self.save_pt(mask[0], "masks")

    def _process_row(self, row: pd.Series):
        case_id = row["case_id"]
        data = self._create_data_dict(row)
        data = self.apply_transforms(data)
        image = data["image"]
        lm = data["lm"]
        assert image.shape == lm.shape, "mismatch in shape"
        assert image.dim() == 4, "images should be cxhxwxd"
        if image.numel() <= MIN_SIZE**3:
            return {
                "case_id": case_id,
                "ok": False,
                "err": "image too small after label crop",
            }
        fn_name = strip_extension(Path(str(row["image"])).name) + ".pt"
        fg, bg = mask_fg_bg_flat_indices(data["mask_image"])
        inds = {
            "fg_indices": fg,
            "bg_indices": bg,
            "meta": image.meta,
        }
        self.save_indices(inds, self.indices_subfolder)
        self.save_pt(image[0], "images")
        self.save_mask_pt(data, image)
        self.save_bbox_sidecar(data, fn_name)
        return {
            "case_id": case_id,
            "ok": True,
            "shape": list(image.shape),
            "n_boxes": int(data[self.box_key].shape[0]),
        }


@ray.remote(num_cpus=CPUS_PER_ACTOR)
class LBDDetWorkerImpl(_LBDDetWorker):
    pass


class LBDDetWorkerLocal(_LBDDetWorker):
    pass


class LabelBoundedDetDataGenerator(LabelBoundedDataGenerator):
    """LBD detection preproc: label crop → standard boxes, crop mask, fp16 image per case."""

    hdf5_shards = True
    actor_cls = LBDDetWorkerImpl
    local_worker_cls = LBDDetWorkerLocal

    def __init__(self, project, plan, data_folder, output_folder=None):
        crop_to_label = int(plan["lbd_crop_label"])
        LabelBoundedDataGenerator.__init__(
            self,
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
            crop_to_label=crop_to_label,
        )
        self.hdf5_shards = True

    def process_hdf5(
        self,
        cases_per_shard=5,
        overwrite_hdf5_shards=False,
        hdf5_compression="gzip",
        hdf5_compression_opts=1,
        num_processes=8,
    ):
        if not self.hdf5_shards:
            return []
        if overwrite_hdf5_shards:
            self.df["hdf5_processed"] = None
        writer = DetHDF5ShardGenerator(
            project=self.project,
            plan=self.plan,
            data_folder=self.output_folder,
            output_folder=self.hdf5_output_folder,
            indices_folder=self.indices_subfolder,
            cases_per_shard=cases_per_shard,
            compression=hdf5_compression,
            compression_opts=hdf5_compression_opts,
        )
        writer.setup(overwrite=overwrite_hdf5_shards)
        writer.run(num_processes=num_processes, overwrite=overwrite_hdf5_shards)

    def extra_worker_kwargs(self, mean_std_mode="dataset"):
        plan = self.plan
        return {
            "crop_to_label": int(plan["lbd_crop_label"]),
            "dusting_threshold": _dusting_threshold(plan),
            "ignore_labels": plan["ignore_labels"],
            "foreground_class_id": int(plan.get("foreground_class_id", 0)),
            "remapping_train": plan.get("remapping_train"),
        }

    def set_input_output_folders(self, data_folder, output_folder):
        self.data_folder = Path(data_folder)
        if output_folder is not None:
            self.output_folder = Path(output_folder)
        else:
            self.output_folder = lbd_det_folder_from_plan(self.project, self.plan)

    def create_output_folders(self):
        maybe_makedirs(
            [
                self.output_folder / "images",
                self.output_folder / "masks",
                self.output_folder / "bboxes",
                self.indices_subfolder,
            ]
        )

    def _register_existing_pt_files(self):
        existing_img = {p.name for p in (self.output_folder / "images").glob("*.pt")}
        existing_mask = {p.name for p in (self.output_folder / "masks").glob("*.pt")}
        bbox_stems = {p.stem for p in (self.output_folder / "bboxes").glob("*.json")}
        self.existing_pt_fnames = {
            fn
            for fn in existing_img.intersection(existing_mask)
            if strip_extension(fn) in bbox_stems
        }
        print("Output folder: ", self.output_folder)
        print(
            "LBD det case files fully processed in a previous session: ",
            len(self.existing_pt_fnames),
        )
        case_ids_done = [strip_extension(fn) for fn in self.existing_pt_fnames]
        self.df.loc[self.df["case_id"].isin(case_ids_done), "pt_processed"] = True

    def postprocess(self, overwrite=False, num_processes=8):
        if overwrite is False and not self.postprocess_artifacts_missing():
            return
        labels_all = set()
        for bbox_fn in (self.output_folder / "bboxes").glob("*.json"):
            sidecar = json.loads(bbox_fn.read_text())
            labels_all.update(int(v) for v in sidecar["label"])
        save_json(sorted(labels_all), self.output_folder / "labels_all.json")
        write_dataset_details_csv(self.output_folder, overwrite=True)

    def postprocess_artifacts_missing(self):
        return not (self.output_folder / "labels_all.json").is_file() or not (
            self.output_folder / "dataset_details.csv"
        ).is_file()


# SECTION:-------------------- setup--------------------------------------------------------------------------------------

if __name__ == "__main__":

    from det3d.configs.parser import ConfigMakerDet
    from det3d.preprocessing.object_bounded import resolve_input_folder
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers import Project

    project_title = "lidca"
    plan_id = 1
    project = Project(project_title=project_title)
    config_maker = ConfigMakerDet(project)
    config_maker.setup(plan_id)
    plan = config_maker.configs["plan_train"]
    debug_ = False
    overwrite=False
    overwrite_hdf5_shards=False
# %%
    input_folder = resolve_input_folder(project, plan, input_folder=None)
    num_processes = 16
# %%

    G = LabelBoundedDetDataGenerator(
        project=project,
        plan=plan,
        data_folder=input_folder,
    )

    G.setup()
    print(G.output_folder)
# %%
    G.run(overwrite=overwrite, num_processes=num_processes, overwrite_hdf5_shards=overwrite_hdf5_shards)

# %%
    cfg = ConfigMakerDet(project=project, plan=plan_id).cfg
    G = LabelBoundedDetDataGenerator(
        project=project,
        plan=cfg,
        data_folder=data_folder,
    )
# %%
    gen.run(num_processes=16)
# %%
