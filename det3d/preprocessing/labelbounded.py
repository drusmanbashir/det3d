from pathlib import Path
import json
import ipdb

tr = ipdb.set_trace

import pandas as pd
import ray
import torch
from fastcore.basics import listify
from fran.preprocessing.labelbounded import (
    LabelBoundedDataGenerator,
    _LBDSamplerWorkerBase,
)
from fran.preprocessing.preprocessor import CPUS_PER_ACTOR, store_label_count
from fran.preprocessing.rayworker_base import MIN_SIZE
from fran.utils.folder_names import FolderNames
from monai.apps.detection.transforms.dictionary import StandardizeEmptyBoxd
from monai.data import MetaTensor
from monai.transforms import Compose, EnsureTyped
from utilz.fileio import maybe_makedirs, save_json
from utilz.imageviewers import ImageBBoxViewer, ImageMaskViewer
from utilz.stringz import info_from_filename, strip_extension

from det3d.preprocessing.dataset_details import write_dataset_details_csv
from det3d.preprocessing.helpers import dusting_threshold
from det3d.preprocessing.hdf5_shards_det import (
    DetHDF5ShardGenerator,
    ensure_hdf5_shards_for_plan,
)
from det3d.transforms.bbox_stats import DetectionBBoxStatsd
from det3d.transforms.detection import (
    ExtendedBBoxesByPatchSizesd,
    build_extended_bboxes_manifest_payload,
)
from det3d.utils.bbox_sidecar import (
    bbox_sidecar_path,
    save_detection_sidecar,
)


class _LBDDetWorker(_LBDSamplerWorkerBase):
    """Label-bounded detection worker: fran LBD + bbox sidecar + bbox RandCrop mask."""

    box_key = "bbox"
    label_key = "label"

    def __init__(
        self,
        project,
        plan,
        data_folder,
        output_folder,
        dusting_threshold=3.0,
        ignore_labels_cc=None,
        extended_bbox_dim_x=(128, 160, 192),
        extended_bbox_dim_y=(128, 160, 192),
        extended_bbox_dim_z=(64, 96, 128),
        extended_bbox_same_xy=True,
        extended_bbox_prepatch_zoom_scales=(1.1, 1.2, 1.4),
    ):
        self.dusting_threshold = dusting_threshold
        self.ignore_labels_cc = [] if ignore_labels_cc is None else listify(ignore_labels_cc)
        self.extended_bbox_dim_x = extended_bbox_dim_x
        self.extended_bbox_dim_y = extended_bbox_dim_y
        self.extended_bbox_dim_z = extended_bbox_dim_z
        self.extended_bbox_same_xy = extended_bbox_same_xy
        self.extended_bbox_prepatch_zoom_scales = extended_bbox_prepatch_zoom_scales
        _LBDSamplerWorkerBase.__init__(
            self,
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
            tfms_keys="LoadT,Chan,Dev,Crop,Remap,Labels,Indx,Stats,E,Ext,H",
        )

    def create_transforms(self):
        super().create_transforms()
        self.Stats = DetectionBBoxStatsd(
            image_key=self.image_key,
            lm_key=self.lm_key,
            dusting_threshold=dusting_threshold(self.plan),
            ignore_labels=self.ignore_labels_cc,
            gt_box_mode=self.plan["gt_box_mode"],
        )
        self.E = StandardizeEmptyBoxd(
            box_keys=[self.box_key],
            box_ref_image_keys=self.image_key,
        )
        self.Ext = ExtendedBBoxesByPatchSizesd(
            keys=self.box_key,
            box_key=self.box_key,
            image_key=self.image_key,
            dim_x=self.extended_bbox_dim_x,
            dim_y=self.extended_bbox_dim_y,
            dim_z=self.extended_bbox_dim_z,
            same_xy=self.extended_bbox_same_xy,
            whole_box=False,
            prepatch_zoom_scales=self.extended_bbox_prepatch_zoom_scales,
        )
        self.H = Compose(
            [
                EnsureTyped(keys=[self.image_key], dtype=torch.float16),
                EnsureTyped(keys=[self.lm_key], dtype=torch.uint8),
                EnsureTyped(keys=[self.box_key], dtype=torch.float32),
                EnsureTyped(keys=[self.label_key], dtype=torch.long),
            ]
        )
        self.transforms_dict["Stats"] = self.Stats
        self.transforms_dict["E"] = self.E
        self.transforms_dict["Ext"] = self.Ext
        self.transforms_dict["H"] = self.H

    def save_extended_bboxes(self, data, case_id):
        out_fn = self.output_folder / "extended_bboxes" / f"{case_id}.json"
        save_json(data["extended_bboxes"], out_fn)

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
            ignore_labels=list(self.ignore_labels_cc),
        )

    def _process_row(self, row: pd.Series):

        case_id = row["case_id"]
        data = row.to_dict()
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
        src_fn = str(row["image"])
        save_meta = dict(image.meta)
        save_meta["filename_or_obj"] = src_fn
        image.meta = save_meta
        lm.meta = save_meta
        self.save_pt(image[0], "images")
        self.save_pt(lm[0], "lms")
        self.save_bbox_sidecar(data, fn_name)
        self.save_extended_bboxes(data, case_id)
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

    def __init__(
        self,
        project,
        plan,
        data_folder,
        output_folder=None,
        extended_bbox_dim_x=(128, 160, 192),
        extended_bbox_dim_y=(128, 160, 192),
        extended_bbox_dim_z=(64, 96, 128),
        extended_bbox_same_xy=True,
        extended_bbox_prepatch_zoom_scales=(1.1, 1.2, 1.4),
    ):
        self.extended_bbox_dim_x = extended_bbox_dim_x
        self.extended_bbox_dim_y = extended_bbox_dim_y
        self.extended_bbox_dim_z = extended_bbox_dim_z
        self.extended_bbox_same_xy = extended_bbox_same_xy
        self.extended_bbox_prepatch_zoom_scales = extended_bbox_prepatch_zoom_scales
        LabelBoundedDataGenerator.__init__(
            self,
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
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
        _, linked = ensure_hdf5_shards_for_plan(
            self.output_folder, self.plan["src_dims"]
        )
        writer = DetHDF5ShardGenerator(
            project=self.project,
            plan=self.plan,
            data_folder=self.output_folder,
            output_folder=self.hdf5_output_folder,
            cases_per_shard=cases_per_shard,
            compression=hdf5_compression,
            compression_opts=hdf5_compression_opts,
        )
        writer.setup(overwrite=overwrite_hdf5_shards)
        if not linked:
            writer.run(num_processes=num_processes, overwrite=overwrite_hdf5_shards)

    def extra_worker_kwargs(self, mean_std_mode="dataset"):
        plan = self.plan
        return {
            "dusting_threshold": dusting_threshold(plan),
            "ignore_labels_cc": plan["ignore_labels_cc"],
            "extended_bbox_dim_x": self.extended_bbox_dim_x,
            "extended_bbox_dim_y": self.extended_bbox_dim_y,
            "extended_bbox_dim_z": self.extended_bbox_dim_z,
            "extended_bbox_same_xy": self.extended_bbox_same_xy,
            "extended_bbox_prepatch_zoom_scales": self.extended_bbox_prepatch_zoom_scales,
        }

    # def set_input_output_folders(self, data_folder, output_folder):
    #     self.data_folder = Path(data_folder)
    #     if output_folder is not None:
    #         self.output_folder = Path(output_folder)
    #
    def create_output_folders(self):
        maybe_makedirs(
            [
                self.output_folder / "images",
                self.output_folder / "lms",
                self.output_folder / "bboxes",
                self.output_folder / "extended_bboxes",
            ]
        )

    def _register_existing_pt_files(self):
        existing_img = {p.name for p in (self.output_folder / "images").glob("*.pt")}
        existing_lm = {p.name for p in (self.output_folder / "lms").glob("*.pt")}
        bbox_stems = {p.stem for p in (self.output_folder / "bboxes").glob("*.json")}
        ext_stems = {p.stem for p in (self.output_folder / "extended_bboxes").glob("*.json")}
        self.existing_pt_fnames = {
            fn
            for fn in existing_img.intersection(existing_lm)
            if strip_extension(fn) in bbox_stems and strip_extension(fn) in ext_stems
        }
        print("Output folder: ", self.output_folder)
        print(
            "LBD det case files fully processed in a previous session: ",
            len(self.existing_pt_fnames),
        )
        case_ids_done = [
            info_from_filename(fn, full_caseid=True)["case_id"]
            for fn in self.existing_pt_fnames
        ]
        self.df.loc[self.df["case_id"].isin(case_ids_done), "pt_processed"] = True

    def postprocess(self, overwrite=False, num_processes=8):
        if overwrite or self.postprocess_artifacts_missing():
            labels_all = set()
            for bbox_fn in (self.output_folder / "bboxes").glob("*.json"):
                sidecar = json.loads(bbox_fn.read_text())
                labels_all.update(int(v) for v in sidecar["label"])
            manifest = build_extended_bboxes_manifest_payload(
                self.extended_bbox_dim_x,
                self.extended_bbox_dim_y,
                self.extended_bbox_dim_z,
                same_xy=self.extended_bbox_same_xy,
                whole_box=False,
                labels_all=labels_all,
                prepatch_zoom_scales=self.extended_bbox_prepatch_zoom_scales,
            )
            save_json(manifest, self.output_folder / "manifest.json")
            write_dataset_details_csv(self.output_folder, overwrite=True)
        store_label_count(self.output_folder, num_processes=num_processes)

    def postprocess_artifacts_missing(self):
        manifest_fn = self.output_folder / "manifest.json"
        if not manifest_fn.is_file():
            return True
        manifest = json.loads(manifest_fn.read_text())
        if "labels_all" not in manifest or "extended_bboxes_patch_sizes" not in manifest:
            return True
        return not (self.output_folder / "dataset_details.csv").is_file()


# %%
# SECTION:-------------------- setup-------------------------------------------------------------------------------------- <CR> <CR>
if __name__ == "__main__":
    from det3d.configs.parser import ConfigMakerDet
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers import Project

    project_title = "lidca"
    plan_id = 3
    project = Project(project_title=project_title)
    config_maker = ConfigMakerDet(project)
    config_maker.setup(plan_id)
    plan = config_maker.configs["plan_train"]
    overwrite = False
    overwrite_hdf5_shards = False
    folders = FolderNames(project=project, plan=plan).folders
    folder_src = folders["data_folder_source"]
    folder_lbd = folders["data_folder_lbd"]
# %%
    num_processes = 1
    num_processes = 16
    G = LabelBoundedDetDataGenerator(
        project=project,
        plan=plan,
        data_folder=folder_src,
    )

# %%
    debug_ = True
    debug_ = False
    G.setup(debug=debug_)
    G.df
    print(G.output_folder)
# %%
    G.run(
        overwrite=overwrite,
        num_processes=num_processes,
        overwrite_hdf5_shards=overwrite_hdf5_shards,
    )
# %%
    G.postprocess()
# %%

# SECTION:-------------------- LBDWorker-------------------------------------------------------------------------------------- <CR>
# %%
    L = LBDDetWorkerLocal(
        project=project,
        plan=plan,
        data_folder=folder_src,
        output_folder=folder_lbd,
        ignore_labels_cc=plan["ignore_labels_cc"],
    )

    L.setup(debug=debug_)
# %%
    row = G.df.iloc[0]
    L._process_row(G.df.iloc[0])
# %%  # T:block_start|_LBDDetWorker._process_row

    tfms_keys="LoadT,Chan,Dev,Crop,Remap,Labels,Indx,Stats,E,L,H",
    tfms = L.transforms_dict

    dici = row.to_dict()

    L = tfms["LoadT"]
    dici = L(dici)

    C = tfms["Chan"]
    dici = C(dici)

    D = tfms["Dev"]
    dici = D(dici)

    Cr = tfms["Crop"]
    dici = Cr(dici)

    R = tfms["Remap"]
    dici = R["lidc"](dici)

    La = tfms["Labels"]
    dici = La(dici)

    I = tfms["Indx"]
    dici = I(dici)

    S = tfms["Stats"]
    dici = S(dici)
    print(dici.keys())
    print(dici['bbox'])

    E = tfms["E"]
    dici = E(dici)

    L2 = tfms["L"]
    dici = L2(dici)
    dici["mask_image"]
    lm = dici["mask_image"]
    im = dici["image"]
    ImageMaskViewer([im, lm],'im')

# %%
    H = tfms["H"]
    dici = H(dici)



    dici = row.to_dict()

#%%

    dici = tfms["LoadT"](dici)
    print(dici['image'].shape)
    dici.keys()

    dici = tfms["Chan"](dici)
    print(dici['image'].shape)

    dici = tfms["Dev"](dici)
    print(dici['image'].shape)

    dici = tfms["Crop"](dici)
    C = tfms["Crop"]
    C.keys
    C.source_key

    dici = C(dici)
    print(dici['image'].shape)
    im = dici['image']
    lm = dici['lm']
    ImageMaskViewer([im, lm],'im')

    dici = tfms["Remap"](dici)
    print(dici['image'].shape)

    dici = tfms["Labels"](dici)
    print(dici['image'].shape)

    dici = tfms["Indx"](dici)
    print(dici['image'].shape)

    dici = tfms["Stats"](dici)
    print(dici['image'].shape)

    dici = tfms["E"](dici)
    print(dici['image'].shape)

    dici = tfms["L"](dici)
    print(dici['image'].shape)

    dici = tfms["H"](dici)
    print(dici['image'].shape)

# %%
#%%
# SECTION:-------------------- _process_row--------------------------------------------------------------------------------------  # T:block_meta|_LBDDetWorker._process_row <CR>
    # requires L = LabelBoundedDetDataGenerator(...) in __main__  # T:requires_alias|L = LabelBoundedDetDataGenerator(...)
    case_id = row["case_id"]
    data = row.to_dict()
    data = L.apply_transforms(data)  # T:self_ref|data = self.apply_transforms(data)
    image = data["image"]
    lm = data["lm"]
    assert image.shape == lm.shape, "mismatch in shape"
    assert image.dim() == 4, "images should be cxhxwxd"

    fn_name = strip_extension(Path(str(row["image"])).name) + ".pt"
    src_fn = str(row["image"])
    save_meta = dict(image.meta)
    save_meta["filename_or_obj"] = src_fn
    image.meta = save_meta
    lm.meta = save_meta
    mask = data["mask_image"]
    fg, bg = map_binary_to_indices(mask, image=None, image_threshold=0.0)
    inds = {
        "fg_indices": fg,
        "bg_indices": bg,
        "meta": image.meta,
    }
    L.save_indices(
        inds, L.indices_subfolder
    )  # T:self_ref|self.save_indices(inds, self.indices_subfolder)
    L.save_pt(image[0], "images")  # T:self_ref|self.save_pt(image[0], "images")
    L.save_pt(lm[0], "lms")  # T:self_ref|self.save_pt(lm[0], "lms")
    L.save_mask_pt(data, image)  # T:self_ref|self.save_mask_pt(data, image)
    L.save_bbox_sidecar(
        data, fn_name
    )  # T:self_ref|self.save_bbox_sidecar(data, fn_name)
    # return {  # T:early_return|return {
    #     "case_id": case_id,
    #     "ok": True,
    #     "shape": list(image.shape),
    #     "n_boxes": int(data[L.box_key].shape[0]),  # T:self_ref|    "n_boxes": int(data[self.box_key].shape[0]),
    # }

# %

    ImageMaskViewer([img, lm], "im")
    ImageBBoxViewer(img, bbox)

    image = data["image"]
    lm = data["lm"]
    assert image.shape == lm.shape, "mismatch in shape"
    assert image.dim() == 4, "images should be cxhxwxd"
    if image.numel() <= MIN_SIZE**3:
        pass  # T:early_return|image too small after label crop
    fn_name = strip_extension(Path(str(row["image"])).name) + ".pt"
    fg, bg = volume_fg_bg_flat_indices(data["mask_image"])
    inds = {
        "fg_indices": fg,
        "bg_indices": bg,
        "meta": image.meta,
    }
    L.save_indices(
        inds, L.indices_subfolder
    )  # T:self_ref|self.save_indices(inds, self.indices_subfolder)
    L.save_pt(image[0], "images")  # T:self_ref|self.save_pt(image[0], "images")
    L.save_pt(lm[0], "lms")  # T:self_ref|self.save_pt(lm[0], "lms")
    L.save_mask_pt(data, image)  # T:self_ref|self.save_mask_pt(data, image)
    L.save_bbox_sidecar(
        data, fn_name
    )  # T:self_ref|self.save_bbox_sidecar(data, fn_name)
    pass  # T:early_return|_process_row ok
    # end PythonMethodScratch  # T:block_end|_LBDDetWorker._process_row
# %%
    L._process_row(row)

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
    row = None
# %%  # T:block_start|LabelBoundedDetDataGenerator._process_row
# /home/ub/code/det3d/det3d/preprocessing/labelbounded.py  # T:block_donor|/home/ub/code/det3d/det3d/preprocessing/labelbounded.py
# SECTION:-------------------- _process_row end --------------------------------------------------------------------------------------  # T:block_meta_end|LabelBoundedDetDataGenerator._process_row <CR>
# end PythonMethodScratch  # T:block_end|LabelBoundedDetDataGenerator._process_row
# %%
