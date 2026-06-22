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
from det3d.preprocessing.hdf5_shards_det import DetHDF5ShardGenerator
from det3d.transforms.bbox_stats import DetectionBBoxStatsd
from det3d.transforms.crop_indices import (
    mask_fg_bg_flat_indices,
    volume_fg_bg_flat_indices,
    volume_fg_flat_indices,
)
from det3d.transforms.detection import GenerateExtendedBoxMask
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
        ignore_labels=None,
    ):
        self.dusting_threshold = dusting_threshold
        self.ignore_labels = [] if ignore_labels is None else listify(ignore_labels)
        _LBDSamplerWorkerBase.__init__(
            self,
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
            tfms_keys="LoadT,Chan,Dev,Crop,Remap,Labels,Indx,Stats,E,L,H",
        )

    def create_transforms(self):
        super().create_transforms()
        plan = self.plan
        patch_size = tuple(int(v) for v in plan["patch_size"])
        self.Stats = DetectionBBoxStatsd(
            image_key=self.image_key,
            lm_key=self.lm_key,
            dusting_threshold=dusting_threshold(plan),
            ignore_labels=self.ignore_labels,
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
                EnsureTyped(keys=[self.lm_key], dtype=torch.uint8),
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
        mask = torch.as_tensor(data["mask_image"], dtype=torch.uint8)
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
        src_fn = str(row["image"])
        save_meta = dict(image.meta)
        save_meta["filename_or_obj"] = src_fn
        image.meta = save_meta
        lm.meta = save_meta
        fg, bg = volume_fg_bg_flat_indices(data["mask_image"])
        inds = {
            "fg_indices": fg,
            "bg_indices": bg,
            "lm_fg_indices": volume_fg_flat_indices(lm[0]),
            "meta": image.meta,
        }
        self.save_indices(inds, self.indices_subfolder)
        self.save_pt(image[0], "images")
        self.save_pt(lm[0], "lms")
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
            "dusting_threshold": dusting_threshold(plan),
            "ignore_labels": plan["ignore_labels_cc"],
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
                self.output_folder / "masks",
                self.output_folder / "bboxes",
                self.indices_subfolder,
            ]
        )

    def _register_existing_pt_files(self):
        existing_img = {p.name for p in (self.output_folder / "images").glob("*.pt")}
        existing_lm = {p.name for p in (self.output_folder / "lms").glob("*.pt")}
        existing_mask = {p.name for p in (self.output_folder / "masks").glob("*.pt")}
        bbox_stems = {p.stem for p in (self.output_folder / "bboxes").glob("*.json")}
        self.existing_pt_fnames = {
            fn
            for fn in existing_img.intersection(existing_lm).intersection(existing_mask)
            if strip_extension(fn) in bbox_stems
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
            save_json(sorted(labels_all), self.output_folder / "labels_all.json")
            write_dataset_details_csv(self.output_folder, overwrite=True)
        store_label_count(self.output_folder, num_processes=num_processes)

    def postprocess_artifacts_missing(self):
        return (
            not (self.output_folder / "labels_all.json").is_file()
            or not (self.output_folder / "dataset_details.csv").is_file()
        )


# %%
# SECTION:-------------------- setup-------------------------------------------------------------------------------------- <CR>
if __name__ == "__main__":
    from det3d.configs.parser import ConfigMakerDet
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers import Project

    project_title = "lidca"
    plan_id = 2
    project = Project(project_title=project_title)
    config_maker = ConfigMakerDet(project)
    config_maker.setup(plan_id)
    plan = config_maker.configs["plan_train"]
    overwrite = False
    overwrite_hdf5_shards =False
    folders = FolderNames(project=project, plan=plan).folders
    folder_src = folders["data_folder_source"]
    folder_lbd = folders["data_folder_lbd"]
# %%
    num_processes = 16
    num_processes = 1
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

#SECTION:-------------------- LBDWorker--------------------------------------------------------------------------------------
    debug_ =True
# %%
    L = LBDDetWorkerLocal(
        project=project,
        plan=plan,
        data_folder=folder_src,
        output_folder=folder_lbd,
    )


    L.setup(debug=debug_)
# %%
    L._process_row(G.df.iloc[0])
# %%  # T:block_start|_LBDDetWorker._process_row
#SECTION:-------------------- _process_row--------------------------------------------------------------------------------------  # T:block_meta|_LBDDetWorker._process_row
    # requires L = _LBDDetWorker(...) in __main__  # T:requires_alias|L = _LBDDetWorker(...)
    for ids,row in G.df.iterrows():
        case_id = row["case_id"]
        data = L._create_data_dict(row)  # T:self_ref|data = self._create_data_dict(row)
        data = L.apply_transforms(data)  # T:self_ref|data = self.apply_transforms(data)
        data.keys()
        img = data['image']
        bbox = data['bbox']
        lm = data['lm']
        LMG = data['LMG']
        data['label']
        print(LMG.nbrhoods)
        tr()
        df = LMG.nbrhoods
        d = {str(k): int(v) for k, v in zip(df.label_cc, df.label_org)}
        LMG.li_cc_sitk
        import SimpleITK as sitk
        lm2 = sitk.GetArrayFromImage(LMG.li_cc_sitk)
        lm3 = torch.as_tensor(lm2)
        lm4 = lm3.permute(2,1,0)
        lm4 = lm4.long()
        im2 = img[0]
        ImageMaskViewer([im2,lm4],'im')
        lm4.unique()
        torch.save(lm4,"/s/tmp/lm4.pt")
        ImageMaskViewer([im2, lm4],'im')
        LMG.li_cc.GetDirection()
        LMG.li_cc.GetSpacing()
        LMG.li_cc.GetOrigin()

        lm.meta
# %%

        




# %

    ImageMaskViewer([img, lm],'im')
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
        "lm_fg_indices": data["lm_fg_indices"],
        "meta": image.meta,
    }
    L.save_indices(inds, L.indices_subfolder)  # T:self_ref|self.save_indices(inds, self.indices_subfolder)
    L.save_pt(image[0], "images")  # T:self_ref|self.save_pt(image[0], "images")
    L.save_pt(lm[0], "lms")  # T:self_ref|self.save_pt(lm[0], "lms")
    L.save_mask_pt(data, image)  # T:self_ref|self.save_mask_pt(data, image)
    L.save_bbox_sidecar(data, fn_name)  # T:self_ref|self.save_bbox_sidecar(data, fn_name)
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
