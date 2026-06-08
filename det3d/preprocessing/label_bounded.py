from pathlib import Path

import pandas as pd
import ray
from fran.preprocessing.helpers import infer_indices_folder
from fran.preprocessing.labelbounded import LabelBoundedDataGenerator
from fran.preprocessing.preprocessor import CPUS_PER_ACTOR, store_label_count
from fran.preprocessing.rayworker_base import MIN_SIZE, RayWorkerBase
from monai.transforms import ScaleIntensityRanged
from utilz.fileio import maybe_makedirs
from utilz.stringz import strip_extension

from det3d.preprocessing.object_bounded import _dusting_threshold
from det3d.transforms.bbox_stats import DetectionBBoxStatsd
from det3d.utils.bbox_sidecar import bbox_sidecar_path, save_detection_sidecar
from det3d.utils.folder_names import lbd_det_folder_from_plan


class _LBDDetWorker(RayWorkerBase):
    """Label-bounded detection worker: fixed_spacing PT in → one cropped volume + bbox sidecar."""

    remapping_key = "remapping_lbd_rbd"

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
            tfms_keys="LoadT,Chan,Dev,Crop,Remap,Labels,Indx,Stats,Int",
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
        self.Stats = DetectionBBoxStatsd(
            image_key=self.image_key,
            lm_key=self.lm_key,
            dusting_threshold=_dusting_threshold(plan),
            ignore_labels=ignore_labels,
            foreground_class_id=self.foreground_class_id,
            remapping_train=self.remapping_train,
            gt_box_mode=plan["gt_box_mode"],
        )
        self.Int = ScaleIntensityRanged(
            keys=[self.image_key],
            a_min=float(plan["intensity_a_min"]),
            a_max=float(plan["intensity_a_max"]),
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
        self.transforms_dict["Stats"] = self.Stats
        self.transforms_dict["Int"] = self.Int

    def save_bbox_sidecar(self, data, fn_name):
        stem = strip_extension(fn_name)
        out_fn = bbox_sidecar_path(self.output_folder / "bboxes", stem)
        save_detection_sidecar(
            out_fn,
            data["detection_box"],
            data["detection_label"],
            ignore_labels=list(self.ignore_labels),
        )

    def _process_row(self, row: pd.Series):
        case_id = row["case_id"]
        data = self._create_data_dict(row)
        data = self.apply_transforms(data)
        data = self.Int(data)
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
        inds = {
            "lm_fg_indices": data["lm_fg_indices"],
            "lm_bg_indices": data["lm_bg_indices"],
            "meta": image.meta,
        }
        self.save_indices(inds, self.indices_subfolder)
        self.save_pt(image[0], "images")
        self.save_pt(lm[0], "lms")
        self.save_bbox_sidecar(data, fn_name)
        return {
            "case_id": case_id,
            "ok": True,
            "shape": list(image.shape),
            "n_boxes": len(data["detection_box"]),
        }


@ray.remote(num_cpus=CPUS_PER_ACTOR)
class LBDDetWorkerImpl(_LBDDetWorker):
    pass


class LBDDetWorkerLocal(_LBDDetWorker):
    pass


class LabelBoundedDetDataGenerator(LabelBoundedDataGenerator):
    """LBD detection preproc: label crop → lesion bboxes on one volume per case."""

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
                self.output_folder / "lms",
                self.output_folder / "bboxes",
                self.indices_subfolder,
            ]
        )

    def _register_existing_pt_files(self):
        existing_img = {p.name for p in (self.output_folder / "images").glob("*.pt")}
        existing_lm = {p.name for p in (self.output_folder / "lms").glob("*.pt")}
        bbox_stems = {p.stem for p in (self.output_folder / "bboxes").glob("*.json")}
        self.existing_pt_fnames = {
            fn
            for fn in existing_img.intersection(existing_lm)
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
        labels_fn = self.output_folder / "labels_all.json"
        if overwrite is False and labels_fn.exists():
            return
        store_label_count(self.output_folder, num_processes=num_processes)

    def postprocess_artifacts_missing(self):
        return not (self.output_folder / "labels_all.json").exists()



# SECTION:-------------------- setup--------------------------------------------------------------------------------------

if __name__ == "__main__":

    from det3d.configs.parser import ConfigMakerDet
    from det3d.preprocessing.object_bounded import resolve_input_folder
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers import Project

    project_title = "lidc"
    plan_id = 1
    project = Project(project_title=project_title)
    config_maker = ConfigMakerDet(project)
    config_maker.setup(plan_id)
    plan = config_maker.configs["plan_train"]
    debug_ = False
    overwrite=False
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
# %%
    G.run(overwrite=overwrite, num_processes=num_processes)
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


