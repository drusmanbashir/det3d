from pathlib import Path

import pandas as pd
import ray
from det3d.transforms.bbox_stats import AttachDetectionGTd
from det3d.transforms.patch_size import NbrhoodsToPatchesOBDD
from det3d.utils.bbox_sidecar import bbox_sidecar_path, save_detection_sidecar
from det3d.utils.folder_names import obd_folder_from_plan
from dot.transforms.transforms import BBoxInfoStatsd2
from fran.preprocessing.preprocessor import CPUS_PER_ACTOR, Preprocessor
from fran.preprocessing.rayworker_base import RayWorkerBase
from fran.transforms.imageio import LoadTorchd
from fran.utils.folder_names import FolderNames
from monai.transforms import ScaleIntensityRanged
from monai.transforms.utility.dictionary import EnsureChannelFirstd, ToDeviced
from utilz.fileio import maybe_makedirs
from utilz.stringz import strip_extension


def _dusting_threshold(plan):
    dusting = plan.get("dusting_mm")
    if dusting is None:
        dusting = 3.0
    return float(dusting)


class _OBJWorker(RayWorkerBase):
    """Object-bounded detection worker: fixed_spacing PT in → strict-bbox patches out.

    expand_by=0; no preproc min/max resize/pad — train batch_size=1, native patch shape.
    """

    def __init__(
        self,
        project,
        plan,
        data_folder,
        output_folder,
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
            debug=debug,
            tfms_keys="LoadT,Chan,Dev,Stats,N2P,AttachGT,Int",
            remapping_key=None,
        )

    def create_transforms(self):
        self.Dev = ToDeviced(device="cpu", keys=self.tnsr_keys)
        self.Chan = EnsureChannelFirstd(keys=self.tnsr_keys, channel_dim="no_channel")
        self.LoadT = LoadTorchd(keys=[self.image_key, self.lm_key])
        self.transforms_dict = {
            "Dev": self.Dev,
            "Chan": self.Chan,
            "LoadT": self.LoadT,
        }
        plan = self.plan
        ignore_labels = plan["ignore_labels"]
        self.Stats = BBoxInfoStatsd2(
            image_key=self.image_key,
            lm_key=self.lm_key,
            dusting_threshold=_dusting_threshold(plan),
            dusting_method=plan.get("dusting_method", "major_axis"),
            ignore_labels=ignore_labels,
        )
        self.N2P = NbrhoodsToPatchesOBDD(
            keys=[self.image_key, self.lm_key],
            nbrhoods_key="nbrhoods",
            expand_mode=plan.get("expand_mode", "mm"),
            expand_by=0,
            bbox_key="bbox",
            nbrhood_outkey="stats",
        )
        self.AttachGT = AttachDetectionGTd(
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
        self.transforms_dict["N2P"] = self.N2P
        self.transforms_dict["AttachGT"] = self.AttachGT
        self.transforms_dict["Int"] = self.Int

    def _create_data_dict(self, row):
        return {"image": row["image"], "lm": row["lm"]}

    def save_bbox_sidecar(self, patch, suffix):
        image = patch[self.image_key]
        fn = Path(image[0].meta["filename_or_obj"])
        stem = strip_extension(fn.name) + "_" + suffix
        out_fn = bbox_sidecar_path(self.output_folder / "bboxes", stem)
        save_detection_sidecar(
            out_fn,
            patch["box"],
            patch["label"],
            ignore_labels=list(self.ignore_labels),
        )

    def _process_row(self, row: pd.Series):
        case_id = row["case_id"]
        data = self._create_data_dict(row)
        data = self.apply_transforms(data)
        row_results = []
        for n, patch in enumerate(data):
            if "box" not in patch or len(patch["box"]) == 0:
                continue
            image = patch["image"]
            lm = patch["lm"]
            assert image.shape == lm.shape, "mismatch in shape"
            assert image.dim() == 4, "images should be cxhxwxd"
            suffix = f"bbox{n}"
            stem = (
                strip_extension(Path(image[0].meta["filename_or_obj"]).name)
                + "_"
                + suffix
            )
            pt_name = stem + ".pt"
            if (
                (self.output_folder / "images" / pt_name).exists()
                and (self.output_folder / "lms" / pt_name).exists()
                and (self.output_folder / "bboxes" / f"{stem}.json").exists()
            ):
                continue
            self.save_pt(image[0], "images", suffix=suffix)
            self.save_pt(lm[0], "lms", suffix=suffix)
            self.save_bbox_sidecar(patch, suffix)
            row_results.append(
                {
                    "case_id": case_id,
                    "ok": True,
                    "shape": list(image.shape),
                    "bbox_index": n,
                    "patch_suffix": suffix,
                }
            )
        return row_results


@ray.remote(num_cpus=CPUS_PER_ACTOR)
class OBJWorkerImpl(_OBJWorker):
    pass


class OBJWorkerLocal(_OBJWorker):
    pass


def resolve_input_folder(project, plan, input_folder=None):
    """Fixed_spacing PT folder (orientation/spacing/remap already applied)."""
    if input_folder is not None:
        return Path(input_folder)
    return Path(FolderNames(project, plan).folders["data_folder_source"])


class ObjectBoundedDataGenerator(Preprocessor):
    hdf5_shards = False

    actor_cls = OBJWorkerImpl
    local_worker_cls = OBJWorkerLocal

    def __init__(self, project, plan, data_folder, output_folder=None):
        super().__init__(
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
        )

    def extra_worker_kwargs(self, mean_std_mode="dataset"):
        plan = self.plan
        return {
            "dusting_threshold": _dusting_threshold(plan),
            "ignore_labels": plan["ignore_labels"],
            "foreground_class_id": int(plan.get("foreground_class_id", 0)),
            "remapping_train": plan.get("remapping_train"),
        }

    def should_use_ray(self, num_processes=8):
        return (num_processes > 1) and (not getattr(self, "debug", False))

    def setup(self, mean_std_mode="dataset", debug=False):
        super().setup(debug=debug, mean_std_mode=mean_std_mode)

    def set_input_output_folders(self, data_folder, output_folder):
        self.data_folder = Path(data_folder)
        if output_folder is not None:
            self.output_folder = Path(output_folder)
        else:
            self.output_folder = obd_folder_from_plan(self.project, self.plan)

    def create_output_folders(self):
        maybe_makedirs(
            [
                self.output_folder / "images",
                self.output_folder / "lms",
                self.output_folder / "bboxes",
            ]
        )

    def postprocess(self, overwrite=False, num_processes=8):
        return

    def postprocess_artifacts_missing(self):
        return False

    def build_preprocessing_log_rows(self, results):
        rows = []
        for mini_df, worker_outs in zip(self.mini_dfs, results):
            if len(worker_outs) != len(mini_df):
                raise ValueError(
                    "Worker output length mismatch: "
                    f"got {len(worker_outs)} rows for mini_df of size {len(mini_df)}"
                )
            for (_, src_row), worker_out in zip(mini_df.iterrows(), worker_outs):
                case_id = self._coerce_log_value(src_row.get("case_id"))
                image = self._coerce_log_value(src_row.get("image"))
                lm = self._coerce_log_value(src_row.get("lm"))
                if isinstance(worker_out, dict) and "_preprocess_error" in worker_out:
                    err_info = worker_out["_preprocess_error"]
                    rows.append(
                        {
                            "case_id": case_id,
                            "status": "ERROR",
                            "image": image,
                            "lm": lm,
                            "error_type": self._coerce_log_value(
                                err_info.get("error_type")
                            ),
                            "error_message": self._coerce_log_value(
                                err_info.get("error_message")
                            ),
                            "traceback": self._coerce_log_value(
                                err_info.get("traceback")
                            ),
                        }
                    )
                    continue
                patch_rows = worker_out if isinstance(worker_out, list) else []
                if not patch_rows:
                    rows.append(
                        {
                            "case_id": case_id,
                            "status": "OK",
                            "image": image,
                            "lm": lm,
                            "error_type": "",
                            "error_message": "",
                            "traceback": "",
                        }
                    )
                    continue
                for patch in patch_rows:
                    rows.append(
                        {
                            "case_id": case_id,
                            "status": "OK",
                            "image": image,
                            "lm": lm,
                            "bbox_index": patch["bbox_index"],
                            "patch_suffix": patch["patch_suffix"],
                            "error_type": "",
                            "error_message": "",
                            "traceback": "",
                        }
                    )
        return rows

    def _register_existing_pt_files(self):
        existing_img = {
            p.name for p in (self.output_folder / "images").glob("*_bbox*.pt")
        }
        existing_lm = {p.name for p in (self.output_folder / "lms").glob("*_bbox*.pt")}
        bbox_stems = {
            p.stem for p in (self.output_folder / "bboxes").glob("*_bbox*.json")
        }
        self.existing_pt_fnames = {
            fn
            for fn in existing_img.intersection(existing_lm)
            if strip_extension(fn) in bbox_stems
        }
        print("Output folder: ", self.output_folder)
        print(
            "OBD patch files fully processed in a previous session: ",
            len(self.existing_pt_fnames),
        )


# %%
# SECTION:-------------------- --------------------------------------------------------------------------------------
if __name__ == "__main__":
    from det3d.configs.parser import ConfigMakerDet
    from fran.managers import Project

    project_title = "lidc"
    plan_id = 1
    project = Project(project_title=project_title)
    config_maker = ConfigMakerDet(project)
    config_maker.setup(plan_id)
    plan = config_maker.configs["plan_train"]
    debug_ = False
# %%
    input_folder = resolve_input_folder(project, plan, input_folder=None)
# %%

    G = ObjectBoundedDataGenerator(
        project=project,
        plan=plan,
        data_folder=input_folder,
    )
    G.setup(debug=debug_, mean_std_mode="dataset")
    overwrite = True
    num_processes = 8
    # G.use_ray=False
# %%
    G.run(overwrite=overwrite, num_processes=num_processes)
# %%
    df = G.df
    df_pt_run = G.df_pt
# %%
    G.initialize_process_state()
    worker_kwargs = G.extra_worker_kwargs(mean_std_mode=G.mean_std_mode)
    G.mini_dfs = [df_pt_run]
    G.local_worker = G.local_worker_cls(
        project=G.project,
        plan=G.plan,
        data_folder=G.data_folder,
        output_folder=G.output_folder,
        debug=G.debug,
        **worker_kwargs,
    )
    L = G.local_worker
    row = df_pt_run.iloc[0]
    L._process_row(row)
# %%


