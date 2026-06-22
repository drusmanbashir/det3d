from copy import deepcopy
from pathlib import Path

import torch
from fran.data.dataset import FillBBoxPatchesd
from fran.inference.cascade import CascadeInferer
from fran.transforms.inferencetransforms import MakeWritabled, SqueezeListofListsd
from fran.transforms.spatialtransforms import RestoreOriginalOrientationd
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToWorldCoordinated,
    ClipBoxToImaged,
    ConvertBoxModed,
)
from monai.data.meta_tensor import MetaTensor
from monai.transforms.io.dictionary import SaveImaged

from det3d.inference.patch import DetPatchInferer
from det3d.inference.transforms import (
    ArgmaxSegd,
    AttachInferenceMetad,
    AttachPredSegPathd,
    CopyBoxKeyd,
    DustSegd,
    OffsetBoxByBBoxd,
    PredBoxToNativeCropViaPointsd,
    PreservePreTfmBoxd,
    SaveInferenceMarkupsd,
    SaveInferenceSidecard,
    ScaleBoxToCropNatived,
    ScaleSegToCropNatived,
    UseFullMetaForImaged,
    WrapPredSegMetad,
)

_SEG_SAVE_KEYS = ",Sav,SavM,RestoreSeg,WriteSeg,SaveSeg,SegPath"
_BOX_WORLD_KEYS = ",Off,VoxCopy,FullMeta,World,WorldCopy,Mode,Meta"


def _decollate_image(img, batch_image):
    if img.dim() == 5:
        img = img[0]
    img = img.detach().cpu()
    if isinstance(img, MetaTensor):
        return img
    if isinstance(batch_image, MetaTensor):
        return MetaTensor(img, meta=deepcopy(batch_image.meta))
    return img


class DetCascadeInferer(CascadeInferer):
    keys_postproc = "Pre,SqL,Clip,Scale,Off,VoxCopy,FullMeta,World,WorldCopy,Mode,Meta"
    keys_postproc_safe = "SqL,Meta"
    pred_run_p = None

    @property
    def output_folder(self):
        run_name = self.pred_run_p if self.pred_run_p is not None else self.run_p
        return self.predictions_folder / run_name

    def setup_patch_inferer(self):
        return DetPatchInferer(
            run_name=self.run_p,
            project_title=self.project_title,
            devices=self.devices,
            patch_overlap=self.patch_overlap,
            safe_mode=self.safe_mode,
            params=self.params,
            debug=self.debug,
            save=False,
        )

    def decollate_patches(self, pa, bboxes, full_metas=None):
        run_name = self.P.run_name
        output = []
        for case_idx, batch in enumerate(pa[run_name]):
            batch_img = batch["image"]
            bb = bboxes[case_idx]
            crop_shape = batch["crop_spatial_shape"]
            item = {
                "image": _decollate_image(batch_img, batch_img),
                "pred_box": batch["pred_box"].detach().cpu(),
                "pred_label": batch["pred_label"].detach().cpu(),
                "pred_score": batch["pred_score"].detach().cpu(),
                "bounding_box": bb,
                "source_image": batch["source_image"],
                "crop_spatial_shape": crop_shape,
            }
            if full_metas is not None:
                item["full_meta"] = full_metas[case_idx]
            output.append(item)
        return output

    def patch_prediction(self, data):
        sources = []
        crop_shapes = []
        for dat in data:
            sources.append(dat["image"].meta["filename_or_obj"])
            crop_shapes.append(tuple(int(v) for v in dat["image"].shape[-3:]))
        preds = super().patch_prediction(data)
        for i, batch in enumerate(preds[self.P.run_name]):
            batch["source_image"] = sources[i]
            batch["crop_spatial_shape"] = crop_shapes[i]
        return preds

    def create_postprocess_transforms(self):
        plan = self.params["configs"]["plan_train"]
        gt_box_mode = plan["gt_box_mode"]
        score_min = float(plan["score_thresh"])
        ik = "image"
        bk = "pred_box"
        lk = "pred_label"
        sk = "pred_score"
        self.postprocess_transforms_dict = {
            "Pre": PreservePreTfmBoxd(box_key=bk, dst_key="pred_box_pre_tfm"),
            "SqL": SqueezeListofListsd(keys=["bounding_box"]),
            "Clip": ClipBoxToImaged(
                box_keys=[bk],
                label_keys=[lk, sk],
                box_ref_image_keys=ik,
                remove_empty=True,
            ),
            "Scale": ScaleBoxToCropNatived(box_keys=[bk], image_key=ik),
            "Off": OffsetBoxByBBoxd(box_keys=[bk]),
            "VoxCopy": CopyBoxKeyd(src_key=bk, dst_key="pred_box_voxel"),
            "FullMeta": UseFullMetaForImaged(keys=[ik]),
            "World": AffineBoxToWorldCoordinated(
                box_keys=[bk],
                box_ref_image_keys=ik,
                affine_lps_to_ras=True,
            ),
            "WorldCopy": CopyBoxKeyd(src_key=bk, dst_key="pred_box_world"),
            "Mode": ConvertBoxModed(
                box_keys=[bk],
                src_mode="xyzxyz",
                dst_mode=gt_box_mode,
            ),
            "Meta": AttachInferenceMetad(
                box_keys=[bk],
                run_w=self.run_w,
                run_p=self.run_p,
            ),
            "Sav": SaveInferenceSidecard(
                box_keys=[bk],
                label_key=lk,
                score_key=sk,
                output_dir=self.output_folder,
                world_box_key="pred_box_world",
                score_min=score_min,
            ),
            "SavM": SaveInferenceMarkupsd(
                label_key=lk,
                score_key=sk,
                output_dir=self.output_folder,
                world_box_key="pred_box_world",
                score_min=score_min,
            ),
        }

    def set_postprocess_tfms_keys(self):
        if self.safe_mode is False:
            self.postprocess_tfms_keys = self.keys_postproc
        else:
            self.postprocess_tfms_keys = self.keys_postproc_safe
        if self.save is True:
            self.postprocess_tfms_keys += ",Sav,SavM"

    def postprocess_iterate(self, batch):
        if isinstance(batch, list):
            batch = batch[0]
        bbox = batch["bounding_box"]
        if isinstance(bbox[0], list):
            batch["bounding_box"] = bbox[0]
        for tfm in self.postprocess_transforms:
            batch = tfm(batch)
        return batch

    def postprocess(self, preds):
        outputs = []
        for item in preds:
            if self.debug is False:
                outputs.append(self.postprocess_compose(item))
            else:
                outputs.append(self.postprocess_iterate(item))
        return outputs

    def process_lbd_sublist(self, pt_paths):
        """Patch inferer + postprocess on pre-cropped LBD .pt (no localiser)."""
        from det3d.inference.hybrid_lbd import load_lbd_pt_patch_data

        self.create_and_set_postprocess_transforms()
        data = load_lbd_pt_patch_data(pt_paths)
        self.bboxes = [dat["bounding_box"] for dat in data]
        full_metas = [dat["full_meta"] for dat in data]
        pred_patches = self.patch_prediction(data)
        decollated = self.decollate_patches(pred_patches, self.bboxes, full_metas)
        output = self.postprocess(decollated)
        self.cuda_clear()
        return output

    def run_lbd(self, pt_paths, chunksize=12, overwrite=False):
        """Run DetPatchInferer on LBD .pt volumes (same path as cascade crop, no localiser)."""
        from pathlib import Path

        from utilz.helpers import chunks
        from utilz.listify import listify

        self.setup()
        pt_paths = [Path(p) for p in listify(pt_paths)]
        if overwrite is False:
            out = Path(self.output_folder)
            pt_paths = [p for p in pt_paths if not (out / f"{p.stem}.json").exists()]
        if len(pt_paths) == 0:
            raise SystemExit("Stopping execution - no LBD cases remain after filtering")
        output = None
        for sublist in chunks(pt_paths, n_sized_chunks=chunksize):
            output = self.process_lbd_sublist(sublist)
        return output


class DetCascadeInfererRetinaUNet(DetCascadeInferer):
    """TotalSeg localiser + RetinaUNet; bbox sidecar JSON + seg-head pred_seg NIfTI."""

    keys_postproc = (
        "Pre,SqL,BoxPts,Off,Reorder,SegScale,Argmax,WrapSeg,FillSeg,Dust,"
        "VoxCopy,FullMeta,World,WorldCopy,Mode,Meta"
    )
    keys_postproc_safe = "Pre,SqL,BoxPts,Off,Reorder,SegScale,Argmax,WrapSeg,FillSeg,Dust"

    def setup_patch_inferer(self):
        from det3d.inference.patch import DetPatchInfererRetinaUNet

        return DetPatchInfererRetinaUNet(
            run_name=self.run_p,
            project_title=self.project_title,
            devices=self.devices,
            patch_overlap=self.patch_overlap,
            safe_mode=self.safe_mode,
            params=self.params,
            debug=self.debug,
            save=False,
            keys_preproc="E,S,Norm,Dtype",
            keys_postproc="Pack,SqL",
        )

    def setup_lbd_patch_inferer(self):
        from det3d.inference.patch import DetPatchInfererRetinaUNetLBD

        return DetPatchInfererRetinaUNetLBD(
            run_name=self.run_p,
            project_title=self.project_title,
            devices=self.devices,
            patch_overlap=self.patch_overlap,
            safe_mode=self.safe_mode,
            params=self.params,
            debug=self.debug,
            save=False,
        )

    def decollate_patches(self, pa, bboxes, full_metas=None):
        output = super().decollate_patches(pa, bboxes, full_metas)
        run_name = self.P.run_name
        for i, batch in enumerate(pa[run_name]):
            output[i]["pred_seg"] = batch["pred_seg"].detach().cpu()
        return output

    def run_lbd(self, pt_paths, chunksize=12, overwrite=False):
        """Run DetPatchInfererRetinaUNetLBD on LBD .pt volumes (no localiser)."""
        from pathlib import Path

        from utilz.helpers import chunks
        from utilz.listify import listify

        self.P = self.setup_lbd_patch_inferer()
        self.predictions_folder = self.P.project.predictions_folder
        pt_paths = [Path(p) for p in listify(pt_paths)]
        if overwrite is False:
            out = Path(self.output_folder)
            pt_paths = [p for p in pt_paths if not (out / f"{p.stem}.json").exists()]
        if len(pt_paths) == 0:
            raise SystemExit("Stopping execution - no LBD cases remain after filtering")
        output = None
        for sublist in chunks(pt_paths, n_sized_chunks=chunksize):
            output = self.process_lbd_sublist(sublist)
        return output

    def create_postprocess_transforms(self):
        from det3d.inference.transforms import NndetBoxToXyzxyzd

        super().create_postprocess_transforms()
        plan = self.params["configs"]["plan_train"]
        dust_mm = plan["dusting_mm"]
        if dust_mm is None:
            dust_mm = 1.0
        else:
            dust_mm = float(dust_mm)
        ik = "image"
        bk = "pred_box"
        sk = "pred_seg"
        del self.postprocess_transforms_dict["Scale"]
        del self.postprocess_transforms_dict["Clip"]
        self.postprocess_transforms_dict["BoxPts"] = PredBoxToNativeCropViaPointsd(
            box_key=bk,
            image_key=ik,
            box_mode="nndet",
        )
        self.postprocess_transforms_dict["Off"] = OffsetBoxByBBoxd(
            box_keys=[bk],
            box_mode="nndet",
        )
        self.postprocess_transforms_dict["Reorder"] = NndetBoxToXyzxyzd(box_key=bk)
        self.postprocess_transforms_dict["SegScale"] = ScaleSegToCropNatived(
            seg_key=sk,
            image_key=ik,
        )
        self.postprocess_transforms_dict["Argmax"] = ArgmaxSegd(seg_key=sk)
        self.postprocess_transforms_dict["WrapSeg"] = WrapPredSegMetad(
            seg_key=sk, image_key=ik
        )
        self.postprocess_transforms_dict["FillSeg"] = FillBBoxPatchesd(keys=[sk])
        self.postprocess_transforms_dict["Dust"] = DustSegd(seg_key=sk, dust_mm=dust_mm)
        self.postprocess_transforms_dict["RestoreSeg"] = RestoreOriginalOrientationd(
            keys=[sk]
        )
        self.postprocess_transforms_dict["WriteSeg"] = MakeWritabled(keys=[sk])
        self.postprocess_transforms_dict["SaveSeg"] = SaveImaged(
            keys=[sk],
            output_dir=self.output_folder,
            separate_folder=False,
            output_dtype=torch.uint8,
            output_postfix="",
            resample=False,
        )
        self.postprocess_transforms_dict["SegPath"] = AttachPredSegPathd(seg_key=sk)

    def set_postprocess_tfms_keys(self):
        if self.safe_mode:
            self.postprocess_tfms_keys = self.keys_postproc_safe
            if self.save:
                self.postprocess_tfms_keys += _BOX_WORLD_KEYS + _SEG_SAVE_KEYS
        else:
            self.postprocess_tfms_keys = self.keys_postproc
            if self.save:
                self.postprocess_tfms_keys += _SEG_SAVE_KEYS


# %%
# SECTION:-------------------- setup--------------------------------------------------------------------------------------
if __name__ == "__main__":
    from fran.inference.cascade import img_bbox_collated
    from fran.inference.helpers import load_images_nifti
    from fran.utils.common import COMMON_PATHS
    from utilz.fileio import load_yaml
    from utilz.helpers import pp
    from utilz.imageviewers import ImageBBoxViewer, ImageMaskViewer


    fldr = Path("/media/UB/datasets/lidc_all/images")
    imgs = [Path("/media/UB/datasets/lidc_all/images/lidc_0008.nii.gz")]
    imgs = list(fldr.glob("*.nii.gz"))

# %%
    def default_run_w():
        fn = Path(COMMON_PATHS["cold_storage_folder"]) / "conf" / "best_runs.yaml"
        return load_yaml(fn)["totalseg"]["whole"]["runs"][0]

    En = DetCascadeInfererRetinaUNet(
        run_w=default_run_w(),
        run_p="LIDCA-GYRO",
        project_title="lidca",
        devices=[1],
        localiser_labels=[6],
        safe_mode=True,
        patch_overlap=0.5,
        save=True,
        debug=False,
    )
# %%
    preds = En.run(imgs, chunksize=4, overwrite=True)
# %%
    imgs_pt = load_images_nifti(imgs)
    img = imgs_pt[0]["image"]

# %%
    preds[0].keys()
    lm =preds[0]['pred_seg'][0]
    # ImageMaskViewer([img, lm],'im')
    bbo = preds[0]['pred_box_voxel']
    scores = preds[0]['pred_score']
    idx=  preds[0]['pred_score'].sort(descending=True)

    scores[idx.indices]
    print(bbo)
    bbox = bbo[idx.indices[0]]
    ImageBBoxViewer(img, bbox)

# %%
    data = imgs_pt
# %%

    # /home/ub/code/det3d/det3d/inference/cascade.py  # T:block_donor|/home/ub/code/det3d/det3d/inference/cascade.py
# SECTION:-------------------- patch_prediction --------------------------------------------------------------------------------------  # T:block_meta|DetCascadeInfererRetinaUNet.patch_prediction
    sources = []
    crop_shapes = []
    dat = next(iter(data))  # T:loop_probe|for dat in data:
    sources.append(
        dat["image"].meta["filename_or_obj"]
    )  # T:indent|    sources.append(dat["image"].meta["filename_or_obj"])
    crop_shapes.append(
        tuple(int(v) for v in dat["image"].shape[-3:])
    )  # T:indent|    crop_shapes.append(tuple(int(v) for v in dat["image"].shape[-3:]))
# %%
    if hasattr(En.W, "model"):  # T:self_ref|if hasattr(self.W, "model"):
        del En.W.model  # T:self_ref|    del self.W.model
    torch.cuda.empty_cache()
# %%
    print("Starting patch data prep and prediction")
    preds_all_runs = {}
    preds_all_runs[
        En.P.run_name
    ] = []  # T:self_ref|preds_all_runs[self.P.run_name] = []
    En.P.setup()  # T:self_ref|self.P.setup()
    En.P.prepare_data(
        data=data, collate_fn=img_bbox_collated
    )  # T:self_ref|self.P.prepare_data(data=data, collate_fn=img_bbox_collated)
    En.P.create_and_set_postprocess_transforms()  # T:self_ref|self.P.create_and_set_postprocess_transforms()
    batch = next(iter(En.P.predict()))  # T:loop_probe|for batch in self.P.predict():
# %%
    batch.keys()
    img = batch["image"][0][0]
    box = batch["merged_boxes"][0]
# %%
    monai = torch.empty_like(box)
    monai[0] = box[0]
    monai[1] = box[3]
    monai[2] = box[2]
    monai[3] = box[1]
    monai[4] = box[4]
    monai[5] = box[5]
    ImageBBoxViewer(img, box)   
# %%
    img.meta
    ImageBBoxViewer(img, bo)

# %%
    batch = En.P.postprocess(batch)  # T:self_ref|    batch = self.P.postprocess(batch)
    preds_all_runs[En.P.run_name].append(
        batch
    )  # T:self_ref|    preds_all_runs[self.P.run_name].append(batch)
    preds = preds_all_runs  # T:return|return preds_all_runs
# %%
    i, batch = next(
        iter(enumerate(preds[En.P.run_name]))
    )  # T:loop_probe|for i, batch in enumerate(preds[self.P.run_name]):
    batch["source_image"] = sources[
        i
    ]  # T:indent|    batch["source_image"] = sources[i]
    batch["crop_spatial_shape"] = crop_shapes[
        i
    ]  # T:indent|    batch["crop_spatial_shape"] = crop_shapes[i]
    patch_prediction_result = preds  # T:return|return preds
# SECTION:-------------------- patch_prediction end --------------------------------------------------------------------------------------  # T:block_meta_end|DetCascadeInfererRetinaUNet.patch_prediction
    # end PythonMethodScratch  # T:block_end|DetCascadeInfererRetinaUNet.patch_prediction

# %%
    imgs_sublist = imgs
# %%
    # /home/ub/code/fran/fran/inference/cascade.py  # T:block_donor|/home/ub/code/fran/fran/inference/cascade.py
# SECTION:-------------------- process_data_sublist --------------------------------------------------------------------------------------  # T:block_meta|DetCascadeInfererRetinaUNet.process_data_sublist
    En.create_and_set_postprocess_transforms()  # T:self_ref|self.create_and_set_postprocess_transforms()
    data = En.load_images(
        imgs_sublist
    )  # T:self_ref|data = self.load_images(imgs_sublist)
    image_paths = imgs_sublist
# %%
    En.bboxes = En.extract_fg_bboxes(  # T:self_ref|self.bboxes = self.extract_fg_bboxes(
        data,
        overwrite=En.localiser_overwrite,  # T:self_ref|    overwrite=self.localiser_overwrite,
    )
# %%
    data = En.load_images(
        image_paths
    )  # T:self_ref|data = self.load_images(image_paths)
    data = En.apply_bboxes(
        data, En.bboxes
    )  # T:self_ref|data = self.apply_bboxes(data, self.bboxes)
    full_metas = [dat["full_meta"] for dat in data]
    pred_patches = En.patch_prediction(
        data
    )  # T:self_ref|pred_patches = self.patch_prediction(data)
# %%
    pred_patches = En.decollate_patches(
        pred_patches, En.bboxes, full_metas
    )  # T:self_ref|pred_patches = self.decollate_patches(pred_patches, self.bboxes, full_metas)
    pred_patches[0].keys()
    pred_patches[0]["pred_seg"].shape
    pred_patches[0]["pred_box"]
    box_p = pred_patches[0]["pred_box"].clone()
# %%
    img = pred_patches[0]["image"]
    im = img[0]
    lm = pred_patches[0]["pred_seg"]
    bbo = pred_patches[0]["pred_box"]
    ImageMaskViewer([im, lm], "im")
# %%
    En.cuda_clear()  # T:self_ref|self.cuda_clear()
    process_data_sublist_result = output  # T:return|return output
# SECTION:-------------------- process_data_sublist end --------------------------------------------------------------------------------------  # T:block_meta_end|DetCascadeInfererRetinaUNet.process_data_sublist
    # end PythonMethodScratch  # T:block_end|DetCascadeInfererRetinaUNet.process_data_sublist

# %%
    data = preds[self.P.run_name]
    chunksize = 12
    overwrite = False
# %%
    # /home/ub/code/fran/fran/inference/cascade.py  # T:block_donor|/home/ub/code/fran/fran/inference/cascade.py
# %%
    out = preds[0]
    bbo_keys = [k for k in out.keys() if "box" in k]
    pp(sorted(out.keys()))
    print("pred_seg fg", int(out["pred_seg"].sum()))
    print("n_boxes", out["pred_box"].shape[0])
    print("sidecar", out["sidecar_path"])
# %%
    out["pred_score"]
    out["pred_box"]
    out["pred_box_voxel"]
    out["pred_box_world"]
    lm = out["pred_seg"]
    out["image"].shape
    bbos = out["pred_box_voxel"]
    idx = out["pred_score"].argsort(descending=True)
    # bb1 =[110.07772827148438, 320.0018615722656, 32.85981750488281, 119.0731430053711, 333.905029296875, 42.902252197265625]
# %%
    bb1 = bbos[4]
    ImageBBoxViewer(img, bb1)
# %%
    ImageMaskViewer([img, lm], "im")

# %%
    from utilz.fileio import load_json

    sc = load_json("/s/agent_rw/tmp/lidc_gyro_sidecar_test/lidc_0007.json")
    sc["markups"][0]
# %%
    sc = load_json("/s/fran_storage/predictions/lidca/LIDCA-GYRO/lidc_0007.json")
    df = pd.DataFrame(sc["predictions"])
    df.to_csv("/s/agent_rw/tmp/lidc_gyro_sidecar_test/lidc_0007.csv", index=False)

# %%
    "Pre,SqL,BoxPts,Clip,SegScale,Argmax,WrapSeg,FillSeg,Dust,Off,VoxCopy,FullMeta,World,WorldCopy,Mode,Meta"
    En.keys_postproc

    dici = pred_patches[0]
# %%
    tfms = En.postprocess_transforms_dict
    dici = tfms["Pre"](dici)
    print(dici["image"].shape)
    print(dici["pred_box_pre_tfm"])

    print(dici["pred_box"])
    dici = tfms["SqL"](dici)
    dici = tfms["BoxPts"](dici)
    print(dici["image"].shape)

    dici = tfms["Clip"](dici)
    print(dici["image"].shape)

    dici = tfms["SegScale"](dici)
    print(dici["image"].shape)

    dici = tfms["Argmax"](dici)
    print(dici["image"].shape)

    dici = tfms["WrapSeg"](dici)
    print(dici["image"].shape)

    dici = tfms["FillSeg"](dici)
    print(dici["image"].shape)

    dici = tfms["Dust"](dici)
    print(dici["image"].shape)

    dici = tfms["Off"](dici)
    print(dici["image"].shape)

    dici = tfms["VoxCopy"](dici)
    print(dici["image"].shape)

    dici = tfms["FullMeta"](dici)
    print(dici["image"].shape)

    dici = tfms["World"](dici)
    print(dici["image"].shape)

    dici = tfms["WorldCopy"](dici)
    print(dici["image"].shape)

    dici = tfms["Mode"](dici)
    print(dici["image"].shape)

    dici = tfms["Meta"](dici)
    print(dici["image"].shape)

# %%


