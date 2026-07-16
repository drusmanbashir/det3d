from copy import deepcopy

from monai.transforms import Compose
from monai.transforms.spatial.dictionary import ConvertBoxToPointsd, ConvertPointsToBoxesd
import numpy as np
import torch
from det3d.inference.post import InvPreprocessBoxd, PackRetinaNetPredsd, PackRetinaUNetPredsd, inverse_preprocess_box
from fran.data.dataset import FillBBoxPatchesd
from fran.inference.cascade import CascadeInferer as FranCascadeInferer
from fran.transforms.inferencetransforms import MakeWritabled, SqueezeListofListsd
from fran.transforms.spatialtransforms import RestoreOriginalOrientationd
from monai.data.meta_tensor import MetaTensor
from monai.transforms.post.dictionary import AsDiscreted
from monai.transforms.utility.dictionary import ApplyTransformToPointsd, CastToTyped

from utilz.imageviewers import ImageMaskBboxViewer

 
def _finalize_postproc_keys(postproc, postproc_safe, *, safe_mode, k_largest, save):
    chain = postproc_safe if safe_mode else postproc
    if k_largest is not None:
        parts = chain.split(",")
        if "F" in parts:
            parts.insert(parts.index("F"), "K")
            chain = ",".join(parts)
    if not save:
        chain = ",".join(k for k in chain.split(",") if k != "S")
    return chain


def decollate_image(img, batch_image):
    if img.dim() == 5:
        img = img[0]
    img = img.detach().cpu()
    if isinstance(img, MetaTensor):
        return img
    if isinstance(batch_image, MetaTensor):
        return MetaTensor(img, meta=deepcopy(batch_image.meta))
    return img


class DetBBoxCascadeInferer(FranCascadeInferer):
    """Localiser bbox → det patch → full-volume boxes (RetinaNet)."""

    keys_postproc = "SqL,Off,BoxR,S"

    def __init__(
        self,
        run_w,
        run_p,
        localiser_labels,
        project_title=None,
        devices=(0,),
        safe_mode=False,
        patch_overlap=0.2,
        profile=None,
        save=True,
        save_localiser=True,
        k_largest=None,
        debug=False,
    ):
        from fran.inference.helpers import load_params
        from fran.utils.misc import parse_devices

        assert profile in [None, "dataloading", "prediction", "all"]
        self.run_w = run_w
        self.run_p = run_p
        self.localiser_labels = localiser_labels
        self.project_title = project_title
        self.devices = devices
        self.safe_mode = safe_mode
        self.patch_overlap = patch_overlap
        self.profile = profile
        self.save_channels = False
        self.save = save
        self.save_localiser = save_localiser
        self.k_largest = k_largest
        self.debug = debug
        self.merge_touching_labels = None
        self.pred_run_p = None
        self.localiser_overwrite = False
        self.device = parse_devices(devices)
        self.params = load_params(run_p)
        self.P = self.setup_patch_inferer()
        self.predictions_folder = self.P.project.predictions_folder
        self.W = self.setup_localiser_inferer()

    @property
    def output_folder(self):
        run_name = self.pred_run_p if self.pred_run_p is not None else self.run_p
        return self.predictions_folder / run_name

    def cuda_clear(self):
        if hasattr(self.P, "model"):
            del self.P.model
        torch.cuda.empty_cache()

    def setup_patch_inferer(self):
        from det3d.inference.patch import DetPatchRetinaNet

        return DetPatchRetinaNet(
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
        output = []
        for case_idx, batch in enumerate(pa[self.P.run_name]):
            batch_img = batch["image"]
            bb = bboxes[case_idx]
            crop_shape = batch["crop_spatial_shape"]
            item = {
                "image": decollate_image(batch_img, batch_img),
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
        from fran.inference import helpers

        sources = helpers.image_paths_from_data(data)
        crop_shapes = []
        for dat in data:
            crop_shapes.append(tuple(int(v) for v in dat["image"].shape[-3:]))
        preds = FranCascadeInferer.patch_prediction(self, data)
        for i, batch in enumerate(preds[self.P.run_name]):
            batch["source_image"] = sources[i]
            batch["crop_spatial_shape"] = crop_shapes[i]
        return preds

    def create_postprocess_transforms(self):
        from det3d.inference.post import BoxRd, Offd, SaveDetOutputd

        self.postprocess_transforms_dict = {
            "SqL": SqueezeListofListsd(keys=["bounding_box"]),
            "Off": Offd(box_keys=["pred_box"]),
            "BoxR": BoxRd(box_key="pred_box"),
            "S": SaveDetOutputd(
                output_dir=self.output_folder,
                run_w=self.run_w,
                run_p=self.run_p,
                write_seg=False,
            ),
        }

    def set_postprocess_tfms_keys(self):
        cls = type(self)
        self.keys_postproc = _finalize_postproc_keys(
            cls.keys_postproc,
            cls.keys_postproc,
            safe_mode=False,
            k_largest=self.k_largest,
            save=self.save,
        )
        self.postprocess_tfms_keys = self.keys_postproc

    def set_postprocess_transforms(self):
        self.postprocess_transforms = self.tfms_from_dict(
            self.keys_postproc, self.postprocess_transforms_dict
        )
        self.postprocess_compose = Compose(self.postprocess_transforms)


class DetSegBBoxCascadeInferer(DetBBoxCascadeInferer):
    """Localiser bbox → RetinaUNet patch → full-volume seg + boxes."""

    keys_postproc = "SqL,MR,A,Int,W,F,R,Off,BoxR,S"
    keys_postproc_safe = "SqL,MR,W,F,R,Off,BoxR,S"

    def setup_patch_inferer(self):
        from det3d.inference.patch import DetPatchRetinaUNet

        return DetPatchRetinaUNet(
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
        output = []
        for case_idx, batch in enumerate(pa[self.P.run_name]):
            batch_img = batch["image"]
            bb = bboxes[case_idx]
            crop_shape = batch["crop_spatial_shape"]
            pred = batch["pred"].detach().cpu()
            if pred.ndim == 5:
                pred = pred[0]
            if pred.ndim == 3:
                pred = pred.unsqueeze(0)
            item = {
                self.run_p: pred,
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

    def create_postprocess_transforms(self):
        FranCascadeInferer.create_postprocess_transforms(self)
        from det3d.inference.post import BoxRd, Offd, SaveDetOutputd

        self.postprocess_transforms_dict["SqL"] = SqueezeListofListsd(keys=["bounding_box"])
        self.postprocess_transforms_dict["Off"] = Offd(box_keys=["pred_box"])
        self.postprocess_transforms_dict["BoxR"] = BoxRd(box_key="pred_box")
        self.postprocess_transforms_dict["S"] = SaveDetOutputd(
            output_dir=self.output_folder,
            run_w=self.run_w,
            run_p=self.run_p,
            write_seg=True,
        )

    def set_postprocess_tfms_keys(self):
        cls = type(self)
        self.keys_postproc = _finalize_postproc_keys(
            cls.keys_postproc,
            cls.keys_postproc_safe,
            safe_mode=self.safe_mode,
            k_largest=self.k_largest,
            save=self.save,
        )
        self.postprocess_tfms_keys = self.keys_postproc


if __name__ == "__main__":
    from pathlib import Path

    import SimpleITK as sitk
    import torch
    from fran.inference import helpers
    from fran.inference.cascade import img_bbox_collated
    from fran.inference.common_vars import *
    from label_analysis.geometry_pt import LabelMapGeometryPT
    from utilz.fileio import load_yaml
    from utilz.helpers import pp
    from utilz.imageviewers import ImageBBoxViewer, ImageMaskViewer

    from fran.utils.common import COMMON_PATHS

    devices = [0]
    safe_mode = True
    patch_overlap = 0.5
    overwrite = True
    debug_ = False
    chunksize = 1

    def default_run_w():
        fn = Path(COMMON_PATHS["cold_storage_folder"]) / "conf" / "best_runs.yaml"
        return load_yaml(fn)["totalseg"]["whole"]["runs"][0]


# %%
# SECTION:-------------------- LIDC cascade — 0 config -------------------------------------------------------------------
    run_w = "TOTALSEG-NJUGU"
    run_p = "LIDCA-QUARK"
    localiser_labels = [6]
    project_title = "lidca"

    lidc_all_fldr = Path("/media/UB/datasets/lidc_all/images")
    lidc_all_lm = Path("/media/UB/datasets/lidc_all/lms")
    fldr_k = Path("/s/xnat_shadow/misc/images")
    imgs_k = sorted(fldr_k.glob("*.nii.gz"), key=lambda p: p.stat().st_mtime)


# %%
# SECTION:-------------------- LIDC cascade — 1 inferer -----------------------------------------------------------------

    imgs = sorted(lidc_all_fldr.glob("*.nii.gz"))
    case_id = "misc_00008"
    print(imgs)
    En = DetSegBBoxCascadeInferer(
        run_w=run_w,
        run_p=run_p,
        project_title=project_title,

        devices=devices,
        localiser_labels=localiser_labels,
        safe_mode=safe_mode,
        patch_overlap=patch_overlap,
        save=True,
        save_localiser=False,
        debug=debug_,
    )
    print("patch postproc", En.P.keys_postproc)
    print("cascade postproc", En.keys_postproc)

# %%
    # imgs = imgs[:10]
    imgs= imgs_k
    imgs = [p for p in imgs_k if case_id in p.name]
    preds = En.run(imgs, chunksize=chunksize, overwrite=overwrite)
# %%
# SECTION:-------------------- LIDC cascade — 2 localiser bboxes --------------------------------------------------------
    imgs_sublist = imgs
    data = En.load_images(imgs_sublist)
    En.bboxes = En.extract_fg_bboxes(data, overwrite=overwrite)
    pp(En.bboxes[0])
    En.create_and_set_postprocess_transforms()
    data = En.load_images(imgs_sublist)
    image_paths = helpers.image_paths_from_data(data)
    En.bboxes = En.extract_fg_bboxes(
        data,
        overwrite=En.localiser_overwrite,
    )
    data = En.load_images(image_paths)
    data = En.apply_bboxes(data, En.bboxes)

    full_metas = [dat["full_meta"] for dat in data]
    pred_patches = En.patch_prediction(data)
    pred_patches = En.decollate_patches(pred_patches, En.bboxes, full_metas)
    pred_patches[0].keys()
    pred = pred_patches[0]["LIDCA-QUARK"]
    bbo = pred_patches[0]["pred_box"]
    scores = pred_patches[0]["pred_score"]
    _,idx = torch.sort(scores, descending=True)
    bbo_sorted = bbo[idx]
    bbo2 = bbo_sorted[:50]
    
    img = data[0]["image"].detach().cpu()
    ImageMaskBboxViewer(img, pred[0], bbo2)
    torch.save(pred_patches[0], "/s/agent_rw/tmp/misc_00008.pt")


    
# %%

    dici = pred_patches[0]
# %%
    En.keys_postproc
    'SqL,MR,W,F,R,Off,BoxR,S'
    tfms = En.postprocess_transforms_dict
# %%
    S = tfms["SqL"]
    dici = S(dici)

    M = tfms["MR"]
    dici = M(dici)

    W = tfms["W"]
    dici = W(dici)
    dici["pred"].shape

    F = tfms["F"]
    dici = F(dici)
    dici['pred'].shape
    b1 = dici["bounding_box"]
    bbo = dici["pred_box"][0]

    R = tfms["R"]
    dici = R(dici)

    O = tfms["Off"]
    dici = O(dici)

    B = tfms["BoxR"]
    dici = B(dici)
    bbo2 = dici["pred_box"][0]
    ImageMaskBboxViewer(dici["pred"][0], dici["pred"][0], bbo2)
    pred = dici["pred"][0]

    S2 = tfms["S"]
    dici = S2(dici)
# %%
# %%
#SECTION:-------------------- postprocess end --------------------------------------------------------------------------------------  # T:block_meta_end|CascadeInferer.postprocess
    # end PythonMethodScratch  # T:block_end|CascadeInferer.postprocess
# %%
# SECTION:-------------------- LIDC cascade — 3 crop + patch (patch post only) -------------------------------------------
    image_paths = helpers.image_paths_from_data(data)
    data = En.load_images(image_paths)
    data = En.apply_bboxes(data, En.bboxes)
    full_metas = [dat["full_meta"] for dat in data]
    print("crop shape", tuple(data[0]["image"].shape))
    data[0]["image"]
    data[0].keys()
    img = data[0]["image"].detach().cpu()
    # ImageMaskViewer([img,img],'im') # works

# %%
    En.P.setup()
    En.P.prepare_data(data=data, collate_fn=img_bbox_collated)
    En.P.create_and_set_postprocess_transforms()

# %%
    batch = next(iter(En.P.predict()))
# %%
    print("pre-postprocess keys", sorted(batch.keys()))
    img = batch["image"]
    img.shape
    batch.keys()
    batch.keys()
    bbo = batch["merged_boxes"]
    scores = batch["merged_scores"]
    pred = batch["stitched_seg"]
    bbo2 = bbo[0]
    lm  =pred[0,1]
    im = img[0,0]
    # ImageMaskViewer([im, lm],'im')
    ImageBBoxViewer(im,bbo2)
# %%
    dici0 = batch
    img = dici0["image"]
    pred = dici0["stitched_seg"]
    dici0.keys()
    torch.save(dici0, "/s/agent_rw/tmp/lidc_020.pt")

# %%

    # img2 = img[0,0]
    # img2.shape
    # ImageBBoxViewer(img2,bbo2) #HACK: perfect bbox and pred

    batch2 = En.P.postprocess(batch)
    batch2['image'].shape
    batch2['pred'].shape
    batch2['pred_box'][0]

    pred = batch2['pred'][0]
    bbo2 = batch2['pred_box'][0]
    ImageMaskBboxViewer(pred, pred, bbo2)
    # end PythonMethodScratch  # T:block_end|PackRetinaUNetPredsd.__call__
# %%
# SECTION:-------------------- LIDC cascade — 4 decollate + cascade post stepwise --------------------------------------
    pred_patches = En.patch_prediction(data)
    decollated = En.decollate_patches(pred_patches, En.bboxes, full_metas)
    dici = decollated[0]
    print("decollated keys", sorted(dici.keys()))
    dici["pred_box"][0]
    dici["pred"].shape

# %%
    En.create_and_set_postprocess_transforms()
    tfms = En.postprocess_transforms_dict
    keys = En.keys_postproc.split(",")
    for key in keys:
        if key == "S":
            continue
        dici = tfms[key](dici)
        fg = int(dici["pred"].sum()) if "pred" in dici else None
        nbox = dici["pred_box"].shape[0] if "pred_box" in dici else None
        print(key, "pred fg", fg, "n_boxes", nbox)

# %%
# SECTION:-------------------- LIDC cascade — 5 viz full volume ----------------------------------------------------------
    src = helpers.load_images_nifti(imgs)[0]["image"].detach().cpu()
    pred_full = dici["pred"][0].detach().cpu()
    boxes_full = dici["pred_box"].detach().cpu()
    src.shape
    pred_full.shape
    ImageMaskViewer([src, pred_full], "im")
    ImageBBoxViewer(src, boxes_full)

# %%
# SECTION:-------------------- LIDC cascade — 6 CC vs GT ----------------------------------------------------------------
    gt_fn = lidc_all_lm / f"{case_id}.nii.gz"
    gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_fn)))
    gt_t = torch.from_numpy(gt_arr).permute(2, 1, 0).contiguous()
    pred_t = pred_full.squeeze(0)
    L_gt = LabelMapGeometryPT(li=gt_t, ignore_labels=[0], compute_feret=False)
    L_pr = LabelMapGeometryPT(li=pred_t, ignore_labels=[0], compute_feret=False)
    if len(L_gt.nbrhoods) and len(L_pr.nbrhoods):
        cc_gt = L_gt.nbrhoods.iloc[0][["centroid_x", "centroid_y", "centroid_z"]].astype(float)
        cc_pr = L_pr.nbrhoods.iloc[0][["centroid_x", "centroid_y", "centroid_z"]].astype(float)
        dist = float(((cc_gt - cc_pr) ** 2).sum() ** 0.5)
        print("CC dist voxels", dist, "gt", cc_gt.tolist(), "pred", cc_pr.tolist())
    else:
        print("CC skip: gt nbrhoods", len(L_gt.nbrhoods), "pred nbrhoods", len(L_pr.nbrhoods))

# %%
# SECTION:-------------------- LIDC cascade — 7 full run (optional save) -----------------------------------------------
    En.save = True
    preds = En.run(imgs, chunksize=chunksize, overwrite=overwrite)
    out = preds[0]
    print("saved keys", sorted(out.keys()))
    print("pred fg", int(out["pred"].sum()), "n_boxes", out["pred_box"].shape[0])
