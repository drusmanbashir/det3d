import numpy as np
import torch
from det3d.detection.nndet_train import (
    offset_nndet_xyxyzz_boxes,
)
from fran.inference.base import BaseInferer
from fran.inference.cascade import img_bbox_collated
from fran.transforms.inferencetransforms import SqueezeListofListsd
from monai.transforms import EnsureTyped, ScaleIntensityRanged
from nndet.core.retina import BaseRetinaNet
from utilz.cprint import cprint

from det3d.managers.retinanet import RetinaNetManager


class DetPatchInferer(BaseInferer):
    keys_postproc = "SqL"

    def __init__(
        self,
        run_name,
        project_title=None,
        patch_overlap=0.25,
        devices=(0,),
        safe_mode=False,
        save=False,
        params=None,
        debug=False,
        keys_preproc="E,S,Norm,Dtype",
        **kwargs,
    ):
        cprint("Setting up detection patch inference", color="red", bold=True)
        self.patch_overlap = patch_overlap
        super().__init__(
            run_name=run_name,
            project_title=project_title,
            patch_overlap=patch_overlap,
            devices=devices,
            safe_mode=safe_mode,
            save=save,
            save_channels=False,
            params=params,
            debug=debug,
            keys_preproc=keys_preproc,
            keys_postproc=self.keys_postproc,
            model_manager=RetinaNetManager,
            **kwargs,
        )

    def check_plan_compatibility(self):
        pass

    def set_preprocess_tfms_keys(self):
        self.preprocess_tfms_keys = self.keys_preproc

    def create_preprocess_transforms(self):
        super().create_preprocess_transforms()
        clip = self.dataset_params["intensity_clip_range"]
        del self.preprocess_transforms_dict["N"]
        self.preprocess_transforms_dict["Norm"] = ScaleIntensityRanged(
            keys=["image"],
            a_min=float(clip[0]),
            a_max=float(clip[1]),
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
        self.preprocess_transforms_dict["Dtype"] = EnsureTyped(
            keys=["image"], dtype=torch.float16
        )

    def create_postprocess_transforms(self, preprocess_transform):
        self.postprocess_transforms_dict = {
            "SqL": SqueezeListofListsd(keys=["bounding_box"]),
        }

    def set_postprocess_tfms_keys(self):
        self.postprocess_tfms_keys = self.keys_postproc

    def prepare_data(self, data, collate_fn=img_bbox_collated):
        super().prepare_data(data, collate_fn=collate_fn)

    def predict_inner(self, batch):
        img = batch["image"].float()
        detector = self.model.detector
        detector.eval()
        if img.dim() == 5:
            val_inputs = [img[i] for i in range(img.shape[0])]
        elif img.dim() == 4:
            val_inputs = [img]
        else:
            val_inputs = [img.unsqueeze(0)]
        use_inferer = val_inputs[0][0, ...].numel() >= int(
            np.prod(self.model.val_patch_size)
        )
        with torch.inference_mode():
            outputs = detector(val_inputs, use_inferer=use_inferer)
        out = outputs[0]
        batch["pred_box"] = out[detector.target_box_key].detach()
        batch["pred_label"] = out[detector.target_label_key].detach()
        batch["pred_score"] = out[detector.pred_score_key].detach()
        return batch


class DetPatchInfererRetinaUNet(DetPatchInferer):
    """RetinaUNet patch inferer (det + seg); use with DetCascadeInfererRetinaUNet."""

    keys_preproc = "L,E,S,O,Norm,Dtype"
    keys_postproc = "Pack,SqL,WrapSeg,Re,BoxInv,R,Int"

    def __init__(self, *args, keys_preproc=None, keys_postproc=None, **kwargs):
        if keys_preproc is None:
            keys_preproc = DetPatchInfererRetinaUNet.keys_preproc
        if keys_postproc is None:
            keys_postproc = DetPatchInfererRetinaUNet.keys_postproc
        kwargs.pop("keys_preproc", None)
        kwargs.pop("keys_postproc", None)
        super().__init__(*args, keys_preproc=keys_preproc, **kwargs)
        self.keys_postproc = keys_postproc
        from det3d.managers.retinaunet import RetinaUNetManager
        self.model_manager = RetinaUNetManager
        self._box_acc = []

    def setup(self):
        super().setup()
        self.inferer.with_coord = True

    def create_preprocess_transforms(self):
        from fran.inference.helpers import get_patch_spacing
        from monai.transforms import EnsureChannelFirstd, EnsureTyped, Orientationd, Spacingd

        from det3d.inference.transforms import LoadInferImaged

        clip = self.dataset_params["intensity_clip_range"]
        norm = ScaleIntensityRanged(
            keys=["image"],
            a_min=float(clip[0]),
            a_max=float(clip[1]),
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
        dtype = EnsureTyped(keys=["image"], dtype=torch.float16)
        preproc_keys = self.keys_preproc.replace(" ", "")
        if "L_pt" in preproc_keys:
            self.preprocess_transforms_dict = {
                "L_pt": LoadInferImaged(keys=["image"]),
                "E": EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                "Norm": norm,
                "Dtype": dtype,
            }
        elif "L" in preproc_keys:
            spacing = get_patch_spacing(self.run_name)
            self.preprocess_transforms_dict = {
                "L": LoadInferImaged(keys=["image"]),
                "E": EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                "S": Spacingd(keys=["image"], pixdim=spacing),
                "O": Orientationd(keys=["image"], axcodes="RAS"),
                "Norm": norm,
                "Dtype": dtype,
            }
        else:
            DetPatchInferer.create_preprocess_transforms(self)
            return
        for key, value in self.preprocess_transforms_dict.items():
            setattr(self, key, value)

    def create_postprocess_transforms(self, preprocess_transform):
        from fran.transforms.spatialtransforms import (
            RestoreOriginalOrientationd,
            ResizeToMetaSpatialShaped,
        )
        from monai.transforms.utility.dictionary import CastToTyped

        from det3d.inference.transforms import (
            PackRetinaUNetPredsd,
            RetinaUNetBoxInversed,
            WrapPredSegMetad,
        )

        postproc_keys = self.keys_postproc.replace(" ", "")
        if "BoxInv" in postproc_keys:
            self.postprocess_transforms_dict = {
                "Pack": PackRetinaUNetPredsd(),
                "SqL": SqueezeListofListsd(keys=["bounding_box"]),
                "WrapSeg": WrapPredSegMetad(seg_key="pred_seg", image_key="image"),
                "Re": ResizeToMetaSpatialShaped(keys=["pred_seg"], mode="nearest"),
                "BoxInv": RetinaUNetBoxInversed(),
                "R": RestoreOriginalOrientationd(keys=["pred_seg"]),
                "Int": CastToTyped(keys=["pred_seg"], dtype=torch.uint8),
            }
        else:
            self.postprocess_transforms_dict = {
                "Pack": PackRetinaUNetPredsd(),
                "SqL": SqueezeListofListsd(keys=["bounding_box"]),
            }
        for key, value in self.postprocess_transforms_dict.items():
            setattr(self, key, value)

    def inference_params(self):
        plan = self.model.plan
        arch = self.model.nndet_plan["architecture"]
        from det3d.detection.nndet_train import ensure_nndet_importable
        ensure_nndet_importable()
        from nndet.inference.detection.model import batched_weighted_nms_model

        params = {
            "model_score_thresh": float(plan["score_thresh"]),
            "model_topk": int(arch["topk_candidates"]),
            "model_detections_per_image": int(arch["detections_per_img"]),
            "remove_small_boxes": float(arch["remove_small_boxes"]),
            "model_iou": 0.1,
            "model_nms_fn": batched_weighted_nms_model,
        }
        return params

    @staticmethod
    def _tile_origin(unravel_entry):
        spatial = unravel_entry[2:]
        return [int(spatial[0].start), int(spatial[1].start), int(spatial[2].start)]

    def _offset_boxes(self, boxes, origin):
        return offset_nndet_xyxyzz_boxes(boxes, origin)

    def _box_tile_weight(self, boxes, tile_size):
        from det3d.detection.nndet_train import ensure_nndet_importable

        ensure_nndet_importable()
        from nndet.core.boxes.ops import box_center
        from nndet.inference.ensembler.detection import BoxEnsemblerSelective

        centers = box_center(boxes) if boxes.numel() else boxes
        return BoxEnsemblerSelective._get_box_in_tile_weight(centers, tile_size)

    def filter_boxes(self, boxes, probs, labels, weights, spatial):
        from det3d.detection.nndet_train import ensure_nndet_importable

        ensure_nndet_importable()
        from nndet.core.boxes.clip import clip_boxes_to_image
        from nndet.core.boxes.ops import remove_small_boxes

        params = self.inference_params()
        p_sorted, idx_sorted = probs.sort(descending=True)
        idx_sorted = idx_sorted[: params["model_topk"]]
        keep_idxs = probs[idx_sorted] > params["model_score_thresh"]
        idx_sorted = idx_sorted[keep_idxs]

        b = boxes[idx_sorted]
        p = probs[idx_sorted]
        l = labels[idx_sorted]
        w = weights[idx_sorted]

        b = clip_boxes_to_image(b, spatial)
        keep = remove_small_boxes(b, min_size=params["remove_small_boxes"])
        b, p, l, w = b[keep], p[keep], l[keep], w[keep]

        b, p, l, _w = params["model_nms_fn"](
            boxes=b,
            scores=p,
            labels=l,
            weights=w,
            iou_thresh=params["model_iou"],
        )
        cap = params["model_detections_per_image"]
        b = b[:cap]
        p = p[:cap]
        l = l[:cap]
        return b, p, l

    def merge_tile_boxes(self, acc, spatial):
        if not acc:
            device = next(self.model.net.parameters()).device
            empty = torch.zeros(0, 6, device=device)
            return (
                empty,
                torch.zeros(0, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            )
        boxes = torch.cat([item["boxes"] for item in acc])
        probs = torch.cat([item["scores"] for item in acc])
        labels = torch.cat([item["labels"] for item in acc])
        weights = torch.cat([item["weights"] for item in acc])
        return self.filter_boxes(boxes, probs, labels, weights, spatial)

    def _needs_swi(self, img):
        patch_size = self.model.forward_patch_size
        spatial = tuple(int(v) for v in img.shape[-3:])
        return any(s > p for s, p in zip(spatial, patch_size))

    def _forward_tile(self, img):
        x = img.unsqueeze(0)
        device_type = x.device.type
        with torch.autocast(device_type, enabled=device_type == "cuda"):
            pred = self.model.net.inference_step(x)
        return pred

    def _swi_predictor(self, win_data, unravel_slice):
        device_type = win_data.device.type
        with torch.autocast(device_type, enabled=device_type == "cuda"):
            pred = self.model.net.inference_step(win_data)
        tile_size = tuple(int(v) for v in win_data.shape[-3:])
        for i in range(win_data.shape[0]):
            boxes = pred["pred_boxes"][i].detach()
            scores = pred["pred_scores"][i].detach()
            labels = pred["pred_labels"][i].detach()
            origin = self._tile_origin(unravel_slice[i])
            boxes = self._offset_boxes(boxes, origin)
            weights = self._box_tile_weight(boxes, tile_size)
            self._box_acc.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                    "weights": weights,
                }
            )
        return pred["pred_seg"]

    def _run_swi(self, img):
        self._box_acc = []
        stitched_seg = self.inferer(inputs=img, network=self._swi_predictor)
        spatial = tuple(int(v) for v in img.shape[-3:])
        boxes, scores, labels = self.merge_tile_boxes(self._box_acc, spatial)
        return stitched_seg, boxes, scores, labels

    def predict_inner(self, batch):
        from det3d.detection.nndet_train import _plain_tensor
        device = next(self.model.net.parameters()).device
        img = _plain_tensor(batch["image"]).float().to(device)
        if img.dim() == 5:
            img = img[0]
        if img.dim() == 3:
            img = img.unsqueeze(0)
        self.model.eval()
        if self._needs_swi(img):
            vol = img.unsqueeze(0)
            seg, boxes, scores, labels = self._run_swi(vol)
            batch["stitched_seg"] = seg
            batch["merged_boxes"] = boxes
            batch["merged_scores"] = scores
            batch["merged_labels"] = labels
            return batch
        batch["raw_pred"] = self._forward_tile(img)
        return batch


class DetPatchInfererRetinaUNetLBD(DetPatchInfererRetinaUNet):
    """RetinaUNet on pre-spaced LBD .pt crops; whole crop is inference domain."""

    keys_preproc = "L_pt,E,Norm,Dtype"
    keys_postproc = "Pack,SqL,Int"

    def __init__(self, *args, keys_preproc=None, keys_postproc=None, **kwargs):
        if keys_preproc is None:
            keys_preproc = DetPatchInfererRetinaUNetLBD.keys_preproc
        if keys_postproc is None:
            keys_postproc = DetPatchInfererRetinaUNetLBD.keys_postproc
        kwargs.pop("keys_preproc", None)
        kwargs.pop("keys_postproc", None)
        super().__init__(*args, keys_preproc=keys_preproc, **kwargs)
        self.keys_postproc = keys_postproc

    def check_plan_compatibility(self):
        pass

    def load_images(self, images):
        from fran.inference.helpers import load_images_pt

        return load_images_pt(images)


# %%
# SECTION:-------------------- setup--------------------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    from fran.inference import helpers
    from utilz.imageviewers import ImageBBoxViewer, ImageMaskViewer

    devices = [1]

    safe_mode = True
    patch_overlap = 0.5
    debug_ = True

    fldr_lidc2 = Path("/media/UB/datasets/lidc2/images/")
    fldr_lidc = Path("/media/UB/datasets/lidc/images/")
    fldr_pt = Path("/r/datasets/preprocessed/lidca/lbd/spc_070_070_125_ex000/images")

# %%
    imgs = sorted(fldr_lidc2.glob("*.nii.gz"))
    imgs = sorted(fldr_pt.glob("*.pt"))
    cid = "lidc_0001"
    img = [im for im in imgs if cid in im.name][0]
    imgs2 = [img]




# %%
# SECTION:-------------------- RetinaUNet patch — 0 setup -----------------------------------------------------
    run_p = "LIDCA-GYRO"
    project_title = "lidca"
    D = DetPatchInfererRetinaUNetLBD(
        run_name=run_p,
        project_title=project_title,
        devices=devices,
        safe_mode=safe_mode,
        patch_overlap=patch_overlap,
        debug=debug_,
    )
    data = helpers.load_images_pt(imgs2)
    D.setup()

    D.model.net.score_thresh =0.3
    D.model.net.detections_per_img = 25
    D.model.net.nms_thresh = 0.15
    D.prepare_data(data=data, collate_fn=None)
    D.create_and_set_postprocess_transforms()
    print("patch_size", D.model.forward_patch_size)
    print(
        "keys_preproc",
        D.keys_preproc,
        "keys_postproc",
        D.keys_postproc,
    )
    
    
# %%
    type(D.model.net)
    D.model.net.nms_thresh

# %%
# SECTION:-------------------- RetinaUNet LBD — 1 preprocessed batch -----------------------------------------------------
    batch = next(iter(D.pred_dl))
# %%
# %%  # T:block_start|DetPatchInfererRetinaUNetLBD.predict_inner
# /home/ub/code/det3d/det3d/inference/patch.py  # T:block_donor|/home/ub/code/det3d/det3d/inference/patch.py
#SECTION:-------------------- predict_inner --------------------------------------------------------------------------------------  # T:block_meta|DetPatchInfererRetinaUNetLBD.predict_inner
    # end PythonMethodScratch  # T:block_end|DetPatchInfererRetinaUNetLBD.predict_inner
    img = batch["image"][0,0]
# %%
    patch_size = [128,128,64]
    print(img.shape)
    start_x = 60
    start_y =00
    start_z = 100
    end_x = start_x+patch_size[0]
    end_y = start_y+patch_size[1]
    end_z = start_z+patch_size[2]
    slc = (slice(start_x, end_x), slice(start_y, end_y), slice(start_z, end_z))
# %%
    im2 = img[slc]
# %%
    ImageMaskViewer([im2, im2],'im')
# %%
# /home/ub/code/det3d/det3d/inference/patch.py  # T:block_donor|/home/ub/code/det3d/det3d/inference/patch.py
#SECTION:-------------------- predict_inner --------------------------------------------------------------------------------------  # T:block_meta|DetPatchInfererRetinaUNetLBD.predict_inner
    im2 = img[0][slc]
    im2 = im2.to("cuda:1")
    im3 = im2.clone().unsqueeze(0)

    batch["image"]=im3
    from det3d.detection.nndet_train import _plain_tensor
    im3 = _plain_tensor(batch["image"]).float()
    if im3.dim() == 5:
        im3 = im3[0]
    if im3.dim() == 3:
        im3 = im3.unsqueeze(0)
    D.model.eval()  # T:self_ref|self.model.eval()
    if D._needs_swi(im3):  # T:self_ref|if self._needs_swi(im3):
        vol = im3.unsqueeze(0)
        seg, boxes, scores, labels = D._run_swi(vol)  # T:self_ref|    seg, boxes, scores, labels = self._run_swi(vol)
        batch["stitched_seg"] = seg
        batch["merged_boxes"] = boxes
        batch["merged_scores"] = scores
        batch["merged_labels"] = labels
        pass  # T:early_return|    return batch
# %%
# %%
#SECTION:-------------------- single patch--------------------------------------------------------------------------------------
    net = D.model.net
    batch["raw_pred"] = D._forward_tile(im3)  # T:self_ref|batch["raw_pred"] = self._forward_tile(img)
    raw_pred = batch["raw_pred"]
    raw_pred["pred_boxes"]


    batch2 = D.postprocess(batch)
    batch2.keys()
    batch2['pred_score']
    bbo = batch2['pred_box']
    n=1
    bb1 = bbo[n,:].detach().cpu()
    img = batch2["image"][0, 0].detach().cpu()
    ImageBBoxViewer(img,bb1)
# %%
#SECTION:-------------------- predict_inner end --------------------------------------------------------------------------------------  # T:block_meta_end|DetPatchInfererRetinaUNetLBD.predict_inner
    # end PythonMethodScratch  # T:block_end|DetPatchInfererRetinaUNetLBD.predict_inner
# %%
#SECTION:-------------------- RetinaUNet internsl: inference_step--------------------------------------------------------------------------------------
    net = D.model.net
    images = img
    images = images.unsqueeze(0)
    device_type = images.device.type
# %%
    with torch.autocast(device_type, enabled=device_type == "cuda"):
        pred_detection, anchors, pred_seg = net(images)
# %%
    prediction = net.postprocess_for_inference(
            images=images,
            pred_detection=pred_detection,
            pred_seg=pred_seg,
            anchors=anchors,
        )
# %%

    image_shapes = [images.shape[2:]] * images.shape[0]
# %%
# %%
# /home/ub/code/nnDetection/nndet/core/retina.py  # T:block_donor|/home/ub/code/nnDetection/nndet/core/retina.py
#SECTION:-------------------- postprocess_detections end --------------------------------------------------------------------------------------  # T:block_meta_end|BaseRetinaNet.postprocess_detections
    # end PythonMethodScratch  # T:block_end|BaseRetinaNet.postprocess_detections
# %%

    boxes, probs, labels = net.postprocess_detections(
        pred_detection=pred_detection,
        anchors=anchors,
        image_shapes=image_shapes,
    )
    prediction = {"pred_boxes": boxes, "pred_scores": probs, "pred_labels": labels}
# %%

    if net.segmenter is not None:
        prediction["pred_seg"] = net.segmenter.postprocess_for_inference(pred_seg)["pred_seg"]
# %%
# SECTION:-------------------- RetinaUNet LBD — 2 predict_inner -----------------------------------------------------
    batch = D.predict_inner(batch)
    if "raw_pred" in batch:
        print("path: fast (single tile)")
    else:
        print("path: swi", batch["merged_boxes"].shape)
# %%
    print(batch.keys())
    img = batch["image"][0, 0].detach().cpu()
    boxes = batch["pred_box"].detach().cpu()
    lm = batch["stitched_seg"].detach().cpu()
    scores = batch["pred_score"].detach().cpu()
    ImageMaskViewer([img, lm[0,1]], "ii")
    ImageBBoxViewer(img, boxes)


# %%
# SECTION:-------------------- RetinaUNet LBD — 3 postprocess -----------------------------------------------------
    batch2 = D.postprocess(batch)
    print("pred_box", batch2["pred_box"].shape)
    print("pred_seg", batch2["pred_seg"].shape, "fg_voxels", int(batch2["pred_seg"].sum()))

# %%
# SECTION:-------------------- RetinaUNet LBD — 4 viz -----------------------------------------------------
    img = batch2["image"][0, 0].detach().cpu()
    boxes = batch2["merged_boxes"].detach().cpu()
    lm = batch2["stitched_seg"].detach().cpu()
    scores = batch2["merged_scores"].detach().cpu()
    box = boxes
# %%
    bbo = nndet_batch_to_xyzxyz(boxes[1, :])
    ImageBBoxViewer(img, bbo)
                    
# %%
# SECTION:-------------------- RetinaUNet nifti — 0 setup -----------------------------------------------------
    D = DetPatchInfererRetinaUNet(
        run_name=run_p,
        project_title=project_title,
        devices=devices,
        safe_mode=safe_mode,
        patch_overlap=patch_overlap,
        debug=debug_,
    )
    data_nii = helpers.load_images_nifti(imgs[:1])
    D.setup()
    D.prepare_data(data=data_nii, collate_fn=None)
    D.create_and_set_postprocess_transforms()
    print(
        "nifti keys_preproc",
        D.keys_preproc,
        "keys_postproc",
        D.keys_postproc,
    )

# %%
# SECTION:-------------------- RetinaUNet nifti — 1 batch + postprocess -----------------------------------------------------
    batch_nii = next(iter(D.pred_dl))
    batch_nii = D.predict_inner(batch_nii)
    batch_nii = D.postprocess(batch_nii)
    print(
        "nifti spatial",
        tuple(batch_nii["image"].shape),
        "pred_seg",
        batch_nii["pred_seg"].shape,
        "pred_box",
        batch_nii["pred_box"].shape,
    )

# %%

# SECTION:-------------------- RetinaNet patch (single LBD crop) --------------------------------------------------------
    run_p = "LIDCA-GYRO"
    project_title = "lidca"
    P = DetPatchInferer(
        run_name=run_p,
        project_title=project_title,
        devices=devices,
        safe_mode=safe_mode,
        patch_overlap=patch_overlap,
        debug=debug_,
    )

    data = helpers.load_images_pt(imgs2)
    P.setup()
    P.prepare_data(data=data, collate_fn=img_bbox_collated)
# %%
    iteri = iter(P.pred_dl)
    batch = next(iter(P.predict()))
    batch = P.predict_inner(batch)
    img = batch["image"][0, 0].detach().cpu()
    boxes = batch["pred_box"].detach().cpu()
    scores = batch["pred_score"].detach().cpu()
    crop = tuple(int(v) for v in img.shape)
    print(
        "crop",
        crop,
        "n_boxes",
        boxes.shape[0],
        "score_max",
        float(scores.max()) if scores.numel() else None,
    )
    ImageBBoxViewer(img, boxes)

