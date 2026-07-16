import torch
from det3d.managers.helpers.nndet_retinaunet import (
    ensure_nndet_importable,
    nndet_batch_to_xyzxyz,
    offset_nndet_xyxyzz_boxes,
)
from det3d.utils.tensor import plain_tensor
from fran.inference.base import BaseInferer
from fran.inference.cascade import img_bbox_collated
from fran.transforms.inferencetransforms import SqueezeListofListsd
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, ScaleIntensityRanged
from monai.transforms.post.dictionary import Invertd
from utilz.cprint import cprint


class DetPatchInfererBase(BaseInferer):
    arch_name = "base"

    def bind_preprocess_transforms(self):
        for key, value in self.preprocess_transforms_dict.items():
            setattr(self, key, value)

    def norm_dtype_transforms(self):
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
        return {"Norm": norm, "Dtype": dtype}

    def set_preprocess_tfms_keys(self):
        self.keys_preproc = type(self).keys_preproc
        self.preprocess_tfms_keys = self.keys_preproc

    def set_postprocess_tfms_keys(self):
        chain = type(self).keys_postproc
        if not self.save:
            chain = ",".join(k for k in chain.split(",") if k != "S")
        self.keys_postproc = chain
        self.postprocess_tfms_keys = chain

    def set_preprocess_transforms(self):
        transform = self.tfms_from_dict(
            self.keys_preproc, self.preprocess_transforms_dict
        )
        self.preprocess_compose = Compose(transform)

    def set_postprocess_transforms(self):
        self.postprocess_transforms = self.tfms_from_dict(
            self.keys_postproc, self.postprocess_transforms_dict
        )
        self.postprocess_compose = Compose(self.postprocess_transforms)

    def check_plan_compatibility(self):
        pass

    def prepare_data(self, data, collate_fn=img_bbox_collated):
        super().prepare_data(data, collate_fn=collate_fn)


class DetPatchRetinaUNet(DetPatchInfererBase):
    keys_preproc = "L,E,S,O,Norm,Dtype"
    keys_postproc = "Pack,SqL,InvP,InvPreBox"

    def __init__(
        self,
        run_name,
        project_title=None,
        patch_overlap=0.25,
        devices=(0,),
        safe_mode=False,
        save=False,
        params=None,
        ckpt=None,
        debug=False,
        **kwargs,
    ):
        cprint("Setting up RetinaUNet Det patch inference", color="red", bold=True)
        from det3d.inference.transforms import LoadInferImaged
        from det3d.managers.retinaunet import RetinaUNetManager
        from fran.inference.helpers import load_params

        if params is None:
            params = load_params(run_name)
        self.patch_overlap = patch_overlap
        self._box_acc = []
        super().__init__(
            run_name=run_name,
            project_title=project_title,
            patch_overlap=patch_overlap,
            devices=devices,
            safe_mode=safe_mode,
            save=save,
            save_channels=False,
            params=params,
            ckpt=ckpt,
            debug=debug,
            keys_preproc=self.keys_preproc,
            keys_postproc=self.keys_postproc,
            model_manager=RetinaUNetManager,
            **kwargs,
        )

    def create_preprocess_transforms(self):
        from monai.transforms import Orientationd, Spacingd
        from utilz.stringz import ast_literal_eval

        from det3d.inference.transforms import LoadInferImaged

        self.preprocess_transforms_dict = {
            "L": LoadInferImaged(keys=["image"]),
            "E": EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            **self.norm_dtype_transforms(),
        }
        spacing = ast_literal_eval(self.params["configs"]["plan_train"]["spacing"])
        self.preprocess_transforms_dict["S"] = Spacingd(keys=["image"], pixdim=spacing)
        self.preprocess_transforms_dict["O"] = Orientationd(keys=["image"], axcodes="RAS")
        self.bind_preprocess_transforms()

    def create_postprocess_transforms(self, preprocess_transform):
        from det3d.inference.post import InvPreprocessBoxd, PackRetinaUNetPredsd

        self.postprocess_transforms_dict = {
            "Pack": PackRetinaUNetPredsd(),
            "SqL": SqueezeListofListsd(keys=["bounding_box"]),
            "InvP": Invertd(
                keys=["pred"],
                transform=preprocess_transform,
                orig_keys=["image"],
            ),
            "InvPreBox": InvPreprocessBoxd(),
        }
        for key, value in self.postprocess_transforms_dict.items():
            setattr(self, key, value)

    def setup(self):
        super().setup()
        self.inferer.with_coord = True

    def inference_params(self):
        net = self.model.net
        ensure_nndet_importable()
        from nndet.inference.detection.model import batched_weighted_nms_model

        params = {
            "model_score_thresh": 0.0,
            "model_topk": int(net.topk_candidates),
            "model_detections_per_image": int(net.detections_per_img),
            "remove_small_boxes": float(net.remove_small_boxes),
            "model_iou": 0.1,
            "model_nms_fn": batched_weighted_nms_model,
        }
        return params

    @staticmethod
    def tile_origin(unravel_entry):
        spatial = unravel_entry[2:]
        return [int(spatial[0].start), int(spatial[1].start), int(spatial[2].start)]

    def offset_boxes(self, boxes, origin):
        return offset_nndet_xyxyzz_boxes(boxes, origin)

    def box_tile_weight(self, boxes, tile_size):
        ensure_nndet_importable()
        from nndet.core.boxes.ops import box_center
        from nndet.inference.ensembler.detection import BoxEnsemblerSelective

        centers = box_center(boxes) if boxes.numel() else boxes
        return BoxEnsemblerSelective._get_box_in_tile_weight(centers, tile_size)

    def clip_boxes_xyzxyz(self, boxes, spatial):
        if boxes.numel() == 0:
            return boxes
        out = boxes.clone()
        out[:, 0] = out[:, 0].clamp(0, spatial[0])
        out[:, 1] = out[:, 1].clamp(0, spatial[1])
        out[:, 2] = out[:, 2].clamp(0, spatial[2])
        out[:, 3] = out[:, 3].clamp(0, spatial[0])
        out[:, 4] = out[:, 4].clamp(0, spatial[1])
        out[:, 5] = out[:, 5].clamp(0, spatial[2])
        return out

    def filter_boxes(self, boxes, probs, labels, weights, spatial):
        ensure_nndet_importable()
        from nndet.core.boxes.clip import clip_boxes_to_image
        from nndet.core.boxes.ops import remove_small_boxes

        params = self.inference_params()
        p_sorted, idx_sorted = probs.sort(descending=True)
        idx_sorted = idx_sorted[: params["model_topk"]]

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
        return nndet_batch_to_xyzxyz(b), p, l

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

    def needs_swi(self, img):
        patch_size = [int(v) for v in self.model.plan["patch_size"]]
        spatial = tuple(int(v) for v in img.shape[-3:])
        return any(s > p for s, p in zip(spatial, patch_size))

    def forward_tile(self, img):
        x = img.unsqueeze(0)
        device_type = x.device.type
        with torch.autocast(device_type, enabled=device_type == "cuda"):
            pred = self.model.net.inference_step(x)
        return pred

    def swi_predictor(self, win_data, unravel_slice):
        device_type = win_data.device.type
        with torch.autocast(device_type, enabled=device_type == "cuda"):
            pred = self.model.net.inference_step(win_data)
        tile_size = tuple(int(v) for v in win_data.shape[-3:])
        for i in range(win_data.shape[0]):
            boxes = pred["pred_boxes"][i].detach()
            scores = pred["pred_scores"][i].detach()
            labels = pred["pred_labels"][i].detach()
            origin = self.tile_origin(unravel_slice[i])
            boxes = self.offset_boxes(boxes, origin)
            weights = self.box_tile_weight(boxes, tile_size)
            self._box_acc.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                    "weights": weights,
                }
            )
        return pred["pred_seg"]

    def run_swi(self, img):
        self._box_acc = []
        stitched_seg = self.inferer(inputs=img, network=self.swi_predictor)
        spatial = tuple(int(v) for v in img.shape[-3:])
        boxes, scores, labels = self.merge_tile_boxes(self._box_acc, spatial)
        return stitched_seg, boxes, scores, labels

    def predict_inner(self, batch):
        device = next(self.model.net.parameters()).device
        img = plain_tensor(batch["image"]).float().to(device)
        if img.dim() == 5:
            img = img[0]
        if img.dim() == 3:
            img = img.unsqueeze(0)
        self.model.eval()
        if self.needs_swi(img):
            vol = img.unsqueeze(0)
            seg, boxes, scores, labels = self.run_swi(vol)
            batch["stitched_seg"] = seg
            batch["merged_boxes"] = boxes
            batch["merged_scores"] = scores
            batch["merged_labels"] = labels
            return batch
        batch["raw_pred"] = self.forward_tile(img)
        return batch


class DetPatchRetinaNet(DetPatchInfererBase):
    keys_preproc = "L,E,S,O,Norm,Dtype"
    keys_postproc = "Pack,SqL,InvPreBox"

    def __init__(
        self,
        run_name,
        project_title=None,
        patch_overlap=0.25,
        devices=(0,),
        safe_mode=False,
        save=False,
        params=None,
        ckpt=None,
        debug=False,
        **kwargs,
    ):
        cprint("Setting up RetinaNet Det patch inference", color="red", bold=True)
        from det3d.managers.retinanet import RetinaNetManager
        from fran.inference.helpers import load_params

        if params is None:
            params = load_params(run_name)
        super().__init__(
            run_name=run_name,
            project_title=project_title,
            patch_overlap=patch_overlap,
            devices=devices,
            safe_mode=safe_mode,
            save=save,
            save_channels=False,
            params=params,
            ckpt=ckpt,
            debug=debug,
            keys_preproc=self.keys_preproc,
            keys_postproc=self.keys_postproc,
            model_manager=RetinaNetManager,
            **kwargs,
        )

    def create_preprocess_transforms(self):
        from monai.transforms import Orientationd, Spacingd
        from utilz.stringz import ast_literal_eval

        from det3d.inference.transforms import LoadInferImaged

        self.preprocess_transforms_dict = {
            "L": LoadInferImaged(keys=["image"]),
            "E": EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            **self.norm_dtype_transforms(),
        }
        spacing = ast_literal_eval(self.params["configs"]["plan_train"]["spacing"])
        self.preprocess_transforms_dict["S"] = Spacingd(keys=["image"], pixdim=spacing)
        self.preprocess_transforms_dict["O"] = Orientationd(keys=["image"], axcodes="RAS")
        self.bind_preprocess_transforms()

    def create_postprocess_transforms(self, preprocess_transform):
        from det3d.inference.post import InvPreprocessBoxd, PackRetinaNetPredsd

        self.postprocess_transforms_dict = {
            "Pack": PackRetinaNetPredsd(),
            "SqL": SqueezeListofListsd(keys=["bounding_box"]),
            "InvPreBox": InvPreprocessBoxd(),
        }
        for key, value in self.postprocess_transforms_dict.items():
            setattr(self, key, value)

    def predict_inner(self, batch):
        device = next(self.model.parameters()).device
        img = plain_tensor(batch["image"]).float().to(device)
        if img.dim() == 5:
            img = img[0]
        if img.dim() == 3:
            img = img.unsqueeze(0)
        detector = self.model.detector
        detector.eval()
        val_input = img.contiguous()
        patch_size = self.model.val_patch_size
        use_inferer = val_input[0, ...].numel() >= int(torch.prod(torch.tensor(patch_size)))
        with torch.inference_mode():
            outputs = detector([val_input], use_inferer=use_inferer)
        batch["raw_pred"] = outputs[0]
        return batch


class DetPatchLBD(DetPatchRetinaUNet):
    """Pre-spaced LBD .pt crops; patch post includes save when save=True."""

    keys_preproc = "L,E,Norm,Dtype"
    keys_postproc = "Pack,SqL,A,Int,W,S"

    def create_preprocess_transforms(self):
        from det3d.inference.transforms import LoadInferImaged

        self.preprocess_transforms_dict = {
            "L": LoadInferImaged(keys=["image"]),
            "E": EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            **self.norm_dtype_transforms(),
        }
        self.bind_preprocess_transforms()

    def create_postprocess_transforms(self, preprocess_transform):
        import numpy as np
        from fran.transforms.inferencetransforms import MakeWritabled
        from monai.transforms.post.dictionary import AsDiscreted
        from monai.transforms.utility.dictionary import CastToTyped

        from det3d.inference.post import PackRetinaUNetPredsd, SaveDetOutputd

        self.postprocess_transforms_dict = {
            "Pack": PackRetinaUNetPredsd(),
            "SqL": SqueezeListofListsd(keys=["bounding_box"]),
            "A": AsDiscreted(keys=["pred"], argmax=True),
            "Int": CastToTyped(keys=["pred"], dtype=np.uint8),
            "W": MakeWritabled(keys=["pred"]),
            "S": SaveDetOutputd(
                output_dir=self.output_folder,
                run_w="",
                run_p=self.run_name,
                write_seg=True,
            ),
        }
        for key, value in self.postprocess_transforms_dict.items():
            setattr(self, key, value)

# %%
# SECTION:-------------------- setup --------------------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    import torch
    from fran.inference import helpers
    from fran.inference.cascade import img_bbox_collated
    from fran.inference.common_vars import *
    from utilz.imageviewers import ImageBBoxViewer, ImageMaskViewer

    devices = [0]
    safe_mode = True
    patch_overlap = 0.25
    debug_ = False


# %%
# SECTION:-------------------- LIDC patch — 0 paths ---------------------------------------------------------------------
    run_p = "LIDCA-QUARK"
    project_title = "lidca"
    case_id = "lidc_0001"

    lidc_all_fldr = Path("/media/UB/datasets/lidc_all/images")
    lbd_pt_fldr = Path(
        "/r/datasets/preprocessed/lidca/lbd/spc_080_080_150_rlb40c36831_rlb40c36831_ex000/images"
    )
    imgs_nii = [p for p in sorted(lidc_all_fldr.glob("*.nii.gz")) if case_id in p.name]
    imgs_pt = sorted(lbd_pt_fldr.glob("*.pt"))
    imgs_pt = [p for p in imgs_pt if case_id in p.name]

# %%
# SECTION:-------------------- LIDC RetinaUNet nifti — 1 setup ----------------------------------------------------------
    D = DetPatchRetinaUNet(
        run_name=run_p,
        project_title=project_title,
        devices=devices,
        safe_mode=safe_mode,
        patch_overlap=patch_overlap,
        debug=debug_,
        save=False,
    )
    data_nii = helpers.load_images_nifti(imgs_nii)
    D.setup()
    D.prepare_data(data=data_nii, collate_fn=img_bbox_collated)
    D.create_and_set_postprocess_transforms()
    print("preproc", D.keys_preproc)
    print("postproc", D.keys_postproc)
    print("patch_size", [int(v) for v in D.model.plan["patch_size"]])

# %%
# SECTION:-------------------- LIDC RetinaUNet nifti — 2 predict_inner ---------------------------------------------------
    batch = next(iter(D.pred_dl))
    batch = D.predict_inner(batch)
    if "raw_pred" in batch:
        print("path: single tile")
        raw = batch["raw_pred"]
        print("raw boxes", raw["pred_boxes"].shape, "raw seg", raw["pred_seg"].shape)
    else:
        print("path: swi", batch["merged_boxes"].shape, batch["stitched_seg"].shape)

# %%
# SECTION:-------------------- LIDC RetinaUNet nifti — 3 postprocess stepwise -------------------------------------------
    tfms = D.postprocess_transforms_dict
    dici = dict(batch)
    for key in D.keys_postproc.split(","):
        dici = tfms[key](dici)
        print(
            key,
            "pred fg",
            int(dici["pred"].sum()),
            "n_boxes",
            dici["pred_box"].shape[0],
        )
    img = dici["image"][0, 0].detach().cpu()
    pred = dici["pred"][0].detach().cpu()
    boxes = dici["pred_box"].detach().cpu()
    ImageMaskViewer([img, pred], "nifti")
    ImageBBoxViewer(img, boxes)

# %%
# SECTION:-------------------- LIDC RetinaUNet LBD .pt — 4 setup --------------------------------------------------------
    D = PatchInfererLBD(
        run_name=run_p,
        project_title=project_title,
        devices=devices,
        safe_mode=safe_mode,
        patch_overlap=patch_overlap,
        debug=debug_,
        save=False,
    )
    from det3d.inference.lbd_pt import load_lbd_pt_patch_data

    data_pt = load_lbd_pt_patch_data(imgs_pt)
    D.setup()
    D.prepare_data(data=data_pt, collate_fn=img_bbox_collated)
    D.create_and_set_postprocess_transforms()
    print("LBD crop shape", tuple(data_pt[0]["image"].shape))

# %%
# SECTION:-------------------- LIDC RetinaUNet LBD .pt — 5 predict + postprocess ---------------------------------------
    batch = next(iter(D.pred_dl))
    batch = D.predict_inner(batch)
    batch = D.postprocess(batch)
    img = batch["image"][0, 0].detach().cpu()
    pred = batch["pred"][0].detach().cpu()
    boxes = batch["pred_box"].detach().cpu()
    scores = batch["pred_score"].detach().cpu()
    print("LBD fg", int(pred.sum()), "n_boxes", boxes.shape[0], "score_max", float(scores.max()))
    ImageMaskViewer([img, pred], "lbd")
    ImageBBoxViewer(img, boxes)

# %%
# SECTION:-------------------- LIDC patch — 6 single-tile probe ---------------------------------------------------------
    batch = next(iter(D.pred_dl))
    img = plain_tensor(batch["image"]).float()
    if img.dim() == 5:
        img = img[0]
    if img.dim() == 3:
        img = img.unsqueeze(0)
    patch_size = [int(v) for v in D.model.plan["patch_size"]]
    sx, sy, sz = 60, 0, 100
    slc = (
        slice(sx, sx + patch_size[0]),
        slice(sy, sy + patch_size[1]),
        slice(sz, sz + patch_size[2]),
    )
    im_tile = img[slc].unsqueeze(0).to(next(D.model.net.parameters()).device)
    raw = D.forward_tile(im_tile)
    print("tile raw boxes", raw["pred_boxes"].shape)

# %%
# SECTION:-------------------- LIDC RetinaNet patch — 7 setup -----------------------------------------------------------
    P = PatchInferer(
        run_name=run_p,
        project_title=project_title,
        devices=devices,
        safe_mode=safe_mode,
        patch_overlap=patch_overlap,
        debug=debug_,
        save=False,
    )
    P.setup()
    P.prepare_data(data=data_pt, collate_fn=img_bbox_collated)
    P.create_and_set_postprocess_transforms()

# %%
    batch = next(iter(P.pred_dl))
    batch = P.predict_inner(batch)
    batch = P.postprocess(batch)
    img = batch["image"][0, 0].detach().cpu()
    boxes = nndet_batch_to_xyzxyz(batch["pred_box"]).detach().cpu()
    print("RetinaNet n_boxes", boxes.shape[0])
    ImageBBoxViewer(img, boxes)
