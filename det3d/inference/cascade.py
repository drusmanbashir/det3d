import ipdb
import torch
from det3d.inference.patch import DetPatchInferer
from det3d.inference.transforms import build_det_postprocess_transforms_dict
from fran.inference.cascade import CascadeInferer
from utilz.stringz import headline

tr = ipdb.set_trace


class DetCascadeInferer(CascadeInferer):
    keys_postproc = "Pre,SqL,Clip,Scale,Off,VoxCopy,FullMeta,World,WorldCopy,Mode,Meta"
    keys_postproc_safe = "SqL,Meta"

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
            img = batch["image"]
            if isinstance(img, torch.Tensor) and img.dim() == 5:
                img = img[0]
            bb = bboxes[case_idx]
            crop_shape = batch.get(
                "crop_spatial_shape",
                tuple(int(s.stop - s.start) for s in bb[1:]),
            )
            item = {
                "image": img.detach().cpu() if isinstance(img, torch.Tensor) else img,
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
        gt_box_mode = self.params["configs"]["plan_train"]["gt_box_mode"]
        self.postprocess_transforms_dict = build_det_postprocess_transforms_dict(
            self,
            gt_box_mode=gt_box_mode,
            affine_lps_to_ras=True,
        )

    def set_postprocess_tfms_keys(self):
        if self.safe_mode is False:
            self.postprocess_tfms_keys = self.keys_postproc
        else:
            self.postprocess_tfms_keys = self.keys_postproc_safe
        if self.save is True:
            self.postprocess_tfms_keys += ",Sav"

    def postprocess_iterate(self, batch):
        if isinstance(batch, list):
            batch = batch[0]
        bbox = batch.get("bounding_box")
        if (
            bbox
            and isinstance(bbox, list)
            and len(bbox) == 1
            and isinstance(bbox[0], list)
            and all(isinstance(x, slice) for x in bbox[0])
        ):
            batch["bounding_box"] = bbox[0]
        for tfm in self.postprocess_transforms:
            headline(tfm)
            tr()
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
