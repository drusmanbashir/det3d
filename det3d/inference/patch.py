import numpy as np
import torch
from det3d.managers.retinanet_bk import RetinaNetManager, VAL_PATCH_SIZE
from fran.inference.base import BaseInferer
from fran.inference.cascade import img_bbox_collated
from fran.transforms.inferencetransforms import SqueezeListofListsd
from monai.transforms import EnsureTyped, ScaleIntensityRanged
from utilz.cprint import cprint


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
        plan = self.plan
        self.preprocess_transforms_dict.pop("N", None)
        self.preprocess_transforms_dict["Norm"] = ScaleIntensityRanged(
            keys=["image"],
            a_min=float(plan["intensity_a_min"]),
            a_max=float(plan["intensity_a_max"]),
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
        use_inferer = val_inputs[0][0, ...].numel() >= int(np.prod(VAL_PATCH_SIZE))
        with torch.inference_mode():
            outputs = detector(val_inputs, use_inferer=use_inferer)
        out = outputs[0]
        batch["pred_box"] = out[detector.target_box_key].detach()
        batch["pred_label"] = out[detector.target_label_key].detach()
        batch["pred_score"] = out[detector.pred_score_key].detach()
        return batch
