import torch
from det.geometry.lmg import DetectionLabelMapGeometryPT
from monai.transforms import MapTransform


def maybe_squeeze(lm, desired_dim=3):
    if lm.dim() == desired_dim + 1 and lm.shape[0] == 1:
        return lm[0].clone()
    if lm.dim() == desired_dim:
        return lm.clone()
    raise ValueError(f"lm must be 3D or 4D, found {lm.dim()}")


class DetectionBBoxStatsd(MapTransform):
    def __init__(
        self,
        image_key="image",
        lm_key="lm",
        dusting_threshold=3.0,
        dusting_method="major_axis",
        ignore_labels=None,
        foreground_class_id=0,
        remapping_train=None,
        gt_box_mode="cccwhd",
    ):
        super().__init__([image_key, lm_key], False)
        self.image_key = image_key
        self.lm_key = lm_key
        self.ignore_labels = ignore_labels or []
        self.dusting_threshold = dusting_threshold
        self.foreground_class_id = foreground_class_id
        self.remapping_train = remapping_train
        self.gt_box_mode = gt_box_mode
        assert dusting_method in ["major_axis", "bbox_smallest_side"]

    def __call__(self, data):
        d = dict(data)
        lm2 = maybe_squeeze(d[self.lm_key], 3)
        L = DetectionLabelMapGeometryPT(
            li=lm2,
            ignore_labels=self.ignore_labels,
            compute_feret=False,
        )
        L.dust(self.dusting_threshold)
        rec = L.to_voxel_detection_records(
            self.gt_box_mode,
            foreground_class_id=self.foreground_class_id,
            remapping_train=self.remapping_train,
        )
        d["LMG"] = L
        d["nbrhoods"] = L.nbrhoods
        d["detection_box"] = rec["box"]
        d["detection_label"] = rec["label"]
        return d


class AttachDetectionGTd(MapTransform):
    """LMG on cropped patch; patch-voxel gt_box_mode box/label on data dict."""

    def __init__(
        self,
        image_key="image",
        lm_key="lm",
        dusting_threshold=3.0,
        ignore_labels=None,
        foreground_class_id=0,
        remapping_train=None,
        gt_box_mode="cccwhd",
    ):
        super().__init__([image_key, lm_key], False)
        self.image_key = image_key
        self.lm_key = lm_key
        self.ignore_labels = ignore_labels or []
        self.dusting_threshold = dusting_threshold
        self.foreground_class_id = foreground_class_id
        self.remapping_train = remapping_train
        self.gt_box_mode = gt_box_mode

    def __call__(self, data):
        d = dict(data)
        lm2 = maybe_squeeze(d[self.lm_key], 3)
        L = DetectionLabelMapGeometryPT(
            li=lm2,
            ignore_labels=self.ignore_labels,
            compute_feret=False,
        )
        L.dust(self.dusting_threshold)
        stats = d.get("stats")
        if stats and "label_cc" in stats:
            label_cc = int(stats["label_cc"])
            matched = L.nbrhoods[L.nbrhoods["label_cc"] == label_cc]
            if len(matched) > 0:
                L.nbrhoods = matched
        rec = L.to_voxel_detection_records(
            self.gt_box_mode,
            foreground_class_id=self.foreground_class_id,
            remapping_train=self.remapping_train,
        )
        if len(rec["box"]) == 0:
            return d
        d["box"] = rec["box"]
        d["label"] = rec["label"]
        return d
