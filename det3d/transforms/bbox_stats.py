import torch
from det3d.geometry.lmg import DetectionLabelMapGeometryPT
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
    ):
        super().__init__([image_key, lm_key], False)
        self.image_key = image_key
        self.lm_key = lm_key
        self.ignore_labels = ignore_labels or []
        self.dusting_threshold = dusting_threshold
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
        # xyzxyz voxels — same contract as bbox sidecars and det3d_batch_to_nndet.
        rec = L.to_voxel_detection_records()
        d["LMG"] = L
        d["nbrhoods"] = L.nbrhoods
        boxes = rec["box"]
        labels = rec["label"]
        if len(boxes) == 0:
            d["bbox"] = torch.zeros((0, 6), dtype=torch.float32)
            d["label"] = torch.zeros((0,), dtype=torch.long)
        else:
            d["bbox"] = torch.stack(boxes)
            d["label"] = torch.tensor(labels, dtype=torch.long)
        return d


class AttachDetectionGTd(MapTransform):
    """LMG on cropped patch; patch-voxel xyzxyz box/label on data dict."""

    def __init__(
        self,
        image_key="image",
        lm_key="lm",
        dusting_threshold=3.0,
        ignore_labels=None,
    ):
        super().__init__([image_key, lm_key], False)
        self.image_key = image_key
        self.lm_key = lm_key
        self.ignore_labels = ignore_labels or []
        self.dusting_threshold = dusting_threshold

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
        rec = L.to_voxel_detection_records()
        if len(rec["box"]) == 0:
            return d
        d["box"] = rec["box"]
        d["label"] = rec["label"]
        return d
