from ast import literal_eval

import torch
from label_analysis.geometry_pt import LabelMapGeometryPT


def voxel_start_size_to_gt_box(bbox, gt_box_mode):
    """ITK nbrhood bbox: [index_x, index_y, index_z, size_x, size_y, size_z] voxels."""
    if isinstance(bbox, str):
        bbox = literal_eval(bbox)
    x0, y0, z0, sx, sy, sz = [float(x) for x in bbox]
    if gt_box_mode == "cccwhd":
        return [x0 + sx / 2, y0 + sy / 2, z0 + sz / 2, sx, sy, sz]
    if gt_box_mode == "xyzxyz":
        return [x0, y0, z0, x0 + sx, y0 + sy, z0 + sz]
    raise ValueError(f"unsupported gt_box_mode {gt_box_mode}")


class DetectionLabelMapGeometryPT(LabelMapGeometryPT):
    def to_voxel_detection_records(
        self,
        gt_box_mode,
        foreground_class_id=0,
        remapping_train=None,
    ):
        boxes = []
        labels = []
        for _, row in self.nbrhoods.iterrows():
            boxes.append(
                torch.tensor(
                    voxel_start_size_to_gt_box(row["bbox"], gt_box_mode),
                    dtype=torch.float32,
                )
            )
            if remapping_train is None:
                labels.append(foreground_class_id)
            else:
                labels.append(remapping_train[int(row["label_org"])])
        return {"box": boxes, "label": labels}
