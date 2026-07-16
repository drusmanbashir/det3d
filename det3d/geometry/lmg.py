from ast import literal_eval

import torch
from label_analysis.geometry_pt import LabelMapGeometryPT


def voxel_start_size_to_xyzxyz(bbox):
    """ITK nbrhood bbox [ix,iy,iz,sx,sy,sz] → patch-voxel xyzxyz [x1,y1,z1,x2,y2,z2]."""
    if isinstance(bbox, str):
        bbox = literal_eval(bbox)
    x0, y0, z0, sx, sy, sz = [float(x) for x in bbox]
    return [x0, y0, z0, x0 + sx, y0 + sy, z0 + sz]


class DetectionLabelMapGeometryPT(LabelMapGeometryPT):
    def to_voxel_detection_records(self):
        boxes = []
        labels = []
        for _, row in self.nbrhoods.iterrows():
            boxes.append(
                torch.tensor(
                    voxel_start_size_to_xyzxyz(row["bbox"]),
                    dtype=torch.float32,
                )
            )
            labels.append(int(row["label_org"]))
        return {"box": boxes, "label": labels}
