"""Detection box preprocessing and spatial tracking in exclusive xyzxyz voxels."""

from itertools import product

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import MapTransform
from monai.transforms.spatial.dictionary import ConvertBoxToPointsd, ConvertPointsToBoxesd
from monai.transforms.utils import to_affine_nd
from monai.utils.type_conversion import convert_to_dst_type
from scipy import ndimage
from scipy.spatial import ConvexHull
from torch.nn.utils.rnn import pad_sequence

from det3d.transforms.detection import ClipBoxToImagedWithDf


def _row_components(lm, rows):  #AI
    """Yield each sidecar row's face-connected voxel indices within its tight box."""
    lm = np.asarray(torch.as_tensor(lm).detach().cpu()).reshape(lm.shape[-3:])
    pending = {}
    for i, row in enumerate(rows.itertuples(index=False)):
        pending.setdefault(int(row.label_org), {}).setdefault(tuple(row.bbox_xyzxyz), []).append(i)
    components = [None] * len(rows)
    for label, boxes in pending.items():
        cc, _ = ndimage.label(lm == label)
        for component, roi in enumerate(ndimage.find_objects(cc), start=1):
            box = tuple(s.start for s in roi) + tuple(s.stop for s in roi)
            matches = boxes.get(box)
            if matches:
                row_index = matches.pop(0)
                components[row_index] = (roi, cc[roi] == component)
    for i, component in enumerate(components):
        if component is None:
            raise ValueError(f"Sidecar row {i} has no matching component in the LBD label map")
        yield component


def voxel_map(before, after, device):  #AI
    """Map pre-transform voxel centres into the new grid."""
    before = before.to(device=device, dtype=torch.float64)
    after = after.to(device=device, dtype=torch.float64)
    result = torch.linalg.inv(after) @ before
    return result


def apply_affine_to_points_gpu(data, affine, dtype=torch.float64):  #AI
    """Apply a voxel-centre affine to groups of points."""
    data_ = data.to(dtype=torch.float64)
    affine = to_affine_nd(
        data_.shape[-1], affine.to(device=data_.device, dtype=torch.float64)
    )
    ones = torch.ones(
        (data_.shape[0], data_.shape[1], 1),
        device=data_.device,
        dtype=torch.float64,
    )
    homogeneous = torch.cat((data_, ones), dim=2)
    transformed = torch.matmul(homogeneous, affine.T)
    out, *_ = convert_to_dst_type(transformed[:, :, :-1], data, dtype=dtype)
    return out


def _warp_edges(points, transform):  #AI
    """Shift exclusive edges into centre coordinates for the voxel map."""
    warped = apply_affine_to_points_gpu(points - 0.5, transform, dtype=torch.float32) + 0.5
    return warped


class BoxTracker(MapTransform):
    """Own one method's preprocessing, spatial updates and final box targets."""

    def __init__(self, box_key="bbox", label_key="label", image_key="image"):  #AI
        """Set the detection target and reference image keys."""
        super().__init__([box_key])
        self.box_key = box_key
        self.label_key = label_key
        self.image_key = image_key
        self.clip = ClipBoxToImagedWithDf(box_key, [label_key], image_key)

    @staticmethod
    def compute(lm, rows):  #AI
        """Return the method's extra preprocessing payload."""
        raise NotImplementedError

    def warp(self, data, transform):  #AI
        """Update tracking state from the voxel-centre map."""
        raise NotImplementedError

    def translate(self, data, shift):  #AI
        """Update tracking after an integer crop or pad shift."""
        transform = torch.eye(4, dtype=torch.float64)
        transform[:3, 3] = torch.as_tensor(shift)
        return self.warp(data, transform)

    def finalize(self, data):  #AI
        """Produce boxes and labels aligned to the final image."""
        return self.clip(data)

    def __call__(self, data):  #AI
        """Finalize the targets at the end of a transform chain."""
        return self.finalize(data)


class CornerTracker(BoxTracker):
    """Warp the eight corners of each current axis-aligned box."""

    @staticmethod
    def compute(lm, rows):  #AI
        """Use the existing bbox column as the entire preprocessing payload."""
        return rows["bbox_xyzxyz"]

    def warp(self, data, transform):  #AI
        """Warp the current boxes and take their axis-aligned extent."""
        if data[self.box_key].numel() == 0:
            return data
        key = "__box_corners__"
        d = ConvertBoxToPointsd(keys=[self.box_key], point_key=key)(data)
        corners = d[key].to(d[self.image_key].device)
        warped = _warp_edges(corners, transform)
        d[key] = MetaTensor(warped, meta=corners.meta)
        d = ConvertPointsToBoxesd(keys=[key], box_key=self.box_key)(d)
        del d[key]
        return d

    def translate(self, data, shift):  #AI
        """Preserve the original float32 addition for integer shifts."""
        box = torch.as_tensor(data[self.box_key], dtype=torch.float32)
        if box.numel() != 0:
            off = torch.as_tensor(shift, dtype=box.dtype, device=box.device)
            box = box.clone()
            box[:, :3] += off
            box[:, 3:] += off
            data[self.box_key] = box
        return data


class HullTracker(BoxTracker):
    """Keep the convex hull vertices through all spatial transforms."""

    def __init__(self, box_key="bbox", label_key="label", image_key="image"):  #AI
        """Keep subvoxel box extents until checking the resampled mask."""
        super().__init__(box_key, label_key, image_key)
        self.clip.remove_empty = False

    @staticmethod
    def compute(lm, rows):  #AI
        """Build the convex hull of each matched component's voxel cubes."""
        offsets = np.array(list(product((0, 1), repeat=3)))
        hulls = []
        for roi, component in _row_components(lm, rows):
            boundary = component & ~ndimage.binary_erosion(component)
            centres = np.argwhere(boundary) + np.array([s.start for s in roi])
            points = (centres[:, None] + offsets).reshape(-1, 3)
            hulls.append(points[ConvexHull(points).vertices].tolist())
        return hulls

    def warp(self, data, transform):  #AI
        """Warp padded vertex lists without reboxing between spatial transforms."""
        hulls = data["hull_xyz"]
        if len(hulls) == 0:
            return data
        dev = data[self.image_key].device
        if not torch.is_tensor(hulls):
            hulls = [torch.as_tensor(h, dtype=torch.float32) for h in hulls]
            points = pad_sequence(hulls, batch_first=True)
            lengths = torch.as_tensor([len(h) for h in hulls])
            padding = torch.arange(points.shape[1])[None, :] >= lengths[:, None]
            hulls = torch.where(padding[..., None], points[:, :1], points)
        data["hull_xyz"] = _warp_edges(hulls.to(dev), transform).contiguous()
        return data

    def finalize(self, data):  #AI
        """Clip each hull's extent and discard boxes without matching voxels."""
        d = dict(data)
        hulls = d.pop("hull_xyz")
        if len(hulls) != 0:
            if not torch.is_tensor(hulls):
                hulls = [torch.as_tensor(h, dtype=torch.float32) for h in hulls]
                d[self.box_key] = torch.stack(
                    [torch.cat((h.amin(0), h.amax(0))) for h in hulls]
                ).to(d[self.image_key].device)
            else:
                d[self.box_key] = torch.cat((hulls.amin(1), hulls.amax(1)), dim=1)
        d = self.clip(d)
        boxes = d[self.box_key]
        if len(boxes) == 0:
            return d
        lm = torch.as_tensor(d["lm"])[0]
        limits = torch.cat((boxes[:, :3].floor(), boxes[:, 3:].ceil()), dim=1)
        labels = d[self.label_key].to(lm.device)
        present = []
        for (x0, y0, z0, x1, y1, z1), label in zip(limits.int().tolist(), labels):
            present.append((lm[x0:x1, y0:y1, z0:z1] == label).any())
        positive = (boxes[:, 3:] > boxes[:, :3]).all(dim=1)
        keep = torch.stack(present) & positive.to(lm.device)
        d[self.box_key] = boxes[keep.to(boxes.device)].contiguous()
        d[self.label_key] = labels[keep].contiguous()
        if d.get("df") is not None:
            d["df"] = d["df"].iloc[keep.cpu().numpy()].reset_index(drop=True)
        return d


class RemeasureTracker(BoxTracker):
    """Resample row IDs and measure the tight extent of surviving instances."""

    @staticmethod
    def compute(lm, rows):  #AI
        """Assign sidecar row index plus one to each retained component."""
        inst = np.zeros(lm.shape[-3:], dtype=np.int32)
        for i, (roi, component) in enumerate(_row_components(lm, rows), start=1):
            inst[roi][component] = i
        return inst

    def warp(self, data, transform):  #AI
        """Leave tracking to the nearest-neighbor spatial transform on inst."""
        return data

    def translate(self, data, shift):  #AI
        """Leave tracking to the crop or pad on inst."""
        return data

    def finalize(self, data):  #AI
        """Remeasure nnDetection boxes and convert them to exclusive xyzxyz."""
        
        import nndet.compat  # noqa: F401
        from nndet.io.transforms.instances import instances_to_boxes

        d = dict(data)
        inst = d.pop("inst")
        labels = d.pop("inst_labels").to(inst.device)
        native, ids = instances_to_boxes(torch.as_tensor(inst)[0])
        boxes = native.reshape(-1, 6)[:, [0, 1, 4, 2, 3, 5]]
        boxes[:, :3] += 1
        rows = ids.long() - 1
        d[self.box_key] = boxes.contiguous()
        d[self.label_key] = labels[rows].contiguous()
        if "lm" not in d:
            lut = torch.cat((labels.new_zeros(1), labels))
            lm = lut[torch.as_tensor(inst).long()].contiguous()
            d["lm"] = MetaTensor(lm, meta=dict(inst.meta))
        if d.get("df") is not None:
            d["df"] = d["df"].iloc[rows.cpu().numpy()].reset_index(drop=True)
        return d


BOX_TRACKERS = {"corner": CornerTracker, "hull": HullTracker, "remeasure": RemeasureTracker}


class HullStatsd(MapTransform):
    """Add each bbox sidecar row's hull after detection box statistics."""

    def __init__(self):  #AI
        """Read the remapped LBD label map."""
        super().__init__(["lm"])

    def __call__(self, data):  #AI
        """Compute hulls in the same row order as the recorded tight boxes."""
        d = dict(data)
        rows = d["nbrhoods"].assign(bbox_xyzxyz=d["bbox"].int().tolist())
        d["hull_xyz"] = BOX_TRACKERS["hull"].compute(d["lm"], rows)
        return d
