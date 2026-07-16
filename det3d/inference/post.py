from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from det3d.managers.data.collate import adjust_boxes_for_pad
from det3d.inference.markups import save_inference_markups
from det3d.utils.bbox_sidecar import save_inference_sidecar
from fran.inference.base import TokenFolderLayout
from fran.transforms.inferencetransforms import MakeWritabled
from fran.transforms.spatialtransforms import _ornt_current_to_original
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToWorldCoordinated,
)
from monai.config import KeysCollection
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import to_affine_nd
from monai.transforms.io.dictionary import SaveImaged
from monai.transforms.transform import MapTransform
from utilz.fileio import maybe_makedirs
from utilz.stringz import info_from_filename, strip_extension


def bbox_slice_starts(bounding_box):
    starts = []
    for sl in bounding_box[1:]:
        starts.append(int(sl.start) if isinstance(sl, slice) else int(sl))
    return starts


def encode_bounding_box(bounding_box):
    out = []
    for sl in bounding_box:
        if isinstance(sl, slice):
            out.append([int(sl.start), int(sl.stop)])
        elif isinstance(sl, (list, tuple)):
            out.append([int(v) for v in sl])
        else:
            out.append(int(sl))
    return out


def _box_corners(box):
    x1, y1, z1, x2, y2, z2 = (float(v) for v in box)
    return torch.tensor(
        [
            [x1, y1, z1],
            [x2, y1, z1],
            [x1, y2, z1],
            [x2, y2, z1],
            [x1, y1, z2],
            [x2, y1, z2],
            [x1, y2, z2],
            [x2, y2, z2],
        ],
        dtype=torch.float64,
    )


def _apply_ornt_points(pts, ornt, spatial_shape):
    sh = torch.tensor([float(v) for v in spatial_shape], dtype=torch.float64)
    coords = pts.clone()
    axes = [int(o[0]) for o in ornt]
    coords = coords[:, axes]
    sh = sh[axes]
    for i, flip in enumerate(ornt[:, 1]):
        if int(flip) == -1:
            coords[:, i] = sh[i] - 1.0 - coords[:, i]
    return coords


def _box_apply_ornt(boxes, ornt, spatial_shape):
    boxes = torch.as_tensor(boxes, dtype=torch.float64)
    if boxes.numel() == 0:
        return boxes.reshape(0, 6)
    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)
    out = []
    for i in range(boxes.shape[0]):
        corners = _apply_ornt_points(_box_corners(boxes[i]), ornt, spatial_shape)
        lo = corners.min(dim=0).values
        hi = corners.max(dim=0).values
        out.append(torch.cat([lo, hi]))
    return torch.stack(out, 0).to(torch.float32)


def apply_affine_row_points(pts, mat):
    """#AI Apply 4x4 affine to Nx3 voxel points (row-vector convention)."""
    mat = to_affine_nd(3, mat)
    mat = np.asarray(mat, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    out = []
    for p in pts:
        ph = np.append(p, 1.0)
        out.append((ph @ mat.T)[:3])
    return np.stack(out, 0)


def apply_ornt_points_np(pts, ornt, shape):
    """#AI Flip + permute Nx3 points like _apply_ornt_points (MONAI orientation indices)."""
    sh = np.array(shape, dtype=np.float64)
    coords = np.asarray(pts, dtype=np.float64).copy()
    if coords.ndim == 1:
        coords = coords.reshape(1, 3)
    tr = np.asarray(ornt)
    axes = [int(o[0]) for o in tr]
    coords = coords[:, axes]
    sh = sh[axes]
    for i, flip in enumerate(tr[:, 1]):
        if int(flip) == -1:
            coords[:, i] = sh[i] - 1.0 - coords[:, i]
    return coords


def inverse_preprocess_points_from_ops(points, applied_operations, spatial_shape, affine):
    """#AI LIFO inverse of Spacingd+Orientationd corner points (matches InvP trace)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    running_shape = tuple(int(v) for v in spatial_shape)
    running_affine = np.asarray(affine, dtype=np.float64)
    for op in reversed(applied_operations):
        if op["class"] == "Orientation":
            orig_affine = op["extra_info"]["original_affine"]
            orig_affine = np.asarray(orig_affine, dtype=np.float64)
            src_o = nib.io_orientation(to_affine_nd(3, orig_affine))
            dst_o = nib.orientations.axcodes2ornt("RAS")
            inv_o = nib.orientations.ornt_transform(dst_o, src_o)
            pts = apply_ornt_points_np(pts, inv_o, running_shape)
            running_shape = tuple(int(v) for v in op["orig_size"])
            running_affine = orig_affine
        elif op["class"] == "SpatialResample":
            orig = np.array([float(v) for v in op["orig_size"]], dtype=np.float64)
            cur = np.array([float(v) for v in running_shape], dtype=np.float64)
            pts = pts * (orig / cur)
            running_shape = tuple(int(v) for v in op["orig_size"])
            running_affine = np.asarray(op["extra_info"]["src_affine"], dtype=np.float64)
    return pts


def box_from_corner_points(corners):
    lo = corners.min(axis=0)
    hi = corners.max(axis=0)
    return np.concatenate([lo, hi])


def inverse_preprocess_box(box, applied_operations, spatial_shape, affine):
    """#AI xyzxyz box: preproc crop voxels -> native crop voxels."""
    corners = inverse_preprocess_points_from_ops(
        _box_corners(box).numpy(), applied_operations, spatial_shape, affine
    )
    return box_from_corner_points(corners)


class PackRetinaNetPredsd(MapTransform):
    """Map MONAI RetinaNet detector outputs to pred_box/label/score."""

    def __init__(self, keys=("image",)):
        super().__init__(keys=keys)

    def __call__(self, data):
        d = dict(data)
        pred = d.pop("raw_pred")
        d["pred_box"] = pred["box"].detach()
        d["pred_label"] = pred["label"].detach()
        d["pred_score"] = pred["label_scores"].detach()
        return d


class PackRetinaUNetPredsd(MapTransform):
    """Map RetinaUNet predict_inner outputs to pred / pred_box / label / score."""

    def __init__(self, keys=("image",)):
        super().__init__(keys=keys)

    def _pack_seg(self, seg, image):
        seg = seg.detach()
        if seg.dim() == 5:
            seg = seg[0]
        if seg.shape[0] == 1:
            out = (seg[0] > 0).to(torch.uint8)
        else:
            out = seg.argmax(dim=0).to(torch.uint8)
        out = out.unsqueeze(0)
        meta = deepcopy(image.meta)
        return MetaTensor(out.contiguous(), meta=meta)

    def __call__(self, data):
        from det3d.managers.helpers.nndet_retinaunet import (
            nndet_batch_to_xyzxyz,
            nndet_pred_to_vis,
        )

        d = dict(data)
        image = d["image"][0, 0]
        if "raw_pred" in d:
            vis = nndet_pred_to_vis(d.pop("raw_pred"))
            d["pred_box"] = nndet_batch_to_xyzxyz(vis["bbox"].detach())
            d["pred_label"] = vis["label"].detach()
            d["pred_score"] = vis["label_scores"].detach()
            seg = vis["pred_seg"].detach()
            d["pred"] = self._pack_seg(seg, image)
            return d
        d["pred_box"] = d.pop("merged_boxes").detach()
        d["pred_label"] = d.pop("merged_labels").detach()
        d["pred_score"] = d.pop("merged_scores").detach()
        d["pred"] = self._pack_seg(d.pop("stitched_seg"), image)
        return d


class InvPreprocessBoxd(MapTransform):
    """Inverse Spacingd+Orientationd on xyzxyz pred_box via 8 corners + image.applied_operations."""

    def __init__(
        self,
        box_key: str = "pred_box",
        image_key: str = "image",
    ):
        super().__init__(keys=[box_key])
        self.box_key = box_key
        self.image_key = image_key

    def __call__(self, data):
        d = dict(data)
        image = d[self.image_key]
        if image.dim() == 5:
            image = image[0, 0]
        image = image.detach().cpu()
        boxes = torch.as_tensor(d[self.box_key], dtype=torch.float32).detach().cpu()
        if boxes.numel() == 0:
            d[self.box_key] = boxes.reshape(0, 6)
            return d
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)
        spatial_shape = tuple(int(v) for v in image.shape[-3:])
        affine = image.affine.numpy()
        ops = image.applied_operations
        out = []
        for i in range(boxes.shape[0]):
            out.append(
                inverse_preprocess_box(boxes[i].numpy(), ops, spatial_shape, affine)
            )
        d[self.box_key] = torch.tensor(np.stack(out, 0), dtype=torch.float32)
        return d


class Offd(MapTransform):
    """Shift pred_box from crop voxels to full-volume voxels (xyzxyz)."""

    def __init__(
        self,
        box_keys: KeysCollection,
        bbox_key: str = "bounding_box",
    ):
        super().__init__(box_keys)
        self.bbox_key = bbox_key

    def __call__(self, data):
        d = dict(data)
        offsets = bbox_slice_starts(d[self.bbox_key])
        for key in self.key_iterator(d):
            d[key] = adjust_boxes_for_pad(
                torch.as_tensor(d[key], dtype=torch.float32), offsets
            )
        return d


class BoxRd(MapTransform):
    """Apply RestoreOriginalOrientationd-equivalent transform to xyzxyz boxes."""

    def __init__(
        self,
        box_key: str = "pred_box",
        full_meta_key: str = "full_meta",
    ):
        super().__init__(keys=[box_key])
        self.box_key = box_key
        self.full_meta_key = full_meta_key

    def __call__(self, data):
        d = dict(data)
        meta = deepcopy(d[self.full_meta_key])
        ornt = _ornt_current_to_original(meta)
        spatial_shape = tuple(int(v) for v in meta["spatial_shape"])
        d[self.box_key] = _box_apply_ornt(d[self.box_key], ornt, spatial_shape)
        return d


def det_output_paths(output_dir, source_image):
    """#AI FRAN TokenFolderLayout: {run}/{proj_title}/{case}.ext"""
    layout = TokenFolderLayout(data_root_dir=Path(output_dir), extension=".nii.gz")
    nii = Path(layout.filename(source_image))
    stem = strip_extension(Path(str(source_image)).name)
    out_json = nii.parent / f"{stem}.json"
    out_mrk = nii.parent / f"{stem}.mrk.json"
    return layout, out_json, out_mrk


class SaveDetOutputd(MapTransform):
    """Write pred seg NIfTI + detection sidecar (all scores; world from pred.meta)."""

    def __init__(
        self,
        output_dir: str | Path,
        run_w: str,
        run_p: str,
        write_seg: bool = True,
        seg_key: str = "pred",
        box_key: str = "pred_box",
        label_key: str = "pred_label",
        score_key: str = "pred_score",
    ):
        super().__init__(keys=[seg_key] if write_seg else [box_key])
        self.output_dir = Path(output_dir)
        self.run_w = run_w
        self.run_p = run_p
        self.write_seg = write_seg
        self.seg_key = seg_key
        self.box_key = box_key
        self.label_key = label_key
        self.score_key = score_key

    def __call__(self, data):
        d = dict(data)
        maybe_makedirs(self.output_dir)
        source_image = d["source_image"]
        fname = Path(str(source_image))
        case_id = info_from_filename(fname.name, full_caseid=True)["case_id"]
        layout, out_json, out_mrk = det_output_paths(self.output_dir, source_image)
        meta = d[self.seg_key].meta if self.write_seg else deepcopy(d["full_meta"])
        if not self.write_seg:
            meta["affine"] = torch.as_tensor(meta["original_affine"]).clone()
        spacing = [float(v) for v in meta["spacing"]]
        affine = torch.as_tensor(meta["affine"]).cpu().tolist()
        lbd_bbox = encode_bounding_box(d["bounding_box"])

        boxes_voxel = torch.as_tensor(d[self.box_key], dtype=torch.float32)
        labels = torch.as_tensor(d[self.label_key])
        scores = torch.as_tensor(d[self.score_key])

        ref_key = self.seg_key if self.write_seg else "_world_ref"
        ref_tensor = d[self.seg_key] if self.write_seg else MetaTensor(
            torch.zeros(1, 1, 1, 1), meta=meta
        )
        world_batch = {
            ref_key: ref_tensor,
            self.box_key: boxes_voxel.clone(),
        }
        world_batch = AffineBoxToWorldCoordinated(
            box_keys=[self.box_key],
            box_ref_image_keys=ref_key,
            affine_lps_to_ras=False,
        )(world_batch)
        boxes_world = world_batch[self.box_key]

        out_json = Path(out_json)
        sidecar = save_inference_sidecar(
            out_json,
            source_image=str(source_image),
            case_id=case_id,
            lbd_bounding_box=lbd_bbox,
            localiser_run=self.run_w,
            det_run=self.run_p,
            spacing=spacing,
            affine=affine,
            boxes_voxel=boxes_voxel,
            boxes_world=boxes_world,
            labels=labels,
            scores=scores,
            boxes_pre_tfm=None,
        )
        d["sidecar_path"] = str(out_json)
        save_inference_markups(
            out_mrk,
            sidecar,
        )
        d["markups_path"] = str(out_mrk)

        if self.write_seg:
            seg_batch = MakeWritabled(keys=[self.seg_key])({self.seg_key: d[self.seg_key]})
            seg_batch = SaveImaged(
                keys=[self.seg_key],
                output_dir=self.output_dir,
                separate_folder=False,
                output_dtype=torch.uint8,
                output_postfix="",
                resample=False,
                folder_layout=layout,
            )(seg_batch)
            d[self.seg_key] = seg_batch[self.seg_key]
            d["pred_seg_nii"] = str(d[self.seg_key].meta["filename_or_obj"])
        return d
