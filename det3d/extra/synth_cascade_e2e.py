"""E2E cascade QA on synthetic lidc_0008 cuboid: GT adapters + staged probes."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import zoom
from utilz.fileio import load_json

from det3d.detection.nndet_train import (
    nndet_batch_to_xyzxyz,
    xyzxyz_exclusive_batch_to_nndet,
)
from det3d.inference.cascade import DetCascadeInfererRetinaUNet
from det3d.inference.markups import roi_center_size_from_voxel_box, voxel_box_to_world_xyzxyz
from det3d.inference.patch import DetPatchInfererRetinaUNet


def lbd_window_to_fran_bbox(lbd_window_xyzxyz):
    """#AI manifest lbd_window xyzxyz exclusive -> fran apply_bboxes slice tuple."""
    x0, y0, z0, x1, y1, z1 = [int(v) for v in lbd_window_xyzxyz]
    return [slice(0, 100), slice(x0, x1), slice(y0, y1), slice(z0, z1)]


def scale_box_xyzxyz(box, native_shape, spaced_shape):
    """#AI Scale xyzxyz exclusive box from native x,y,z crop to spaced x,y,z crop."""
    box = [float(v) for v in box]
    sx = spaced_shape[0] / native_shape[0]
    sy = spaced_shape[1] / native_shape[1]
    sz = spaced_shape[2] / native_shape[2]
    x0, y0, z0, x1, y1, z1 = box
    out = [x0 * sx, y0 * sy, z0 * sz, x1 * sx, y1 * sy, z1 * sz]
    return out


def box_tensor_list(box):
    """#AI Flatten pred_box tensor to python list."""
    t = torch.as_tensor(box).detach().cpu().reshape(-1)
    return t.tolist()


def assert_box_close(name, got, want, tol):
    """#AI Assert 6-float box corners within tolerance."""
    got = np.asarray(got, dtype=float).reshape(-1)
    want = np.asarray(want, dtype=float).reshape(-1)
    if got.size == 0:
        raise AssertionError(f"{name}: empty box, want {want.tolist()}")
    diff = np.abs(got - want)
    if diff.max() > tol:
        raise AssertionError(f"{name}: max corner diff {diff.max():.4f} > {tol}: got {got.tolist()} want {want.tolist()}")


@dataclass
class ExpectedAnchors:
    """#AI Analytical bbox anchors from synth MANIFEST."""

    lbd_window: list
    fg_lbd_native: list
    fg_full_voxel: list
    fg_full_world: list
    native_crop_shape: tuple
    fg_spaced: list | None = None

    @classmethod
    def from_manifest(cls, manifest):
        full_img = manifest["outputs"]["full_image"]
        lbd = manifest["lbd_window_xyzxyz"]
        fg_lbd = manifest["fg_cuboid_voxel_xyzxyz_lbd"]
        fg_full = manifest["fg_cuboid_voxel_xyzxyz_full"]
        lx, ly, lz, _, _, _ = lbd
        native = (lbd[3] - lx, lbd[4] - ly, lbd[5] - lz)
        world = voxel_box_to_world_xyzxyz(full_img, fg_full)
        return cls(
            lbd_window=list(lbd),
            fg_lbd_native=list(fg_lbd),
            fg_full_voxel=list(fg_full),
            fg_full_world=world,
            native_crop_shape=native,
        )

    def set_fg_spaced(self, spaced_shape):
        self.fg_spaced = scale_box_xyzxyz(
            self.fg_lbd_native, self.native_crop_shape, spaced_shape
        )


class StageProbe:
    """#AI Per-transform checks; fail fast with stage name."""

    def __init__(self, anchors: ExpectedAnchors, full_image: str):
        self.anchors = anchors
        self.full_image = full_image
        self.log = []

    def _record(self, stage, batch):
        box = batch.get("pred_box")
        n = 0 if box is None else int(torch.as_tensor(box).numel() // 6)
        img_shape = tuple(int(v) for v in batch["image"].shape[-3:])
        crop_shape = batch.get("crop_spatial_shape")
        line = f"{stage}: pred_box={box_tensor_list(box) if n else '[]'} image={img_shape} crop={crop_shape}"
        self.log.append(line)
        print(line)

    def check_patch(self, stage, batch):
        self._record(stage, batch)
        if stage == "predict_inner":
            assert_box_close("P0", box_tensor_list(batch["raw_pred"]["pred_boxes"][0]), self._nndet_from_spaced(), tol=1.5)
        elif stage == "PackRetinaUNetPredsd":
            assert_box_close("P1", box_tensor_list(batch["pred_box"]), self._nndet_from_spaced(), tol=1.5)

    def check_decollate(self, item):
        self._record("decollate", item)
        native = self.anchors.native_crop_shape
        crop = tuple(int(v) for v in item["crop_spatial_shape"])
        if crop != native:
            raise AssertionError(f"P2b: crop_spatial_shape {crop} != native {native}")
        spaced = tuple(int(v) for v in item["image"].shape[-3:])
        if self.anchors.fg_spaced is None:
            self.anchors.set_fg_spaced(spaced)
        assert_box_close("P2b-box", box_tensor_list(item["pred_box"]), self._nndet_from_spaced(), tol=1.5)

    def check_cascade(self, stage, batch):
        self._record(stage, batch)
        if stage == "NndetBoxToXyzxyzd":
            target = xyzxyz_exclusive_batch_to_nndet(
                torch.tensor([self.anchors.fg_full_voxel], dtype=torch.float32)
            )
            want = nndet_batch_to_xyzxyz(target)[0].tolist()
            assert_box_close("Cfmt", box_tensor_list(batch["pred_box"]), want, tol=1.5)
        elif stage == "PreservePreTfmBoxd":
            assert_box_close("C0", box_tensor_list(batch["pred_box_pre_tfm"]), box_tensor_list(batch["pred_box"]), tol=0.01)
        elif stage == "PredBoxToNativeCropViaPointsd":
            target = xyzxyz_exclusive_batch_to_nndet(
                torch.tensor([self.anchors.fg_lbd_native], dtype=torch.float32)
            )
            assert_box_close("C1", box_tensor_list(batch["pred_box"]), target[0].tolist(), tol=2.0)
        elif stage == "ClipBoxToImaged":
            box = torch.as_tensor(batch["pred_box"]).reshape(-1, 6)
            if box.numel() == 0:
                raise AssertionError("C2: Clip removed all boxes")
            img = batch["image"].shape[-3:]
            for row in box:
                x0, y0, z0, x1, y1, z1 = [float(v) for v in row]
                if x0 < -0.5 or y0 < -0.5 or z0 < -0.5:
                    raise AssertionError(f"C2b: box min below zero: {row.tolist()}")
                if x1 > img[0] + 0.5 or y1 > img[1] + 0.5 or z1 > img[2] + 0.5:
                    raise AssertionError(f"C2b: box exceeds native crop {img}: {row.tolist()}")
        elif stage == "OffsetBoxByBBoxd":
            target = xyzxyz_exclusive_batch_to_nndet(
                torch.tensor([self.anchors.fg_full_voxel], dtype=torch.float32)
            )
            assert_box_close("C3", box_tensor_list(batch["pred_box"]), target[0].tolist(), tol=2.0)
        elif stage == "CopyBoxKeyd":
            if "pred_box_voxel" in batch and batch["pred_box_voxel"] is not batch.get("_vox_checked"):
                assert_box_close("C4", box_tensor_list(batch["pred_box_voxel"]), box_tensor_list(batch["pred_box"]), tol=0.01)
                batch["_vox_checked"] = batch["pred_box_voxel"]
            if "pred_box_world" in batch:
                assert_box_close("C5", box_tensor_list(batch["pred_box_world"]), self.anchors.fg_full_world, tol=2.0)
        elif stage == "SaveInferenceSidecard":
            sidecar = load_json(batch["sidecar_path"])
            if len(sidecar["predictions"]) < 1:
                raise AssertionError("C6: sidecar predictions empty")
        elif stage == "SaveInferenceMarkupsd":
            from utilz.stringz import strip_extension

            mrk_fn = Path(batch["markups_path"])
            ref_center, _ref_size = roi_center_size_from_voxel_box(
                self.full_image, self.anchors.fg_full_voxel
            )
            got_center, _got_size = read_mrk_center_size(mrk_fn)
            diff = np.linalg.norm(np.array(got_center) - np.array(ref_center))
            if diff > 2.0:
                raise AssertionError(f"C6b: mrk center diff {diff:.3f} mm > 2.0")

    def _nndet_from_spaced(self):
        boxes = xyzxyz_exclusive_batch_to_nndet(torch.tensor([self.anchors.fg_spaced], dtype=torch.float32))
        return box_tensor_list(boxes)


def read_mrk_center_size(mrk_fn):
    """#AI Read first ROI markup center and size."""
    payload = load_json(mrk_fn)
    roi = payload["markups"][0]
    return roi["center"], roi["size"]


class SynthGtPatchInfererRetinaUNet(DetPatchInfererRetinaUNet):
    """Patch inferer stub: manifest box + LM seg, no checkpoint."""

    def __init__(self, *args, manifest=None, anchors=None, probe=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifest = manifest
        self.anchors = anchors
        self.probe = probe

    def prepare_model(self):
        return

    def setup(self):
        self.create_and_set_preprocess_transforms()

    def prepare_data(self, data, collate_fn=None):
        from monai.data import DataLoader, Dataset

        nw, bs = 0, 1
        self.ds = Dataset(data=data, transform=self.preprocess_compose)
        self.pred_dl = DataLoader(self.ds, num_workers=nw, batch_size=bs, collate_fn=collate_fn)

    def predict(self):
        with torch.inference_mode():
            for batch in self.pred_dl:
                batch = self.predict_inner(batch)
                if self.probe is not None:
                    self.probe.check_patch("predict_inner", batch)
                yield batch

    def postprocess(self, batch):
        if isinstance(batch, list):
            batch = batch[0]
        for tfm in self.postprocess_transforms:
            batch = tfm(batch)
            if self.probe is not None:
                self.probe.check_patch(tfm.__class__.__name__, batch)
        return batch

    def _bbox_slices(self, bbox):
        if isinstance(bbox[0], list):
            bb = bbox[0]
        elif isinstance(bbox[0], slice):
            bb = bbox
        else:
            bb = bbox[0]
        return bb[1], bb[2], bb[3]

    def _spaced_lm_crop(self, batch, lm_path):
        xsl, ysl, zsl = self._bbox_slices(batch["bounding_box"])
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(lm_path)))
        crop = np.transpose(arr[zsl, ysl, xsl], (2, 1, 0))
        native_shape = self.anchors.native_crop_shape
        spaced_shape = tuple(int(v) for v in batch["image"].shape[-3:])
        factors = [spaced_shape[i] / native_shape[i] for i in range(3)]
        spaced = zoom(crop.astype(np.float32), factors, order=0)
        return spaced

    def predict_inner(self, batch):
        spaced_shape = tuple(int(v) for v in batch["image"].shape[-3:])
        self.anchors.set_fg_spaced(spaced_shape)
        lm_path = self.manifest["outputs"]["full_lm"]
        spaced = self._spaced_lm_crop(batch, lm_path)
        device = batch["image"].device
        boxes = xyzxyz_exclusive_batch_to_nndet(
            torch.tensor([self.anchors.fg_spaced], dtype=torch.float32)
        ).to(device)
        seg = torch.from_numpy(spaced).float().unsqueeze(0).unsqueeze(0).to(device)
        batch["raw_pred"] = {
            "pred_boxes": [boxes],
            "pred_labels": [torch.tensor([0], dtype=torch.long, device=device)],
            "pred_scores": [torch.tensor([1.0], dtype=torch.float32, device=device)],
            "pred_seg": seg,
        }
        return batch


class SynthGtCascadeInfererRetinaUNet(DetCascadeInfererRetinaUNet):
    """Cascade stub: manifest LBD window + GT patch inferer, real cascade postprocess."""

    def __init__(self, manifest_path, output_dir, probe=None, anchors=None, **kwargs):
        self.manifest = load_json(manifest_path)
        self.anchors = anchors if anchors is not None else ExpectedAnchors.from_manifest(self.manifest)
        self.probe = probe
        self._output_dir = Path(output_dir)
        super().__init__(**kwargs)

    @property
    def output_folder(self):
        return self._output_dir

    def setup_patch_inferer(self):
        return SynthGtPatchInfererRetinaUNet(
            run_name=self.run_p,
            project_title=self.project_title,
            devices=self.devices,
            patch_overlap=self.patch_overlap,
            safe_mode=self.safe_mode,
            params=self.params,
            debug=self.debug,
            save=False,
            keys_preproc="E,S,Norm,Dtype",
            keys_postproc="Pack,SqL",
            manifest=self.manifest,
            anchors=self.anchors,
            probe=self.probe,
        )

    def extract_fg_bboxes(self, data, overwrite=None):
        bb = lbd_window_to_fran_bbox(self.manifest["lbd_window_xyzxyz"])
        return [bb] * len(data)

    def postprocess(self, preds):
        outputs = []
        for item in preds:
            outputs.append(self._postprocess_probed(item))
        return outputs

    def _postprocess_probed(self, batch):
        bbox = batch["bounding_box"]
        if isinstance(bbox[0], list):
            batch["bounding_box"] = bbox[0]
        for tfm in self.postprocess_transforms:
            batch = tfm(batch)
            if self.probe is not None:
                self.probe.check_cascade(tfm.__class__.__name__, batch)
        return batch


def run_e2e(manifest_path, output_dir, devices=(0,)):
    """#AI Full probed E2E run; returns final batch."""
    from fran.utils.common import COMMON_PATHS
    from utilz.fileio import load_yaml

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(manifest_path)
    anchors = ExpectedAnchors.from_manifest(manifest)
    probe = StageProbe(anchors, manifest["outputs"]["full_image"])

    def default_run_w():
        fn = Path(COMMON_PATHS["cold_storage_folder"]) / "conf" / "best_runs.yaml"
        return load_yaml(fn)["totalseg"]["whole"]["runs"][0]

    En = SynthGtCascadeInfererRetinaUNet(
        manifest_path=manifest_path,
        output_dir=output_dir,
        probe=probe,
        anchors=anchors,
        run_w=default_run_w(),
        run_p="LIDCA-GYRO",
        project_title="lidca",
        devices=list(devices),
        localiser_labels=[6],
        safe_mode=True,
        save=True,
        debug=False,
    )
    En.setup()
    data = En.load_images([manifest["outputs"]["full_image"]])
    En.bboxes = En.extract_fg_bboxes(data)
    image_paths = [manifest["outputs"]["full_image"]]
    data = En.load_images(image_paths)
    data = En.apply_bboxes(data, En.bboxes)
    full_metas = [dat["full_meta"] for dat in data]
    En.create_and_set_postprocess_transforms()
    pred_patches = En.patch_prediction(data)
    decollated = En.decollate_patches(pred_patches, En.bboxes, full_metas)
    if probe is not None:
        probe.check_decollate(decollated[0])
    output = En.postprocess(decollated)
    return output[0], probe


# %%
# SECTION:--- run e2e ---
if __name__ == "__main__":
    manifest_path = Path("/s/agent_rw/tmp/lidc_0008_synth_bbox_qa/MANIFEST.json")
    output_dir = Path("/s/agent_rw/tmp/lidc_0008_synth_bbox_qa/e2e")
    out, probe = run_e2e(manifest_path, output_dir)
    print("ALL PROBES PASSED")
    print(f"  pred_box_voxel: {box_tensor_list(out['pred_box_voxel'])}")
