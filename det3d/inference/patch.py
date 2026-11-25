import hashlib
import json
import subprocess
from collections import Counter
from functools import cached_property
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch
from det3d.geometry.nndet_boxes import from_nndet, offset_nndet_xyxyzz_boxes, to_nndet
from det3d.inference.transforms import LoadInferImaged
from det3d.managers.retinanet import RetinaNetManager
from det3d.managers.retinaunet import RetinaUNetManager
from det3d.utils.tensor import plain_tensor
from fran.inference.base import BaseInferer
from fran.inference.cascade import img_bbox_collated
from fran.inference.helpers import channel_first
from fran.transforms.inferencetransforms import SqueezeListofListsd
from fran.transforms.intensitytransforms import NormaliseClipd
from monai.transforms import EnsureChannelFirstd, EnsureTyped, Orientationd, Spacingd, SpatialPadd
from monai.transforms.post.dictionary import Invertd
from tqdm import tqdm
from utilz.cprint import cprint
from utilz.stringz import ast_literal_eval


class DetPatchInfererBase(BaseInferer):
    """Det patch inference on localiser crops (or stored LBD crops when stored_input)."""

    arch_name = "base"
    stored_input = False  # True: inputs come from the preprocessed store (LBD .pt)
    model_manager = None
    keys_preproc = "L,E,S,O,Norm,Dtype"

    def __init__(
        self,
        run_name,
        project_title=None,
        patch_overlap=None,
        devices=(0,),
        safe_mode=False,
        save=False,
        params=None,
        ckpt=None,
        debug=False,
        save_states=None,
        **kwargs,
    ):
        self.save_states = save_states
        cprint(f"Setting up {self.arch_name} Det patch inference", color="red", bold=True)
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
            model_manager=self.model_manager,
            **kwargs,
        )

    def create_preprocess_transforms(self):
        plan = self.params["configs"]["plan_train"]
        self.preprocess_transforms_dict = {
            "L": LoadInferImaged(keys=["image"]),
            "E": EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            "S": Spacingd(keys=["image"], pixdim=ast_literal_eval(plan["spacing"])),
            "O": Orientationd(keys=["image"], axcodes="RAS"),
            "Norm": NormaliseClipd.from_plan(
                keys=["image"], plan=plan, detection_params=self.dataset_params,
                stored_plan=plan if self.stored_input else None,
            ),
            "Dtype": EnsureTyped(keys=["image"], dtype=torch.float16),
        }
        self.bind_transforms(self.preprocess_transforms_dict)

    def set_postprocess_tfms_keys(self):
        self.postprocess_tfms_keys = ",".join(
            k for k in self.keys_postproc.split(",") if self.save or k != "Sav"
        )

    def check_plan_compatibility(self):
        pass

    def prepare_data(self, data, collate_fn=img_bbox_collated):
        super().prepare_data(data, collate_fn=collate_fn)

    def image_on_model_device(self, batch, device):
        return channel_first(plain_tensor(batch["image"]).float().to(device))


class DetPatchRetinaUNet(DetPatchInfererBase):
    save_states = None
    arch_name = "RetinaUNet"
    model_manager = RetinaUNetManager
    keys_postproc = "Pack,SqL,InvP,InvPreBox"

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
        self.bind_transforms(self.postprocess_transforms_dict)

    def setup(self):
        from det3d.inference.mirror_tta import inference_tta

        super().setup()
        self.inferer.with_coord = True
        self.num_tta = inference_tta(self.params["configs"])

    def inference_params(self):
        net = self.model.net
        import nndet.compat  # noqa: F401
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
        import nndet.compat  # noqa: F401
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
        import nndet.compat  # noqa: F401
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
        return from_nndet(b), p, l

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
        """Predict every mirror view (inference_tta); the seg is the mean over views.
        Without TTA boxes are tile-weighted into one pool; with TTA each view goes to
        its own nnDetection BoxEnsemblerSelective model slot."""
        from det3d.inference.mirror_tta import mirror_tta_step

        device_type = win_data.device.type
        with torch.autocast(device_type, enabled=device_type == "cuda"):
            views, seg = mirror_tta_step(self.model.net, win_data, self.num_tta)
        if self.num_tta > 0:
            origins = [self.tile_origin(entry) for entry in unravel_slice]
            batch = {"tile_origin": list(zip(*origins)), "data": win_data}
            for t, pred in enumerate(views):
                self._ensembler.model_current = f"model0_t{t}"
                self._ensembler.process_batch(result=pred, batch=batch)
            return seg
        tile_size = tuple(int(v) for v in win_data.shape[-3:])
        for pred in views:
            for i in range(win_data.shape[0]):
                boxes = pred["pred_boxes"][i].detach()
                scores = pred["pred_scores"][i].detach()
                labels = pred["pred_labels"][i].detach()
                origin = self.tile_origin(unravel_slice[i])
                weights = self.box_tile_weight(boxes, tile_size)
                boxes = self.offset_boxes(boxes, origin)
                self._box_acc.append(
                    {
                        "boxes": boxes,
                        "scores": scores,
                        "labels": labels,
                        "weights": weights,
                    }
                )
        return seg

    @staticmethod
    def case_properties(batch):
        """Keep native ITK geometry and both centre-coordinate RAS affines for grid replay."""
        image = batch["image"]
        meta = getattr(image, "meta", {})
        case_id = batch.get("case_id", meta.get("case_id"))
        if case_id is None and "filename_or_obj" in meta:
            case_id = Path(meta["filename_or_obj"]).name.removesuffix(".gz").removesuffix(".nii").removesuffix(".pt")
        props = dict(case_id=case_id, original_size_of_raw_data=None, itk_origin=None,
                     itk_spacing=None, itk_direction=None)
        if "original_affine" in meta and "spatial_shape" in meta:
            native = np.asarray(meta["original_affine"]).reshape(4, 4)
            itk = np.diag([-1, -1, 1, 1]) @ native
            spacing = np.linalg.norm(itk[:3, :3], axis=0)
            props.update(original_size_of_raw_data=tuple(np.asarray(meta["spatial_shape"]).ravel()[::-1]),
                         itk_origin=tuple(itk[:3, 3]), itk_spacing=tuple(spacing),
                         itk_direction=tuple((itk[:3, :3] / spacing).ravel()),
                         native_affine=native, model_affine=np.asarray(image.affine).reshape(4, 4))
        return props

    def make_tta_ensembler(self, spatial, properties=None):
        """Build one upstream model slot per view, using every effective merge parameter."""
        import nndet.compat  # noqa: F401
        from nndet.inference.ensembler.detection import BoxEnsemblerSelective

        parameters = BoxEnsemblerSelective.get_default_parameters()
        parameters.update(self.inference_params())
        if properties is None:
            properties = self.case_properties({"image": None})
        ensembler = BoxEnsemblerSelective(properties=dict(properties, shape=spatial), parameters=parameters)
        for t in range(len(self._tta_view_names())):
            ensembler.add_model(name=f"model0_t{t}", model_weight=1.0)
        return ensembler

    def _tta_view_names(self):
        from nndet.inference.transforms import get_tta_transforms

        return get_tta_transforms(self.num_tta, True)[0]

    def merge_tta_views(self, ensembler):
        """Per view: topk -> clip -> small boxes -> weighted NMS; then WBC across views
        (BoxEnsembler.process_model / process_ensemble)."""
        pred = ensembler.get_case_result(restore=False)
        return from_nndet(pred["pred_boxes"]), pred["pred_scores"], pred["pred_labels"].long()

    def save_ensembler_state(self, ensembler):
        from nndet.inference.ensembler.detection import BoxEnsembler

        properties = ensembler.properties
        case_id = properties["case_id"]
        if not case_id or "model_affine" not in properties:
            raise ValueError(f"save_states={self.save_states}, case_id={case_id!r}, geometry keys={sorted(properties)}")
        folder = Path(self.save_states)
        folder.mkdir(parents=True, exist_ok=True)
        # The selective writer truncates and reorders tied scores. Its parent saves the full slot.
        with TemporaryDirectory(dir=folder) as temp:
            BoxEnsembler.save_state(ensembler, Path(temp), case_id)
            name = f"{case_id}_{ensembler.ID}.pt"
            (Path(temp) / name).replace(folder / name)

    def run_swi(self, img, properties=None):
        self._box_acc = []
        spatial = tuple(int(v) for v in img.shape[-3:])
        if self.num_tta > 0:
            if properties is None:
                properties = self.case_properties({"image": img})
            self._ensembler = self.make_tta_ensembler(spatial, properties)
        elif self.save_states is not None:
            raise ValueError(f"save_states={self.save_states} requires mirror slots, num_tta={self.num_tta}")
        stitched_seg = self.inferer(inputs=img, network=self.swi_predictor)
        if self.num_tta > 0:
            if self.save_states is not None:
                self.save_ensembler_state(self._ensembler)
            boxes, scores, labels = self.merge_tta_views(self._ensembler)
        else:
            boxes, scores, labels = self.merge_tile_boxes(self._box_acc, spatial)
        return stitched_seg, boxes, scores, labels

    def detection_limit(self):
        return self.model.plan.get("detections_per_img")

    def predict_inner(self, batch):
        img = self.image_on_model_device(batch, next(self.model.net.parameters()).device)
        self.model.eval()
        if self.needs_swi(img) or self.num_tta > 0 or self.save_states is not None:  # TTA views meet in the tile box merge
            vol = img.unsqueeze(0)
            seg, boxes, scores, labels = self.run_swi(vol, self.case_properties(batch) if self.num_tta > 0 else None)
            limit = self.detection_limit()
            if limit is not None:
                keep = scores.argsort(descending=True)[:int(limit)]
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            batch["stitched_seg"] = seg
            batch["merged_boxes"] = boxes
            batch["merged_scores"] = scores
            batch["merged_labels"] = labels
            return batch
        batch["raw_pred"] = self.forward_tile(img)
        return batch


class NNDetInferer(DetPatchRetinaUNet):
    """Whole-scan Retina U-Net with nnDetection tile settings and replayable view slots."""

    keys_preproc = DetPatchRetinaUNet.keys_preproc + ",Pad"
    target_metric = "mAP_IoU_0.10_0.50_0.05_MaxDet_100"

    def __init__(
        self, run_name, project_title=None, devices=(0,), safe_mode=False, save=False,
        params=None, ckpt=None, debug=False, bs=8, patch_overlap=0.5, mirror_views=8,
        tile_score_thresh=0.0, tile_nms_thresh=0.6, topk_candidates=10000,
        tile_min_box=0.01, detections_per_img=100, model_topk=1000,
        model_iou=0.1, model_score_thresh=0.0, save_states=None,
    ):
        """mirror_views accepts 1 (identity), 4 (two axes), or 8 (three axes)."""
        if mirror_views not in (1, 4, 8):
            raise ValueError(f"mirror_views={mirror_views}, expected 1, 4 or 8")
        super().__init__(run_name, project_title=project_title, devices=devices, safe_mode=safe_mode,
                         save=save, params=params, ckpt=ckpt, debug=debug, bs=bs,
                         patch_overlap=patch_overlap, save_states=save_states)
        self.source_checkpoint = Path(self.ckpt).resolve()
        self.mirror_views = mirror_views
        self.tile_settings = dict(score_thresh=tile_score_thresh, nms_thresh=tile_nms_thresh,
                                  topk_candidates=topk_candidates, remove_small_boxes=tile_min_box,
                                  detections_per_img=detections_per_img)
        self.merge_settings = dict(model_topk=model_topk, model_iou=model_iou,
                                   model_score_thresh=model_score_thresh)

    def create_preprocess_transforms(self):
        super().create_preprocess_transforms()
        self.Pad = SpatialPadd(keys=["image"], spatial_size=self.plan["patch_size"])
        self.preprocess_transforms_dict["Pad"] = self.Pad

    def setup(self):
        super().setup()
        self.num_tta = self.mirror_views
        for key, value in self.tile_settings.items():
            setattr(self.model.net, key, value)

    def detection_limit(self):
        return self.model.net.detections_per_img

    def inference_params(self):
        from nndet.inference.ensembler.detection import BoxEnsemblerSelective

        params = super().inference_params()
        params.update(BoxEnsemblerSelective.get_default_parameters())
        params.update(self.merge_settings)
        return params

    @staticmethod
    def transform_box(box, centre_transform):
        """Transform all eight exclusive XYZXYZ voxel-edge corners between two grids."""
        from det3d.inference.post import _box_corners, apply_affine_row_points, box_from_corner_points

        corners = _box_corners(box).numpy() - 0.5
        points = apply_affine_row_points(corners, centre_transform) + 0.5
        return box_from_corner_points(points)

    def prepare_case(self, case_id, image_path):
        """Use the native NIfTI header and MONAI's traced, padded model grid."""
        native = nib.load(image_path)
        batch = self.preprocess_compose({"image": image_path, "case_id": case_id})
        image = batch["image"].unsqueeze(0)
        image.meta["original_affine"] = native.affine
        image.meta["spatial_shape"] = native.shape
        batch["image"] = image
        return batch

    @staticmethod
    def write_ground_truth(case_id, records, properties, label_map, gt_dir):
        """Native exclusive XYZXYZ edges to model-grid xyxyzz with minima padded by one, unclipped."""
        if properties["case_id"] != case_id:
            raise ValueError(f'{case_id}: properties.case_id={properties["case_id"]}')
        transform = np.linalg.solve(properties["model_affine"], properties["native_affine"])
        boxes, classes = [], []
        for record in records:
            label = record["label"]
            if label not in label_map:
                raise ValueError(f"{case_id}: label={label}, label_map={label_map}")
            boxes.append(NNDetInferer.transform_box(record["bbox_xyzxyz"], transform))
            classes.append(label_map[label])
        boxes = to_nndet(torch.as_tensor(np.asarray(boxes), dtype=torch.float32))
        folder = Path(gt_dir)
        folder.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=folder, delete=False) as temp:
            np.savez(temp, boxes=boxes.numpy(), classes=np.asarray(classes, dtype=int))
        Path(temp.name).replace(folder / f"{case_id}_boxes_gt.npz")

    @torch.inference_mode()
    def collect_states(self, cases, pred_dir, gt_dir, reference_dir, label_map):
        """Collect states and GT for a mapping of case IDs to image paths after setup()."""
        previous = self.save_states
        self.save_states = pred_dir
        try:
            for case_id, path in tqdm(cases.items(), unit="scan"):
                batch = self.prepare_case(case_id, path)
                records = json.loads((Path(reference_dir) / f"{case_id}.json").read_text())["boxes"]
                self.write_ground_truth(case_id, records, self.case_properties(batch), label_map, gt_dir)
                self.predict_inner(batch)
        finally:
            self.save_states = previous
        self.check_cohort(cases, pred_dir, gt_dir)

    @staticmethod
    def check_cohort(case_ids, pred_dir, gt_dir):
        from nndet.inference.ensembler.detection import BoxEnsemblerSelective

        counts = Counter(iter(case_ids))
        expected = set(counts)
        states = set(BoxEnsemblerSelective.get_case_ids(pred_dir))
        gt = {p.name.removesuffix("_boxes_gt.npz") for p in Path(gt_dir).glob("*_boxes_gt.npz")}
        duplicates = sorted(key for key, count in counts.items() if count != 1)
        if not expected or duplicates or states != expected or gt != expected:
            raise ValueError(f"cohort={len(expected)}, duplicates={duplicates}, "
                             f"states missing={sorted(expected-states)}, extra={sorted(states-expected)}, "
                             f"GT missing={sorted(expected-gt)}, extra={sorted(gt-expected)}")

    @cached_property
    def checkpoint_sha256(self):
        digest = hashlib.sha256()
        with self.source_checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def winners_path(self, mode):
        """mode accepts 'whole-scan' or 'cascade', each with a separate winners file."""
        if mode not in ("whole-scan", "cascade"):
            raise ValueError(f"mode={mode!r}, expected 'whole-scan' or 'cascade'")
        return self.source_checkpoint.parent / f"inference_plan_{self.checkpoint_sha256}_{mode}.json"

    def sweep(self, case_ids, pred_dir, gt_dir, label_map, mode="whole-scan"):
        """Score the upstream sweep on the exact cohort. mode accepts 'whole-scan' or 'cascade'."""
        from nndet.inference.ensembler.detection import BoxEnsemblerSelective
        from nndet.inference.sweeper import BoxSweeper

        path = self.winners_path(mode)
        case_ids = list(case_ids)
        self.check_cohort(case_ids, pred_dir, gt_dir)
        classes = [str(label) for label in self.plan["fg_labels"]]
        if set(label_map.values()) != set(range(len(classes))):
            raise ValueError(f"label_map={label_map}, fg_labels={self.plan['fg_labels']}")
        # BoxSweeper asks its state reader for the starting parameters as well as the grid.
        # Keep constructor overrides (e.g. model_topk) during every candidate evaluation.
        _, grid = BoxEnsemblerSelective.sweep_parameters()
        reader = SimpleNamespace(get_case_ids=BoxEnsemblerSelective.get_case_ids,
                                 from_checkpoint=BoxEnsemblerSelective.from_checkpoint,
                                 sweep_parameters=lambda: (self.inference_params(), grid))
        sweeper = BoxSweeper(classes, pred_dir, gt_dir, self.target_metric, reader)
        state = sweeper.run_postprocessing_sweep()
        score = float(sweeper._evaluate_value(state)[self.target_metric])
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
        epoch = torch.load(self.source_checkpoint, map_location="cpu", weights_only=False)["epoch"]
        record = dict(checkpoint_sha256=self.checkpoint_sha256, mode=mode, epoch=epoch,
                      det3d_commit=commit, case_ids=sorted(case_ids), case_count=len(case_ids),
                      label_map=label_map, metric=self.target_metric, score=score,
                      parameters={k: v.__name__ if callable(v) else v for k, v in state.items()})
        with NamedTemporaryFile("w", dir=path.parent, delete=False) as temp:
            json.dump(record, temp, indent=2, allow_nan=False)
            temp.write("\n")
        Path(temp.name).replace(path)
        return record

    def load_winners(self, mode="whole-scan"):
        """Apply scored winners to subsequent merges. mode accepts 'whole-scan' or 'cascade'."""
        from nndet.inference.ensembler.detection import BoxEnsemblerSelective

        path = self.winners_path(mode)
        record = json.loads(path.read_text())
        for key, expected in (("checkpoint_sha256", self.checkpoint_sha256), ("mode", mode)):
            if record[key] != expected:
                raise ValueError(f"{path}: {key}={record[key]!r}, expected {expected!r}")
        defaults, grid = BoxEnsemblerSelective.sweep_parameters()
        params = record["parameters"].copy()
        if params.keys() != defaults.keys():
            raise ValueError(f"{path}: parameters={sorted(params)}, expected {sorted(defaults)}")
        for key, value in defaults.items():
            if callable(value):
                choices = {fn.__name__: fn for fn in grid.get(key, [value])}
                if params[key] not in choices:
                    raise ValueError(f"{path}: {key}={params[key]!r}, expected {sorted(choices)}")
                params[key] = choices[params[key]]
        self.merge_settings.update(params)
        return record


class DetPatchRetinaNet(DetPatchInfererBase):
    arch_name = "RetinaNet"
    model_manager = RetinaNetManager
    keys_postproc = "Pack,SqL,InvPreBox"

    def create_postprocess_transforms(self, preprocess_transform):
        from det3d.inference.post import InvPreprocessBoxd, PackRetinaNetPredsd

        self.postprocess_transforms_dict = {
            "Pack": PackRetinaNetPredsd(),
            "SqL": SqueezeListofListsd(keys=["bounding_box"]),
            "InvPreBox": InvPreprocessBoxd(),
        }
        self.bind_transforms(self.postprocess_transforms_dict)

    def predict_inner(self, batch):
        img = self.image_on_model_device(batch, next(self.model.parameters()).device)
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
    stored_input = True
    keys_postproc = "Pack,SqL,A,Int,W,Sav"

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
            "Sav": SaveDetOutputd(
                output_dir=self.output_folder,
                run_w="",
                run_p=self.run_name,
                write_seg=True,
            ),
        }
        self.bind_transforms(self.postprocess_transforms_dict)

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

    lidc_fldr = Path("/media/UB/datasets/lidc/images")
    lbd_pt_fldr = Path(
        "/r/datasets/preprocessed/lidca/lbd/spc_080_080_150_rlb40c36831_rlb40c36831_ex000/images"
    )
    imgs_nii = [p for p in sorted(lidc_fldr.glob("*.nii.gz")) if case_id in p.name]
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
    boxes = batch["pred_box"].detach().cpu()
    print("RetinaNet n_boxes", boxes.shape[0])
    ImageBBoxViewer(img, boxes)
