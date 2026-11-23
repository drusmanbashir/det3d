from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import wandb
from fran.callback.wandb.wandb import WandbImageGridCallback, _annotate_wandb_grid_image
from utilz.stringz import info_from_filename

from det3d.geometry.nndet_boxes import from_nndet
from det3d.detection.visualize_image import (
    annotate_snippet_grid,
    draw_slice_boxes,
    draw_slice_boxes_scored,
    draw_slice_box_overlap_heatmap,
    draw_slice_seg_with_gt_boxes,
    draw_slice_seg_with_pred_boxes_scored,
    filter_detection_pred,
    overlay_panel_label,
    pick_slice_index_max_visible,
    count_boxes_on_slice,
)
from det3d.utils.tensor import sanitize_tensor_for_numpy, to_numpy

_PRED_BOX_KEY = "bbox"
_PRED_LABEL_KEY = "label"
_PRED_SCORE_KEY = "label_scores"
_PRED_SEG_KEY = "pred_seg"


def grid_shape_for_case_count(n_cases):
    n_tiles = max(1, int(np.ceil(np.sqrt(n_cases / 3))))
    grid_rows = max(1, int(np.ceil(n_cases / n_tiles)))
    return n_tiles, grid_rows


def _case_ids_from_batch(batch):
    if "keys" in batch:
        case_ids = batch["keys"]
        if isinstance(case_ids, (str, Path)):
            case_ids = [case_ids]
        return [str(c) for c in case_ids]
    fns = batch["image"].meta["filename_or_obj"]
    if isinstance(fns, (str, Path)):
        fns = [fns]
    out = []
    for fn in fns:
        name = Path(str(fn)).name
        out.append(info_from_filename(name, full_caseid=True)["case_id"])
    return out


def _panel_rgb(panel_bgr):
    panel_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(panel_rgb).permute(2, 0, 1)


def _resize_panel_tensor(panel, height, width):
    if panel.shape[1] == height and panel.shape[2] == width:
        return panel
    arr = to_numpy(panel.permute(1, 2, 0))
    arr = cv2.resize(arr, (width, height))
    return torch.from_numpy(arr).permute(2, 0, 1)


def adapt_retinaunet_pred_boxes(pred):
    """Convert nnDetection predictions to exclusive xyzxyz for ImageBBoxViewer."""
    pred[_PRED_BOX_KEY] = from_nndet(pred[_PRED_BOX_KEY])
    return pred


def _items_from_batch(batch, preds, score_min, top_k, slice_axis, retinaunet=False, adapt_nndet_boxes=True, gt_seg_key=None):
    """Grid items holding only the slice the grid draws: it is picked from the boxes, cut on the device, then copied.

    Volumes are one-slice slabs and boxes are shifted onto them, so the grid draws slice 0.
    """
    batch_size = int(batch["image"].shape[0])
    case_ids = _case_ids_from_batch(batch)
    while len(case_ids) < batch_size:
        case_ids.append(str(len(case_ids)))
    items = []
    for b in range(batch_size):
        pred_cpu = {
            k: sanitize_tensor_for_numpy(v) if isinstance(v, torch.Tensor) else v
            for k, v in preds[b].items()
            if k != _PRED_SEG_KEY
        }
        pred = filter_detection_pred(pred_cpu, score_min=score_min, top_k=top_k)
        pred_overlap = filter_detection_pred(pred_cpu, score_min=score_min, top_k=None)
        if retinaunet and adapt_nndet_boxes:
            pred = adapt_retinaunet_pred_boxes(pred)
            pred_overlap = adapt_retinaunet_pred_boxes(pred_overlap)
        gt_box = batch["bbox"][b].detach().cpu()
        vol = batch["image"][b, 0]
        s = pick_slice_index_max_visible(pred[_PRED_BOX_KEY], gt_box, vol.shape, slice_axis)
        shift = torch.zeros(6)
        shift[[slice_axis, slice_axis + 3]] = s
        pred[_PRED_BOX_KEY] = pred[_PRED_BOX_KEY] - shift
        pred_overlap[_PRED_BOX_KEY] = pred_overlap[_PRED_BOX_KEY] - shift
        item = {
            "case_id": case_ids[b],
            "vol": to_numpy(vol.narrow(slice_axis, s, 1)).astype(np.float32),
            "bbox": gt_box - shift,
            "label": batch["label"][b].detach().cpu(),
            "pred": pred,
            "pred_overlap": pred_overlap,
        }
        if retinaunet and _PRED_SEG_KEY in preds[b]:
            item["pred_seg"] = sanitize_tensor_for_numpy(preds[b][_PRED_SEG_KEY].narrow(slice_axis + 1, s, 1))
        if gt_seg_key:
            item["gt_seg"] = sanitize_tensor_for_numpy(batch[gt_seg_key][b][0].narrow(slice_axis, s, 1))
        items.append(item)
    return items


class WandbDetImageGridCallback(WandbImageGridCallback):
    def __init__(
        self,
        patch_size,
        grid_rows=6,
        imgs_per_batch=4,
        epoch_freq=5,
        slice_axis=2,
        auto_grid=True,
        n_tiles=None,
        local_folder=None,
        score_min=0.3,
        score_low_min=None,
        score_mid_min=0.5,
        score_high_min=0.8,
        tiny_side_px=4,
        pred_top_k=None,
        show_fg_heatmap=True,
    ):
        super().__init__(
            classes=1,
            patch_size=patch_size,
            grid_rows=grid_rows,
            imgs_per_batch=imgs_per_batch,
            epoch_freq=epoch_freq,
        )
        self.slice_axis = int(slice_axis)
        self.auto_grid = auto_grid
        self.n_tiles = n_tiles
        self.local_folder = Path(local_folder) if local_folder is not None else None
        if self.local_folder is not None:
            self.local_folder.mkdir(parents=True, exist_ok=True)
        self.score_min = float(score_min)
        self.score_low_min = float(score_low_min if score_low_min is not None else score_min)
        self.score_mid_min = float(score_mid_min)
        self.score_high_min = float(score_high_min)
        self.tiny_side_px = int(tiny_side_px)
        self.pred_top_k = pred_top_k
        self.show_fg_heatmap = show_fg_heatmap
        self._train_items = []
        self._val_items = []

    def reset_grid(self):
        super().reset_grid()
        self._train_items = []
        self._val_items = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if trainer.store_preds and len(self._val_items) < self.imgs_per_batch * self.grid_rows:
            self._val_items.extend(self._det_items(batch))

    def populate_grid(self, pl_module, batch):
        self._train_items.extend(self._det_items(batch))

    def _det_items(self, batch):
        items = _items_from_batch(
            batch,
            batch["pred"],
            self.score_min,
            top_k=self.pred_top_k,
            slice_axis=self.slice_axis,
        )
        return items

    def _log_grid(self, trainer, epoch, key, items) -> None:
        if len(items) == 0:
            return
        rendered_image, triplet_case_ids, tile_w, tile_h, n_tiles = self._render_grid(items)
        rendered_image = _annotate_wandb_grid_image(
            img=rendered_image,
            case_ids=triplet_case_ids,
            tile_w=tile_w,
            tile_h=tile_h,
            nrow=n_tiles,
            padding=2,
        )

        if self.local_folder is not None:
            fig_fname = self.local_folder / f"{key.split('/')[-1]}_epoch{epoch - 1}.png"
            cv2.imwrite(str(fig_fname), cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR))

        if trainer.logger is not None and hasattr(trainer.logger, "log_image"):
            caption = f"epoch {epoch} cases={len(triplet_case_ids)}"
            trainer.logger.log_image(
                key=key,
                images=[wandb.Image(rendered_image, caption=caption)],
                step=trainer.global_step,
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.store_preds:
            return
        epoch = trainer.current_epoch + 1
        self._log_grid(trainer, epoch, "images/grid_train", self._train_items)
        self._log_grid(trainer, epoch, "images/grid", self._val_items)

    def _draw_gt_panel(self, item, vol, slice_idx, gt_box, gt_label):
        if "gt_seg" in item:
            panel = draw_slice_seg_with_gt_boxes(
                vol,
                item["gt_seg"],
                slice_idx,
                gt_box,
                gt_label,
                slice_axis=self.slice_axis,
                tiny_side_px=self.tiny_side_px,
            )
        else:
            panel = draw_slice_boxes(
                vol,
                slice_idx,
                gt_box,
                gt_label,
                slice_axis=self.slice_axis,
                tiny_side_px=self.tiny_side_px,
            )
        return overlay_panel_label(panel, "gt")

    def _draw_heatmap_panel(self, item, vol, slice_idx):
        overlap = item["pred_overlap"]
        panel, max_count = draw_slice_box_overlap_heatmap(
            vol,
            slice_idx,
            overlap[_PRED_BOX_KEY],
            slice_axis=self.slice_axis,
        )
        label = f"heatmap max={max_count}" if max_count else "heatmap"
        panel = overlay_panel_label(panel, label)
        return panel

    def _draw_pred_panel(
        self, item, vol, slice_idx, pred_box, pred_label, pred_scores, panel_label="pred"
    ):
        if "pred_seg" in item:
            panel = draw_slice_seg_with_pred_boxes_scored(
                vol,
                item["pred_seg"],
                slice_idx,
                pred_box,
                pred_label,
                pred_scores,
                slice_axis=self.slice_axis,
                low_min=self.score_low_min,
                mid_min=self.score_mid_min,
                high_min=self.score_high_min,
                tiny_side_px=self.tiny_side_px,
                show_off_slice_markers=self.pred_top_k is not None,
            )
        else:
            panel = draw_slice_boxes_scored(
                vol,
                slice_idx,
                pred_box,
                pred_label,
                pred_scores,
                slice_axis=self.slice_axis,
                low_min=self.score_low_min,
                mid_min=self.score_mid_min,
                high_min=self.score_high_min,
                tiny_side_px=self.tiny_side_px,
                show_off_slice_markers=self.pred_top_k is not None,
            )
        return overlay_panel_label(panel, panel_label)

    def _render_grid(self, items):
        n_unique = len({item["case_id"] for item in items})
        if self.n_tiles is not None:
            n_tiles = self.n_tiles
            grid_rows = self.grid_rows
        elif self.auto_grid:
            n_tiles, grid_rows = grid_shape_for_case_count(n_unique)
        else:
            n_tiles = 4
            grid_rows = self.grid_rows

        panels = []
        triplet_case_ids = []
        padding = 2
        max_items = self.imgs_per_batch * 10

        selected = []
        seen_case_ids = set()
        for item in items:
            cid = item["case_id"]
            if cid in seen_case_ids:
                continue
            seen_case_ids.add(cid)
            selected.append(item)
            if len(selected) >= min(n_tiles * grid_rows, max_items):
                break

        slice_idx = 0
        for item in selected:
            vol = item["vol"]
            gt_box = item["bbox"]
            gt_label = item["label"]
            pred = item["pred"]
            pred_box = pred[_PRED_BOX_KEY]
            pred_label = pred[_PRED_LABEL_KEY]
            pred_scores = pred[_PRED_SCORE_KEY]

            n_pred = int(pred_box.shape[0])
            n_on_slice = count_boxes_on_slice(pred_box, slice_idx, self.slice_axis)
            pred_label_text = (
                f"pred {n_on_slice}/{n_pred}" if self.pred_top_k is not None else "pred"
            )

            panel_a = self._draw_gt_panel(item, vol, slice_idx, gt_box, gt_label)
            panel_b = self._draw_heatmap_panel(item, vol, slice_idx)
            panel_c = self._draw_pred_panel(
                item, vol, slice_idx, pred_box, pred_label, pred_scores, pred_label_text
            )

            triplet_case_ids.append(item["case_id"])
            panels.append(_panel_rgb(panel_a))
            panels.append(_panel_rgb(panel_b))
            panels.append(_panel_rgb(panel_c))

        tile_h = max(p.shape[1] for p in panels)
        tile_w = max(p.shape[2] for p in panels)
        panels = [_resize_panel_tensor(p, tile_h, tile_w) for p in panels]
        grid = torchvision.utils.make_grid(
            torch.stack(panels),
            nrow=n_tiles * 3,
            padding=padding,
        )
        grid = to_numpy(grid.permute(1, 2, 0))
        grid = np.clip(grid, 0, 255).astype(np.uint8)
        grid = annotate_snippet_grid(
            grid, triplet_case_ids, tile_w, tile_h, n_tiles, padding=padding
        )
        return grid, triplet_case_ids, tile_w, tile_h, n_tiles


class WandbDetImageGridTrainCallback(WandbDetImageGridCallback):
    """Run-through grid (no val dataloader): the train grid is logged at train epoch end."""

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_freq != 0:
            return
        self._log_grid(trainer, epoch, "images/grid_train", self._train_items)


class WandbRetinaUNetImageGridCallback(WandbDetImageGridCallback):
    """RetinaUNet wandb grid: gt lm+box | overlap heatmap | pred_seg+top-k boxes."""

    def __init__(
        self,
        patch_size,
        grid_rows=6,
        imgs_per_batch=4,
        epoch_freq=5,
        slice_axis=2,
        auto_grid=True,
        n_tiles=None,
        local_folder=None,
        score_min=0.3,
        score_low_min=None,
        score_mid_min=0.5,
        score_high_min=0.8,
        tiny_side_px=4,
        pred_top_k=5,
        show_pred_seg=True,
        show_fg_heatmap=True,
        show_gt_seg=True,
        gt_seg_key="lm",
        adapt_nndet_boxes=True,
    ):
        super().__init__(
            patch_size=patch_size,
            grid_rows=grid_rows,
            imgs_per_batch=imgs_per_batch,
            epoch_freq=epoch_freq,
            slice_axis=slice_axis,
            auto_grid=auto_grid,
            n_tiles=n_tiles,
            local_folder=local_folder,
            score_min=score_min,
            score_low_min=score_low_min,
            score_mid_min=score_mid_min,
            score_high_min=score_high_min,
            tiny_side_px=tiny_side_px,
            pred_top_k=pred_top_k,
            show_fg_heatmap=show_fg_heatmap,
        )
        self.show_pred_seg = show_pred_seg
        self.show_gt_seg = show_gt_seg
        self.gt_seg_key = gt_seg_key
        self.adapt_nndet_boxes = adapt_nndet_boxes

    def _det_items(self, batch):
        gt_seg_key = self.gt_seg_key if self.show_gt_seg else None
        items = _items_from_batch(
            batch,
            batch["pred"],
            self.score_min,
            top_k=self.pred_top_k,
            slice_axis=self.slice_axis,
            retinaunet=True,
            adapt_nndet_boxes=self.adapt_nndet_boxes,
            gt_seg_key=gt_seg_key,
        )
        return items


class WandbRetinaUNetImageGridTrainCallback(WandbDetImageGridTrainCallback):
    """RetinaUNet train-path wandb grid."""

    def __init__(
        self,
        patch_size,
        grid_rows=6,
        imgs_per_batch=4,
        epoch_freq=5,
        slice_axis=2,
        auto_grid=True,
        n_tiles=None,
        local_folder=None,
        score_min=0.3,
        score_low_min=None,
        score_mid_min=0.5,
        score_high_min=0.8,
        tiny_side_px=4,
        pred_top_k=5,
        show_pred_seg=True,
        show_fg_heatmap=True,
        show_gt_seg=True,
        gt_seg_key="lm",
        adapt_nndet_boxes=True,
    ):
        super().__init__(
            patch_size=patch_size,
            grid_rows=grid_rows,
            imgs_per_batch=imgs_per_batch,
            epoch_freq=epoch_freq,
            slice_axis=slice_axis,
            auto_grid=auto_grid,
            n_tiles=n_tiles,
            local_folder=local_folder,
            score_min=score_min,
            score_low_min=score_low_min,
            score_mid_min=score_mid_min,
            score_high_min=score_high_min,
            tiny_side_px=tiny_side_px,
            pred_top_k=pred_top_k,
            show_fg_heatmap=show_fg_heatmap,
        )
        self.show_pred_seg = show_pred_seg
        self.show_gt_seg = show_gt_seg
        self.gt_seg_key = gt_seg_key
        self.adapt_nndet_boxes = adapt_nndet_boxes

    def _det_items(self, batch):
        gt_seg_key = self.gt_seg_key if self.show_gt_seg else None
        items = _items_from_batch(
            batch,
            batch["pred"],
            self.score_min,
            top_k=self.pred_top_k,
            slice_axis=self.slice_axis,
            retinaunet=True,
            adapt_nndet_boxes=self.adapt_nndet_boxes,
            gt_seg_key=gt_seg_key,
        )
        return items
