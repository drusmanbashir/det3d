from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import wandb
from fran.callback.wandb.wandb import WandbImageGridCallback, _annotate_wandb_grid_image
from utilz.stringz import info_from_filename

from det3d.detection.nndet_train import nndet_batch_to_xyzxyz
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
    pick_slice_index,
    count_boxes_on_slice,
)
from det3d.utils.tensor import to_numpy

_PRED_BOX_KEY = "bbox"
_PRED_LABEL_KEY = "label"
_PRED_SCORE_KEY = "label_scores"
_PRED_SEG_KEY = "pred_seg"


def grid_shape_for_case_count(n_cases):
    n_tiles = max(1, int(np.ceil(np.sqrt(n_cases / 3))))
    grid_rows = max(1, int(np.ceil(n_cases / n_tiles)))
    return n_tiles, grid_rows


def _case_ids_from_batch(batch):
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
    arr = panel.permute(1, 2, 0).numpy()
    arr = cv2.resize(arr, (width, height))
    return torch.from_numpy(arr).permute(2, 0, 1)


def adapt_retinaunet_pred_boxes(pred):
    """nnDet xyxyzz -> xyzxyz layout for ImageBBoxViewer (no +/-1)."""
    pred[_PRED_BOX_KEY] = nndet_batch_to_xyzxyz(pred[_PRED_BOX_KEY])
    return pred


def _items_from_batch(batch, preds, score_min, top_k, retinaunet=False, adapt_nndet_boxes=True, gt_seg_key=None):
    batch_size = int(batch["image"].shape[0])
    case_ids = _case_ids_from_batch(batch)
    while len(case_ids) < batch_size:
        case_ids.append(str(len(case_ids)))
    items = []
    for b in range(batch_size):
        pred_cpu = {k: v.detach().cpu() for k, v in preds[b].items()}
        pred = filter_detection_pred(pred_cpu, score_min=score_min, top_k=top_k)
        if retinaunet and adapt_nndet_boxes:
            pred = adapt_retinaunet_pred_boxes(pred)
        pred_overlap = filter_detection_pred(pred_cpu, score_min=score_min, top_k=None)
        if retinaunet and adapt_nndet_boxes:
            pred_overlap = adapt_retinaunet_pred_boxes(pred_overlap)
        item = {
            "case_id": case_ids[b],
            "vol": to_numpy(batch["image"][b, 0]).astype(np.float32),
            "bbox": batch["bbox"][b].detach().cpu(),
            "label": batch["label"][b].detach().cpu(),
            "pred": pred,
            "pred_overlap": pred_overlap,
        }
        if retinaunet:
            item["pred_seg"] = pred[_PRED_SEG_KEY]
        if gt_seg_key:
            gt_seg = batch[gt_seg_key]
            if isinstance(gt_seg, list):
                item["gt_seg"] = gt_seg[b].detach().cpu()
            else:
                item["gt_seg"] = gt_seg[b].detach().cpu()
                if item["gt_seg"].ndim == 4:
                    item["gt_seg"] = item["gt_seg"][0]
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
        self._cached_items = []
        self._grid_collect_target = 0

    def reset_grid(self):
        super().reset_grid()

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_freq == 0:
            self._cached_items = []
        super().on_train_epoch_start(trainer, pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        super().on_validation_epoch_start(trainer, pl_module)
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_freq == 0:
            self._grid_collect_target = self.imgs_per_batch * self.grid_rows
        else:
            self._grid_collect_target = 0

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_freq != 0:
            return
        if len(self._cached_items) >= self._grid_collect_target:
            return
        self.populate_grid_val(pl_module, batch)

    def populate_grid(self, pl_module, batch):
        return

    def populate_grid_val(self, pl_module, batch):
        self._append_det_batch(batch)

    def _append_det_batch(self, batch):
        self._cached_items.extend(
            _items_from_batch(
                batch,
                batch["pred"],
                self.score_min,
                top_k=self.pred_top_k,
            )
        )

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_freq != 0 or len(self._cached_items) == 0:
            return

        rendered_image, triplet_case_ids, tile_w, tile_h, n_tiles = self._render_grid(
            self._cached_items
        )
        padding = 2
        rendered_image = _annotate_wandb_grid_image(
            img=rendered_image,
            case_ids=triplet_case_ids,
            tile_w=tile_w,
            tile_h=tile_h,
            nrow=n_tiles,
            padding=padding,
            val_start_idx=self.val_start_idx,
        )

        if self.local_folder is not None:
            fig_fname = self.local_folder / f"grid_epoch{epoch - 1}.png"
            cv2.imwrite(str(fig_fname), cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR))

        if trainer.logger is not None:
            caption = f"epoch {epoch} cases={len(triplet_case_ids)}"
            img = wandb.Image(rendered_image, caption=caption)
            trainer.logger.experiment.log(
                {"images/grid": img},
                step=trainer.global_step,
            )

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
        rng = np.random.default_rng()
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

        order = rng.permutation(len(items))
        selected = []
        seen_case_ids = set()
        for idx in order:
            idx = int(idx)
            cid = items[idx]["case_id"]
            if cid in seen_case_ids:
                continue
            seen_case_ids.add(cid)
            selected.append(items[idx])
            if len(selected) >= min(n_tiles * grid_rows, max_items):
                break

        for item in selected:
            vol = item["vol"]
            gt_box = item["bbox"]
            gt_label = item["label"]
            pred = item["pred"]
            pred_box = pred[_PRED_BOX_KEY]
            pred_label = pred[_PRED_LABEL_KEY]
            pred_scores = pred[_PRED_SCORE_KEY]

            if gt_box.shape[0] or pred_box.shape[0]:
                slice_idx = pick_slice_index_max_visible(
                    pred_box, gt_box, vol.shape, self.slice_axis, rng
                )
            else:
                slice_idx = pick_slice_index(
                    np.zeros((0, 6)), vol.shape, self.slice_axis, rng
                )

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
        grid = grid.permute(1, 2, 0).cpu().numpy()
        grid = np.clip(grid, 0, 255).astype(np.uint8)
        grid = annotate_snippet_grid(
            grid, triplet_case_ids, tile_w, tile_h, n_tiles, padding=padding
        )
        return grid, triplet_case_ids, tile_w, tile_h, n_tiles


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

    def _append_det_batch(self, batch):
        gt_seg_key = self.gt_seg_key if self.show_gt_seg else None
        self._cached_items.extend(
            _items_from_batch(
                batch,
                batch["pred"],
                self.score_min,
                top_k=self.pred_top_k,
                retinaunet=True,
                adapt_nndet_boxes=self.adapt_nndet_boxes,
                gt_seg_key=gt_seg_key,
            )
        )
