from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision
from fran.callback.case_recorder import CaseIDRecorder, CaseIDRecorderSnapshot
from fran.utils.misc import freq_epoch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from utilz.cprint import cprint
from utilz.stringz import info_from_filename

from det3d.detection.visualize_image import (
    annotate_snippet_grid,
    draw_slice_boxes,
    overlay_grid_stage_banner,
    overlay_panel_label,
    pick_slice_index,
)


def _volume_numpy(image_tensor, batch_idx):
    vol = image_tensor[batch_idx, 0]
    if isinstance(vol, torch.Tensor):
        vol = vol.detach().cpu().numpy()
    return np.asarray(vol, dtype=np.float32)


def _panel_rgb(panel_bgr):
    panel_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(panel_rgb).permute(2, 0, 1)


def _case_ids_from_batch(batch):
    image = batch["image"]
    if not hasattr(image, "meta"):
        return []
    meta = image.meta
    fns = meta["filename_or_obj"] if "filename_or_obj" in meta else meta["src_filename"]
    if isinstance(fns, (str, Path)):
        fns = [fns]
    out = []
    for fn in fns:
        name = Path(str(fn)).name
        out.append(info_from_filename(name, full_caseid=True)["case_id"])
    return out


def _empty_snippet_cache():
    return {"train": [], "valid": []}


def grid_shape_for_case_count(n_cases):  # AI
    n_tiles = max(1, int(np.ceil(np.sqrt(n_cases / 3))))
    grid_rows = max(1, int(np.ceil(n_cases / n_tiles)))
    return n_tiles, grid_rows


def _flatten_cached_batches(cached_batches):
    items = []
    for cache in cached_batches:
        batch_size = int(cache["image"].shape[0])
        case_ids = _case_ids_from_batch(cache)
        if len(case_ids) < batch_size:
            case_ids = case_ids + [str(i) for i in range(len(case_ids), batch_size)]
        for b in range(batch_size):
            vol = cache["image"][b, 0]
            if isinstance(vol, torch.Tensor):
                vol = vol.detach().cpu().numpy()
            items.append(
                {
                    "case_id": case_ids[b],
                    "vol": np.asarray(vol, dtype=np.float32),
                    "bbox": cache["bbox"][b],
                    "label": cache["label"][b],
                    "pred": cache["preds"][b],
                }
            )
    return items


def _resize_panel_tensor(panel, height, width):
    if panel.shape[1] == height and panel.shape[2] == width:
        return panel
    arr = panel.permute(1, 2, 0).numpy()
    arr = cv2.resize(arr, (width, height))
    return torch.from_numpy(arr).permute(2, 0, 1)


class CaseIDRecorderDet(CaseIDRecorder):
    def __init__(
        self,
        freq=5,
        monitor_dl="valid",
        local_folder="/tmp",
        dpi=300,
        grid_rows=None,
        n_tiles=None,
        auto_grid=True,
        slice_axis=2,
        max_cached_batches=8,
    ):
        super().__init__(
            vip_label=1,
            freq=freq,
            monitor_dl=monitor_dl,
            local_folder=local_folder,
            dpi=dpi,
        )
        self.auto_grid = bool(auto_grid)
        self.grid_rows = None if grid_rows is None else int(grid_rows)
        self.n_tiles = None if n_tiles is None else int(n_tiles)
        self.slice_axis = int(slice_axis)
        self.max_cached_batches = int(max_cached_batches)
        self._snippet_cache = _empty_snippet_cache()
        self.dfs = {}

    def on_fit_start(self, trainer, pl_module):
        self._snippet_cache = _empty_snippet_cache()
        self.dfs = {}

    def on_train_epoch_start(self, trainer, pl_module):
        self._snippet_cache["train"] = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self._snippet_cache["valid"] = []

    def reset(self):
        self._snippet_cache = _empty_snippet_cache()
        self.dfs = {}

    def _cache_batch(self, pl_module, batch, stage):
        pl_module.detector.eval()
        with torch.no_grad():
            preds = pl_module.val_forward(pl_module.val_inputs(batch))
        bbox_cpu = []
        label_cpu = []
        for box, label in zip(batch["bbox"], batch["label"]):
            if isinstance(box, torch.Tensor):
                bbox_cpu.append(box.detach().cpu())
            else:
                bbox_cpu.append(torch.as_tensor(box))
            if isinstance(label, torch.Tensor):
                label_cpu.append(label.detach().cpu())
            else:
                label_cpu.append(torch.as_tensor(label))
        self._snippet_cache[stage].append(
            {
                "image": batch["image"].detach().cpu(),
                "bbox": bbox_cpu,
                "label": label_cpu,
                "preds": [{k: v.detach().cpu() for k, v in p.items()} for p in preds],
            }
        )

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._should_collect_batch(trainer, "train"):
            self._cache_batch(pl_module, batch, "train")

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self._should_collect_batch(trainer, "valid"):
            self._cache_batch(pl_module, batch, "valid")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.monitor_dl in {"valid", "both"}:
            epoch = trainer.current_epoch
            if freq_epoch(epoch, self.freq):
                self._store_snippets(trainer, "valid", epoch)
                self._snippet_cache["valid"] = []

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.monitor_dl in {"train", "both"}:
            epoch = trainer.current_epoch
            if freq_epoch(epoch, self.freq):
                self._store_snippets(trainer, "train", epoch)
                self._snippet_cache["train"] = []

    def store_results(self, trainer):
        epoch = trainer.current_epoch
        if self.monitor_dl in {"valid", "both"}:
            self._store_snippets(trainer, "valid", epoch)
            self._snippet_cache["valid"] = []
        if self.monitor_dl in {"train", "both"}:
            self._store_snippets(trainer, "train", epoch)
            self._snippet_cache["train"] = []

    def _pred_box_label_keys(self, pred_item):
        box_key = None
        label_key = None
        for key in pred_item:
            val = pred_item[key]
            if not isinstance(val, torch.Tensor):
                continue
            if val.ndim == 2 and val.shape[-1] == 6:
                box_key = key
            elif val.ndim == 1 and val.dtype in {torch.long, torch.int, torch.int64}:
                label_key = key
        return box_key, label_key

    def snippets(  # AI
        self,
        batch,
        preds,
        n_tiles=None,
        grid_rows=None,
        slice_axis=None,
        rng=None,
        stage=None,
        epoch=None,
    ):
        cache = {
            "image": batch["image"],
            "bbox": batch["bbox"],
            "label": batch["label"],
            "preds": preds,
        }
        items = _flatten_cached_batches([cache])
        return self._snippets_from_items(
            items,
            n_tiles=n_tiles,
            grid_rows=grid_rows,
            slice_axis=slice_axis,
            rng=rng,
            stage=stage,
        )

    def _snippets_from_items(
        self,
        items,
        n_tiles=None,
        grid_rows=None,
        slice_axis=None,
        rng=None,
        stage=None,
    ):
        slice_axis = self.slice_axis if slice_axis is None else int(slice_axis)
        rng = np.random.default_rng() if rng is None else rng
        box_key, label_key = self._pred_box_label_keys(items[0]["pred"])

        unique_case_ids = {item["case_id"] for item in items}
        n_unique = len(unique_case_ids)
        if n_tiles is not None and grid_rows is not None:
            n_tiles = int(n_tiles)
            grid_rows = int(grid_rows)
        elif self.auto_grid:
            n_tiles, grid_rows = grid_shape_for_case_count(n_unique)
        else:
            n_tiles = int(self.n_tiles if self.n_tiles is not None else 4)
            grid_rows = int(self.grid_rows if self.grid_rows is not None else 6)

        panels = []
        triplet_case_ids = []
        n_slots = n_tiles * grid_rows
        padding = 2

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
            if len(selected) >= n_slots:
                break

        for item in selected:
            vol = item["vol"]
            gt_box = item["bbox"]
            gt_label = item["label"]
            pred_box = item["pred"][box_key]
            pred_label = item["pred"][label_key]

            if gt_box.numel() > 0:
                slice_idx = pick_slice_index(gt_box, vol.shape, slice_axis, rng)
            elif pred_box.numel() > 0:
                slice_idx = pick_slice_index(pred_box, vol.shape, slice_axis, rng)
            else:
                slice_idx = pick_slice_index(
                    np.zeros((0, 6)), vol.shape, slice_axis, rng
                )

            panel_a = overlay_panel_label(
                draw_slice_boxes(
                    vol, slice_idx, gt_box, gt_label, slice_axis=slice_axis
                ),
                "gt",
            )
            panel_b = overlay_panel_label(
                draw_slice_boxes(
                    vol, slice_idx, pred_box, pred_label, slice_axis=slice_axis
                ),
                "pred",
            )

            if pred_box.numel() > 0:
                slice_idx3 = pick_slice_index(pred_box, vol.shape, slice_axis, rng)
            else:
                slice_idx3 = slice_idx
            panel_c = overlay_panel_label(
                draw_slice_boxes(
                    vol,
                    slice_idx3,
                    pred_box,
                    pred_label,
                    slice_axis=slice_axis,
                ),
                "pred",
            )

            triplet_case_ids.append(item["case_id"])
            panels.append(_panel_rgb(panel_a))
            panels.append(_panel_rgb(panel_b))
            panels.append(_panel_rgb(panel_c))

        if len(panels) == 0:
            raise ValueError("snippets: no cases in batch")

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
        if stage is not None:
            grid = overlay_grid_stage_banner(grid, stage)
        return grid

    def _store_snippets(self, trainer, stage, epoch):
        cached_batches = self._snippet_cache[stage]
        if len(cached_batches) == 0:
            return
        items = _flatten_cached_batches(cached_batches[: self.max_cached_batches])
        grid = self._snippets_from_items(items, stage=stage)
        fig_fname = self.local_folder / f"{stage}_{epoch}_snippets.png"
        cv2.imwrite(
            str(fig_fname),
            cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
        )
        self.dfs["epoch"] = epoch
        self.dfs[stage] = str(fig_fname)
        trainer.dfs = self.dfs
        try:
            trainer.logger.log_image(
                key=f"{stage}_snippets",
                images=[str(fig_fname)],
            )
        except AttributeError as e:
            cprint(e, color="red")


class CaseIDRecorderSnapshotDet(CaseIDRecorderDet):
    def __init__(
        self,
        freq=5,
        local_folder="/tmp",
        dpi=300,
        monitor_dl="valid",
        dl_idx=0,
        grid_rows=None,
        n_tiles=None,
        auto_grid=True,
        slice_axis=2,
        max_cached_batches=8,
    ):
        super().__init__(
            freq=freq,
            monitor_dl=monitor_dl,
            local_folder=local_folder,
            dpi=dpi,
            grid_rows=grid_rows,
            n_tiles=n_tiles,
            auto_grid=auto_grid,
            slice_axis=slice_axis,
            max_cached_batches=max_cached_batches,
        )
        self.dl_idx = int(dl_idx)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx != self.dl_idx:
            return
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
