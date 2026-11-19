# Detection labels: preproc → class channels

Semantic LM values → detection class indices. Seg remaps and bbox sidecars are separate.

## Plan keys by stage

| Key | fixed_spacing | LBD det | Training |
|-----|---------------|---------|----------|
| `remapping_source` | `Remap` on source LM | — | — |
| `remapping_lbd_rbd` | — | `Remap` on LBD LM | — |
| `fg_indices_exclude` | `Indx` + `indices_fg_exclude_*` | `Indx` again | loads saved indices |
| `ignore_labels` | — | `DetectionBBoxStatsd` / LMG | Foreground box classes from `manifest.json` |
| `remapping_train` | — | — | FRAN `MapLabelValued` on `lm` (not for det) |

Preproc order: source remap → foreground indices → LBD crop → LBD remap → box filtering.

## Preproc → manifest

`ignore_labels` filters connected components. Survivors keep semantic `label_org` in `bboxes/*.csv`.

Example: ignore `[1, 5]` → boxes only for **2, 3, 4, 6**.

Postproc unions all `label_org` → `manifest.json` `labels_all` (e.g. `[0, 2, 3, 4, 6]`).

## Training — `fg_labels` → class index

`infer_det_labels_from_data_folder`: `fg_labels = [v for v in labels_all if v != 0]` → plans. Dataloader reads `label_org` unchanged.

Dense index = position in `fg_labels` (`label_to_idx` in `RetinaNetManager` / `nndet_target_classes`):

| `label_org` | class |
|-------------|-------|
| 2 | 0 |
| 3 | 1 |
| 4 | 2 |
| 6 | 3 |

Head width: `len(fg_labels)` (`create_detector.py`), not `max(label_org)+1`. Seg `lm` / `target_seg` stay semantic — separate from det classes.

**LIDC single-class:** all nodules label `1` → `fg_labels = [1]`, class 0. Keep the lesion label out of `ignore_labels`.

## `remapping_train`

Inherited FRAN on-the-fly `MapLabelValued` on `lm`. **Not intended for det** — sidecars keep preproc `label_org`; train-time LM remap does not update boxes and would desync det targets. Leave Excel-None.

## In code

| Thing | Module |
|-------|--------|
| CC ignore | `det3d/transforms/bbox_stats.py`, `labelbounded.py` |
| `labels_all` / `fg_labels` | `labelbounded.py` postprocess, `managers/data/labels.py` |
| Bbox load + class remap | `managers/data/main.py`, `retinanet.py`, `helpers/nndet_retinaunet.py` |

## See also

- [preprocessing.md](preprocessing.md), [plans-and-patches.md](plans-and-patches.md)
- fran [training.md](../../fran/fran/docs/training/training.md) — `remapping_train` for seg
