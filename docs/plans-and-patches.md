# Plans, patch dims, and src_dims

## Excel

`experiment_configs_det.xlsx` → sheet `plans_det` uses **`patch_dim0`** and **`patch_dim1`** (same as fran seg plans), not a literal `patch_size` column.

`ConfigMakerDet` calls fran `parse_plan_row` → `derive_patch_size` → `make_patch_size`:

```
patch_size = [patch_dim0, patch_dim0, patch_dim1]
```

Then det adds plan `src_dims` via `make_src_dims_from_patch_size` (each dim × 1.1, even).

Example (lidc lungs plans 3 vs 4 — only patch dims differ):

| | plan 3 | plan 4 |
|---|--------|--------|
| patch_dim0 × patch_dim1 | 160 × 96 | 128 × 64 |
| patch_size | [160,160,96] | [128,128,64] |
| plan src_dims | [176,176,106] | [140,140,70] |
| data_folder_lbd | same | same |

Edit `plans_det` in `~/code/fran/configurations/experiment_configs_det.xlsx` directly — use `patch_dim0`/`patch_dim1`, not a `patch_size` column.

## Three different `src_dims`

| Symbol | Where | Used for |
|--------|-------|----------|
| `plan["src_dims"]` | `ConfigMakerDet._apply_src_dims_from_patch_size` | HDF5 path tag `hdf5_shards/src_{tag}/`, `_assert_patch_fits_src_dims` |
| `manifest["src_dims"]` | `manifest.json` inside shard dir | Build metadata only |
| `d["src_dims"]` on sample | `LoadHDF5DetShardExtendedBBoxd` from `case_grp["image"].shape` | `RandCropExtendedBBoxd` clamping — **per-case volume**, not plan tag |

Training crop window = `patch_size_prezoom` (`patch_size` × `prezoom_scale`). RandCrop reads sample `src_dims`, not plan `src_dims`.

## Train transform order (`DataManagerDetSource`)

`keys_tr`: **Ld → Rtr → L2** → …

```
Ld   paths + image.shape + extended_center_boxes (no voxels)
Rtr  crop_slices / crop_start (metadata only)
L2   first HDF5 pixel read via crop_slices; loads bbox here
```

Sample dict keys are stripped after each stage (see `preprocessing.md` / code in `managers/data/main.py`).

`extended_center_boxes` come from LBD `extended_bboxes/` (grid of prezoom patch sizes). Multi patch plans share one LBD build.

## See also

- [preprocessing.md](preprocessing.md) — HDF5 symlink when only patch plan changes
- [fran patch-and-folders.md](../../fran/fran/docs/plans/patch-and-folders.md)
