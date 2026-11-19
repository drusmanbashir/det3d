# Plans, patch dimensions, and training crops

## Excel

The `plans` sheet in `experiment_configs_det.xlsx` uses `patch_dim0` and `patch_dim1`, like FRAN segmentation plans.

`ConfigMakerDet` calls fran `parse_plan_row` → `derive_patch_size` → `make_patch_size`:

```
patch_size = [patch_dim0, patch_dim0, patch_dim1]
```

Setting the patch dimensions to `auto` delegates the training patch to BoxC002.

## Crop geometry

The random crop requests `patch_size × prezoom_scale` and uses the stored case’s `image.shape` as its bounds. After augmentation, crop or pad produces the final `patch_size`. HDF5 chunk dimensions use the shared storage default described in [preprocessing.md](preprocessing.md).

## Train transform order (`DataManagerDetSource`)

`keys_tr`: **Ld → Rtr → L2** → …

```
Ld   paths + image.shape + extended_center_boxes (no voxels)
Rtr  crop_slices / crop_start (metadata only)
L2   first HDF5 pixel read via crop_slices; loads bbox here
```

Sample dict keys are stripped after each stage (see `preprocessing.md` / code in `managers/data/main.py`).

`extended_center_boxes` come from the per-case LBD bounding-box CSVs for the requested crop size.

## See also

- [preprocessing.md](preprocessing.md) — HDF5 chunking and existing shards
- [fran patch-and-folders.md](../../fran/fran/docs/plans/patch-and-folders.md)
