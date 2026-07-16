# Preprocessing: LBD and HDF5 shards

## Why plan change can skip HDF5 rebuild

**LBD folder** (`FolderNames.lbd_folder`) depends on spacing, remapping, `expand_by` — **not** patch dims. Plans that differ only in `patch_dim0`/`patch_dim1` share `data_folder_lbd`.

**HDF5 folder** is tagged by **plan** `src_dims`:

```
{data_folder_lbd}/hdf5_shards/src_{plan_src_dims}/manifest.json
```

Shard **content** is LBD `.pt` volumes + bboxes; per-case `image.shape` is stored in HDF5, not plan `src_dims`. Rebuilding for every patch plan is redundant.

## Shard reuse (`ensure_hdf5_shards_for_plan`)

`det3d/preprocessing/hdf5_shards_det.py`

1. Target = `lbd_folder/hdf5_shards/src_{tag(plan_src_dims)}`
2. If `target/manifest.json` exists → done
3. Else scan `src_*/manifest.json` under same `hdf5_shards`
4. Match **case_id set** to LBD `images/` + `bboxes/`
5. `target.symlink_to(candidate.resolve())` — path tag follows active plan; manifest inside target is build metadata only

Wired in:

- `LabelBoundedDetDataGenerator.process_hdf5` — before `DetHDF5ShardGenerator.setup`; skip `run()` if symlink just created
- `DataManagerDet._require_shard_manifest` — training auto-resolves without preproc rerun

Only within the same LBD folder — never across spacing/remapping.

## Rapid access path

Preproc writes HDF5 under `project.rapid_access_folder / {mode} / {lbd_name} / hdf5_shards` (same as `hdf5_output_folder` on preprocessor). Training uses `data_folder_lbd / hdf5_shards` after `ensure_rapid_data_folder`.

## See also

- [labels.md](labels.md) — remapping keys, `ignore_labels_cc`, semantic → class channels
- [plans-and-patches.md](plans-and-patches.md)
- fran `HDF5ShardGenerator.shards_folder` — same `src_{tag}` pattern for seg
