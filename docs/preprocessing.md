# Preprocessing: LBD and HDF5 shards

## Shards

FRAN and det3d use a shared HDF5 chunk target of `(128, 128, 96)`. Each axis is capped by the actual case dimension. The detection writer stores the LBD image, mask, instance map, boxes and labels under each case ID.

```
{data_folder_lbd}/hdf5_shards/chunks_128_128_96/manifest.json
```

Training prefers the new shard folder. It can also open an older `src_*` shard folder in the same LBD build if the available manifests agree on the case set. Preprocessing writes the new layout alongside those older folders.

## Rapid access path

Preprocessing writes HDF5 under `project.rapid_access_folder / {mode} / {lbd_name} / hdf5_shards`. Training looks under `data_folder_lbd / hdf5_shards`.

## See also

- [labels.md](labels.md) — remapping keys, `ignore_labels_cc`, semantic → class channels
- [plans-and-patches.md](plans-and-patches.md)
