# det3d docs

Short notes on non-obvious design. CLI and run scripts: repo `README.md` and `FUNCTIONS.md`.

| Topic | File |
|-------|------|
| Plans, `patch_dim`, `src_dims` (three meanings) | [plans-and-patches.md](plans-and-patches.md) |
| Label remaps, `ignore_labels_cc`, det class channels | [labels.md](labels.md) |
| LBD vs HDF5, shard reuse across plans | [preprocessing.md](preprocessing.md) |
| Affine voxel ↔ world, InvP vs InvB | [affine-voxel-world.md](affine-voxel-world.md) |
| Coordinate math handoff (agent teaching) | [coordinate-transforms-handoff.md](coordinate-transforms-handoff.md) |

Shared plan parsing (`parse_plan_row`, `make_patch_size`): [fran `patch-and-folders.md`](../../fran/fran/docs/plans/patch-and-folders.md).
