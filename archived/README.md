# archived branch

Frozen snapshot of OBD patch sizing before strict `expand_by=0` / collate-only padding.

Includes:
- `PATCH_SIZE_MIN` / `PATCH_SIZE_MAX` pre-crop margin, aspect-ratio resize, center-pad
- `OBDResizeToMaxd`, `OBDPadToMind`, `patch_size_log` / `PATCH_SIZE_ADJUSTED`
- Compose chain: `LoadT,Chan,Dev,Stats,N2P,Resize,PadMin,AttachGT,Int`
- `expand_by` from plan (not forced to 0)

Use branch `archived` — not maintained.
