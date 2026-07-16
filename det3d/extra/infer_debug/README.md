# infer_debug — inference alignment debug suite

Stepwise postprocessing snapshots, pseudo + real fixtures, LM-as-pred roundtrips.  
Ad hoc validation only (`/s/agent_rw/tmp/`); not repo `test_*.py`.

## When to use

- Cascade / patch **seg or bbox desync** vs image or GT
- Pinpoint **which post key** (`InvP`, `F`, `R`, `Off`, …) first changes fg count or boxes
- Roundtrip **known mask** through production transforms (no model weights)

## Layout

```
infer_debug/
  run.py                 # CLI entry
  step_runner.py         # generic per-key transform chain + TSV report
  metrics.py             # fg count, LMG component masks, box deltas
  fixtures/
    pseudo_cuboid.py     # in-memory 2-cuboid LM (exact roundtrip target)
    lidc_0001.py         # real CT + LM (`/media/UB/datasets/lidc_all/`)
  streams/
    cascade_retinaunet.py    # patch + cascade post dict builders, fake batch
    cascade_lm_roundtrip.py  # end-to-end roundtrip + stage reports
```

Related (older, narrower):

- `extra/bbox_geom_roundtrip.py` — box-only synthetic
- `extra/bbox_lmg_roundtrip.py` — box-only real LM
- `inference/cascade.py` `__main__` — interactive `# %%` blocks with real model

## CLI

```bash
# plan entry — lidc_0001 LM roundtrip (strict asserts + Slicer NIfTI)
conda run -n dl python det3d/extra/cascade_mask_roundtrip.py

# exact roundtrip on pseudo (should PASS)
conda run -n dl python -m det3d.extra.infer_debug.run --fixture pseudo_cuboid

# real lidc — stepwise TSV + summary (CHECK if resample loss)
conda run -n dl python -m det3d.extra.infer_debug.run --fixture lidc_0001
```

Outputs: `{--out}/{fixture}/{run_p}/patch_stages.tsv`, `cascade_stages.tsv`, `summary.txt`.

## Reading stage reports

Watch `pred_fg` and `n_boxes` columns. First row where fg jumps → suspect transform on previous key.

| Chain | Keys (RetinaUNet safe) |
|-------|-------------------------|
| patch | `Pack,SqL,InvP,InvPreBox` |
| cascade | `SqL,MR,W,F,R,Off,BoxR` |

## Adding a fixture

1. `fixtures/my_case.py` → `build_my_case() -> InferFixtureCase`
2. Register in `fixtures/__init__.py` `FIXTURES`
3. Optional `crop_fixture` if crop logic differs

## Adding a stream

1. `streams/my_stream.py` — build transform dicts + adapter batch
2. Wire in `run.py` `--stream` choices
3. Use `run_transform_chain` for stepwise reports
