
# HANDOFF — det3d (alignment priority)

See plan: `~/.cursor/plans/cascade_seg_alignment_fix_0e552c86.plan.md`

## Bug

Cascade seg/boxes wrong vs GT on `lidc_0001` — missing patch `InvP`; resize-only box/seg scale.

## Target (`*`)

| | RetinaNet | RetinaUNet |
|---|-----------|------------|
| Patch post | `Pack,SqL,InvB` | `Pack,SqL,InvP,InvB` |
| Cascade | `SqL,Off,BoxR,S` | `SqL,A,Int,W,F,Off,R,BoxR,S` |
| LBD | `DetPatchLBD` patch-only — no cascade | same |

Split `PackRetinaNetPredsd` / `PackRetinaUNetPredsd`. Fran `F`+`R` → `pred.meta` = source grid; `SaveDetOutputd` uses that. Mismatch = parity bug, not extra meta step.

## Implement

1. `post.py` — Pack*, Off, BoxR, SaveDetOutputd
2. `patch.py` / `cascade.py` / `lbd.py`
3. `infer_det.py` routing
