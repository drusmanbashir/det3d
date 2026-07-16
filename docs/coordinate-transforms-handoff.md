# Coordinate transforms handoff (agent teaching doc)

Teach voxel ↔ world ↔ resampled-grid math for MONAI/NIfTI 3D tensors. All examples use **homogeneous row vectors** unless noted.

**Audience:** user learning; agent should walk examples numerically, not skip to APIs.

## Agent instructions

1. Always state **which space** a number lives in before interpreting it (native full voxel, crop-local voxel, preproc spaced voxel, world mm).
2. Distinguish **affine on indices** vs **resample inverse** vs **integer crop offset** — three different ops; order matters.
3. Use the worked matrices below; run [`run/tools/affine_voxel_world_demo.py`](../run/tools/affine_voxel_world_demo.py) or [`det3d/extra/bbox_geom_roundtrip.py`](../det3d/extra/bbox_geom_roundtrip.py) when user wants code proof.

---

## Vocabulary

| Symbol | Meaning |
|--------|---------|
| \(v\) | Voxel **index** \([i,j,k]\) in some grid (units: voxels, origin at corner of voxel 0) |
| \(p\) | **World** mm in scanner/RAS+ \([x,y,z]\) |
| \(A\) | 4×4 **affine** mapping that grid’s indices → world: \(p = A v\) (column \(v\), MONAI/NIfTI convention) |
| `spatial_shape` | Tensor spatial size \((D,H,W)\) for that grid |
| `applied_operations` | MONAI trace stack; **InvP** inverts this on volumes LIFO |

**MetaTensor:** `image` carries `data` + `affine` + `applied_operations`. Slicing/cropping the tensor does **not** automatically fix `affine` or indices — crop offsets are separate.

---

## Rule 1 — Affine forward and inverse

\[
\begin{bmatrix}x\\y\\z\\1\end{bmatrix}
=
A
\begin{bmatrix}i\\j\\k\\1\end{bmatrix},
\qquad
v = A^{-1} p
\]

For \(A = \begin{bmatrix}R & t \\ 0 & 1\end{bmatrix}\) with diagonal \(R=\mathrm{diag}(s_x,s_y,s_z)\):

\[
A^{-1} =
\begin{bmatrix}
R^{-1} & -R^{-1}t \\
0 & 1
\end{bmatrix},
\quad
i = \frac{x - t_x}{s_x},\;
j = \frac{y - t_y}{s_y},\;
k = \frac{z - t_z}{s_z}
\]

### Toy matrix

\[
A =
\begin{bmatrix}
2 & 0 & 0 & 10 \\
0 & 2 & 0 & 20 \\
0 & 0 & 3 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\]

| step | \(v\) or \(p\) | result |
|------|----------------|--------|
| forward | \(v=[5,4,2]\) | \(p=[20,28,6]\) mm |
| inverse | \(p=[20,28,6]\) | \(v=[5,4,2]\) |

---

## Rule 2 — Same anatomy, two grids (spacing change)

World is fixed; **voxel index changes** when spacing/size changes.

| grid | \(A\) | index | world mm |
|------|-------|-------|----------|
| native | \(I\) | 20 | 20 |
| 2 mm spacing | \(\mathrm{diag}(2)\) | 10 | 20 |

Cross-grid (same world):

\[
v_B = A_B^{-1} A_A\, v_A
\]

**Trap:** \(A_B^{-1}[10,0,0]\) treats **10 as world mm** → wrong index 5. Ten is a **B-grid voxel index**, not world.

**InvP / volume inverse:** resample so world anatomy aligns; index jumps 10→20. **Not** a single \(A^{-1}\) on the index alone.

---

## Rule 3 — MONAI Spacingd (resample between affines)

Forward resample uses (MONAI `spatial_resample`):

\[
X = \mathrm{solve}(A_{\text{src}}, A_{\text{dst}})
\]

Voxel index map (continuous corners): \(v_{\text{dst}} \sim X\, v_{\text{src}}\) (then interpolation on the grid).

Inverse spacing on **points:** \(v_{\text{src}} = X^{-1} v_{\text{dst}}\).

### lidc_0001 load → spaced crop (before RAS flip)

\[
A_{\text{load}} =
\begin{bmatrix}
-0.703 & 0 & 0 & 0 \\
0 & -0.703 & 0 & 0 \\
0 & 0 & 2.5 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\]

After `Spacingd(0.8,0.8,1.5)` on crop (still pre-orientation): diagonal magnitudes become 0.8, 0.8, 1.5 (signs preserved until `Orientationd`).

After `Orientationd(RAS)` on crop:

\[
A_{\text{pre}} \approx
\begin{bmatrix}
0.8 & 0 & 0 & -258.4 \\
0 & 0.8 & 0 & -196 \\
0 & 0 & 1.5 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\]

(Translation \(-258.4,-196\) from axis flips on cropped grid ~324×246; exact values depend on crop size.)

**Forward** preproc index → world:

\[
\begin{aligned}
x &= 0.8\cdot 92 - 258.4 = -184.8 \\
y &= 0.8\cdot 13 - 196 = -185.6 \\
z &= 1.5\cdot 112 = 168.0
\end{aligned}
\]

**Correct inverse** world → preproc index: \((92,13,112)\).

**Wrong (old InvB):** apply \(A_{\text{pre}}^{-1}\) to **voxel** \([92,13,112]\) as if world:

\[
A_{\text{pre}}^{-1}[92,13,112] \approx [438,\ 261,\ 75]
\]

MONAI `ApplyTransformToPoints(invert_affine=True)` expects **world mm** input; `ConvertBoxToPointsd` emits **voxel** corners → space mismatch.

---

## Rule 4 — Orientation (flip + permute on indices)

`Orientationd` is **not** only a 4×4 multiply on indices. On voxel data it:

1. flips axes where LPS→RAS requires sign \(-1\): \(i' = \text{size}_i - 1 - i\)
2. permutes axes to RAS order

On **box corners:** apply the same flip/permute inverse LIFO from `applied_operations` (see `InvPreprocessBoxd` in [`det3d/inference/post.py`](../det3d/inference/post.py)).

---

## Rule 5 — Crop (integer offset, separate from affine)

Localiser crop (native full → native crop-local):

```text
bounding_box[1:] = slice(74,442), slice(122,402), slice(18,133)
starts = [74, 122, 18]
v_full = v_crop + starts   (per axis, on all corner coords)
```

Crop does **not** replace Spacing/Orientation inverse. Order on boxes:

```text
bbo2 (preproc crop voxels)
  → InvPreBox  (inverse O, then S on 8 corners)
  → Off        (+74, +122, +18)
  → BoxR       (restore load orientation on full volume)
  → Save       (AffineBoxToWorldCoordinated)
```

**Never** `Off` before inverse preprocess.

---

## lidc_0001 reference numbers

| item | value |
|------|-------|
| full `spatial_shape` | 512 × 512 × 133 |
| crop size | 368 × 280 × 115 |
| preproc crop shape after S,O | ~324 × 246 × 191 |
| `bbo2` xyzxyz (model, preproc crop) | `[91.85, 12.65, 111.84, 126.40, 51.47, 127.32]` |
| inverse → native crop (InvPreBox) | `[172.80, 170.09, 186.40, 203.16, 204.21, 212.20]` |
| + Off → native full (detection box) | `[246.80, 292.09, 204.40, 277.16, 326.21, 230.20]` |
| LMG GT `[x0,y0,z0,sx,sy,sz]` | `[299, 343, 86, 37, 45, 8]` |
| LMG GT xyzxyz full | `[299, 343, 86, 336, 388, 94]` |

GT and top detection box are **different objects**; geometry roundtrip is validated separately for each.

---

## Boxes: 6-vector ↔ 8 corners

**xyzxyz:** `[x1,y1,z1, x2,y2,z2]` — two opposite corners, same voxel space.

**ConvertBoxToPointsd:** unpacks to 8×3 (all min/max combos); **no** coordinate change.

Axis-aligned box after arbitrary corner transform: min/max per axis of the 8 corners.

---

## Tensor cheat sheet

```python
# image: MetaTensor (1, D, H, W) or (B, 1, D, H, W)
image.affine          # 4×4, float64, index → world for THIS grid
image.applied_operations  # list of {class, orig_size, extra_info}
image.shape[-3:]      # current spatial_shape

# bbox tensor (N, 6) float, xyzxyz in ONE named space
# pred_box[i] meaningless without knowing post-pipeline stage
```

**InvP (seg):** copy `image.applied_operations` → `pred`; `Invertd` resamples volume LIFO.

**InvPreBox (boxes):** same trace; apply \(X^{-1}\) and inverse orientation to **8 corners**, not grid sample.

---

## Common mistakes (agent checklist)

| mistake | symptom |
|---------|-----------|
| \(A^{-1}\) on voxel indices | indices ~400+ on 512 grid; box outside image |
| Off before InvPreBox | slice starts added in wrong space |
| overlay post-InvB box on preproc image | box “vanishes” in viewer |
| assume `spatial_shape` meta = tensor `.shape` after preprocess | meta can be stale; trust tensor shape for indexing |
| one affine for crop without offset | crop-local 0 ≠ full-volume 0 |
| ignore orientation | x/y flips swap; y offset ~200 voxels |

---

## In code

| topic | path |
|-------|------|
| InvPreBox | [`det3d/inference/post.py`](../det3d/inference/post.py) `InvPreprocessBoxd` |
| patch postproc | [`det3d/inference/patch.py`](../det3d/inference/patch.py) `Pack,SqL,InvP,InvPreBox` |
| cascade postproc | [`det3d/inference/cascade.py`](../det3d/inference/cascade.py) `Off,BoxR,S` |
| numeric demo | [`run/tools/affine_voxel_world_demo.py`](../run/tools/affine_voxel_world_demo.py) |
| roundtrip scratch | [`det3d/extra/bbox_geom_roundtrip.py`](../det3d/extra/bbox_geom_roundtrip.py) |
| LMG idx+size → xyzxyz | [`det3d/geometry/lmg.py`](../det3d/geometry/lmg.py) `voxel_start_size_to_gt_box` |
| short reference | [affine-voxel-world.md](affine-voxel-world.md) |

## External refs

- [NiBabel — Coordinate systems and affines](https://nipy.org/nibabel/coordinate_systems.html)
- [MONAI — ApplyTransformToPoints](https://docs.monai.io/en/stable/transforms.html#applytransformtopoints)
- [MONAI MetaTensor guide](https://github.com/Project-MONAI/MONAI/wiki/MetaTensor-guide)

## Suggested teaching order

1. Toy \(A\) forward/inverse (Rule 1).
2. Two grids, same world (Rule 2) — explain InvP vs \(A^{-1}\).
3. lidc \(A_{\text{load}}\), \(A_{\text{pre}}\), \([92,13,112]\) forward/inverse (Rule 3).
4. InvB trap \([438,261,75]\) (Rule 3).
5. Crop offsets + production box chain (Rules 4–5).
6. Optional: run roundtrip script; compare `bbo2` vs LMG GT.
