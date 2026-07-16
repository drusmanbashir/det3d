# Affine: voxel index ↔ world mm

One-page reference for MONAI/NIfTI affine forward and inverse, and why **InvP** (volume inverse) differs from **`A⁻¹` on voxel indices** (InvB trap).

## Convention

Homogeneous voxel index \(v = [i,j,k,1]^\top\), world mm \(p = [x,y,z,1]^\top\) (RAS+):

\[
p = A\,v
\qquad
v = A^{-1}\,p
\]

Split \(A = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}\):

\[
p = Rv_{3} + t
\qquad
v_{3} = R^{-1}(p_{3} - t)
\]

Diagonal \(R = \mathrm{diag}(s_x,s_y,s_z)\):

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

## Worked example 1 — toy scale + shift

\[
A =
\begin{bmatrix}
2 & 0 & 0 & 10 \\
0 & 2 & 0 & 20 \\
0 & 0 & 3 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\]

**Forward** \(v=[5,4,2,1]^\top\):

\[
Av =
\begin{bmatrix}
2\cdot5+10 \\
2\cdot4+20 \\
3\cdot2
\end{bmatrix}
=
\begin{bmatrix}
20 \\ 28 \\ 6
\end{bmatrix}
\]

**Inverse** on world \(p=[20,28,6,1]^\top\):

\[
A^{-1} =
\begin{bmatrix}
0.5 & 0 & 0 & -5 \\
0 & 0.5 & 0 & -10 \\
0 & 0 & \tfrac{1}{3} & 0 \\
0 & 0 & 0 & 1
\end{bmatrix},
\quad
A^{-1}p =
\begin{bmatrix}
5 \\ 4 \\ 2
\end{bmatrix}
= v_{3}
\]

## Worked example 2 — same world, two grids

Grid **A** (native): \(A_0 = I\), size 100.  
Grid **B** (2 mm spacing): \(A_1 = \mathrm{diag}(2)\), size 50.

World of native voxel 20:

\[
p = A_0 v = [20,0,0]^\top
\]

Index on B with same world:

\[
20 = 2\,i_B \;\Rightarrow\; i_B = 10
\]

| grid | affine | voxel | world mm |
|------|--------|-------|------------|
| A | \(I\) | 20 | 20 |
| B | \(\mathrm{diag}(2)\) | 10 | 20 |

**InvP-style:** inverse resample B→A moves marker **10 → 20** (preserves world).

**Wrong:** \(A_1^{-1}[10,0,0,1] = [5,0,0]\) — treats **10 as world mm**, not as a B voxel index.

Cross-grid index map (same world, two affines):

\[
v_B = A_B^{-1} A_A\, v_A
\]

## Worked example 3 — preproc crop affine

\[
A_{\text{pre}} =
\begin{bmatrix}
0.8 & 0 & 0 & -258.4 \\
0 & 0.8 & 0 & -196 \\
0 & 0 & 1.5 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\]

**Forward** \(v=[92,13,112,1]^\top\):

\[
\begin{aligned}
x &= 0.8\cdot92 - 258.4 = -184.8 \\
y &= 0.8\cdot13 - 196 = -185.6 \\
z &= 1.5\cdot112 = 168.0
\end{aligned}
\]

**Correct inverse** (world → voxel):

\[
\begin{aligned}
i &= 1.25(-184.8 + 258.4) = 92 \\
j &= 1.25(-185.6 + 196) = 13 \\
k &= \tfrac{2}{3}\cdot168 = 112
\end{aligned}
\]

**InvB mistake** (`ApplyTransformToPointsd`, `invert_affine=True` on **voxel** corners):

\[
A_{\text{pre}}^{-1}[92,13,112,1] =
\begin{bmatrix}
1.25\cdot92 + 323 \\
1.25\cdot13 + 245 \\
\tfrac{2}{3}\cdot112
\end{bmatrix}
\approx
\begin{bmatrix}
438 \\ 261 \\ 75
\end{bmatrix}
\]

MONAI documents `ApplyTransformToPoints` input as **world** coordinates when `invert_affine=True`; `ConvertBoxToPointsd` emits **voxel** corners → space mismatch.

## Correct box chain (cascade)

Production forward on **crop** (native load → localiser `bounding_box` crop → `Spacingd` + `Orientationd`):

\[
\text{bbox}_\text{full native} \xrightarrow{\text{crop-local}} \text{bbox}_\text{crop native} \xrightarrow{S,O} \text{bboxt2 (bbo2)}
\]

Reverse (implemented as `InvPreprocessBoxd` + `Offd`):

\[
\text{bbo2} \xrightarrow{\text{InvPreBox (inv O, S on 8 corners)}} \text{bbox}_\text{crop native} \xrightarrow{+\text{slice starts}} \text{bbox}_\text{full native} \xrightarrow{\text{BoxR}} \text{sidecar voxels}
\]

Patch postproc: `Pack, SqL, InvP, InvPreBox`. Cascade: `Off, BoxR, S`. **Do not** apply `Off` before inverse preprocess.

Scratch roundtrip + LMG GT check: [`det3d/extra/bbox_geom_roundtrip.py`](../det3d/extra/bbox_geom_roundtrip.py). **Agent teaching doc:** [coordinate-transforms-handoff.md](coordinate-transforms-handoff.md).

## InvP (MONAI `Invertd`) — not `A⁻¹` on indices

1. Copy `image.applied_operations` onto `pred`.
2. `Compose.inverse`: LIFO on invertible transforms (`Orientationd`, `Spacingd`, …).
3. Each step **resamples the volume** using traced `src_affine` / `dst_affine` and `orig_size` (`SpatialResample.inverse` swaps affines and grid size).

Spacing forward (MONAI `spatial_resample`):

\[
X = \mathrm{solve}(A_{\text{src}}, A_{\text{dst}})
\]

then grid-sample warp. Inverse swaps src↔dst and restores `orig_size`.

## In code

| Thing | Location |
|-------|----------|
| Runnable numeric demos | [`run/tools/affine_voxel_world_demo.py`](../run/tools/affine_voxel_world_demo.py) |
| InvP wiring | `det3d/inference/patch.py` (`Invertd` on `pred`, trace from `image`) |
| InvPreBox wiring | `det3d/inference/post.py` (`InvPreprocessBoxd`) |
| Roundtrip scratch | [`det3d/extra/bbox_geom_roundtrip.py`](../det3d/extra/bbox_geom_roundtrip.py) |
| Small world-box helper | `det3d/geometry/affine.py` |

## See also (online)

- [NiBabel — Coordinate systems and affines](https://nipy.org/nibabel/coordinate_systems.html) — canonical \(p=Av\), \(v=A^{-1}p\), cross-image \(v_B=A_B^{-1}A_A v_A\)
- [MONAI — `ApplyTransformToPoints`](https://docs.monai.io/en/stable/transforms.html#applytransformtopoints) — world ↔ image space on point sets
- [MONAI tutorial — dict inference + `Invertd`](https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/torch/unet_inference_dict.py)
- [MONAI MetaTensor guide (wiki)](https://github.com/Project-MONAI/MONAI/wiki/MetaTensor-guide) — `.affine`, lazy pending affine
