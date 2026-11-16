# Affine: voxel index ↔ world mm

**Motivation.** A predicted box must be expressed in the voxel grid used by its output image. An image affine connects voxel coordinates to world coordinates; a change between two voxel grids requires both grids' affines. This reference derives those maps and explains the distinction between **InvP** (volume inverse resampling) and applying an inverse image affine to voxel indices (the **InvB** trap).

## Convention

All vectors below are **column vectors**. Let $v_3=[i,j,k]^\top\in\mathbb{R}^3$ denote voxel coordinates and $p_3=[x,y,z]^\top\in\mathbb{R}^3$ world coordinates in millimetres. Integer voxel indices refer to voxel centres; continuous coordinates are allowed between them. The world convention is RAS+: positive axes point right, anterior and superior.

Appending a final coordinate of one gives the homogeneous vectors $v=[i,j,k,1]^\top$ and $p=[x,y,z,1]^\top$. For an invertible $4\times4$ image affine $A$:

$$$$
p = A\,v
\qquad
v = A^{-1}\,p
$$$$

Write $A = \begin{bmatrix} R & t \\ 0^\top & 1 \end{bmatrix}$, where $R\in\mathbb{R}^{3\times3}$ is invertible and $t\in\mathbb{R}^3$. The columns of $R$ give world displacement per voxel step along each image axis; $t$ is the world position of voxel zero. In three coordinates:

$$$$
p_3 = Rv_3 + t
\qquad
v_3 = R^{-1}(p_3 - t)
$$$$

Subtracting the translation and multiplying by $R^{-1}$ gives the inverse affine:

$$$$
A^{-1} =
\begin{bmatrix}
R^{-1} & -R^{-1}t \\
0^\top & 1
\end{bmatrix}
$$$$

For the axis-aligned case $R=\mathrm{diag}(s_x,s_y,s_z)$, with each scale nonzero:

$$$$
i = \frac{x - t_x}{s_x},\;
j = \frac{y - t_y}{s_y},\;
k = \frac{z - t_z}{s_z}
$$$$

The scales are signed millimetres per voxel; their absolute values are the voxel spacings. The general block inverse also applies when $R$ includes rotations or shear. See [NiBabel, affine representation and inverse mapping](https://nipy.org/nibabel/coordinate_systems.html#the-affine-matrix-as-a-transformation-between-spaces).

## Worked example 1 — toy scale + shift

This toy affine scales each voxel axis and then translates the result in world space:

$$$$
A =
\begin{bmatrix}
2 & 0 & 0 & 10 \\
0 & 2 & 0 & 20 \\
0 & 0 & 3 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$$$

**Forward** $v=[5,4,2,1]^\top$:

$$$$
\begin{aligned}
Av &=
\begin{bmatrix}
2\cdot5+10 \\
2\cdot4+20 \\
3\cdot2 \\
1
\end{bmatrix} &&\text{(i)} \\
&=
\begin{bmatrix}
20 \\ 28 \\ 6 \\ 1
\end{bmatrix} &&\text{(ii)}
\end{aligned}
$$$$

- (i) Apply the scale and translation to the voxel coordinates.
- (ii) Evaluate the arithmetic to obtain the homogeneous world vector $p$.

**Inverse** on world $p=[20,28,6,1]^\top$:

$$$$
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
5 \\ 4 \\ 2 \\ 1
\end{bmatrix}
=v
$$$$

## Worked example 2 — same world, two grids

Changing spacing changes a point's voxel coordinates while preserving its world coordinates. Consider two toy grids with zero translation:

- Grid **A** (native): $A_A=I_4$, spacing $1\ \mathrm{mm}$, length $100$ along the first axis.
- Grid **B**: $A_B=\mathrm{diag}(2,2,2,1)$, spacing $2\ \mathrm{mm}$, length $50$ along the first axis.

World position of native voxel $v_A=[20,0,0,1]^\top$:

$$$$
p=A_Av_A=[20,0,0,1]^\top
$$$$

The first coordinate on B follows by equating world coordinates:

$$$$
20 = 2\,i_B \;\Rightarrow\; i_B = 10
$$$$

| grid | affine | first-axis voxel index | first-axis world coordinate (mm) |
|------|--------|-------|------------|
| A | $I_4$ | $20$ | $20$ |
| B | $\mathrm{diag}(2,2,2,1)$ | $10$ | $20$ |

Thus $v_B=[10,0,0,1]^\top$. An inverse resample from B to A places this world position at A's first-axis index $20$; B's corresponding index is $10$.

**InvB mistake:** $A_B^{-1}[10,0,0,1]^\top=[5,0,0,1]^\top$. This calculation interprets $10$ as a world coordinate in millimetres, although the input is a B voxel index.

For any two invertible image affines sharing the same world reference, the coordinate maps follow from the shared world position:

$$$$
\begin{aligned}
A_Bv_B &= A_Av_A &&\text{(i)} \\
v_B &= A_B^{-1}A_Av_A &&\text{(ii)}
\end{aligned}
$$$$

- (i) Both voxel coordinates represent the same world point.
- (ii) Left-multiplication by $A_B^{-1}$ solves for the B coordinate.

The reverse map is $v_A=A_A^{-1}A_Bv_B$. Coordinate conversion does not itself interpolate image values.

## Worked example 3 — preproc crop affine

This numerical affine illustrates the effect of mistaking preprocessed crop indices for world coordinates:

$$$$
A_{\text{pre}} =
\begin{bmatrix}
0.8 & 0 & 0 & -258.4 \\
0 & 0.8 & 0 & -196 \\
0 & 0 & 1.5 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$$$

**Forward** $v=[92,13,112,1]^\top$:

$$$$
\begin{aligned}
x &= 0.8\cdot92 - 258.4 = -184.8 \\
y &= 0.8\cdot13 - 196 = -185.6 \\
z &= 1.5\cdot112 = 168.0
\end{aligned}
$$$$

**Correct inverse** (world → voxel):

$$$$
\begin{aligned}
i &= 1.25(-184.8 + 258.4) = 92 \\
j &= 1.25(-185.6 + 196) = 13 \\
k &= \tfrac{2}{3}\cdot168 = 112
\end{aligned}
$$$$

**InvB mistake** (`ApplyTransformToPointsd`, `invert_affine=True` on **voxel** corners):

$$$$
\begin{aligned}
A_{\text{pre}}^{-1}[92,13,112,1]^\top &=
\begin{bmatrix}
1.25\cdot92 + 323 \\
1.25\cdot13 + 245 \\
\tfrac{2}{3}\cdot112 \\
1
\end{bmatrix} &&\text{(i)} \\
&=
\begin{bmatrix}
438 \\ 261.25 \\ \tfrac{224}{3} \\ 1
\end{bmatrix} &&\text{(ii)} \\
&\approx
\begin{bmatrix}
438 \\ 261 \\ 75 \\ 1
\end{bmatrix} &&\text{(iii)}
\end{aligned}
$$$$

- (i) The inverse uses $R^{-1}=\mathrm{diag}(1.25,1.25,2/3)$ and $-R^{-1}t=[323,245,0]^\top$.
- (ii) The matrix multiplication is valid algebra, but its input has been interpreted in the wrong coordinate space.
- (iii) Rounding is for display; it does not repair the coordinate mismatch.

For an image affine, MONAI's [`ApplyTransformToPoints`, argument `invert_affine`](https://github.com/Project-MONAI/MONAI/blob/dev/monai/transforms/utility/array.py) maps ordinary world-coordinate points to image coordinates when `invert_affine=True`. Its `affine_lps_to_ras` setting must also match the coordinate convention; points carrying their own affine metadata require that metadata to be accounted for. `ConvertBoxToPointsd` enumerates corners without changing their coordinate space, so corners of a voxel box remain voxel coordinates. Applying the inverse image affine directly to those corners causes the space mismatch illustrated above.

## Correct box chain (cascade)

**Motivation.** Crop offsets are measured in the grid where the crop was taken. A box must return to that grid before those offsets can be added.

The forward sequence is native load, localiser `bounding_box` crop, `Spacingd`, then `Orientationd`:

$$$$
\text{bbox}_{\text{full native}}
\xrightarrow{-\text{slice starts}}
\text{bbox}_{\text{crop native}}
\xrightarrow{\text{Spacingd, Orientationd}}
\text{bboxt2 (bbo2)}
$$$$

Reverse (implemented as `InvPreprocessBoxd` + `Offd`):

$$$$
\begin{aligned}
\text{bbo2}
&\xrightarrow{\text{InvPreBox: inverse orientation, then spacing}}
\text{bbox}_{\text{crop native}} \\
&\xrightarrow{+\text{slice starts}}
\text{bbox}_{\text{full native}} \\
&\xrightarrow{\text{BoxR}}
\text{bbox}_{\text{original file voxels}}
\end{aligned}
$$$$

`InvPreBox` transforms all eight corners and reconstructs the enclosing axis-aligned box using the minimum and maximum along each axis. The current [`inverse_preprocess_points_from_ops`](../det3d/inference/post.py) undoes orientation from the trace and handles spacing by multiplying each coordinate by `orig_size / current_shape`. This size-ratio rule equals the physical affine map only when its scale and translation agree with that map; image sizes alone do not establish equality.

RetinaUNet patch postprocessing is `Pack,SqL,InvP,InvPreBox`; the RetinaNet variant omits `InvP`. The final cascade box sequence is `Off,BoxR,S`, where `S` is `SaveDetOutputd`. `BoxR` restores the original file orientation; saving converts the resulting voxel boxes to world coordinates. These stages are defined in [`patch.py`](../det3d/inference/patch.py), [`cascade.py`](../det3d/inference/cascade.py) and [`post.py`](../det3d/inference/post.py). **Apply `Off` after inverse preprocessing**, because its offsets belong to the native crop grid.

Scratch roundtrip + LMG GT check: [`det3d/extra/bbox_geom_roundtrip.py`](../det3d/extra/bbox_geom_roundtrip.py). **Agent teaching doc:** [coordinate-transforms-handoff.md](coordinate-transforms-handoff.md).

## InvP (MONAI `Invertd`) — volume inverse resampling

**Motivation.** A volume contains samples on a grid. Returning it to an earlier grid requires assigning a value to each output voxel using the corresponding position in the input volume.

1. Copy `image.applied_operations` onto `pred`.
2. `Compose.inverse` applies invertible transforms in reverse order (`Orientationd`, `Spacingd`, …).
3. The orientation inverse flips or permutes the array. The spacing inverse resamples it using the saved affine and `orig_size`; `SpatialResample.inverse` makes the original source affine the new destination affine.

Let $A_{\text{src}}$ and $A_{\text{dst}}$ map the input and output grid coordinates to the same world space. MONAI's [`spatial_resample`](https://github.com/Project-MONAI/MONAI/blob/dev/monai/transforms/spatial/functional.py) computes:

$$$$
X = \mathrm{solve}(A_{\text{src}}, A_{\text{dst}})
=A_{\text{src}}^{-1}A_{\text{dst}}
$$$$

This is the map **from an output voxel coordinate to an input sampling coordinate**:

$$$$
v_{\text{src}}=Xv_{\text{dst}}
$$$$

The resampler evaluates the input at that coordinate using the selected interpolation and padding rules. Moving a point from the input grid to the output grid instead uses $v_{\text{dst}}=X^{-1}v_{\text{src}}$. Inverse spacing swaps the grid roles and restores `orig_size`; it restores the grid geometry, while interpolation or downsampling can prevent exact recovery of the original voxel values. See MONAI's [`SpatialResample.inverse`](https://github.com/Project-MONAI/MONAI/blob/dev/monai/transforms/spatial/array.py).

## In code

| Thing | Location |
|-------|----------|
| Runnable numeric demos | [`det3d/run/tools/affine_voxel_world_demo.py`](../det3d/run/tools/affine_voxel_world_demo.py) |
| InvP wiring | `det3d/inference/patch.py` (`Invertd` on `pred`, trace from `image`) |
| InvPreBox wiring | `det3d/inference/post.py` (`InvPreprocessBoxd`) |
| Roundtrip scratch | [`det3d/extra/bbox_geom_roundtrip.py`](../det3d/extra/bbox_geom_roundtrip.py) |
| Small world-box helper | `det3d/geometry/affine.py` |

## See also (online)

- [NiBabel — Coordinate systems and affines](https://nipy.org/nibabel/coordinate_systems.html) — canonical $p=Av$, $v=A^{-1}p$, cross-image $v_B=A_B^{-1}A_A v_A$
- [MONAI — `ApplyTransformToPoints`](https://docs.monai.io/en/stable/transforms.html#applytransformtopoints) — world ↔ image space on point sets
- [MONAI tutorial — dict inference + `Invertd`](https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/torch/unet_inference_dict.py)
- [MONAI MetaTensor guide (wiki)](https://github.com/Project-MONAI/MONAI/wiki/MetaTensor-guide) — `.affine`, lazy pending affine
