"""Worked affine forward/inverse examples (voxel index ↔ world mm).

Run:
    conda activate dl
    python det3d/run/tools/affine_voxel_world_demo.py
"""

import numpy as np
from nibabel.affines import apply_affine


def affine_scale_shift(sx, sy, sz, tx=0.0, ty=0.0, tz=0.0):
    return np.array(
        [
            [sx, 0, 0, tx],
            [0, sy, 0, ty],
            [0, 0, sz, tz],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def show(label, v, A):
    v4 = np.array([*v, 1.0], dtype=np.float64)
    p = apply_affine(A, v)
    back = apply_affine(np.linalg.inv(A), p)
    print(f"\n=== {label} ===")
    print("A =\n", np.array2string(A, precision=4, suppress_small=True))
    print("voxel v =", v)
    print("forward  p = Av     =", np.round(p, 4))
    print("inverse  v' = A^-1 p =", np.round(back, 4))
    print("roundtrip error", np.max(np.abs(back - v)))


def two_grids_same_world():
    A_native = np.eye(4, dtype=np.float64)
    A_spaced = affine_scale_shift(2, 2, 2)
    v_native = np.array([20.0, 0.0, 0.0])
    v_spaced = np.array([10.0, 0.0, 0.0])
    p_native = apply_affine(A_native, v_native)
    p_spaced = apply_affine(A_spaced, v_spaced)
    cross = apply_affine(np.linalg.inv(A_spaced), apply_affine(A_native, v_native))
    wrong = apply_affine(np.linalg.inv(A_spaced), v_spaced)
    print("\n=== two grids, same world ===")
    print("native voxel", v_native, "-> world", p_native)
    print("spaced voxel", v_spaced, "-> world", p_spaced)
    print("world delta mm", p_spaced - p_native)
    print("cross-grid v_B = A_B^-1 A_A v_A =", np.round(cross, 4))
    print("wrong: A_B^-1 @ v_B_as_if_world =", np.round(wrong, 4))


def preproc_crop_example():
    A = affine_scale_shift(0.8, 0.8, 1.5, -258.4, -196.0, 0.0)
    v = np.array([92.0, 13.0, 112.0])
    p = apply_affine(A, v)
    back = apply_affine(np.linalg.inv(A), p)
    invb_wrong = apply_affine(np.linalg.inv(A), v)
    print("\n=== preproc crop affine (InvB trap) ===")
    print("voxel v =", v)
    print("forward world p = Av =", np.round(p, 4))
    print("correct inverse v = A^-1 p =", np.round(back, 4))
    print("InvB mistake A^-1 v =", np.round(invb_wrong, 4))


def main():
    show("toy scale+shift", np.array([5.0, 4.0, 2.0]), affine_scale_shift(2, 2, 3, 10, 20, 0))
    two_grids_same_world()
    preproc_crop_example()


if __name__ == "__main__":
    main()
