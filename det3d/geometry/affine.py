import numpy as np
from nibabel.affines import apply_affine


def affine_to_numpy(affine):
    if hasattr(affine, "numpy"):
        return affine.numpy()
    return np.asarray(affine, dtype=np.float64)


def voxel_spacing_from_affine(affine):
    affine = affine_to_numpy(affine)
    return np.linalg.norm(affine[:3, :3], axis=0)


def voxel_bbox_to_world_box(bbox, affine):
    affine = affine_to_numpy(affine)
    starts = np.array(bbox[:3], dtype=np.float64)
    sizes_vox = np.array(bbox[3:], dtype=np.float64)
    spacing = voxel_spacing_from_affine(affine)
    center_vox = starts + 0.5 * sizes_vox
    center_world = apply_affine(affine, center_vox).tolist()
    size_world = (sizes_vox * spacing).tolist()
    return center_world + size_world
