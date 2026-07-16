"""LBD .pt volume I/O for Det patch/cascade (not hybrid RetinaNet)."""


def load_lbd_pt(path):
    import torch

    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        img = obj
    elif hasattr(obj, "as_tensor"):
        img = obj.as_tensor()
    else:
        img = obj
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return img.float()


def intensity_clip_range(project=None, plan=None):
    if project is not None:
        return project.global_properties["intensity_clip_range"]
    if plan is not None and "intensity_clip_range" in plan:
        return plan["intensity_clip_range"]
    return [-1024.0, 300.0]


def normalize_lbd_image(img, clip_range):
    a_min = float(clip_range[0])
    a_max = float(clip_range[1])
    img = img.clone()
    img = img.clamp(a_min, a_max)
    img = (img - a_min) / (a_max - a_min)
    return img


def full_volume_bounding_box(img):
    sh = tuple(int(v) for v in img.shape)
    if len(sh) == 4:
        depth, height, width = sh[1], sh[2], sh[3]
    elif len(sh) == 3:
        depth, height, width = sh[0], sh[1], sh[2]
    else:
        raise ValueError(f"Expected 3D or 4D LBD image, got shape {sh}")
    return (
        slice(0, 100, None),
        slice(0, depth),
        slice(0, height),
        slice(0, width),
    )


def load_lbd_pt_patch_data(pt_paths):
    from fran.inference.helpers import apply_bboxes, load_images_pt

    paths = [str(p) for p in pt_paths]
    data = load_images_pt(paths)
    bboxes = [full_volume_bounding_box(dat["image"]) for dat in data]
    return apply_bboxes(data, bboxes)
