import torch
from monai.transforms import MapTransform, SpatialCrop
from utilz.stringz import ast_literal_eval


def _ensure_4d(t):
    if t.dim() == 3:
        out = t.unsqueeze(0)
        if hasattr(t, "meta"):
            out.meta = t.meta
        return out
    return t


class NbrhoodsToPatchesOBDD(MapTransform):
    """N2P with strict lesion bbox crop (expand_by=0); 4D channel for downstream tfms."""

    def __init__(  #AI
        self,
        keys,
        expand_mode="mm",
        expand_by=0,
        nbrhoods_key="nbrhoods",
        bbox_key="bbox",
        nbrhood_outkey="nbrhood",
        allow_missing_keys=False,
    ):
        """Configure neighborhood cropping and optional bbox expansion."""
        super().__init__(keys, allow_missing_keys)
        self.nbrhoods_key = nbrhoods_key
        self.bbox_key = bbox_key
        self.nbrhood_outkey = nbrhood_outkey
        assert expand_mode in ["mm", "pct"]
        self.expand_mode = expand_mode
        self.expand_by = (
            [expand_by] * 3 if isinstance(expand_by, int | float) else expand_by
        )

    def expanded_bbox(self, dat, row):  #AI
        """Convert an ITK start-size bbox to expanded spatial slices."""
        bbox = ast_literal_eval(row["bbox"])
        if self.expand_mode == "pct":
            expand_vox = [int(size * ex) for size, ex in zip(bbox[3:], self.expand_by)]
        else:
            spacing = torch.linalg.norm(dat[self.keys[0]].meta["affine"][:3, :3], dim=0)
            expand_vox = [int(ex / sp) for ex, sp in zip(self.expand_by, spacing)]
        expand_before = [ex // 2 for ex in expand_vox]
        expand_after = [ex - before for ex, before in zip(expand_vox, expand_before)]
        roi = tuple(
            slice(max(0, start - before), start + size + after)
            for start, size, before, after in zip(
                bbox[:3], bbox[3:], expand_before, expand_after
            )
        )
        return roi

    def crop_patch_tensors(self, dat, bbox):  #AI
        """Crop each image and label map while retaining its channel dimension."""
        crop = SpatialCrop(roi_slices=bbox)
        out = {self.bbox_key: bbox}
        for key in self.key_iterator(dat):
            out[key] = crop(_ensure_4d(dat[key])).contiguous()
        return out

    def __call__(self, data):  #AI
        """Create one patch and its lesion statistics per neighborhood row."""
        dat = dict(data)
        outputs = []
        for _, row in dat[self.nbrhoods_key].iterrows():
            bbox = self.expanded_bbox(dat, row)
            out = self.crop_patch_tensors(dat, bbox)
            stats = row.to_dict()
            del stats["bbox"]
            out[self.nbrhood_outkey] = stats
            outputs.append(out)
        return outputs
