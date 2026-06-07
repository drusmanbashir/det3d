import torch
from dot.transforms.transforms import NbrhoodsToPatchesd


def _ensure_4d(t):
    if t.dim() == 3:
        out = t.unsqueeze(0)
        if hasattr(t, "meta"):
            out.meta = t.meta
        return out
    return t


class NbrhoodsToPatchesOBDD(NbrhoodsToPatchesd):
    """N2P with strict lesion bbox crop (expand_by=0); 4D channel for downstream tfms."""

    def crop_patch_tensors(self, dat, bbox):
        out = super().crop_patch_tensors(dat, bbox)
        for key in self.keys:
            out[key] = _ensure_4d(out[key])
        return out
