"""Phase 5 deferred: mirror TTA transforms for detection inference."""

import torch


def mirror_tta_forward(model, image: torch.Tensor, forward_fn):
    """Run original + mirrored views; average logits (placeholder)."""
    outputs = [forward_fn(model, image)]
    for dim in (-1, -2):
        flipped = torch.flip(image, dims=[dim])
        out = forward_fn(model, flipped)
        outputs.append(torch.flip(out, dims=[dim]))
    return torch.stack(outputs, dim=0).mean(dim=0)
