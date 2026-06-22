"""Phase 5 deferred: Jaeger logit aggregation behind MONAI inferer shell."""

import torch


def jaeger_aggregate_logits(logit_maps: list[torch.Tensor], overlap_counts: torch.Tensor) -> torch.Tensor:
    """Placeholder Jaeger-style weighted aggregation."""
    stacked = torch.stack(logit_maps, dim=0)
    weights = overlap_counts.clamp(min=1).to(stacked.dtype)
    return stacked.sum(dim=0) / weights
