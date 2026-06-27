from collections import defaultdict

import torch

_NUMPY_UNSUPPORTED_FLOAT = frozenset({torch.float16, torch.bfloat16})


def plain_tensor(x):
    if hasattr(x, "as_tensor"):
        return x.as_tensor().contiguous()
    return torch.as_tensor(x).contiguous()


def sanitize_tensor_for_numpy(t: torch.Tensor) -> torch.Tensor:
    """CPU tensor safe for ``.numpy()`` (bf16/fp16 floats → float32)."""
    t = plain_tensor(t).detach().cpu()
    if t.is_floating_point() and t.dtype in _NUMPY_UNSUPPORTED_FLOAT:
        t = t.float()
    return t


def sanitize_for_numpy(inp):
    """Recursively sanitize tensors in nested batch/pred dicts for numpy export."""
    if isinstance(inp, (tuple, list)):
        return type(inp)(sanitize_for_numpy(i) for i in inp)
    if isinstance(inp, dict) and not isinstance(inp, defaultdict):
        return type(inp)({k: sanitize_for_numpy(i) for k, i in inp.items()})
    if isinstance(inp, torch.Tensor):
        return sanitize_tensor_for_numpy(inp)
    return inp


def to_numpy(inp):
    if isinstance(inp, (tuple, list)):
        return type(inp)(to_numpy(i) for i in inp)
    if isinstance(inp, dict) and not isinstance(inp, defaultdict):
        return type(inp)({k: to_numpy(i) for k, i in inp.items()})
    if isinstance(inp, torch.Tensor):
        return sanitize_tensor_for_numpy(inp).numpy()
    return inp
