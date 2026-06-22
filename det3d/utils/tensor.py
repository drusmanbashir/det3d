from collections import defaultdict

import torch


def plain_tensor(x):
    if hasattr(x, "as_tensor"):
        return x.as_tensor().contiguous()
    return torch.as_tensor(x).contiguous()


def to_numpy(inp):
    if isinstance(inp, (tuple, list)):
        return type(inp)(to_numpy(i) for i in inp)
    if isinstance(inp, dict) and not isinstance(inp, defaultdict):
        return type(inp)({k: to_numpy(i) for k, i in inp.items()})
    if isinstance(inp, torch.Tensor):
        t = plain_tensor(inp).detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()
    return inp
