"""Tensors between the host ACE-Step process and the TT DiT worker.

The two sides run different numpy/torch builds (ACE-Step venv vs the TT image), so tensors cross the socket as raw
little-endian bytes with dtype and shape, inside plain dicts that pickle identically on both sides."""
from __future__ import annotations

import torch

_DTYPES = {"float32": torch.float32}


def encode(t: torch.Tensor) -> dict:
    t = t.detach().to("cpu", torch.float32).contiguous()
    return {"dtype": "float32", "shape": list(t.shape), "data": t.numpy().tobytes()}


def decode(d: dict) -> torch.Tensor:
    return torch.frombuffer(bytearray(d["data"]), dtype=_DTYPES[d["dtype"]]).reshape(d["shape"])
