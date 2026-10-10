# SPDX-License-Identifier: Apache-2.0
"""Device and host helpers shared with the ACE-Step port.

Extracted verbatim from `tt-acestep` (Apache-2.0), where the same code opens the card, reports
DRAM, moves tensors, and prepares checkpoint tensors for ttnn: the matmul layout, the
adjacent-pair head permutation ttnn's rotary kernel expects, and the tiled rotation matrix.

The two ports also share `worker.py`, `remote.py`, `wire.py` and `oobleck_vae.py`; this module
exists so that this repository stands alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

MEM = ttnn.DRAM_MEMORY_CONFIG


def open_device(l1_small_size: int = 98304):  # convs (VAE) keep their config tensors in L1_SMALL
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1),
                                 dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
                                 l1_small_size=l1_small_size)


def close_device(dev):
    ttnn.close_mesh_device(dev)


def dram_stats(dev) -> dict:
    try:
        s = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        return {"allocated_bytes": s.total_bytes_allocated_per_bank * s.num_banks,
                "free_bytes": s.total_bytes_free_per_bank * s.num_banks}
    except Exception:  # pragma: no cover - diagnostics only
        return {}


@dataclass
class Precision:
    weight_dtype: ttnn.DataType = ttnn.bfloat16
    mm_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    sdpa_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4
    sdpa_fp32_acc: bool = True


def _dev(dev, t: torch.Tensor, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=MEM)


def _row(dev, v: torch.Tensor):
    """[D] -> [1, 1, 1, D] bf16 row (norm gamma, shift, gate, bias)."""
    return _dev(dev, v.to(torch.bfloat16).reshape(1, 1, 1, -1))


TILE = 32
SEQ_MULTIPLE = 128  # sequence lengths are padded to multiples of this at least (SDPA chunk sizes divide it)


NEG = -1e9  # additive mask value for excluded keys (finite, representable in bf16)


def linear_to_mm(w: torch.Tensor) -> torch.Tensor:
    """nn.Linear weight [out, in] -> matmul weight [in, out]."""
    return w.t().contiguous()


def interleave_pairs_permutation(head_dim: int) -> torch.Tensor:
    """new[j] = old[p[j]]: rotate-half pairs (i, i + D/2) -> adjacent pairs (2i, 2i + 1).

    Applying it to the rows of q_proj/k_proj (per head) and to q_norm/k_norm leaves every q.k product unchanged and
    lets tt's rotary_embedding_llama (adjacent pairs) replace upstream's rotate_half RoPE."""
    half = head_dim // 2
    p = torch.empty(head_dim, dtype=torch.long)
    p[0::2] = torch.arange(half)
    p[1::2] = torch.arange(half) + half
    return p


def permute_heads_rows(w_out_in: torch.Tensor, n_heads: int, head_dim: int, perm: torch.Tensor) -> torch.Tensor:
    out, inp = w_out_in.shape
    assert out == n_heads * head_dim
    return w_out_in.view(n_heads, head_dim, inp)[:, perm, :].reshape(out, inp).contiguous()


def rot_transformation_mat(tile: int = TILE) -> torch.Tensor:
    """[1, 1, 32, 32] T with (x @ T)[2k] = -x[2k+1], (x @ T)[2k+1] = x[2k] (adjacent-pair rotation)."""
    m = torch.zeros(1, 1, tile, tile)
    m[..., torch.arange(0, tile, 2), torch.arange(1, tile, 2)] = 1.0
    m[..., torch.arange(1, tile, 2), torch.arange(0, tile, 2)] = -1.0
    return m


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def round_up(n: int, m: int = SEQ_MULTIPLE) -> int:
    return (n + m - 1) // m * m


def swiglu_interleave(gate_out_in: torch.Tensor, up_out_in: torch.Tensor, tile: int = TILE) -> torch.Tensor:
    """[K, 2N] weight for minimal_matmul(fuse_swiglu=True): column tile 2p = gate tile p, 2p+1 = up tile p."""
    g, u = linear_to_mm(gate_out_in), linear_to_mm(up_out_in)
    K, N = g.shape
    assert N % tile == 0, N
    return torch.stack([g.view(K, N // tile, tile), u.view(K, N // tile, tile)], dim=2).reshape(K, 2 * N).contiguous()


def apply_rope_adjacent(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
    return x * cos + rot * sin
