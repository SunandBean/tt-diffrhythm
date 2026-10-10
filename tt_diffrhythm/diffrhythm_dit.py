# SPDX-License-Identifier: Apache-2.0
"""DiffRhythm 1.2 DiT on ONE Blackhole (P100a), TTNN, batch 1.

Mirrors diffrhythm_host.RefDecoder op for op, with the block kernels of acestep_dit.py (fused QKV minimal_matmul,
adjacent-pair rotary_embedding_llama, fused SwiGLU, fp32-accumulated SDPA).

Per song (prepare): for each CFG branch, the timestep-invariant part of the input projection
[text_embed @ Wtext + cond @ Wcond] and the 8 text-fusion residuals silu(text_embed @ Wf + bf) are computed on the
device from the host's text embedding and kept there; plus the RoPE tables and the masks.
Per call (forward): x @ Wx + that base + the call's row (style/time), zero past the song, two grouped k31 convs with
Mish (zeroing the padding before each), 16 Llama layers (32 x 64 heads, padded keys excluded), the fusion residuals
after the first 8, then LayerNorm * (1 + scale) + shift and proj_out.

Speed (dit_profile.py, sdpa_check.py, conv_speed.py, mm_sweep.py): the attention excludes the padding with
block-diagonal windows [0, frames, S] built on the device instead of reading an [S, S] mask for every head (30 % faster
at 6144 frames); each position conv is one matmul over the 31 shifted rows (unfold) instead of ttnn's grouped conv,
which expands the 16 groups densely anyway; the matmuls use blockings measured for these shapes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import ttnn

from . import diffrhythm_host as host
from .common import MEM, Precision, _dev, _row, close_device, dram_stats, open_device  # noqa: F401
from .common import rot_transformation_mat

try:
    from models.tt_dit.utils.matmul import get_matmul_config
except ImportError:  # pragma: no cover - only on hosts without tt-metal models/
    get_matmul_config = None


@dataclass
class DRPrecision(Precision):
    """Precision on top of acestep_dit.Precision. CFG (pred + 4 (pred - null)) amplifies the error of the small
    difference between the two branches, and 31 steps accumulate it, so a DiT call that matches the GPU at PCC 0.9999
    still drifts over a song. Closed-loop check (device_check.py --closed-loop, 96 s song, final latents vs the GPU's;
    CPU float32 0.99997, CPU all-bfloat16 0.991): all bfloat16 0.970; the defaults below 0.994-0.995 for +1.2 s per
    song. Splitting the weights (bf16 hi + lo) or keeping the branch tensors in float32 gained nothing more
    (experiments/tt-diffrhythm/results/precision_sweep_d96.json)."""

    mm_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4
    x_fp32: bool = True  # upload the noisy latents in float32 (the input projection reads them)
    residual_fp32: bool = True  # keep the residual stream in float32 (branch outputs are added into it)
    out_fp32: bool = True  # the final projection writes float32
    rows_fp32: bool = True  # the per-call rows (style/time input row, output scale and shift) in float32: their
    # rounding is the same for every frame and step, so it adds up instead of averaging out
    consts_fp32: bool = True  # the same for the constant per-channel rows: RMSNorm weights and proj_out's bias
    windowed_attention: bool = True  # padded keys excluded by on-device windows (False: an [S, S] additive mask)
    unfold_conv: bool = True  # position convs as one matmul over the unfolded rows (False: ttnn.conv1d, groups=16)


# minimal_matmul blocks (M, K, N, (subblock h, w)) per (K, N) on the P100a's 11x10 grid (mm_sweep.py: 5-15 % faster
# than tt-metal's 8x8x8 default); a list of (largest M, blocks) where the best depends on the length
MM_BLOCKS = {
    (2048, 6144): [(None, (4, 8, 4, (4, 1)))],  # qkv
    (2048, 2048): [(None, (4, 4, 4, (4, 1)))],  # o
    (2048, 16384): [(3072, (8, 8, 8, (2, 2))), (None, (4, 8, 8, (4, 1)))],  # gate/up, fused SwiGLU
    (8192, 2048): [(None, (4, 8, 4, (4, 1)))],  # down
}


def conv_unfold_weight(w: torch.Tensor, groups: int) -> torch.Tensor:
    """Grouped conv1d weight [C, C / groups, k] -> matmul weight [k * C, C] for the unfolded input (row t holds the
    input rows t - k // 2 ... t + k // 2 side by side): block-diagonal per tap, row j * C + (g * cg + i), column
    g * cg + o."""
    C, cg, k = w.shape
    out = torch.zeros(k, C, C)
    wg = w.view(groups, cg, cg, k)  # g, o, i, tap
    for g in range(groups):
        out[:, g * cg:(g + 1) * cg, g * cg:(g + 1) * cg] = wg[g].permute(2, 1, 0)  # tap, i, o
    return out.reshape(k * C, C)


class Layer:
    def __init__(self, dev, w: host.LayerWeights, prec: Precision):
        wd = prec.weight_dtype
        self.wqkv = _dev(dev, w.wqkv.to(torch.bfloat16), wd)
        self.wo = _dev(dev, w.wo.to(torch.bfloat16), wd)
        self.w_gateup = _dev(dev, w.w_gateup.to(torch.bfloat16), wd)
        self.w_down = _dev(dev, w.w_down.to(torch.bfloat16), wd)
        if getattr(prec, "consts_fp32", False):
            self.ln1, self.ln2 = (_dev(dev, v.float().reshape(1, 1, 1, -1), ttnn.float32) for v in (w.ln1, w.ln2))
        else:
            self.ln1, self.ln2 = _row(dev, w.ln1), _row(dev, w.ln2)


@dataclass
class Song:
    """Device state of one generation: per-branch base and fusion residuals, tables and masks."""

    geo: host.Geometry
    branches: Dict[bool, dict]  # drop_text -> {"base": tensor, "fusion": [8 tensors]}
    cos: ttnn.Tensor
    sin: ttnn.Tensor
    key_mask: Optional[ttnn.Tensor]  # [S, S] additive mask, or None with windowed attention
    windows: Optional[ttnn.Tensor]  # cu_window_seqlens [0, frames, S] (or [0, S]) for windowed attention
    row_mask: ttnn.Tensor

    def tensors(self) -> List[ttnn.Tensor]:
        out = [t for t in (self.cos, self.sin, self.key_mask, self.windows, self.row_mask) if t is not None]
        for b in self.branches.values():
            out += [b["base"], *b["fusion"]]
        return out


class DiffRhythmDiT:
    def __init__(self, dev, ckpt: host.Checkpoint, prec: Optional[Precision] = None, keep_ref: bool = False):
        self.dev, self.cfg = dev, ckpt.cfg
        self.prec = prec or DRPrecision()
        self.res_dtype = ttnn.float32 if getattr(self.prec, "residual_fp32", False) else ttnn.bfloat16
        self.ref = host.RefDecoder(ckpt)  # host weights: layouts and the per-call rows
        self.grid = dev.compute_with_storage_grid_size()
        r = self.ref
        self.layers = [Layer(dev, w, self.prec) for w in r.layers]
        bf = lambda t: _dev(dev, t.to(torch.bfloat16))
        self.wx, self.wcond, self.wtext = bf(r.iw.wx), bf(r.iw.wcond), bf(r.iw.wtext)
        self.fusion = [(bf(w), _row(dev, b)) for w, b in r.fusion]
        self.unfold = getattr(self.prec, "unfold_conv", False)
        if self.unfold:
            self.conv = [(bf(conv_unfold_weight(w, self.cfg.conv_groups)), _row(dev, b)) for w, b in r.conv]
        else:
            self.conv = [(ttnn.from_torch(w.unsqueeze(2).contiguous(), dtype=ttnn.float32),
                          ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=ttnn.float32)) for w, b in r.conv]
        self.w_out, self.b_out = bf(r.w_out), _row(dev, r.b_out)
        self.b_out32 = (_dev(dev, r.b_out.float().reshape(1, 1, 1, -1), ttnn.float32)
                        if getattr(self.prec, "consts_fp32", False) else None)
        self.trans_mat = _dev(dev, rot_transformation_mat())
        arch = dev.arch()
        ck = lambda fid, fp32: ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False)
        self.ck_mm = ck(self.prec.mm_fidelity, True)
        self.ck_norm = ck(ttnn.MathFidelity.HiFi4, True)
        self.ck_sdpa = ck(self.prec.sdpa_fidelity, self.prec.sdpa_fp32_acc)
        self.ck_rope = ck(ttnn.MathFidelity.HiFi4, True)
        self.ck_conv = ck(ttnn.MathFidelity.HiFi4, True)
        self._mm_cfg_cache: Dict[tuple, object] = {}
        self.windowed = getattr(self.prec, "windowed_attention", False)
        if not self.unfold:
            # ttnn's conv moves its weights onto the device (in its own layout) on the first call. Done now, before
            # any song, so they do not land behind a song's tensors (the VAE's later allocations, and so its compiled
            # kernels, would then depend on the song length). The prepared weights serve every sequence bucket
            # (bit-identical to weights prepared at each length).
            S = host.seq_buckets()[0]
            z = _dev(dev, torch.zeros(1, 1, S, self.cfg.dim, dtype=torch.bfloat16))
            for i in range(2):
                ttnn.deallocate(self._conv(z, i, S))
            ttnn.deallocate(z)
        if not keep_ref:  # the per-call rows need only the small input/output weights on the host
            r.layers = None

    # ------------------------------------------------------------------ song
    def prepare(self, text_embed: Dict[bool, torch.Tensor], cond: torch.Tensor) -> Song:
        """text_embed {drop_text: [S, 512]} from upstream's TextEmbedding, cond [S, 64] -> device state."""
        c = self.cfg
        geo = host.Geometry(frames=cond.shape[0])
        S = geo.seq_pad
        branches = {}
        cond_d = _dev(self.dev, host.pad_rows(cond.float(), S).to(torch.bfloat16).reshape(1, 1, S, -1))
        for null, te in text_embed.items():
            t = _dev(self.dev, host.pad_rows(te.float(), S).to(torch.bfloat16).reshape(1, 1, S, -1))
            bd = ttnn.bfloat16
            base = ttnn.linear(t, self.wtext, compute_kernel_config=self.ck_mm, memory_config=MEM, dtype=bd)
            if not null:  # the CFG branch drops the audio condition (zeros)
                bc = ttnn.linear(cond_d, self.wcond, compute_kernel_config=self.ck_mm, memory_config=MEM, dtype=bd)
                base = self._add_free(base, bc)
            fusion = []
            for w, b in self.fusion:
                f = ttnn.linear(t, w, bias=b, compute_kernel_config=self.ck_mm, memory_config=MEM, dtype=bd)
                fusion.append(ttnn.silu(f, memory_config=MEM))
                ttnn.deallocate(f)
            ttnn.deallocate(t)
            branches[null] = {"base": base, "fusion": fusion}
        ttnn.deallocate(cond_d)
        cos, sin = host.rope_tables(S, c)
        f = lambda m: _dev(self.dev, m.to(torch.bfloat16)[None, None])
        rows = host.row_mask(geo).expand(S, c.dim)
        if self.windowed:
            bounds = [0, geo.frames, S] if geo.frames < S else [0, S]
            windows = ttnn.from_torch(torch.tensor(bounds, dtype=torch.int32), dtype=ttnn.int32,
                                      layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev)
            key_mask = None
        else:
            windows, key_mask = None, f(host.key_mask(geo))
        return Song(geo, branches, f(cos), f(sin), key_mask, windows, f(rows))

    def release(self, song: Song):
        for t in song.tensors():
            ttnn.deallocate(t)

    # ------------------------------------------------------------------ ops
    def _mm_config(self, M, K, N):
        for max_m, (mb, kb, nb, (sh, sw)) in MM_BLOCKS.get((K, N), []):
            if max_m is None or M <= max_m:
                return ttnn.MinimalMatmulConfig(M_block_size=mb, K_block_size=kb, N_block_size=nb, subblock_h=sh,
                                                subblock_w=sw, compute_with_storage_grid_size=self.grid)
        return get_matmul_config(M, K, N, self.grid)

    def _mm(self, x, w, M, K, N, fuse_swiglu=False, dtype=ttnn.bfloat16):
        key = (M, K, N)
        if key not in self._mm_cfg_cache:
            self._mm_cfg_cache[key] = self._mm_config(M, K, N)
        out = ttnn.experimental.minimal_matmul(x, w, config=self._mm_cfg_cache[key], compute_kernel_config=self.ck_mm,
                                               dtype=dtype, memory_config=MEM, fuse_swiglu=fuse_swiglu)
        if len(out.shape) != 4:
            out = ttnn.reshape(out, [1, 1, M, out.shape[-1]])
        return out

    @staticmethod
    def _add_free(a, b, dtype=None):
        out = ttnn.add(a, b, memory_config=MEM, dtype=dtype) if dtype is not None else ttnn.add(a, b, memory_config=MEM)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        return out

    def _residual(self, h, o):
        """h + o in the residual stream's dtype; frees both."""
        if self.res_dtype == ttnn.bfloat16:
            return self._add_free(h, o)
        if o.dtype != ttnn.float32:
            o32 = ttnn.typecast(o, ttnn.float32)
            ttnn.deallocate(o)
            o = o32
        return self._add_free(h, o)

    def _bf16(self, x):
        """x as bfloat16 for a matmul; frees x when it had to be converted."""
        if x.dtype == ttnn.bfloat16:
            return x
        out = ttnn.typecast(x, ttnn.bfloat16)
        ttnn.deallocate(x)
        return out

    def _rms(self, x, w):
        n = ttnn.rms_norm(x, epsilon=self.cfg.eps, weight=w, compute_kernel_config=self.ck_norm, memory_config=MEM)
        return self._bf16(n)

    def _mish(self, x):
        if hasattr(ttnn, "mish"):
            return ttnn.mish(x, memory_config=MEM)
        sp = ttnn.softplus(x, memory_config=MEM)
        th = ttnn.tanh(sp, memory_config=MEM)
        ttnn.deallocate(sp)
        out = ttnn.multiply(x, th, memory_config=MEM)
        ttnn.deallocate(th)
        return out

    def _conv(self, x, i, S):
        if self.unfold:
            return self._conv_unfold(x, i, S)
        c = self.cfg
        w, b = self.conv[i]
        cfg = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16, config_tensors_in_dram=True)
        out, out_len, (w, b) = ttnn.conv1d(
            input_tensor=x, weight_tensor=w, bias_tensor=b, device=self.dev, in_channels=c.dim, out_channels=c.dim,
            batch_size=1, input_length=S, kernel_size=c.conv_kernel, stride=1, padding=c.conv_kernel // 2, dilation=1,
            groups=c.conv_groups, dtype=ttnn.bfloat16, compute_config=self.ck_conv, conv_config=cfg,
            return_output_dim=True, return_weights_and_bias=True)
        self.conv[i] = (w, b)  # prepared on the device by the first call
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return ttnn.reshape(out, [1, 1, out_len, c.dim])

    def _conv_unfold(self, x, i, S):
        """[1, 1, S, C] -> the same conv as one matmul: rows padded by k // 2 zeros at both ends, the k shifted
        copies side by side ([S, k * C]), times the block-diagonal weight."""
        c = self.cfg
        k, pad = c.conv_kernel, c.conv_kernel // 2
        w, b = self.conv[i]
        xr = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [S, c.dim])
        xp = ttnn.pad(xr, [(pad, pad), (0, 0)], 0.0)
        ttnn.deallocate(xr)
        parts = [ttnn.slice(xp, [j, 0], [j + S, c.dim]) for j in range(k)]
        ttnn.deallocate(xp)
        xu = ttnn.concat(parts, dim=1, memory_config=MEM)
        for t in parts:
            ttnn.deallocate(t)
        xt = ttnn.to_layout(ttnn.reshape(xu, [1, 1, S, k * c.dim]), ttnn.TILE_LAYOUT)
        ttnn.deallocate(xu)
        y = self._mm(xt, w, S, k * c.dim, c.dim)
        ttnn.deallocate(xt)
        out = ttnn.add(y, b, memory_config=MEM)
        ttnn.deallocate(y)
        return out

    def _conv_pos(self, h, song: Song, S):
        """Two k31 convs with Mish; the rows past the song are zeroed before each (upstream's sequence ends there)."""
        y = h
        for i in range(2):
            m = ttnn.multiply(y, song.row_mask, memory_config=MEM)
            if y is not h:
                ttnn.deallocate(y)
            conv = self._conv(m, i, S)
            ttnn.deallocate(m)
            y = self._mish(conv)
            ttnn.deallocate(conv)
        return y

    def _sdpa(self, q, k, v, song: Song, S):
        # sdpa_check.py: 256 x 256 chunks are fastest from 3072 frames on; at 2560 windowed attention prefers 128-row Q
        # chunks (S is a multiple of SEQ_BUCKET = 256, so both divide it)
        q_chunk = 128 if self.windowed and S <= 2560 else 256
        cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=self.grid, q_chunk_size=q_chunk, k_chunk_size=256,
                                     exp_approx_mode=False)
        exclude = {"cu_window_seqlens": song.windows} if self.windowed else {"attn_mask": song.key_mask}
        return ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=self.cfg.head_dim ** -0.5, program_config=cfg,
            compute_kernel_config=self.ck_sdpa, memory_config=MEM, **exclude)

    def _attention(self, h, L: Layer, S, song: Song):
        c = self.cfg
        qkv = self._mm(h, L.wqkv, S, c.dim, 3 * c.dim)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=c.heads, num_kv_heads=c.heads,
                                                         transpose_k_heads=False, memory_config=MEM)
        ttnn.deallocate(qkv)
        roped = []
        for t in (q, k):
            roped.append(ttnn.experimental.rotary_embedding_llama(t, song.cos, song.sin, self.trans_mat,
                                                                  is_decode_mode=False, compute_kernel_config=self.ck_rope))
            ttnn.deallocate(t)
        a = self._sdpa(roped[0], roped[1], v, song, S)
        for t in (*roped, v):
            ttnn.deallocate(t)
        a2 = ttnn.transformer.concatenate_heads(a, memory_config=MEM)
        ttnn.deallocate(a)
        o = self._mm(a2, L.wo, S, c.dim, c.dim, dtype=self.res_dtype)  # added into the residual stream as is
        ttnn.deallocate(a2)
        return o

    # ------------------------------------------------------------------ public
    def forward(self, x_t: torch.Tensor, drop_text: bool, style: torch.Tensor, c_emb: torch.Tensor, song: Song,
                taps: Optional[list] = None) -> torch.Tensor:
        """[S, 64] noisy latents, the branch, style [512] and c = time + start + duration embeddings [512]
        -> [S, 64] float32 prediction."""
        c, geo = self.cfg, song.geo
        S = geo.seq_pad
        branch = song.branches[drop_text]
        rows = self.ref.rows(style.float(), c_emb.float())
        if getattr(self.prec, "rows_fp32", False):
            row = lambda v: _dev(self.dev, v.float().reshape(1, 1, 1, -1), ttnn.float32)
        else:
            row = lambda v: _row(self.dev, v)
        in_row, gamma, shift = row(rows.input_row), row(1 + rows.out_scale), row(rows.out_shift)
        p = self.prec
        xs = host.pad_rows(x_t.float(), S).reshape(1, 1, S, -1)
        if getattr(p, "x_fp32", False):
            x = _dev(self.dev, xs, ttnn.float32)
        else:
            x = _dev(self.dev, xs.to(torch.bfloat16))
        xw = ttnn.linear(x, self.wx, compute_kernel_config=self.ck_mm, memory_config=MEM, dtype=self.res_dtype)
        ttnn.deallocate(x)
        h0 = ttnn.add(xw, branch["base"], memory_config=MEM, dtype=self.res_dtype)
        ttnn.deallocate(xw)
        h1 = ttnn.add(h0, in_row, memory_config=MEM, dtype=self.res_dtype)
        ttnn.deallocate(h0)
        h = ttnn.multiply(h1, song.row_mask, memory_config=MEM, dtype=self.res_dtype)
        ttnn.deallocate(h1)
        conv_in = ttnn.typecast(h, ttnn.bfloat16) if self.res_dtype != ttnn.bfloat16 else h
        pos = self._conv_pos(conv_in, song, S)
        if conv_in is not h:
            ttnn.deallocate(conv_in)
        h = self._residual(h, pos)
        tap = (lambda v: taps.append(ttnn.to_torch(v)[0, 0].float())) if taps is not None else (lambda v: None)
        for i, L in enumerate(self.layers):
            a = self._rms(h, L.ln1)
            o = self._attention(a, L, S, song)
            ttnn.deallocate(a)
            h = self._residual(h, o)
            m = self._rms(h, L.ln2)
            gu = self._mm(m, L.w_gateup, S, c.dim, 2 * c.intermediate, fuse_swiglu=True)
            ttnn.deallocate(m)
            d = self._mm(gu, L.w_down, S, c.intermediate, c.dim, dtype=self.res_dtype)
            ttnn.deallocate(gu)
            h = self._residual(h, d)
            if i < c.fusion_layers:
                h2 = ttnn.add(h, branch["fusion"][i], memory_config=MEM, dtype=self.res_dtype)
                ttnn.deallocate(h)
                h = h2
            tap(h)
        n = ttnn.layer_norm(h, epsilon=c.eps, compute_kernel_config=self.ck_norm, memory_config=MEM)
        ttnn.deallocate(h)
        g = ttnn.multiply(n, gamma, memory_config=MEM)
        ttnn.deallocate(n)
        mod = self._bf16(ttnn.add(g, shift, memory_config=MEM))
        ttnn.deallocate(g)
        if self.b_out32 is not None:
            o = ttnn.linear(mod, self.w_out, compute_kernel_config=self.ck_mm, memory_config=MEM, dtype=ttnn.float32)
            out = ttnn.add(o, self.b_out32, memory_config=MEM)
            ttnn.deallocate(o)
        else:
            out = ttnn.linear(mod, self.w_out, bias=self.b_out, compute_kernel_config=self.ck_mm, memory_config=MEM,
                              dtype=ttnn.float32 if getattr(p, "out_fp32", False) else ttnn.bfloat16)
        ttnn.deallocate(mod)
        host_out = ttnn.to_torch(out)[0, 0, :geo.frames, : c.mel_dim].float()
        for t in (out, in_row, gamma, shift):
            ttnn.deallocate(t)
        return host_out
