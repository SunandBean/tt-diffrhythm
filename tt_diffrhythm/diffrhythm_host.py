"""Host (torch, CPU) side of the DiffRhythm 1.2 DiT port to one Tenstorrent P100a.

Upstream (DiffRhythm model/dit.py DiT.forward) as used by CFM.sample, per call:
  c = time_embed(t) + start_time_embed(start) + duration_time_embed(duration)                    [512]
  text_embed = TextEmbedding(text)  (embedding + sinusoid + 4 ConvNeXtV2 blocks; zeros in for drop_text)  [S, 512]
  h = proj(cat(x, cond, text_embed, style, c))  -> [S, 2048];  h = ConvPos(h) + h   (two k31 groups-16 convs + Mish)
  16 x LlamaDecoderLayer (RMSNorm, 32 x 64 heads, RoPE theta 1e4, SwiGLU 8192; no mask, bidirectional);
       after each of the first 8: h += silu(text_fusion_linears[i](text_embed))
  out = proj_out(LayerNorm(h) * (1 + scale) + shift),  scale, shift = linear(silu(c)).chunk(2)
The CFG branch (drop_audio_cond + drop_text) zeroes cond and the text tokens and uses the negative style.

Split: the timestep-invariant text embedding and the small time MLPs stay in the DiffRhythm process on the host
(upstream modules, exact); everything weight-heavy runs on the card. The input projection is split by input:
  h = x @ Wx + [cond @ Wcond + text_embed @ Wtext]  (per song and branch)  + [style @ Ws + c @ Wc + b]  (per call row)
RefDecoder runs this formulation in float32 so it can be checked against upstream without a device.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .common import (NEG, TILE, apply_rope_adjacent, interleave_pairs_permutation, linear_to_mm,
                     permute_heads_rows, rms, round_up, swiglu_interleave)

SEQ_BUCKET = 256  # frames padded to multiples of this: 96-285 s songs (2067-6136 frames) -> 16 shapes
LATENT_RATE = 44100 / 2048  # latent frames per second (get_lrc_token: int(seconds * 44100 / 2048))
MIN_FRAMES, MAX_FRAMES = int(96 * LATENT_RATE), int(285 * LATENT_RATE)  # the 96-285 s the port was exercised over


def seq_buckets():
    """Every padded length a DiffRhythm song can have (warm_cache.py compiles them ahead of time)."""
    return sorted({round_up(n, SEQ_BUCKET) for n in range(MIN_FRAMES, MAX_FRAMES + 1)})


@dataclass(frozen=True)
class Config:
    dim: int = 2048
    depth: int = 16
    heads: int = 32
    head_dim: int = 64
    intermediate: int = 8192
    text_dim: int = 512
    cond_dim: int = 512
    mel_dim: int = 64
    fusion_layers: int = 8
    conv_kernel: int = 31
    conv_groups: int = 16
    rope_theta: float = 10000.0
    eps: float = 1e-6


class Checkpoint:
    """transformer.* of DiffRhythm's cfm_model.pt (fp16 on disk), read as float32."""

    def __init__(self, path):
        sd = torch.load(str(path), weights_only=True, map_location="cpu")["model_state_dict"]
        self._sd = {k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.")}
        self.cfg = Config()

    def get(self, name: str) -> torch.Tensor:
        return self._sd[name].float()


@dataclass
class LayerWeights:
    wqkv: torch.Tensor  # [2048, 3 * 2048], q/k rows permuted to adjacent pairs
    wo: torch.Tensor
    w_gateup: torch.Tensor  # [2048, 2 * 8192] tile-interleaved
    w_down: torch.Tensor
    ln1: torch.Tensor
    ln2: torch.Tensor


def layer_weights(ckpt: Checkpoint, i: int) -> LayerWeights:
    c = ckpt.cfg
    g = lambda k: ckpt.get(f"transformer_blocks.{i}.{k}")
    perm = interleave_pairs_permutation(c.head_dim)
    wq = permute_heads_rows(g("self_attn.q_proj.weight"), c.heads, c.head_dim, perm)
    wk = permute_heads_rows(g("self_attn.k_proj.weight"), c.heads, c.head_dim, perm)
    return LayerWeights(
        wqkv=torch.cat([linear_to_mm(wq), linear_to_mm(wk), linear_to_mm(g("self_attn.v_proj.weight"))], dim=1),
        wo=linear_to_mm(g("self_attn.o_proj.weight")),
        w_gateup=swiglu_interleave(g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")),
        w_down=linear_to_mm(g("mlp.down_proj.weight")),
        ln1=g("input_layernorm.weight"), ln2=g("post_attention_layernorm.weight"))


@dataclass
class InputWeights:
    """input_embed.proj split by input (matmul layout [in, out])."""

    wx: torch.Tensor  # [64, 2048]
    wcond: torch.Tensor  # [64, 2048]
    wtext: torch.Tensor  # [512, 2048]
    wstyle: torch.Tensor  # [512, 2048]
    wc: torch.Tensor  # [512, 2048]
    b: torch.Tensor


def input_weights(ckpt: Checkpoint) -> InputWeights:
    c = ckpt.cfg
    w = linear_to_mm(ckpt.get("input_embed.proj.weight"))  # [1664, 2048]
    edges = [0, c.mel_dim, 2 * c.mel_dim, 2 * c.mel_dim + c.text_dim, 2 * c.mel_dim + c.text_dim + c.cond_dim]
    parts = [w[a:b] for a, b in zip(edges, edges[1:] + [w.shape[0]])]
    return InputWeights(*[p.contiguous() for p in parts], ckpt.get("input_embed.proj.bias"))


@dataclass(frozen=True)
class Geometry:
    frames: int

    @property
    def seq_pad(self) -> int:
        return round_up(self.frames, SEQ_BUCKET)


def key_mask(geo: Geometry) -> torch.Tensor:
    """Additive [S_pad, S_pad] mask excluding padded keys (upstream attends over exactly S frames)."""
    valid = torch.arange(geo.seq_pad)[None, :] < geo.frames
    return torch.where(valid, 0.0, NEG).expand(geo.seq_pad, -1).contiguous()


def row_mask(geo: Geometry) -> torch.Tensor:
    """[S_pad, 1]: 1 on real frames, 0 on padding (the position convs must see zeros past the song, like upstream)."""
    return (torch.arange(geo.seq_pad) < geo.frames).float()[:, None]


def rope_tables(seq_pad: int, cfg: Config, dtype=torch.float32):
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float64) / cfg.head_dim))
    ang = torch.outer(torch.arange(seq_pad, dtype=torch.float64), inv).float()
    return torch.cos(ang).repeat_interleave(2, -1).to(dtype), torch.sin(ang).repeat_interleave(2, -1).to(dtype)


def pad_rows(t: torch.Tensor, rows: int) -> torch.Tensor:
    out = torch.zeros(rows, t.shape[-1], dtype=t.dtype)
    out[: t.shape[0]] = t
    return out


@dataclass
class CallRows:
    """What changes per call: the style/time part of the input projection and the output modulation."""

    input_row: torch.Tensor  # [2048]
    out_scale: torch.Tensor  # [2048]
    out_shift: torch.Tensor  # [2048]


def call_rows(ckpt_or_iw, norm_linear_w, norm_linear_b, style: torch.Tensor, c: torch.Tensor) -> CallRows:
    """style [512], c [512] (time + start + duration embeddings, from upstream on the host)."""
    iw = ckpt_or_iw
    input_row = style @ iw.wstyle + c @ iw.wc + iw.b
    emb = torch.nn.functional.silu(c) @ norm_linear_w.t() + norm_linear_b
    scale, shift = emb.chunk(2, dim=-1)
    return CallRows(input_row, scale, shift)


class RefDecoder:
    """float32 CPU DiT in the TT formulation."""

    def __init__(self, ckpt: Checkpoint):
        self.ckpt, self.cfg = ckpt, ckpt.cfg
        c = self.cfg
        self.layers = [layer_weights(ckpt, i) for i in range(c.depth)]
        self.iw = input_weights(ckpt)
        self.conv = [(ckpt.get(f"input_embed.conv_pos_embed.conv1d.{i}.weight"),
                      ckpt.get(f"input_embed.conv_pos_embed.conv1d.{i}.bias")) for i in (0, 2)]
        self.fusion = [(linear_to_mm(ckpt.get(f"text_fusion_linears.{i}.0.weight")),
                        ckpt.get(f"text_fusion_linears.{i}.0.bias")) for i in range(c.fusion_layers)]
        self.w_out, self.b_out = linear_to_mm(ckpt.get("proj_out.weight")), ckpt.get("proj_out.bias")
        self.norm_w, self.norm_b = ckpt.get("norm_out.linear.weight"), ckpt.get("norm_out.linear.bias")

    def rows(self, style: torch.Tensor, c: torch.Tensor) -> CallRows:
        return call_rows(self.iw, self.norm_w, self.norm_b, style, c)

    def branch(self, text_embed: torch.Tensor, cond, geo: Geometry) -> Dict[str, object]:
        """Per song and CFG branch: [cond @ Wcond + text @ Wtext] and the 8 text-fusion residuals, padded."""
        t = pad_rows(text_embed.float(), geo.seq_pad)
        base = t @ self.iw.wtext
        if cond is not None:
            base = base + pad_rows(cond.float(), geo.seq_pad) @ self.iw.wcond
        fusion = [torch.nn.functional.silu(t @ w + b) for w, b in self.fusion]
        return {"base": base, "fusion": fusion}

    def conv_pos(self, h: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Two k31 convs with Mish. Upstream's sequence ends at the song, so each conv sees zeros past it: the rows
        past the song are zeroed before every conv (the first conv's Mish output is not zero there)."""
        c = self.cfg
        y = h
        for w, b in self.conv:
            if mask is not None:
                y = y * mask
            y = torch.nn.functional.mish(torch.nn.functional.conv1d(y.t()[None], w, b, padding=c.conv_kernel // 2,
                                                                    groups=c.conv_groups))[0].t()
        return y

    def forward(self, x_t: torch.Tensor, branch: Dict[str, object], rows: CallRows, geo: Geometry, taps: list = None):
        """[S, 64] noisy latents -> [S, 64] prediction."""
        c = self.cfg
        mask = row_mask(geo)
        h = (pad_rows(x_t.float(), geo.seq_pad) @ self.iw.wx + branch["base"] + rows.input_row) * mask
        h = self.conv_pos(h, mask) + h
        cos, sin = rope_tables(geo.seq_pad, c)
        kmask = key_mask(geo)
        for i, w in enumerate(self.layers):
            a = rms(h, w.ln1, c.eps)
            q, k, v = (a @ w.wqkv).split(c.dim, -1)
            q = apply_rope_adjacent(q.view(-1, c.heads, c.head_dim).transpose(0, 1), cos, sin)
            k = apply_rope_adjacent(k.view(-1, c.heads, c.head_dim).transpose(0, 1), cos, sin)
            v = v.view(-1, c.heads, c.head_dim).transpose(0, 1)
            scores = q @ k.transpose(-1, -2) * c.head_dim ** -0.5 + kmask
            att = (torch.softmax(scores, -1) @ v).transpose(0, 1).reshape(geo.seq_pad, -1)
            h = h + att @ w.wo
            m = rms(h, w.ln2, c.eps)
            gu = (m @ w.w_gateup).view(geo.seq_pad, -1, 2, TILE)
            h = h + (torch.nn.functional.silu(gu[:, :, 0]) * gu[:, :, 1]).reshape(geo.seq_pad, -1) @ w.w_down
            if i < c.fusion_layers:
                h = h + branch["fusion"][i]
            if taps is not None:
                taps.append(h.clone())
        h = torch.nn.functional.layer_norm(h, (c.dim,), eps=c.eps) * (1 + rows.out_scale) + rows.out_shift
        return (h @ self.w_out + self.b_out)[: geo.frames]
