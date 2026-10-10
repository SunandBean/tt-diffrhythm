# SPDX-License-Identifier: Apache-2.0
"""Oobleck audio VAE decoders on ONE Blackhole (P100a), TTNN: ACE-Step's (diffusers AutoencoderOobleck, 48 kHz
stereo, strides 10, 6, 4, 4, 2) and DiffRhythm's (stable-audio-tools, TorchScript, 44.1 kHz stereo, strides
8, 8, 4, 4, 2). Same architecture, different weight files:

  conv1 (64 -> 2048, k7)
  5 x block: Snake -> ConvTranspose1d (k = 2s, stride s) -> 3 x residual unit (dilation 1, 3, 9):
             x + conv_k1(Snake(conv_k7_dilated(Snake(x))))
  Snake -> conv2 (128 -> 2, k7, no bias)
Snake(x) = x + 1 / (exp(beta) + 1e-9) * sin(exp(alpha) * x)^2 per channel. Weight norm is folded on the host
(diffusers) or already folded (the TorchScript export).

Activations are [1, 1, L, C] (channels last), as ttnn.conv1d / conv_transpose2d (height 1) expect.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import ttnn

MEM = ttnn.DRAM_MEMORY_CONFIG


def fold_weight_norm(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """torch weight_norm(dim=0): w = g * v / ||v|| over every dim but the first."""
    norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
    return g * v / norm


class VaeWeights:
    """decoder.* of checkpoints/vae, float32, weight norm folded."""

    def __init__(self, vae_dir):
        from safetensors import safe_open

        self.dir = Path(vae_dir)
        self.config = json.loads((self.dir / "config.json").read_text())
        self._f = safe_open(str(self.dir / "diffusion_pytorch_model.safetensors"), framework="pt")
        self.keys = set(self._f.keys())

    def get(self, name):
        return self._f.get_tensor(f"decoder.{name}").float()

    def conv(self, name):
        w = fold_weight_norm(self.get(f"{name}.weight_g"), self.get(f"{name}.weight_v"))
        b = self.get(f"{name}.bias") if f"decoder.{name}.bias" in self.keys else None
        return w, b

    def snake(self, name):
        alpha = torch.exp(self.get(f"{name}.alpha")).flatten()
        inv_beta = 1.0 / (torch.exp(self.get(f"{name}.beta")).flatten() + 1e-9)
        return alpha, inv_beta


class StableAudioVaeWeights:
    """decoder.* of DiffRhythm's vae_model.pt (a stable-audio-tools OobleckDecoder exported with TorchScript, weight
    norm already folded), under VaeWeights' diffusers names."""

    def __init__(self, path):
        sd = torch.jit.load(str(path), map_location="cpu").state_dict()
        self._sd = {k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")}
        strides = [self._sd[f"layers.{i + 1}.layers.1.weight"].shape[-1] // 2 for i in range(5)]
        self.config = {"downsampling_ratios": strides}

    @staticmethod
    def _key(name):
        parts = name.split(".")
        if name == "conv1":
            return "layers.0"
        if name == "snake1":
            return "layers.6"
        if name == "conv2":
            return "layers.7"
        block = f"layers.{int(parts[1]) + 1}"
        if parts[2] == "snake1":
            return f"{block}.layers.0"
        if parts[2] == "conv_t1":
            return f"{block}.layers.1"
        unit = f"{block}.layers.{int(parts[2][len('res_unit'):]) + 1}"
        return f"{unit}.layers.{ {'snake1': 0, 'conv1': 1, 'snake2': 2, 'conv2': 3}[parts[3]] }"

    def conv(self, name):
        k = self._key(name)
        return self._sd[f"{k}.weight"].float(), (self._sd[f"{k}.bias"].float() if f"{k}.bias" in self._sd else None)

    def snake(self, name):
        k = self._key(name)
        alpha = torch.exp(self._sd[f"{k}.alpha"].float()).flatten()
        inv_beta = 1.0 / (torch.exp(self._sd[f"{k}.beta"].float()).flatten() + 1e-9)
        return alpha, inv_beta


def vae_weights(path):
    """A diffusers VAE directory (ACE-Step) or a stable-audio TorchScript file (DiffRhythm)."""
    path = Path(path)
    if path.is_dir() and (path / "config.json").exists():
        return VaeWeights(path)
    return StableAudioVaeWeights(path / "vae_model.pt" if path.is_dir() else path)


@dataclass
class Conv:
    weight: object  # ttnn host tensor, replaced by the prepared device tensor after the first call
    bias: object
    in_ch: int
    out_ch: int
    kernel: int
    pad: int
    dilation: int = 1
    stride: int = 1
    transpose: bool = False


@dataclass
class Snake:
    alpha: ttnn.Tensor
    inv_beta: ttnn.Tensor


@dataclass
class ResUnit:
    snake1: Snake
    conv1: Conv
    snake2: Snake
    conv2: Conv


@dataclass
class Block:
    snake: Snake
    up: Conv
    units: List[ResUnit]


@dataclass
class Precision:
    fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4  # HiFi2/LoFi are no faster here: the convs are not math-bound
    act_dtype: ttnn.DataType = ttnn.bfloat16
    # vae_conv_sweep.py: activation blocks of 64 rows make the 48 kHz k7 convs 4.9x and the output conv 6x faster
    # than the default block height, same output; double buffering and bfp8 weights were slower. Applied to the
    # layers with <= 256 channels (the long, late ones); the wide early layers keep the default (the override fails
    # or is slower there).
    act_block_h: int = 64
    transpose_act_block_h: int = 64
    act_block_h_max_channels: int = 256
    config_tensors_in_dram: bool = True
    fused_snake: bool = True


class OobleckTT:
    def __init__(self, dev, vae_path, prec: Optional[Precision] = None):
        self.dev = dev
        self.prec = prec or Precision()
        w = vae_weights(vae_path)
        self.upsample = math.prod(w.config["downsampling_ratios"])
        self.conv1 = self._conv(*w.conv("conv1"), pad=3)
        self.blocks = []
        for i in range(len(w.config["downsampling_ratios"])):
            p = f"block.{i}"
            wt, bt = w.conv(f"{p}.conv_t1")  # ConvTranspose1d weight [in, out, 2s]
            stride = wt.shape[-1] // 2
            units = []
            for j, dilation in enumerate((1, 3, 9)):
                u = f"{p}.res_unit{j + 1}"
                units.append(ResUnit(self._snake(*w.snake(f"{u}.snake1")),
                                     self._conv(*w.conv(f"{u}.conv1"), pad=3 * dilation, dilation=dilation),
                                     self._snake(*w.snake(f"{u}.snake2")), self._conv(*w.conv(f"{u}.conv2"), pad=0)))
            self.blocks.append(Block(self._snake(*w.snake(f"{p}.snake1")),
                                     self._conv(wt, bt, pad=math.ceil(stride / 2), stride=stride, transpose=True), units))
        self.snake_out = self._snake(*w.snake("snake1"))
        self.conv2 = self._conv(*w.conv("conv2"), pad=3)
        self.ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=self.prec.fidelity,
                                                         math_approx_mode=False, fp32_dest_acc_en=True,
                                                         packer_l1_acc=False)
        self.profile = None  # a list to record (where, op, length, channels, seconds) per op, synchronizing each
        self._where = ""

    def _timed(self, op, length, channels, fn, *args):
        if self.profile is None:
            return fn(*args)
        import time

        ttnn.synchronize_device(self.dev)
        started = time.perf_counter()
        out = fn(*args)
        ttnn.synchronize_device(self.dev)
        self.profile.append((self._where, op, length, channels, time.perf_counter() - started))
        return out

    # ------------------------------------------------------------------ weights
    def _conv(self, w, b, pad, dilation=1, stride=1, transpose=False):
        if transpose:  # [in, out, k] -> [in, out, 1, k]
            in_ch, out_ch, k = w.shape
        else:  # [out, in, k] -> [out, in, 1, k]
            out_ch, in_ch, k = w.shape
        weight = ttnn.from_torch(w.unsqueeze(2).contiguous(), dtype=ttnn.float32)
        bias = ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=ttnn.float32) if b is not None else None
        return Conv(weight, bias, in_ch, out_ch, k, pad, dilation, stride, transpose)

    def _snake(self, alpha, inv_beta):
        row = lambda v: ttnn.from_torch(v.float().reshape(1, 1, 1, -1), dtype=self.prec.act_dtype,
                                        layout=ttnn.TILE_LAYOUT, device=self.dev, memory_config=MEM)
        return Snake(row(alpha), row(inv_beta))

    # ------------------------------------------------------------------ ops
    def _run_conv(self, x, c: Conv, length: int):
        block_h = self.prec.transpose_act_block_h if c.transpose else self.prec.act_block_h
        if max(c.in_ch, c.out_ch) > self.prec.act_block_h_max_channels:
            block_h = 0
        # smaller activation blocks need larger per-core config tensors; keep them in DRAM, not the 96 KiB L1_SMALL
        cfg = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16, config_tensors_in_dram=self.prec.config_tensors_in_dram)
        if block_h:
            cfg.act_block_h_override = block_h
        common = dict(device=self.dev, in_channels=c.in_ch, out_channels=c.out_ch, batch_size=1, bias_tensor=c.bias,
                      dtype=self.prec.act_dtype, compute_config=self.ck, conv_config=cfg,
                      return_weights_and_bias=True)
        if c.transpose:
            out, out_len, (c.weight, c.bias) = ttnn.conv_transpose2d(
                input_tensor=x, weight_tensor=c.weight, input_height=1, input_width=length, kernel_size=(1, c.kernel),
                stride=(1, c.stride), padding=(0, c.pad), output_padding=(0, 0), dilation=(1, 1), groups=1,
                return_output_dim=True, **common)
            out_len = out_len[1] if isinstance(out_len, (tuple, list)) else out_len
        else:
            out, out_len, (c.weight, c.bias) = ttnn.conv1d(
                input_tensor=x, weight_tensor=c.weight, input_length=length, kernel_size=c.kernel, stride=1,
                padding=c.pad, dilation=c.dilation, groups=1, return_output_dim=True, **common)
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return ttnn.reshape(out, [1, 1, out_len, c.out_ch]), out_len

    def _snake_op(self, x, s: Snake):
        if self.prec.fused_snake:  # sin(alpha x)^2 in one multiply, then x + inv_beta * that in one addcmul: 2 passes
            unary = lambda op: ttnn.UnaryWithParam(op)
            sq = ttnn.multiply(x, s.alpha, activations=[unary(ttnn.UnaryOpType.SIN), unary(ttnn.UnaryOpType.SQUARE)],
                               memory_config=MEM)
            out = ttnn.addcmul(x, sq, s.inv_beta, value=1.0, memory_config=MEM)
            ttnn.deallocate(sq)
            return out
        ax = ttnn.multiply(x, s.alpha, memory_config=MEM)
        sn = ttnn.sin(ax, memory_config=MEM)
        ttnn.deallocate(ax)
        sq = ttnn.multiply(sn, sn, memory_config=MEM)
        ttnn.deallocate(sn)
        t = ttnn.multiply(sq, s.inv_beta, memory_config=MEM)
        ttnn.deallocate(sq)
        out = ttnn.add(x, t, memory_config=MEM)
        ttnn.deallocate(t)
        return out

    def _unit(self, x, u: ResUnit, length):
        c = u.conv1.out_ch
        h = self._timed("snake", length, c, self._snake_op, x, u.snake1)
        y, _ = self._timed(f"conv_k7_d{u.conv1.dilation}", length, c, self._run_conv, h, u.conv1, length)
        ttnn.deallocate(h)
        h = self._timed("snake", length, c, self._snake_op, y, u.snake2)
        ttnn.deallocate(y)
        y, _ = self._timed("conv_k1", length, c, self._run_conv, h, u.conv2, length)
        ttnn.deallocate(h)
        out = self._timed("add", length, c, lambda a, b: ttnn.add(a, b, memory_config=MEM), x, y)
        ttnn.deallocate(y)
        ttnn.deallocate(x)
        return out

    # ------------------------------------------------------------------ public
    def decode(self, latents: torch.Tensor, taps: Optional[list] = None) -> torch.Tensor:
        """[64, T] latents -> [2, T * upsample] float32 audio (1920 for ACE-Step, 2048 for DiffRhythm)."""
        length = latents.shape[-1]
        x = ttnn.from_torch(latents.t().contiguous().reshape(1, 1, length, -1).float(), dtype=self.prec.act_dtype,
                            layout=ttnn.TILE_LAYOUT, device=self.dev, memory_config=MEM)
        tap = (lambda t, n: taps.append(ttnn.to_torch(t)[0, 0, :n].float())) if taps is not None else (lambda t, n: None)
        self._where = "in"
        h, length = self._timed("conv_in", length, self.conv1.out_ch, self._run_conv, x, self.conv1, length)
        ttnn.deallocate(x)
        tap(h, length)
        for i, b in enumerate(self.blocks):
            self._where = f"block{i}"
            s = self._timed("snake", length, b.up.in_ch, self._snake_op, h, b.snake)
            ttnn.deallocate(h)
            h, length = self._timed(f"conv_t_s{b.up.stride}", length, b.up.out_ch, self._run_conv, s, b.up, length)
            ttnn.deallocate(s)
            for u in b.units:
                h = self._unit(h, u, length)
            tap(h, length)
        self._where = "out"
        s = self._timed("snake", length, self.snake_out.alpha.shape[-1], self._snake_op, h, self.snake_out)
        ttnn.deallocate(h)
        out, length = self._timed("conv_out", length, 2, self._run_conv, s, self.conv2, length)
        ttnn.deallocate(s)
        audio = ttnn.to_torch(out)[0, 0, :length, :2].float().t().contiguous()
        ttnn.deallocate(out)
        return audio
