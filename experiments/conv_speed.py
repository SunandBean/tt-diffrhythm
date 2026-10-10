#!/usr/bin/env python3
"""The DiT's grouped position conv (2048 channels, 16 groups of 128, k31, padding 15) three ways on the card,
against torch on the CPU, per sequence length:

  conv     ttnn.conv1d(groups=16), as diffrhythm_dit.py does now (expands the groups densely: 16x the work)
  bmm      unfold per group + one batched matmul: groups as the batch, x [16, S, 31 * 128] @ w [16, 31 * 128, 128]
  dense    unfold the whole row + one matmul: x [S, 31 * 2048] @ w [31 * 2048, 2048] (block-sparse weight)

Start it with run_on_card.sh (any writable directory as the second argument):
  python experiments/tt-diffrhythm/conv_speed.py /golden [--lengths 2560 6144] [--methods conv bmm dense]
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tt import diffrhythm_dit as tdit, diffrhythm_host as host  # noqa: E402

ttnn = tdit.ttnn
MEM = tdit.MEM
G, CG, K, PAD, C = 16, 128, 31, 15, 2048


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def bmm_weight(w):
    """[2048, 128, 31] (out, in per group, k) -> [16, 31 * 128, 128]: row k * 128 + i, column o of group g."""
    return w.view(G, CG, CG, K).permute(0, 3, 2, 1).reshape(G, K * CG, CG).contiguous()


def dense_weight(w):
    """[2048, 128, 31] -> [31 * 2048, 2048] block-diagonal per tap: row k * 2048 + (g * 128 + i), column g * 128 + o."""
    out = torch.zeros(K, C, C)
    wg = w.view(G, CG, CG, K)  # g, o, i, k
    for g in range(G):
        out[:, g * CG:(g + 1) * CG, g * CG:(g + 1) * CG] = wg[g].permute(2, 1, 0)  # k, i, o
    return out.reshape(K * C, C)


class Methods:
    def __init__(self, dev, w, b):
        self.dev = dev
        self.ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                         math_approx_mode=False, fp32_dest_acc_en=True,
                                                         packer_l1_acc=False)
        self.grid = dev.compute_with_storage_grid_size()
        self.conv_w = ttnn.from_torch(w.unsqueeze(2).contiguous(), dtype=ttnn.float32)
        self.conv_b = ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=ttnn.float32)
        self.bmm_w = tdit._dev(dev, bmm_weight(w).to(torch.bfloat16)[None])
        self.dense_w = tdit._dev(dev, dense_weight(w).to(torch.bfloat16))
        self.bias = tdit._row(dev, b)
        self._cfg = {}

    def conv(self, x, S):
        cfg = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16, config_tensors_in_dram=True)
        out, n, (self.conv_w, self.conv_b) = ttnn.conv1d(
            input_tensor=x, weight_tensor=self.conv_w, bias_tensor=self.conv_b, device=self.dev, in_channels=C,
            out_channels=C, batch_size=1, input_length=S, kernel_size=K, stride=1, padding=PAD, dilation=1, groups=G,
            dtype=ttnn.bfloat16, compute_config=self.ck, conv_config=cfg, return_output_dim=True,
            return_weights_and_bias=True)
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return ttnn.reshape(out, [1, 1, n, C])

    @staticmethod
    def _unfold(xp, S, dim):
        shape = list(xp.shape)
        parts = [ttnn.slice(xp, [0] * dim + [k, 0], shape[:dim] + [k + S, shape[-1]]) for k in range(K)]
        out = ttnn.concat(parts, dim=dim + 1, memory_config=MEM)
        for p in parts:
            ttnn.deallocate(p)
        return out

    def bmm(self, x, S):
        xr = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        xr = ttnn.reshape(xr, [S, G, CG])
        xg = ttnn.permute(xr, (1, 0, 2), memory_config=MEM)  # [16, S, 128]
        xp = ttnn.pad(xg, [(0, 0), (PAD, PAD), (0, 0)], 0.0)
        ttnn.deallocate(xg)
        xu = self._unfold(xp, S, 1)  # [16, S, 31 * 128]
        ttnn.deallocate(xp)
        xt = ttnn.to_layout(ttnn.reshape(xu, [1, G, S, K * CG]), ttnn.TILE_LAYOUT)
        ttnn.deallocate(xu)
        y = ttnn.matmul(xt, self.bmm_w, compute_kernel_config=self.ck, memory_config=MEM, dtype=ttnn.bfloat16)
        ttnn.deallocate(xt)
        y = ttnn.permute(y, (0, 2, 1, 3), memory_config=MEM)  # [1, S, 16, 128]
        y = ttnn.reshape(y, [1, 1, S, C])
        out = ttnn.add(y, self.bias, memory_config=MEM)
        ttnn.deallocate(y)
        return out

    def dense(self, x, S):
        xr = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        xr = ttnn.reshape(xr, [S, C])
        xp = ttnn.pad(xr, [(PAD, PAD), (0, 0)], 0.0)
        xu = self._unfold(xp, S, 0)  # [S, 31 * 2048]
        ttnn.deallocate(xp)
        xt = ttnn.to_layout(ttnn.reshape(xu, [1, 1, S, K * C]), ttnn.TILE_LAYOUT)
        ttnn.deallocate(xu)
        key = (S, K * C, C)
        if key not in self._cfg:
            self._cfg[key] = tdit.get_matmul_config(S, K * C, C, self.grid)
        y = ttnn.experimental.minimal_matmul(xt, self.dense_w, config=self._cfg[key], compute_kernel_config=self.ck,
                                             dtype=ttnn.bfloat16, memory_config=MEM)
        ttnn.deallocate(xt)
        y = ttnn.reshape(y, [1, 1, S, C])
        out = ttnn.add(y, self.bias, memory_config=MEM)
        ttnn.deallocate(y)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("--checkpoint", default="/checkpoint/cfm_model.pt")
    parser.add_argument("--lengths", type=int, nargs="*", default=[2560, 6144])
    parser.add_argument("--methods", nargs="*", default=["conv", "bmm", "dense"])
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    ckpt = host.Checkpoint(args.checkpoint)
    w, b = ckpt.get("input_embed.conv_pos_embed.conv1d.0.weight"), ckpt.get("input_embed.conv_pos_embed.conv1d.0.bias")
    g = torch.Generator().manual_seed(0)
    dev = tdit.open_device()
    rows = []
    try:
        m = Methods(dev, w, b)
        for S in args.lengths:
            x = torch.randn(S, C, generator=g)
            x[S - 40:] = 0  # rows past the song are zero, as in the DiT
            expected = torch.nn.functional.conv1d(x.t()[None], w, b, padding=PAD, groups=G)[0].t()
            xd = tdit._dev(dev, x.to(torch.bfloat16).reshape(1, 1, S, C))
            for name in args.methods:
                row = {"S": S, "method": name}
                try:
                    fn = getattr(m, name)
                    out = fn(xd, S)  # compiles
                    got = ttnn.to_torch(out)[0, 0, :S].float()
                    ttnn.deallocate(out)
                    ttnn.synchronize_device(dev)
                    t0 = time.monotonic()
                    for _ in range(args.repeat):
                        ttnn.deallocate(fn(xd, S))
                    ttnn.synchronize_device(dev)
                    row.update(ms=round((time.monotonic() - t0) / args.repeat * 1000, 2), pcc=pcc(got, expected),
                               max_abs=float((got - expected).abs().max()))
                except Exception as exc:  # report and go on with the other methods
                    row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                print(json.dumps(row), flush=True)
                rows.append(row)
            ttnn.deallocate(xd)
    finally:
        tdit.close_device(dev)
        (Path(args.golden) / "conv_speed.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
