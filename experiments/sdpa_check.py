#!/usr/bin/env python3
"""Padded keys in the DiT's attention two ways on the P100a, against torch:

  mask      additive [S, S] mask (diffrhythm_dit.py now), read for every head
  windowed  cu_window_seqlens = [0, frames, S]: block-diagonal attention with the mask built on the device, so real
            frames see only real frames (padded rows see only padding; their outputs are never used)

Start it with run_on_card.sh (any writable directory as the second argument):
  python experiments/tt-diffrhythm/sdpa_check.py /golden
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
H, D = 32, 64


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("--cases", nargs="*", default=["2067:2560", "2560:2560", "3001:3072", "6136:6144"])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--chunks", nargs="*", default=["256x256"], help="q_chunk x k_chunk sizes to try")
    args = parser.parse_args()
    dev = tdit.open_device()
    grid = dev.compute_with_storage_grid_size()
    ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    g = torch.Generator().manual_seed(0)
    rows = []
    try:
        for case in args.cases:
            frames, S = (int(v) for v in case.split(":"))
            q, k, v = (torch.randn(1, H, S, D, generator=g) for _ in range(3))
            ref = torch.nn.functional.scaled_dot_product_attention(q[:, :, :frames], k[:, :, :frames], v[:, :, :frames])[0]
            qd, kd, vd = (tdit._dev(dev, t.to(torch.bfloat16)) for t in (q, k, v))
            mask = tdit._dev(dev, host.key_mask(host.Geometry(frames)).to(torch.bfloat16)[None, None]) if S == host.round_up(frames, host.SEQ_BUCKET) else \
                tdit._dev(dev, torch.where(torch.arange(S)[None, :] < frames, 0.0, -1e9).expand(S, S).to(torch.bfloat16)[None, None])
            cu = ttnn.from_torch(torch.tensor([0, frames, S] if frames < S else [0, S], dtype=torch.int32),
                                 dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            for chunks in args.chunks:
                qc, kc = (int(v) for v in chunks.split("x"))
                cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc,
                                             exp_approx_mode=False)
                common = dict(scale=D ** -0.5, program_config=cfg, compute_kernel_config=ck, memory_config=tdit.MEM)
                methods = {"mask": lambda: ttnn.transformer.scaled_dot_product_attention(qd, kd, vd, attn_mask=mask, is_causal=False, **common),
                           "windowed": lambda: ttnn.transformer.scaled_dot_product_attention(qd, kd, vd, is_causal=False, cu_window_seqlens=cu, **common)}
                for name, fn in methods.items():
                    row = {"frames": frames, "S": S, "chunks": chunks, "method": name}
                    try:
                        out = fn()
                        got = ttnn.to_torch(out)[0, :, :frames].float()
                        ttnn.deallocate(out)
                        ttnn.synchronize_device(dev)
                        t0 = time.monotonic()
                        for _ in range(args.repeat):
                            ttnn.deallocate(fn())
                        ttnn.synchronize_device(dev)
                        row.update(ms=round((time.monotonic() - t0) / args.repeat * 1000, 2), pcc=pcc(got, ref),
                                   max_abs=float((got - ref).abs().max()))
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    print(json.dumps(row), flush=True)
                    rows.append(row)
            for t in (qd, kd, vd, mask, cu):
                ttnn.deallocate(t)
    finally:
        tdit.close_device(dev)
        (Path(args.golden) / "sdpa_check.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
