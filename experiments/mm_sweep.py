#!/usr/bin/env python3
"""minimal_matmul blockings for the DiffRhythm DiT's matmuls on the P100a (11x10 grid), per sequence length.

tt-metal's table (models/tt_dit/utils/matmul.py) has none of these shapes, so get_matmul_config falls back to
8x8x8 blocks. This times candidate (M, K, N) blocks and subblocks, plus the table's heuristic, with the DiT's
compute config (HiFi4, fp32 accumulation) and bf16 operands:

  qkv [S, 2048] x [2048, 6144]   o [S, 2048] x [2048, 2048]
  gateup [S, 2048] x [2048, 16384] with fuse_swiglu   down [S, 8192] x [8192, 2048]

Start it with run_on_card.sh (any writable directory as the second argument):
  python experiments/tt-diffrhythm/mm_sweep.py /golden [--lengths 2560 6144]
"""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tt import diffrhythm_dit as tdit  # noqa: E402

ttnn = tdit.ttnn
from models.tt_dit.utils import matmul as mmu  # noqa: E402

SHAPES = {"qkv": (2048, 6144, False), "o": (2048, 2048, False), "gateup": (2048, 16384, True), "down": (8192, 2048, False)}
BLOCKS = list(itertools.product((4, 8, 16), (4, 8, 16), (4, 8, 16)))
SUBBLOCKS = ((2, 2), (4, 1), (1, 4))


def time_config(x, w, cfg, ck, swiglu, repeat):
    out = ttnn.experimental.minimal_matmul(x, w, config=cfg, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                                           memory_config=tdit.MEM, fuse_swiglu=swiglu)
    ttnn.deallocate(out)
    ttnn.synchronize_device(x.device())
    t0 = time.monotonic()
    for _ in range(repeat):
        ttnn.deallocate(ttnn.experimental.minimal_matmul(x, w, config=cfg, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                                                         memory_config=tdit.MEM, fuse_swiglu=swiglu))
    ttnn.synchronize_device(x.device())
    return (time.monotonic() - t0) / repeat * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("--lengths", type=int, nargs="*", default=[2560, 6144])
    parser.add_argument("--shapes", nargs="*", default=list(SHAPES))
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    dev = tdit.open_device()
    grid = dev.compute_with_storage_grid_size()
    ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    g = torch.Generator().manual_seed(0)
    report = {"grid": [grid.x, grid.y], "results": []}
    try:
        for name in args.shapes:
            K, N, swiglu = SHAPES[name]
            w = tdit._dev(dev, (torch.randn(K, N, generator=g) * 0.02).to(torch.bfloat16))
            for S in args.lengths:
                x = tdit._dev(dev, torch.randn(1, 1, S, K, generator=g).to(torch.bfloat16))
                rows = []
                default = mmu.get_matmul_config(S, K, N, grid)
                heuristic = mmu.get_matmul_config(S, K, N, grid, use_heuristic=True)
                candidates = [("default", default), ("heuristic", heuristic)]
                for (mb, kb, nb), (sh, sw) in itertools.product(BLOCKS, SUBBLOCKS):
                    if mb % sh or nb % sw:
                        continue
                    candidates.append((f"{mb}x{kb}x{nb}/{sh}x{sw}", ttnn.MinimalMatmulConfig(
                        M_block_size=mb, K_block_size=kb, N_block_size=nb, subblock_h=sh, subblock_w=sw,
                        compute_with_storage_grid_size=grid)))
                for label, cfg in candidates:
                    try:
                        ms = time_config(x, w, cfg, ck, swiglu, args.repeat)
                    except Exception as exc:
                        rows.append({"cfg": label, "error": str(exc)[:120]})
                        continue
                    rows.append({"cfg": label, "ms": round(ms, 3)})
                ok = sorted((r for r in rows if "ms" in r), key=lambda r: r["ms"])
                dflt = next(r for r in rows if r["cfg"] == "default")
                heur = next(r for r in rows if r["cfg"] == "heuristic")
                rec = {"shape": name, "S": S, "K": K, "N": N, "default_ms": dflt.get("ms"), "heuristic_ms": heur.get("ms"),
                       "heuristic_cfg": str(heuristic), "best": ok[:5], "tried": len(rows),
                       "errors": sum("error" in r for r in rows),
                       "tflops_best": round(2 * S * K * N / (ok[0]["ms"] / 1000) / 1e12, 1)}
                print(json.dumps(rec), flush=True)
                report["results"].append(rec)
                ttnn.deallocate(x)
            ttnn.deallocate(w)
    finally:
        tdit.close_device(dev)
        out = Path(args.golden) / "mm_sweep.jsonl"
        with out.open("a") as f:
            for rec in report["results"]:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
