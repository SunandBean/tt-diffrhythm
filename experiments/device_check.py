#!/usr/bin/env python3
"""TT DiT (tt_diffrhythm/diffrhythm_dit.py) against the GPU calls recorded by dump_reference.py (teacher forcing), with the
inputs export_inputs.py prepared. Runs inside the TT image with the P100a; start it with run_on_card.sh:

  python experiments/tt-diffrhythm/device_check.py /golden [--calls 0 1 60 61] [--repeat 2] [--taps]

Writes device_check.json next to the golden data.
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


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def closed_loop(dit, song, inp, golden_dir, cfg_strength=4.0):
    """CFM.sample's loop (torchdiffeq euler on linspace(0, 1, 32)) with the TT DiT, from the GPU run's noise."""
    calls = inp["calls"]
    times = [calls[i]["t"] for i in range(0, len(calls), 2)] + [1.0]
    x = calls[0]["x"].clone()
    cond, uncond = calls[0], calls[1]
    started = time.monotonic()
    per_step, cfg_pcc = [], []
    for k in range(len(times) - 1):  # teacher-forced: the CFG-combined velocity from the recorded inputs
        a, b = calls[2 * k], calls[2 * k + 1]
        pred = dit.forward(a["x"], False, a["style"], a["c"], song)
        null = dit.forward(b["x"], True, b["style"], b["c"], song)
        cfg_pcc.append(pcc(pred + 4 * (pred - null), a["gpu"] + 4 * (a["gpu"] - b["gpu"])))
    started = time.monotonic()
    for k, t in enumerate(times[:-1]):
        c_emb = calls[2 * k]["c"]
        pred = dit.forward(x, False, cond["style"], c_emb, song)
        null = dit.forward(x, True, uncond["style"], calls[2 * k + 1]["c"], song)
        x = x + (times[k + 1] - t) * (pred + (pred - null) * cfg_strength)
        if 2 * k + 2 < len(calls):
            per_step.append(pcc(x, calls[2 * k + 2]["x"]))
    gpu = torch.load(golden_dir / "calls.pt")["vae"]["latents"][0].t()  # [S, 64]
    out = {"seconds": round(time.monotonic() - started, 2), "final_pcc": pcc(x, gpu),
           "final_rel_rms": float((x - gpu).pow(2).mean().sqrt() / gpu.pow(2).mean().sqrt()),
           "step_pcc": [round(p, 5) for p in per_step], "cfg_teacher_pcc_min": min(cfg_pcc),
           "cfg_teacher_pcc": [round(p, 5) for p in cfg_pcc]}
    torch.save(x, golden_dir / "tt_latents.pt")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("--checkpoint", default="/checkpoint/cfm_model.pt")
    parser.add_argument("--calls", type=int, nargs="*", help="recorded calls to check (default: all)")
    parser.add_argument("--repeat", type=int, default=2, help="passes over the calls (the first one compiles)")
    parser.add_argument("--taps", action="store_true", help="per-layer hidden states of the first call vs RefDecoder")
    parser.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE",
                        help="override DRPrecision fields, e.g. mm_fidelity=HiFi2 residual_fp32=0")
    parser.add_argument("--closed-loop", action="store_true",
                        help="also sample the whole song on the card (euler over the recorded times, CFG 4) from the "
                             "recorded noise and compare the final latents with the GPU's")
    parser.add_argument("--out", default="device_check.json")
    args = parser.parse_args()
    golden_dir = Path(args.golden)
    inp = torch.load(golden_dir / "tt_inputs.pt")
    picks = args.calls if args.calls else list(range(len(inp["calls"])))
    result = {"frames": inp["frames"], "passes": []}
    t0 = time.monotonic()
    dev = tdit.open_device()
    try:
        overrides = {}
        for item in args.set:
            field, value = item.split("=", 1)
            overrides[field] = (getattr(tdit.ttnn.MathFidelity, value) if field == "mm_fidelity"
                                else value.lower() in ("1", "true", "yes"))
        prec = tdit.DRPrecision(**overrides)
        result["precision"] = {k: str(v) for k, v in vars(prec).items()}
        dit = tdit.DiffRhythmDiT(dev, host.Checkpoint(args.checkpoint), prec, keep_ref=args.taps)
        result["load_s"] = round(time.monotonic() - t0, 2)
        result["dram_after_load"] = tdit.dram_stats(dev)
        t0 = time.monotonic()
        song = dit.prepare(inp["text_embed"], inp["cond"])
        result.update(prepare_s=round(time.monotonic() - t0, 2), seq_pad=song.geo.seq_pad,
                      dram_after_prepare=tdit.dram_stats(dev))
        for p in range(args.repeat):
            rows = []
            for i in picks:
                call = inp["calls"][i]
                started = time.monotonic()
                v = dit.forward(call["x"], call["drop_text"], call["style"], call["c"], song)
                rows.append({"call": i, "t": round(call["t"], 4), "null": call["drop_text"], "pcc": pcc(v, call["gpu"]),
                             "max_abs": float((v - call["gpu"]).abs().max()),
                             "seconds": round(time.monotonic() - started, 3)})
                print(json.dumps({"pass": p, **rows[-1]}), flush=True)
            result["passes"].append({"min_pcc": min(r["pcc"] for r in rows), "calls": rows,
                                     "total_s": round(sum(r["seconds"] for r in rows), 2)})
        if args.taps:
            call = inp["calls"][picks[0]]
            taps, ref_taps = [], []
            dit.forward(call["x"], call["drop_text"], call["style"], call["c"], song, taps=taps)
            geo = song.geo
            branch = dit.ref.branch(inp["text_embed"][call["drop_text"]], None if call["drop_text"] else inp["cond"], geo)
            dit.ref.forward(call["x"], branch, dit.ref.rows(call["style"], call["c"]), geo, taps=ref_taps)
            result["taps"] = [{"layer": j, "pcc": pcc(a[:geo.frames], b[:geo.frames])} for j, (a, b) in enumerate(zip(taps, ref_taps))]
            print(json.dumps(result["taps"]), flush=True)
        if args.closed_loop:
            result["closed_loop"] = closed_loop(dit, song, inp, golden_dir)
            print(json.dumps(result["closed_loop"]), flush=True)
        dit.release(song)
        result["dram_after_release"] = tdit.dram_stats(dev)
    finally:
        tdit.close_device(dev)
        (golden_dir / args.out).write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "passes"}), flush=True)


if __name__ == "__main__":
    main()
