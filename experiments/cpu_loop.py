#!/usr/bin/env python3
"""Baseline for device_check.py --closed-loop: the same sampling loop with upstream's DiT on the CPU (float32, or
bfloat16 with --bf16), from the GPU run's noise, compared with the GPU's final latents (and with the card's, when
tt_latents.pt is there). Shows how far any other numerics drift from the GPU fp16 run.

  DIFFRHYTHM_DIR=vendor/DiffRhythm vendor/DiffRhythm/.venv/bin/python experiments/tt-diffrhythm/cpu_loop.py golden/d96
"""
import json
import os
from pathlib import Path
import sys
import time


def pcc(a, b):
    import torch
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    golden_dir = Path(sys.argv[1]).resolve()
    bf16 = "--bf16" in sys.argv
    repo = Path(os.environ["DIFFRHYTHM_DIR"]).resolve()
    sys.path[:0] = [str(repo), str(repo / "infer")]
    os.chdir(repo)
    import torch
    from infer_utils import prepare_model

    torch.set_grad_enabled(False)
    golden = torch.load(golden_dir / "calls.pt")
    fixed, calls = golden["fixed"], golden["calls"]
    dtype = torch.bfloat16 if bf16 else torch.float32
    cfm, *_ = prepare_model(6144, "cpu")
    dit = cfm.transformer.to(dtype).eval()
    times = [float(calls[i]["time"].flatten()[0]) for i in range(0, len(calls), 2)] + [1.0]
    x = calls[0]["x"].to(dtype)
    kw = dict(cond=fixed["cond"].to(dtype), text=fixed["text"].long(), start_time=fixed["start_time"].to(dtype),
              duration=fixed["duration"].to(dtype), drop_prompt=False)
    started = time.monotonic()
    for k, t in enumerate(times[:-1]):
        tt = torch.tensor([t], dtype=dtype)
        pred = dit(x=x, time=tt, drop_audio_cond=False, drop_text=False, style_prompt=fixed["style_prompt"].to(dtype), **kw)
        null = dit(x=x, time=tt, drop_audio_cond=True, drop_text=True, style_prompt=fixed["negative_style_prompt"].to(dtype), **kw)
        x = x + (times[k + 1] - t) * (pred + (pred - null) * 4.0)
    x = x[0].float()
    gpu = golden["vae"]["latents"][0].t()
    out = {"dtype": str(dtype), "seconds": round(time.monotonic() - started, 1), "final_pcc_vs_gpu": pcc(x, gpu),
           "rel_rms_vs_gpu": float((x - gpu).pow(2).mean().sqrt() / gpu.pow(2).mean().sqrt())}
    tt_path = golden_dir / "tt_latents.pt"
    if tt_path.exists():
        tt = torch.load(tt_path)
        out.update(final_pcc_vs_tt=pcc(x, tt), rel_rms_vs_tt=float((x - tt).pow(2).mean().sqrt() / x.pow(2).mean().sqrt()))
    torch.save(x, golden_dir / f"cpu_latents_{'bf16' if bf16 else 'fp32'}.pt")
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
