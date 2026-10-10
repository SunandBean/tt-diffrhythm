#!/usr/bin/env python3
"""TT Oobleck decoder (tt_diffrhythm/oobleck_vae.py with DiffRhythm's stable-audio weights) against the TorchScript VAE itself
on the CPU (float32, decode_export), on a window of the latents a golden run gave its VAE. Runs inside the TT image
with the P100a; start it with run_on_card.sh:

  python experiments/tt-diffrhythm/vae_check.py /golden [--frames 512] [--start 768] [--timing 512 1024]
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tt import acestep_dit, oobleck_vae  # noqa: E402


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def spectral_db(a, b, n_fft=2048, hop=512):
    spec = lambda w: torch.stft(w.mean(0), n_fft, hop, window=torch.hann_window(n_fft), return_complex=True).abs()
    da, db = 20 * torch.log10(spec(a) + 1e-7), 20 * torch.log10(spec(b) + 1e-7)
    quiet = da < da.max() - 60
    return {"all_db": float((da - db).abs().mean()), "quiet_db": float((da - db)[quiet].abs().mean()),
            "quiet_fraction": float(quiet.float().mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("--vae", default="/vae/vae_model.pt")
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--start", type=int, default=768)
    parser.add_argument("--timing", type=int, nargs="*", default=[])
    parser.add_argument("--out", default="vae_check.json")
    args = parser.parse_args()
    golden_dir = Path(args.golden)
    latents = torch.load(golden_dir / "calls.pt")["vae"]["latents"][0]  # [64, T]
    lat = latents[:, args.start: args.start + args.frames].contiguous()
    result = {"frames": lat.shape[-1], "start": args.start}
    ref = torch.jit.load(args.vae, map_location="cpu").eval()
    t0 = time.monotonic()
    with torch.inference_mode():
        expected = ref.decode_export(lat[None])[0].float()
    result["cpu_s"] = round(time.monotonic() - t0, 2)
    dev = acestep_dit.open_device()
    try:
        t0 = time.monotonic()
        vae = oobleck_vae.OobleckTT(dev, args.vae)
        result.update(load_s=round(time.monotonic() - t0, 2), upsample=vae.upsample)
        runs = []
        for p in range(2):
            t0 = time.monotonic()
            audio = vae.decode(lat)
            runs.append(round(time.monotonic() - t0, 2))
        result.update(tt_s=runs, shape=list(audio.shape), expected_shape=list(expected.shape),
                      audio_pcc=pcc(audio, expected), max_abs=float((audio - expected).abs().max()),
                      spectral=spectral_db(audio, expected))
        print(json.dumps(result), flush=True)
        timing = []
        for n in args.timing:
            z = torch.randn(64, n)
            vae.decode(z)
            t0 = time.monotonic()
            vae.decode(z)
            timing.append({"frames": n, "s": round(time.monotonic() - t0, 2)})
            print(json.dumps(timing[-1]), flush=True)
        result["timing"] = timing
    finally:
        acestep_dit.close_device(dev)
        (golden_dir / args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
