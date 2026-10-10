#!/usr/bin/env python3
"""Compile every P100a kernel shape this port can hit, ahead of the first songs.

The card compiles kernels per shape and keeps them in TT_METAL_CACHE (data/tt-cache). Without this, the first song
of each new length bucket pays that compilation (tens of seconds). Shapes, per model (--model):
           ENC_BUCKET), and both VAE windows (remote.py RemoteVae)
  diffrhythm: every DiT sequence bucket of a 96-285 s song (diffrhythm_host.seq_buckets), both CFG branches, and
           both VAE windows
Random inputs: only the shapes matter. Run it again after changing this package or the TT image.
"""
import argparse
import json
import time

import torch

from . import diffrhythm_dit, diffrhythm_host as dhost, oobleck_vae
from .shapes import VAE_WINDOWS

def warm_vae(vae, report):
    for window in VAE_WINDOWS:  # exactly as the worker does right after loading (worker.py)
        t0 = time.monotonic()
        audio = vae.decode(torch.zeros(64, window))
        rec = {"vae_window": window, "s": round(time.monotonic() - t0, 2), "finite": bool(torch.isfinite(audio).all())}
        report["vae"].append(rec)
        print(json.dumps(rec), flush=True)


def warm_diffrhythm(args):
    g = torch.Generator().manual_seed(0)
    started = time.monotonic()
    dev = diffrhythm_dit.open_device()
    report = {"seq_buckets": dhost.seq_buckets(), "dit": [], "vae": []}
    try:
        dit = diffrhythm_dit.DiffRhythmDiT(dev, dhost.Checkpoint(args.checkpoint or "/checkpoint/cfm_model.pt"))
        vae = oobleck_vae.OobleckTT(dev, args.vae or "/vae/vae_model.pt")
        report["load_s"] = round(time.monotonic() - started, 1)
        warm_vae(vae, report)
        for seq_pad in report["seq_buckets"]:
            frames = min(seq_pad, dhost.MAX_FRAMES)  # any length in the bucket gives the same shapes
            t0 = time.monotonic()
            song = dit.prepare({null: torch.randn(frames, 512, generator=g) for null in (False, True)},
                               torch.randn(frames, 64, generator=g))
            assert song.geo.seq_pad == seq_pad
            finite = True
            for null in (False, True):
                v = dit.forward(torch.randn(frames, 64, generator=g), null, torch.randn(512, generator=g),
                                torch.randn(512, generator=g), song)
                finite &= bool(torch.isfinite(v).all())
            dit.release(song)
            rec = {"seq_pad": seq_pad, "s": round(time.monotonic() - t0, 2), "finite": finite}
            report["dit"].append(rec)
            print(json.dumps(rec), flush=True)
    finally:
        diffrhythm_dit.close_device(dev)
    report["total_s"] = round(time.monotonic() - started, 1)
    print(json.dumps({k: report[k] for k in ("seq_buckets", "load_s", "total_s")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--vae")
    args = parser.parse_args()
    return warm_diffrhythm(args)


if __name__ == "__main__":
    main()
