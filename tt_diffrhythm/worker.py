#!/usr/bin/env python3
"""TT worker: loads a music model's DiT and its audio VAE decoder onto the P100a and answers one host pipeline.

Runs inside the TT image; the host (remote.py) owns the container for its whole life and talks to
this process over a Unix socket in a directory bind-mounted into both:

  python /work/tt_diffrhythm/worker.py --socket /ipc/worker.sock [--model diffrhythm] --checkpoint ... --vae ...

Messages are plain dicts (tensors via wire.py):
  -> {"op": "ready", ...}                                           once the weights are on the device
  <- {"op": "prepare", ...}                                         -> {"op": "prepared", ...}
       DiffRhythm: "text_embed": {drop_text: [T, 512]}, "cond": [T, 64]
  <- {"op": "forward", ...}                                         -> {"op": "velocity", "v": [T, 64], ...}
       DiffRhythm: "x": [T, 64], "drop_text": bool, "style": [512], "c": [512]
  <- {"op": "vae", "z": [64, L]}                                    -> {"op": "audio", "audio": [2, L * upsample], ...}
  <- {"op": "close"}
Any failure is sent back as {"op": "error", "error": "..."} and ends the worker.
"""
import argparse
from multiprocessing.connection import Listener
import os
import signal
import sys
import time
import traceback

import torch

from . import diffrhythm_dit, oobleck_vae, wire
from .shapes import VAE_WINDOWS


class DiffRhythm:
    def __init__(self, dev, checkpoint):
        from . import diffrhythm_host

        self.dit = diffrhythm_dit.DiffRhythmDiT(dev, diffrhythm_host.Checkpoint(checkpoint))

    def prepare(self, msg):
        text_embed = {bool(k): wire.decode(v) for k, v in msg["text_embed"].items()}
        song = self.dit.prepare(text_embed, wire.decode(msg["cond"]))
        return song, {"frames": song.geo.frames, "seq_pad": song.geo.seq_pad}

    def forward(self, msg, song):
        return self.dit.forward(wire.decode(msg["x"]), bool(msg["drop_text"]), wire.decode(msg["style"]),
                                wire.decode(msg["c"]), song)


MODELS = {"diffrhythm": (DiffRhythm, "/checkpoint/cfm_model.pt", "/vae/vae_model.pt")}


def serve(conn, model, vae):
    dit = model.dit
    song = None
    while True:
        msg = conn.recv()
        op = msg.get("op")
        started = time.monotonic()
        if op == "prepare":
            if song is not None:
                dit.release(song)
                song = None
            song, geometry = model.prepare(msg)
            conn.send({"op": "prepared", "seconds": time.monotonic() - started, "geometry": geometry})
        elif op == "forward":
            if song is None:
                raise RuntimeError("forward before prepare")
            v = model.forward(msg, song)
            conn.send({"op": "velocity", "v": wire.encode(v), "seconds": time.monotonic() - started})
        elif op == "vae":
            if song is not None:  # sampling is over; freeing the song's tensors first gives the VAE the same device
                dit.release(song)  # allocation state for every song length, so its compiled conv kernels are reused
                song = None
            audio = vae.decode(wire.decode(msg["z"]))
            conn.send({"op": "audio", "audio": wire.encode(audio), "seconds": time.monotonic() - started})
        elif op == "close":
            if song is not None:
                dit.release(song)
            return
        else:
            raise RuntimeError(f"unknown op {op!r}")


def main():
    # docker stop / a signal: leave through the finally below so the device is closed, not dropped
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--model", choices=sorted(MODELS), default="diffrhythm")
    parser.add_argument("--checkpoint")
    parser.add_argument("--vae")
    args = parser.parse_args()
    model_class, checkpoint, vae_path = MODELS[args.model]
    authkey = os.environ.get("MUSIC_TT_AUTHKEY", "").encode() or None
    listener = Listener(args.socket, family="AF_UNIX", authkey=authkey)
    started = time.monotonic()
    dev = diffrhythm_dit.open_device()
    conn = None
    try:
        model = model_class(dev, args.checkpoint or checkpoint)
        vae = oobleck_vae.OobleckTT(dev, args.vae or vae_path)
        # Build the VAE programs now, in the same device state for every song: their conv kernels then compile
        # once (TT_METAL_CACHE) and later decodes reuse the in-process programs. Built after a song's DiT steps
        # instead, the device state (and so the kernels) depends on the song length and recompiles for 20-40 s.
        warm = time.monotonic()
        for window in VAE_WINDOWS:
            vae.decode(torch.zeros(64, window))
        ready = {"op": "ready", "model": args.model, "load_s": time.monotonic() - started,
                 "vae_warm_s": time.monotonic() - warm, "upsample": vae.upsample, "dram": diffrhythm_dit.dram_stats(dev)}
        conn = listener.accept()  # one pipeline per container
        conn.send(ready)
        serve(conn, model, vae)
    except BaseException as exc:
        traceback.print_exc()
        if conn is not None:
            try:
                conn.send({"op": "error", "error": f"{type(exc).__name__}: {exc}"})
            except OSError:
                pass
        raise
    finally:
        diffrhythm_dit.close_device(dev)
        listener.close()


if __name__ == "__main__":
    main()
