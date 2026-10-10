#!/usr/bin/env python3
"""Where a DiffRhythm DiT call's time goes on the P100a: the DiT's methods and ttnn ops wrapped with device
synchronization and timed (inclusive and exclusive per name). Synchronizing adds a little per op, so the
profiled total is above a normal call's. Start it with run_on_card.sh (any writable directory as the second
argument):

  python experiments/tt-diffrhythm/dit_profile.py /golden [--lengths 2560 6144]
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tt import diffrhythm_dit as tdit, diffrhythm_host as host  # noqa: E402

ttnn = tdit.ttnn


class Profiler:
    def __init__(self, dev):
        self.dev, self.stack, self.incl, self.excl, self.count = dev, [], defaultdict(float), defaultdict(float), defaultdict(int)
        self.on = False

    def wrap(self, owner, attr, name):
        fn = getattr(owner, attr)

        def timed(*a, **kw):
            if not self.on:
                return fn(*a, **kw)
            ttnn.synchronize_device(self.dev)
            self.stack.append(0.0)
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            ttnn.synchronize_device(self.dev)
            dt = time.perf_counter() - t0
            inner = self.stack.pop()
            self.incl[name] += dt
            self.excl[name] += dt - inner
            self.count[name] += 1
            if self.stack:
                self.stack[-1] += dt
            return out

        setattr(owner, attr, timed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("--checkpoint", default="/checkpoint/cfm_model.pt")
    parser.add_argument("--lengths", type=int, nargs="*", default=[2560, 6144])
    args = parser.parse_args()
    dev = tdit.open_device()
    prof = Profiler(dev)
    report = []
    try:
        dit = tdit.DiffRhythmDiT(dev, host.Checkpoint(args.checkpoint))
        for attr in ("_conv_pos", "_attention", "_rms", "_residual", "_mm", "_sdpa", "_conv", "_mish", "_bf16"):
            prof.wrap(dit, attr, attr)
        for owner, attr in ((ttnn, "add"), (ttnn, "multiply"), (ttnn, "typecast"), (ttnn, "layer_norm"), (ttnn, "linear"),
                            (ttnn, "to_torch"), (ttnn, "from_torch"), (ttnn, "concat"), (ttnn, "slice"), (ttnn, "to_layout"), (ttnn, "pad"),
                            (ttnn.experimental, "nlp_create_qkv_heads"), (ttnn.experimental, "rotary_embedding_llama"),
                            (ttnn.transformer, "concatenate_heads")):
            prof.wrap(owner, attr, f"ttnn.{attr}")
        g = torch.Generator().manual_seed(0)
        for S in args.lengths:
            frames = min(S, host.MAX_FRAMES)
            song = dit.prepare({null: torch.randn(frames, 512, generator=g) for null in (False, True)},
                               torch.randn(frames, 64, generator=g))
            x, style, c = torch.randn(frames, 64, generator=g), torch.randn(512, generator=g), torch.randn(512, generator=g)
            dit.forward(x, False, style, c, song)  # compile
            t0 = time.monotonic()
            for _ in range(3):
                dit.forward(x, False, style, c, song)
            plain = (time.monotonic() - t0) / 3
            prof.incl.clear(), prof.excl.clear(), prof.count.clear()
            prof.on = True
            t0 = time.monotonic()
            dit.forward(x, False, style, c, song)
            profiled = time.monotonic() - t0
            prof.on = False
            rows = sorted(({"name": k, "excl_ms": round(prof.excl[k] * 1000, 2), "incl_ms": round(prof.incl[k] * 1000, 2),
                            "count": prof.count[k]} for k in prof.excl), key=lambda r: -r["excl_ms"])
            rec = {"S": S, "call_ms": round(plain * 1000, 1), "profiled_ms": round(profiled * 1000, 1), "ops": rows}
            print(json.dumps(rec), flush=True)
            report.append(rec)
            dit.release(song)
    finally:
        tdit.close_device(dev)
        (Path(args.golden) / "dit_profile.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
