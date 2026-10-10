#!/usr/bin/env python3
"""The DiT's grouped position convs (k31, groups 16, 2048 channels) on the card, per sequence bucket: weights the
conv prepared on the device for one length reused at the others, against torch on the CPU, plus timing. Answers
whether one prepared weight tensor can serve every song length. Start it with run_on_card.sh:

  python experiments/tt-diffrhythm/conv_check.py /golden [--buckets 2560 3072 6144]
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


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


class Stub(tdit.DiffRhythmDiT):
    """Only what _conv needs: the two conv weights, no transformer on the device."""

    def __init__(self, dev, ckpt):
        self.dev, self.cfg, self.unfold = dev, ckpt.cfg, False  # ttnn.conv1d, as the DiT used it then
        self.host_conv = [(ckpt.get(f"input_embed.conv_pos_embed.conv1d.{i}.weight"),
                           ckpt.get(f"input_embed.conv_pos_embed.conv1d.{i}.bias")) for i in (0, 2)]
        self.reset()
        self.ck_conv = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                              math_approx_mode=False, fp32_dest_acc_en=True,
                                                              packer_l1_acc=False)

    def reset(self):
        self.conv = [(ttnn.from_torch(w.unsqueeze(2).contiguous(), dtype=ttnn.float32),
                      ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=ttnn.float32)) for w, b in self.host_conv]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("--checkpoint", default="/checkpoint/cfm_model.pt")
    parser.add_argument("--buckets", type=int, nargs="*",
                        default=sorted({host.round_up(n, host.SEQ_BUCKET) for n in range(96 * 21 + 51, 285 * 21 + 54)}))
    args = parser.parse_args()
    ckpt = host.Checkpoint(args.checkpoint)
    g = torch.Generator().manual_seed(0)
    dev = tdit.open_device()
    result = {"buckets": args.buckets, "rows": []}
    try:
        stub = Stub(dev, ckpt)
        w, b = stub.host_conv[0]
        first = None
        for S in args.buckets:
            x = torch.randn(S, 2048, generator=g)
            expected = torch.nn.functional.conv1d(x.t()[None], w, b, padding=15, groups=16)[0].t()
            xd = tdit._dev(dev, x.to(torch.bfloat16).reshape(1, 1, S, -1))
            if first is None:
                first = S
            else:  # weights prepared at the first bucket
                reused = ttnn.to_torch(stub._conv(xd, 0, S))[0, 0, :S].float()
            shared = stub.conv[0]
            stub.reset()
            fresh_out = stub._conv(xd, 0, S)
            ttnn.synchronize_device(dev)
            t0 = time.monotonic()
            for _ in range(5):
                o = stub._conv(xd, 0, S)
                ttnn.deallocate(o)
            ttnn.synchronize_device(dev)
            fresh = ttnn.to_torch(fresh_out)[0, 0, :S].float()
            row = {"S": S, "fresh_pcc": pcc(fresh, expected), "conv_ms": round((time.monotonic() - t0) / 5 * 1000, 2)}
            if S != first:
                row["reused_pcc"] = pcc(reused, expected)
                row["reused_equals_fresh"] = bool(torch.equal(reused, fresh))
                stub.conv[0] = first_weights
            else:
                first_weights = stub.conv[0]
            print(json.dumps(row), flush=True)
            result["rows"].append(row)
            ttnn.deallocate(xd)
    finally:
        tdit.close_device(dev)
        (Path(args.golden) / "conv_check.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
