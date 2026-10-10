#!/usr/bin/env python3
"""RefDecoder (tt_diffrhythm/diffrhythm_host.py, float32, TT formulation) against upstream's DiT on the CPU in float32 and
against the GPU (fp16) calls recorded by dump_reference.py, for a few recorded calls (teacher forcing).

  DIFFRHYTHM_DIR=vendor/DiffRhythm vendor/DiffRhythm/.venv/bin/python experiments/tt-diffrhythm/check_ref.py golden/d96
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def pcc(a, b):
    import torch
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    golden_dir = Path(sys.argv[1]).resolve()
    picks = [int(i) for i in sys.argv[2:]] or [0, 1, 30, 31, 60, 61]
    repo = Path(os.environ["DIFFRHYTHM_DIR"]).resolve()
    sys.path[:0] = [str(ROOT), str(repo), str(repo / "infer")]
    os.chdir(repo)
    import torch
    from infer_utils import prepare_model
    from tt import diffrhythm_host as host

    torch.set_grad_enabled(False)
    golden = torch.load(golden_dir / "calls.pt")
    fixed, calls = golden["fixed"], golden["calls"]
    cfm, *_ = prepare_model(6144, "cpu")
    dit = cfm.transformer.float().eval()
    ckpt = host.Checkpoint(next((repo / "pretrained").glob("models--ASLP-lab--DiffRhythm-1_2-full/snapshots/*/cfm_model.pt")))
    ref = host.RefDecoder(ckpt)
    S = fixed["cond"].shape[1]
    geo = host.Geometry(frames=S)
    text, cond = fixed["text"].long(), fixed["cond"]
    st, dur = fixed["start_time"], fixed["duration"]
    branches = {}
    for null in (False, True):
        te = dit.text_embed(text, S, drop_text=null)
        branches[null] = ref.branch(te[0], None if null else cond[0], geo)
    results = []
    for i in picks:
        call = calls[i]
        null = call["drop_text"]
        style = fixed["negative_style_prompt"] if null else fixed["style_prompt"]
        t = call["time"].reshape(-1)
        c = dit.time_embed(t) + dit.start_time_embed(st) + dit.duration_time_embed(dur)
        rows = ref.rows(style[0], c[0])
        started = time.monotonic()
        mine = ref.forward(call["x"][0], branches[null], rows, geo)
        ref_s = time.monotonic() - started
        started = time.monotonic()
        up = dit(x=call["x"], cond=cond, text=text, time=t, drop_audio_cond=call["drop_audio_cond"],
                 drop_text=null, drop_prompt=False, style_prompt=style, start_time=st, duration=dur)[0]
        up_s = time.monotonic() - started
        rec = {"call": i, "null": null, "t": round(float(t[0]), 4), "ref_vs_upstream_fp32": pcc(mine, up),
               "upstream_fp32_vs_gpu_fp16": pcc(up, call["output"][0]), "ref_vs_gpu_fp16": pcc(mine, call["output"][0]),
               "max_abs_ref_vs_upstream": float((mine - up).abs().max()), "ref_s": round(ref_s, 1), "upstream_s": round(up_s, 1)}
        results.append(rec)
        print(json.dumps(rec), flush=True)
    (golden_dir / "check_ref.json").write_text(json.dumps({"frames": S, "seq_pad": geo.seq_pad, "calls": results}, indent=2))


if __name__ == "__main__":
    main()
