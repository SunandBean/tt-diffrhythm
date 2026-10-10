#!/usr/bin/env python3
"""What the TT DiT needs from the DiffRhythm process, for recorded golden calls: the text embedding of each CFG
branch (upstream TextEmbedding, float32 CPU), the audio condition, and per call x, the branch, the style prompt and
c = time + start + duration embeddings; plus the GPU output. The TT image has no DiffRhythm code, so
device_check.py reads this file (plain torch tensors) instead.

  DIFFRHYTHM_DIR=vendor/DiffRhythm vendor/DiffRhythm/.venv/bin/python experiments/tt-diffrhythm/export_inputs.py golden/d96
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    golden_dir = Path(sys.argv[1]).resolve()
    repo = Path(os.environ["DIFFRHYTHM_DIR"]).resolve()
    sys.path[:0] = [str(repo), str(repo / "infer")]
    os.chdir(repo)
    import torch
    from infer_utils import prepare_model

    torch.set_grad_enabled(False)
    golden = torch.load(golden_dir / "calls.pt")
    fixed, calls = golden["fixed"], golden["calls"]
    cfm, *_ = prepare_model(6144, "cpu")
    dit = cfm.transformer.float().eval()
    S = fixed["cond"].shape[1]
    text = fixed["text"].long()
    st, dur = fixed["start_time"], fixed["duration"]
    out = {"frames": S, "cond": fixed["cond"][0].clone(),
           "text_embed": {null: dit.text_embed(text, S, drop_text=null)[0].clone() for null in (False, True)},
           "calls": []}
    for call in calls:
        null = call["drop_text"]
        style = fixed["negative_style_prompt"] if null else fixed["style_prompt"]
        t = call["time"].reshape(-1)
        c = dit.time_embed(t) + dit.start_time_embed(st) + dit.duration_time_embed(dur)
        out["calls"].append({"x": call["x"][0].clone(), "drop_text": null, "t": float(t[0]), "style": style[0].clone(),
                             "c": c[0].clone(), "gpu": call["output"][0].clone()})
    torch.save(out, golden_dir / "tt_inputs.pt")
    print(f"{len(out['calls'])} calls, {S} frames -> {golden_dir / 'tt_inputs.pt'}")


if __name__ == "__main__":
    main()
