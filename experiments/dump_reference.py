#!/usr/bin/env python3
"""Golden data for the DiffRhythm 1.2 port to the P100a: the production GPU path, recorded.

Calls the host pipeline's run() unchanged (GPU, fp16) and wraps upstream's prepare_model and decode_audio to
record every call of the DiT (cfm.transformer): its inputs and output, per step and CFG branch; plus the latents
given to the VAE and the audio it returns. Run with DiffRhythm's own environment and the GPU free:

  CUDA_VISIBLE_DEVICES=<gpu> DIFFRHYTHM_DIR=vendor/DiffRhythm PHONEMIZER_ESPEAK_LIBRARY=... \
      vendor/DiffRhythm/.venv/bin/python experiments/tt-diffrhythm/dump_reference.py --duration 96 --out ...
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROMPT = "warm indie pop, gentle guitars, soft female vocal, 100 bpm"
LYRICS = """[Verse]
Lights along the harbor
Paper boats and summer rain
[Chorus]
We take turns with the night
Hold the melody again"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(os.environ["DIFFRHYTHM_DIR"]).resolve()
    sys.path[:0] = [str(ROOT / "runners"), str(repo), str(repo / "infer")]
    os.chdir(repo)
    import torch
    import infer_utils
    import diffrhythm as runner

    calls, fixed, vae_io = [], {}, {}
    prepare, decode = infer_utils.prepare_model, infer_utils.decode_audio

    def recording_prepare(*a, **kw):
        cfm, tokenizer, muq, vae = prepare(*a, **kw)

        def before(module, args_, kwargs):
            kw_ = dict(kwargs)
            if not fixed:
                fixed.update({k: kw_[k].detach().float().cpu().clone() for k in ("cond", "text", "start_time", "duration")})
                fixed["style_prompt"] = kw_["style_prompt"].detach().float().cpu().clone()
            if kw_["drop_text"] and "negative_style_prompt" not in fixed:
                fixed["negative_style_prompt"] = kw_["style_prompt"].detach().float().cpu().clone()
            calls.append({"x": kw_["x"].detach().float().cpu().clone(), "time": kw_["time"].detach().float().cpu().clone(),
                          "drop_audio_cond": bool(kw_["drop_audio_cond"]), "drop_text": bool(kw_["drop_text"]),
                          "drop_prompt": bool(kw_.get("drop_prompt", False)), "started": time.monotonic()})

        def after(module, args_, kwargs, output):
            calls[-1]["output"] = output.detach().float().cpu().clone()
            calls[-1]["seconds"] = time.monotonic() - calls[-1].pop("started")

        cfm.transformer.register_forward_pre_hook(before, with_kwargs=True)
        cfm.transformer.register_forward_hook(after, with_kwargs=True)
        return cfm, tokenizer, muq, vae

    def recording_decode(latents, vae, *a, **kw):
        audio = decode(latents, vae, *a, **kw)
        vae_io.update(latents=latents.detach().float().cpu().clone(), audio=audio.detach().float().cpu().clone())
        return audio

    infer_utils.prepare_model, infer_utils.decode_audio = recording_prepare, recording_decode
    request = {"prompt": PROMPT, "lyrics": LYRICS, "duration": args.duration, "instrumental": False, "seed": args.seed}
    started = time.monotonic()
    runner.run(request, out / "reference.wav")
    torch.save({"fixed": fixed, "calls": calls, "vae": vae_io}, out / "calls.pt")
    meta = {"prompt": PROMPT, "duration": args.duration, "seed": args.seed, "calls": len(calls),
            "latent_shape": list(calls[0]["x"].shape), "cond_shape": list(fixed["cond"].shape),
            "text_shape": list(fixed["text"].shape), "times": [round(float(c["time"].flatten()[0]), 4) for c in calls[::2]],
            "gpu_call_seconds": round(sum(c["seconds"] for c in calls), 2),
            "vae_latents": list(vae_io["latents"].shape), "vae_audio": list(vae_io["audio"].shape),
            "total_seconds": round(time.monotonic() - started, 1), "torch": torch.__version__}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
