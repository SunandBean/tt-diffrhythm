#!/usr/bin/env python3
"""Decode saved final latents (tt_latents.pt from device_check.py --closed-loop, cpu_latents_*.pt from cpu_loop.py)
with upstream's VAE exactly as the GPU runner does (decode_audio, chunked, on the CPU) and compare each with the GPU
run's audio (reference.wav): same starting noise, so differences are numerics only. Writes <name>.wav for listening.

  DIFFRHYTHM_DIR=vendor/DiffRhythm vendor/DiffRhythm/.venv/bin/python experiments/tt-diffrhythm/decode_latents.py \\
      golden/d96 tt_latents.pt cpu_latents_bf16.pt
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    golden_dir = Path(sys.argv[1]).resolve()
    repo = Path(os.environ["DIFFRHYTHM_DIR"]).resolve()
    sys.path[:0] = [str(ROOT / "runners"), str(repo), str(repo / "infer")]
    os.chdir(repo)
    import soundfile as sf
    import torch
    import diffrhythm
    from infer_utils import decode_audio

    torch.set_grad_enabled(False)
    vae_path = next((repo / "pretrained").glob("models--ASLP-lab--DiffRhythm-vae/snapshots/*/vae_model.pt"))
    vae = torch.jit.load(str(vae_path), map_location="cpu").eval()
    ref, sr = sf.read(str(golden_dir / "reference.wav"), dtype="float32")
    ref = torch.from_numpy(ref.T.copy())

    def spectral(a, b, n_fft=2048, hop=512):
        spec = lambda w: torch.stft(w.mean(0), n_fft, hop, window=torch.hann_window(n_fft), return_complex=True).abs()
        da, db = 20 * torch.log10(spec(a) + 1e-7), 20 * torch.log10(spec(b) + 1e-7)
        return float((da - db).abs().mean())

    out = {}
    for name in sys.argv[2:]:
        z = torch.load(golden_dir / name).float()  # [S, 64]
        audio = decode_audio(z.t()[None].contiguous(), vae, chunked=True)[0]
        wav = golden_dir / (Path(name).stem + ".wav")
        diffrhythm.save_audio(audio, wav, "")
        mine, _ = sf.read(str(wav), dtype="float32")
        mine = torch.from_numpy(mine.T.copy())
        n = min(mine.shape[-1], ref.shape[-1])
        a, b = mine[:, :n], ref[:, :n]
        out[name] = {"audio_pcc": float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1]),
                     "spectral_db": spectral(a, b), "wav": str(wav)}
        print(json.dumps({name: out[name]}), flush=True)


if __name__ == "__main__":
    main()
