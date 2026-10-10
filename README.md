# tt-diffrhythm

**DiffRhythm 1.2 Full** ported to a single **Tenstorrent Blackhole p100a**. The CFM transformer and
the stable-audio VAE decoder run on the card in TTNN.

Model card and the full numbers:
**[sunandbean/diffrhythm-1.2-p100a](https://huggingface.co/sunandbean/diffrhythm-1.2-p100a)**

**What this port adds** — the CFM transformer and the stable-audio-tools Oobleck decoder on the card, sharing the worker, transport and host helpers with the sibling [tt-acestep](https://github.com/SunandBean/tt-acestep) port.
**What it builds on** — DiffRhythm's own pipeline (Apache-2.0), which runs unchanged on the CPU; only those two stages move.

**A 96-second song in 10.0 s** end to end in one process — the CFM transformer 7.7 s across 62
calls, the audio VAE decode 2.0 s. One DiT call at sequence 2560 is 121 ms.

An RTX 5070 Ti takes 9.7–10.1 s on the same song, so the two are level. The card gets there with
the transformer and the VAE decoder in TTNN; the GPU gets there by moving three models in and out
of 16 GB of VRAM. Numbers and method on the model card.

## One transport, two homes for it

`RemoteDiffRhythmDiT` and `RemoteVae` reach the card through exactly one method, `worker.call(msg)`,
and there are two things that answer it. `TTWorker` runs the card in its own container behind a Unix
socket, for when DiffRhythm's environment and the tt-metal runtime cannot share an interpreter — the
deployment this port came from. `LocalWorker` answers in-process, for when they can. The shims are
the same code either way.

```
  DiffRhythm venv (CPU)                    TT image (p100a)
  ─────────────────────                    ────────────────
  style encoder                ──socket──► worker.py
  CFM sampler loop                         DiffRhythmDiT  (on the card)
  post-processing                          OobleckTT      (on the card)
```

`remote.py` replaces `cfm.transformer` with a `RemoteDiffRhythmDiT`, so the upstream `CFM.sample` loop
runs **unmodified**. Tensors cross as raw little-endian bytes with a dtype and shape (`wire.py`).

## Accuracy, and why precision matters here

The torch reference in this port against upstream DiffRhythm is **PCC 0.9999997–0.9999999** per CFM
call — closer to upstream than upstream float32 is to its own GPU fp16 run (0.999995–0.999999). The
card against that reference is minimum PCC **0.99994** across a whole 96-second song.

Final latents against the GPU fp16 run:

| | Final latent PCC |
|---|---:|
| CPU float32 | 0.99997 |
| **p100a (HiFi4, default)** | **0.9954** |
| p100a (HiFi2) | 0.9931 |
| CPU bf16 | 0.99129 |

**The card beats a CPU bf16 run.** Decoded audio against the GPU is PCC 0.9721 / 2.06 dB spectral,
against 0.9691 / 3.07 dB for CPU bf16.

CFG is the reason the margin matters: `pred + 4 (pred − null)` multiplies the difference between the
two branches by four. `DRPrecision` therefore keeps activations, residual and output in fp32 and
defaults to HiFi4 matmuls — HiFi2 is 14% faster and measurably worse.

## Install

```bash
pip install -e .
```

Host side, inside the DiffRhythm environment:

```python
from tt_diffrhythm import TTWorker, RemoteDiffRhythmDiT, RemoteVae

worker = TTWorker(cfm_model_pt, vae=vae_model_pt).start()
worker.wait_ready()
cfm.transformer = RemoteDiffRhythmDiT(worker, cfm.transformer)
audio = RemoteVae(worker, upsample=2048)(latents)
```

Worker side, inside the tt-metal image:

```bash
python -m tt_diffrhythm.worker --socket /ipc/worker.sock \
  --checkpoint /checkpoint/cfm_model.pt --vae /vae/vae_model.pt
python -m tt_diffrhythm.warm_cache
```

See [`PYTHON.md`](PYTHON.md) for the protocol and the full API.

## Layout

| Path | |
|---|---|
| `tt_diffrhythm/` | the port — host side, worker side, CFM transformer, VAE, transport |
| `tt_diffrhythm/common.py` | device and host helpers shared with the tt-acestep port |
| `tt-model.yaml` | the container manifest the published image is built from — `tt-model package --container tt-model.yaml` |
| `PYTHON.md` | API reference and the socket protocol |
| `experiments/` | the verification scripts behind the published numbers — most import this port under the name it had in the private tree it was written in, so read [`experiments/README.md`](experiments/README.md) before running them |

## Related

The ACE-Step port is [tt-acestep](https://github.com/SunandBean/tt-acestep); it shares this worker,
transport and audio VAE. `oobleck_vae.py` implements both DiffRhythm's stable-audio-tools decoder and
ACE-Step's diffusers Oobleck decoder.

## Licence

Apache-2.0. The weights
([ASLP-lab/DiffRhythm-1_2-full](https://huggingface.co/ASLP-lab/DiffRhythm-1_2-full), Apache-2.0) are
not redistributed here.
