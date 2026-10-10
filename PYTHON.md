# tt_diffrhythm — Python reference

DiffRhythm 1.2 Full on one Tenstorrent Blackhole p100a. The CFM transformer and the stable-audio VAE
decoder run on the card; everything else stays on the CPU in DiffRhythm's own environment.

## Two processes

DiffRhythm's environment and the tt-metal runtime need different torch and numpy builds, so they run
as two processes talking over a Unix socket in a directory bind-mounted into both. Tensors cross as
raw little-endian bytes with a dtype and a shape (`wire.py`).

## Host side (inside the DiffRhythm environment)

### `TTWorker(checkpoint, image=None, log=None, reply_timeout_s=120.0, model="diffrhythm", vae=None)`

`checkpoint` is `cfm_model.pt` and `vae` is `vae_model.pt`; Hugging Face cache links are resolved.
`image` is the tt-metal / ttnn image the port was built against; it has no portable default, so set
it here or through `MUSIC_TT_IMAGE`. `.start()` starts the worker container on the card,
`.wait_ready(timeout_s)` blocks until the weights are on the device, `.close()` stops the container and
waits for it to exit, so the card is free only after that.

### `RemoteDiffRhythmDiT(worker, transformer)`

A drop-in for `cfm.transformer` in `CFM.sample` (batch 1): same call, same output. Assign it and the
upstream sampler runs unchanged.

### `RemoteVae(worker, upsample=2048)`

`[B, 64, T]` latents to `[B, 2, T * upsample]` float32 audio on the CPU. It picks the largest VAE
window from `shapes.VAE_WINDOWS` that fits the song.

## Worker side (inside the tt-metal image)

```bash
python -m tt_diffrhythm.worker --socket /ipc/worker.sock \
  --checkpoint /checkpoint/cfm_model.pt --vae /vae/vae_model.pt
```

Protocol (plain dicts, tensors via `wire`):

| | |
|---|---|
| `-> {"op": "ready", ...}` | once the weights are on the device |
| `<- {"op": "prepare", "text_embed": {drop_text: [T, 512]}, "cond": [T, 64]}` | `-> {"op": "prepared", ...}` |
| `<- {"op": "forward", "x": [T, 64], "drop_text": bool, "style": [512], "c": [512]}` | `-> {"op": "velocity", "v": [T, 64]}` |

## Direct use on the card

```python
from tt_diffrhythm import DiffRhythmDiT, DRPrecision, OobleckTT, open_device, close_device
from tt_diffrhythm.diffrhythm_host import Checkpoint

dev = open_device()
dit = DiffRhythmDiT(dev, Checkpoint("cfm_model.pt"))     # DRPrecision() by default
vae = OobleckTT(dev, "vae_model.pt")
...
close_device(dev)
```

### `DRPrecision`

Extends the shared `Precision` with the choices CFG forces. `pred + 4 (pred − null)` multiplies the
difference between the two branches by four, so the activations, the residual and the output stay in
fp32 and matmuls default to **HiFi4**. HiFi2 is about 14% faster and measurably worse
(final latent PCC 0.9931 against 0.9954).

## Kernel cache

```bash
python -m tt_diffrhythm.warm_cache
```

Compiles every sequence bucket of a 96–285 s song, both CFG branches, and the VAE windows in
`shapes.VAE_WINDOWS`. The first decode at a VAE shape costs 33.9 s against 0.33 s warm, so this is
worth running after any change to the code or the image.

## Modules

| Module | Role |
|---|---|
| `remote.py` | host side: `TTWorker`, `RemoteDiffRhythmDiT`, `RemoteVae`, the worker container |
| `worker.py` | worker side: loads the CFM transformer and the VAE and answers one host pipeline |
| `wire.py` | the tensor transport between the two torch builds |
| `shapes.py` | shapes both sides need, with no torch or ttnn import |
| `diffrhythm_dit.py` | the CFM transformer on the card, and `DRPrecision` |
| `diffrhythm_host.py` | checkpoint access, the torch reference and the sequence buckets |
| `oobleck_vae.py` | the audio VAE decoder on the card |
| `common.py` | device and host helpers shared with the tt-acestep port |
| `warm_cache.py` | ahead-of-time kernel compilation for every bucket |
