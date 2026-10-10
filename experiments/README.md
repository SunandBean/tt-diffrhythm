# Verification scripts

| Script | What it checks |
|---|---|
| `dump_reference.py`, `export_inputs.py` | Record the upstream DiffRhythm run everything is compared against |
| `check_ref.py` | This port's torch reference against upstream float32 and against the GPU fp16 run |
| `cpu_loop.py` | The CPU float32 and bf16 baselines the device is measured against |
| `device_check.py` | Every CFM call on the card against the reference, across a whole song |
| `decode_latents.py` | Decoded audio against the GPU: PCC and spectral distance |
| `dit_profile.py` | Where a 121 ms DiT call goes — matmul 66.6 ms, SDPA 21.9 ms, and the rest |
| `mm_sweep.py`, `sdpa_check.py` | Matmul fidelity and SDPA chunking / masking sweeps |
| `conv_check.py`, `conv_speed.py` | The convolution path: dense against bmm |
| `vae_check.py` | The audio VAE on the card against the CPU |
| `run_on_card.sh` | Runs any of these in the TT image (`MUSIC_TT_IMAGE`) |



## Running these outside the tree they were written in

These are the scripts as they were run, inside the private working tree this port was developed in.
They are published as the record behind the numbers on the model card, and most of them need two
edits before they will run from a clone of this repo:

1. **The package name.** Nine of them (`check_ref.py`, `conv_check.py`, `conv_speed.py`,
   `device_check.py`, `dit_profile.py`, `mm_sweep.py`, `sdpa_check.py`, `vae_check.py`) do
   `from tt import diffrhythm_dit, ...`. `tt` was this port's package inside the private monorepo;
   it is published here as **`tt_diffrhythm`**.
2. **The `ROOT` line.** Those scripts set `ROOT = Path(__file__).resolve().parents[2]`, the monorepo
   root, and put it on `sys.path`. From a clone the repository root is `parents[1]`.

`check_ref.py`, `cpu_loop.py`, `decode_latents.py`, `dump_reference.py` and `export_inputs.py` import
`infer_utils` and friends from an upstream [DiffRhythm](https://github.com/ASLP-lab/DiffRhythm)
checkout — set `DIFFRHYTHM_DIR` to it. `decode_latents.py` and `dump_reference.py` also import the
deployment's own runner, which is not published. `mm_sweep.py` needs tt-metal's `models/` tree.
`run_on_card.sh` uses the same monorepo layout (`vendor/DiffRhythm/pretrained`,
`experiments/tt-diffrhythm/`). The device scripts need a p100a.
