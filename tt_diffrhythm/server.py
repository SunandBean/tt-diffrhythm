# SPDX-License-Identifier: Apache-2.0
"""HTTP server for DiffRhythm 1.2 Full on one Blackhole p100a, everything in one process.

    uvicorn tt_diffrhythm.server:app --host 0.0.0.0 --port 20000

DiffRhythm's own sampling loop runs here on the CPU and the CFM transformer and the stable-audio
VAE decoder run on the card, joined by `LocalWorker` instead of the Unix socket `remote.py` uses.
That split existed because DiffRhythm and tt-metal could not share an interpreter in the
deployment this port came from; an image built for this model alone resolves one environment that
satisfies both, so the socket is unnecessary there. See `local.py`.

Requires DiffRhythm itself (its `model`, `infer` and `g2p` trees) importable beside this one -- it
is the model, and this package only moves two of its stages onto the card.
"""
from __future__ import annotations

import base64
import io
import os
import random
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .local import LocalWorker
from .remote import RemoteDiffRhythmDiT, RemoteVae

LICENSE = "apache-2.0"
WEIGHTS_REPO = "ASLP-lab/DiffRhythm-1_2-full"
VAE_REPO = "ASLP-lab/DiffRhythm-vae"
STYLE_REPO = "OpenMuQ/MuQ-MuLan-large"
STEPS = 32  # the schedule this port's shapes are compiled for
CFG_STRENGTH = 4.0
MAX_FRAMES = 6144  # the "full" checkpoint: 285 seconds at 21.5 frames per second
MIN_DURATION, MAX_DURATION = 96, 285
SAMPLE_RATE = 44100
UPSAMPLE = 2048  # VAE frames -> audio samples
TURN_WAIT_S = float(os.environ.get("MUSIC_TURN_WAIT_S", "1800"))

# DiffRhythm reads `./config/diffrhythm-1b.json` and `./g2p/g2p/vocab.json` relative to the working
# directory, so the server runs from the checkout rather than rewriting upstream's path handling.
DIFFRHYTHM_ROOT = os.environ.get("DIFFRHYTHM_ROOT", "")

TIMESTAMP = re.compile(r"^\[(\d{1,2}):(\d{2}(?:\.\d{1,3})?)\]\s*(.*)$")
SECTION = re.compile(r"^\[[A-Za-z][^\]]*\]$")

STATE: dict = {"status": "loading", "error": None, "generating": False, "load_s": None, "dram": None}
WORKER: Optional[LocalWorker] = None
CFM = TOKENIZER = STYLE = None
LOCK = Lock()


def _root() -> Path:
    """The DiffRhythm checkout: DIFFRHYTHM_ROOT, or the one shipped beside this package."""
    if DIFFRHYTHM_ROOT:
        return Path(DIFFRHYTHM_ROOT).resolve()
    here = Path(__file__).resolve().parent.parent
    if (here / "infer" / "infer_utils.py").is_file():
        return here
    raise RuntimeError("set DIFFRHYTHM_ROOT to the DiffRhythm checkout (the directory holding infer/)")


def _point_phonemizer_at_a_bundled_espeak() -> None:
    """Let phonemizer find a libespeak-ng that came from a wheel instead of a system package.

    `g2p.utils.g2p` builds its `EspeakBackend`s at import, so espeak has to be resolvable before
    upstream is imported at all. Everywhere but a container that means `apt install espeak-ng`;
    the `espeakng-loader` wheel carries the library and its data instead. Both have to be set --
    the library alone leaves espeak looking for `phontab` under the path it was built at, which
    does not exist here.

    Does nothing when the wheel is absent, or when PHONEMIZER_ESPEAK_LIBRARY already names a
    library: a system espeak-ng is the better answer where there is one, and an explicit choice
    is not ours to override. (Asking `EspeakWrapper` whether one is set is not the way to decide
    -- `library_path` is a property on the class, so reading it off the class hands back the
    property object, which is never None.)
    """
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except ImportError:
        return
    EspeakWrapper.set_library(str(espeakng_loader.get_library_path()))
    if hasattr(EspeakWrapper, "set_data_path"):
        EspeakWrapper.set_data_path(str(espeakng_loader.get_data_path()))
    else:  # older phonemizer: espeak reads the data path from its own environment variable
        os.environ.setdefault("ESPEAK_DATA_PATH", str(espeakng_loader.get_data_path()))


def _enter_checkout() -> Path:
    """Make upstream importable and its relative paths resolve, the way its own scripts do."""
    root = _root()
    _point_phonemizer_at_a_bundled_espeak()
    for path in (root, root / "infer"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.chdir(root)
    return root


def _load_cfm(root: Path, device: str = "cpu"):
    """`infer_utils.prepare_model` without the VAE (the card decodes) and without its
    CWD-relative download cache, so the weights land in the ordinary HF cache."""
    import json

    from huggingface_hub import hf_hub_download
    from infer_utils import CNENTokenizer, load_checkpoint
    from model import CFM as CFMModel
    from model import DiT

    with open(root / "config" / "diffrhythm-1b.json") as f:
        config = json.load(f)
    cfm = CFMModel(transformer=DiT(**config["model"], max_frames=MAX_FRAMES),
                   num_channels=config["model"]["mel_dim"], max_frames=MAX_FRAMES).to(device)
    cfm = load_checkpoint(cfm, hf_hub_download(WEIGHTS_REPO, "cfm_model.pt"), device=device, use_ema=False)
    return cfm, CNENTokenizer()


def _style_model(device: str = "cpu"):
    from muq import MuQMuLan

    return MuQMuLan.from_pretrained(STYLE_REPO).to(device).eval().float()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global WORKER, CFM, TOKENIZER, STYLE
    started = time.monotonic()
    try:
        import gc

        from huggingface_hub import hf_hub_download

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        root = _enter_checkout()
        WORKER = LocalWorker(hf_hub_download(WEIGHTS_REPO, "cfm_model.pt"),
                             vae=hf_hub_download(VAE_REPO, "vae_model.pt")).start()
        cfm, TOKENIZER = _load_cfm(root)
        # Replacing the transformer drops the host copy of the big weights; the card holds them.
        cfm.transformer = RemoteDiffRhythmDiT(WORKER, cfm.transformer)
        CFM = cfm.float().eval()
        STYLE = _style_model()
        gc.collect()
        STATE.update(status="ok", load_s=round(time.monotonic() - started, 2),
                     dram=WORKER.stats.get("dram"), vae_warm_s=WORKER.stats.get("vae_warm_s"))
    except Exception as exc:
        STATE.update(status="error", error=f"{type(exc).__name__}: {exc}")
    try:
        yield
    finally:
        CFM = TOKENIZER = STYLE = None
        if WORKER is not None:
            WORKER.close()
            WORKER = None


app = FastAPI(title="DiffRhythm 1.2 Full on p100a", lifespan=lifespan, docs_url=None, redoc_url=None)


class Request(BaseModel):
    model_config = ConfigDict(extra="ignore")
    prompt: str = Field(min_length=1, max_length=4000, description="the style description")
    lyrics: str = Field(default="", max_length=20000, description="LRC, or plain lines to be timed")
    instrumental: bool = False
    duration: int = 180
    seed: Optional[int] = Field(default=None, ge=0, le=4294967295)
    inference_steps: Optional[int] = None


def _stamp(seconds: float) -> str:
    centiseconds = round(seconds * 100)
    minutes, remainder = divmod(centiseconds, 6000)
    return f"[{minutes:02d}:{remainder / 100:05.2f}]"


def _lrc(lyrics: str, duration: int, instrumental: bool) -> str:
    """Real LRC as given, or approximate starts derived for untimed lines.

    The model needs a timestamp on every line. Deriving them by word count is a convenience,
    not learned lyric alignment -- pass LRC when the timing matters."""
    if instrumental:
        return ""
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    lines = [line for line in lines if not SECTION.fullmatch(line)]
    if not lines:
        raise ValueError("vocals need lyrics; set instrumental=true for music without them")
    if len(lines) > 400:
        raise ValueError("at most 400 lyric lines")
    matches = [TIMESTAMP.match(line) for line in lines]
    if any(matches):
        if not all(matches):
            raise ValueError("use either plain lyrics or a timestamp on every line (LRC)")
        out, previous = [], -1.0
        for match in matches:
            minutes, seconds, line = match.groups()
            start = int(minutes) * 60 + float(seconds)
            if float(seconds) >= 60 or not previous <= start < duration:
                raise ValueError("LRC timestamps must be ordered and shorter than the song")
            previous = start
            if line.strip():  # an empty timed line marks a break; the tokenizer cannot encode it
                out.append(f"{_stamp(start)}{line.strip()}")
        if not out:
            raise ValueError("the LRC contains no singable lyrics")
        return "\n".join(out)
    if any(re.match(r"^\[\d", line) for line in lines):
        raise ValueError("invalid LRC timestamp; use [mm:ss.xx] before each line")
    intro, outro = min(12.0, duration * 0.06), min(10.0, duration * 0.05)
    weights = [max(3, len(line.split())) for line in lines]
    total, elapsed, out = sum(weights), 0, []
    for line, weight in zip(lines, weights):
        out.append(f"{_stamp(intro + (duration - intro - outro) * elapsed / total)}{line}")
        elapsed += weight
    return "\n".join(out)


def _wav(waveform: torch.Tensor) -> bytes:
    """[channels, samples] float -> peak-normalized 16-bit 44.1 kHz WAV."""
    import soundfile as sf

    waveform = waveform.float().cpu()
    if not torch.isfinite(waveform).all():
        raise RuntimeError("non-finite audio; retry with a different seed")
    peak = waveform.abs().max().item()
    if peak < 1e-7:
        raise RuntimeError("silent audio; retry with a different seed")
    buf = io.BytesIO()
    sf.write(buf, (waveform / peak).clamp(-1, 1).T.numpy(), SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return buf.getvalue()


@app.get("/health")
def health():
    return dict(STATE)


@app.get("/info")
def info():
    return dict(STATE) | {
        "model": WEIGHTS_REPO,
        "license": LICENSE,
        "device": "Tenstorrent Blackhole p100a",
        "on_card": ["CFM transformer", "stable-audio VAE decoder"],
        "on_cpu": ["MuQ-MuLan style encoder", "lyric tokenizer", "sampler", "post-processing"],
        "inference_steps": STEPS,
        "cfg_strength": CFG_STRENGTH,
        "task_modes": ["text-to-music"],
        "duration_limits": {"min": MIN_DURATION, "max": MAX_DURATION},
    }


@app.post("/predict")
def predict(request: Request):
    if not request.prompt.strip():
        raise HTTPException(400, "Empty prompt")
    try:
        if request.inference_steps is not None and request.inference_steps != STEPS:
            raise ValueError(f"this port runs {STEPS} steps (the schedule its shapes are compiled for)")
        if not MIN_DURATION <= request.duration <= MAX_DURATION:
            raise ValueError(f"duration must be between {MIN_DURATION} and {MAX_DURATION} seconds "
                             "(use ACE-Step for longer songs)")
        lrc = _lrc(request.lyrics, request.duration, request.instrumental)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if STATE["status"] != "ok" or CFM is None:
        raise HTTPException(503, STATE["error"] or "Model is not ready")
    if not LOCK.acquire(timeout=TURN_WAIT_S):
        raise HTTPException(409, "Another song is in progress")
    try:
        import numpy as np
        from infer_utils import (get_lrc_token, get_negative_style_prompt, get_reference_latent,
                                 get_style_prompt)

        STATE["generating"] = True
        t0 = time.perf_counter()
        seed = request.seed if request.seed is not None else random.SystemRandom().randrange(2**32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        with torch.inference_mode():
            style = get_style_prompt(STYLE, prompt=request.prompt).float()
            text, start_time, end_frame, song_duration = get_lrc_token(
                MAX_FRAMES, lrc, TOKENIZER, request.duration, "cpu")
            negative = get_negative_style_prompt("cpu").float()
            cond, segments = get_reference_latent("cpu", MAX_FRAMES, False, None, None, None)
            latents, _ = CFM.sample(
                cond=cond, text=text, duration=end_frame, max_duration=end_frame,
                song_duration=song_duration, style_prompt=style, negative_style_prompt=negative,
                steps=STEPS, cfg_strength=CFG_STRENGTH, start_time=start_time,
                latent_pred_segments=segments, batch_infer_num=1, seed=seed)
            latent = latents[0].float().transpose(1, 2).contiguous()
            vae = RemoteVae(WORKER, upsample=UPSAMPLE)
            waveform = vae(latent)[0]
        wav = _wav(waveform)
        timing = {
            "dit_s": round(sum(CFM.transformer.seconds), 2),
            "dit_calls": len(CFM.transformer.seconds),
            "vae_s": round(sum(vae.seconds), 2),
            "total_s": round(time.perf_counter() - t0, 2),
        }
        CFM.transformer.seconds.clear()
        # The card holds no song once the VAE has run, so the shim's "already prepared" cache has
        # to go with it -- the next request's text and conditioning differ and would otherwise be
        # skipped. A per-song shim would not need this; a server keeps one across songs.
        CFM.transformer.reset()
        return {
            "audio": base64.b64encode(wav).decode(),
            "format": "wav",
            "model": WEIGHTS_REPO,
            "license": LICENSE,
            "seed": seed,
            "duration": request.duration,
            "lrc": lrc,
            "inference_steps": STEPS,
            "timing_s": timing,
        }
    except Exception as exc:
        if CFM is not None:
            CFM.transformer.reset()
        if WORKER is not None and WORKER.failed:
            STATE.update(status="error", error=str(exc))
        raise HTTPException(503, str(exc)) from exc
    finally:
        STATE["generating"] = False
        LOCK.release()
