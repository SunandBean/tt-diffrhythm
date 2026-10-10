# SPDX-License-Identifier: Apache-2.0
"""DiffRhythm 1.2 Full on one Tenstorrent Blackhole p100a.

The CFM transformer and the stable-audio VAE decoder run on the card in TTNN. The style
encoder, the sampler and the post-processing stay on the CPU inside DiffRhythm's own
environment. `LocalWorker` runs both in one process when one environment satisfies DiffRhythm
and ttnn at once; `TTWorker` keeps them apart, talking to the card through a worker process over
a Unix socket, when it does not.
"""
from .common import close_device, dram_stats, open_device
from .diffrhythm_dit import DiffRhythmDiT, DRPrecision
from .local import LocalWorker
from .oobleck_vae import OobleckTT
from .remote import RemoteDiffRhythmDiT, RemoteVae, TTWorker

__all__ = ["DiffRhythmDiT", "DRPrecision", "OobleckTT", "TTWorker", "LocalWorker", "RemoteDiffRhythmDiT",
           "RemoteVae", "open_device", "close_device", "dram_stats"]
__version__ = "0.1.0"
