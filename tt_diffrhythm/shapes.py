"""Shapes shared by the host pipeline and the TT worker (no torch or ttnn imports)."""

# VAE windows in latent frames (remote.py RemoteVae picks the largest that fits the song); the worker builds
# these programs at start and warm_cache.py compiles them ahead of time
VAE_WINDOWS = (1024, 512)
