"""QUAN-ASR Server: an OpenAI-compatible ASR API server for QUAN-Transcribe.

The package exposes a FastAPI application (:mod:`app.server`) that wraps
``quan_transcribe.QuanTranscribe`` behind OpenAI's audio transcription API,
adds SSE and WebSocket streaming, and ships as a container image that GPUStack
can launch as a custom inference backend.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
