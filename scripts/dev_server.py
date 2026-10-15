"""Start the server with a stub model, for a real HTTP smoke test.

This is a development aid: it exercises the actual ASGI stack, SSE framing and
WebSocket handshake over the network without downloading the model. Not used in
production, where the real ``app.server:app`` is served by ``quan-asr-server``.

    .venv/bin/python scripts/dev_server.py --port 8123
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.audio import SAMPLE_RATE, Segment
from app.engine import StreamEvent


class StubLive:
    def __init__(self, engine):
        self.engine = engine
        self.consumed_seconds = 0.0
        self._samples = 0
        self._closed = False

    @property
    def partial(self):
        return ""

    def accept(self, samples):
        self._samples += len(samples)
        self.consumed_seconds = self._samples / SAMPLE_RATE
        if self._samples < int(0.48 * SAMPLE_RATE):
            return [StreamEvent("partial", "xin", 0.0, self.consumed_seconds)]
        return [StreamEvent("final", "stub final", 0.0, self.consumed_seconds)]

    def flush(self):
        self._closed = True
        return "stub flush"

    def close(self):
        self._closed = True


class StubEngine:
    """Mirrors ``TranscriptionEngine``'s public surface."""

    def __init__(self, settings):
        self.settings = settings
        self.loaded_at = 1_700_000_000.0
        self.load_error = None
        self.requests = 0

    @property
    def is_loaded(self):
        return True

    def load(self):
        return self

    def warmup(self):
        return None

    def close(self):
        return None

    def describe_device(self):
        return "stub"

    def info(self):
        return {
            "model": self.settings.served_model_name,
            "model_path": self.settings.model_path,
            "loaded": True,
            "device": "stub",
            "profile": self.settings.profile,
            "language": "vi",
            "lm": False,
            "beam_width": self.settings.beam_width,
            "chunk_seconds": 0.48,
            "stream_step_seconds": self.settings.stream_step_seconds,
            "left_chunks": self.settings.left_chunks,
            "microbatch_seconds": self.settings.microbatch_seconds,
            "microbatch_max_requests": self.settings.microbatch_max_requests,
            "microbatch_max_wait_ms": self.settings.microbatch_max_wait_ms,
            "max_queue": self.settings.max_queue,
            "max_interactive": self.settings.max_interactive,
            "subbatch_seconds": self.settings.subbatch_seconds,
            "tf32": self.settings.tf32,
            "load_error": None,
            "load_seconds": 0.0,
            "loaded_at": self.loaded_at,
        }

    def metrics(self):
        return {
            "requests_total": self.requests,
            "loaded": True,
            "core": None,
        }

    def transcribe(self, wav, **kwargs):
        self.requests += 1
        duration = len(wav) / SAMPLE_RATE
        mid = duration / 2
        return [
            Segment("xin chào", 0.0, mid),
            Segment("tôi tên là nam", mid, duration),
        ]

    def iter_stream(self, wav, **kwargs):
        self.requests += 1
        total = len(wav) / SAMPLE_RATE
        yield StreamEvent("partial", "xin", 0.0, 0.48)
        yield StreamEvent("final", "xin chào", 0.0, total / 2)
        yield StreamEvent("partial", "tôi tên", total / 2, total / 2 + 0.48)
        yield StreamEvent("final", "tôi tên là nam", total / 2, total)
        yield StreamEvent("done", "", 0.0, total)

    def new_live_session(
        self, left_chunks=None, chunk_seconds=None, use_lm=None, beam_width=None
    ):
        return StubLive(self)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    import uvicorn

    import app.server as server

    server.engine = StubEngine(server.settings)
    _ = np  # keep the import meaningful for readers of the stub above

    print(f"stub server on http://{args.host}:{args.port} (no model loaded)")
    uvicorn.run(server.app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
