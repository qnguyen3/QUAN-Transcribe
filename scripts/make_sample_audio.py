#!/usr/bin/env python3
"""Generate a small WAV for smoke-testing, using only the standard library.

This is not speech: it is a short tone with silence either side, which is enough
to exercise upload, decoding, splitting and the streaming path. Real accuracy
checks need real Vietnamese audio.

    python scripts/make_sample_audio.py tests/audio/hello.wav
"""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 16000


def build(duration: float = 2.0, tone: float = 220.0, silence: float = 0.4) -> bytes:
    frames = bytearray()
    total = int(duration * SAMPLE_RATE)
    lead = int(silence * SAMPLE_RATE)
    tail = max(0, total - lead)

    for i in range(total):
        if i < lead or i >= tail:
            amplitude = 0.0
        else:
            # Fade in and out so there are no clicks at the boundaries.
            position = (i - lead) / max(1, tail - lead)
            envelope = math.sin(math.pi * position)
            amplitude = 0.25 * envelope * math.sin(2 * math.pi * tone * i / SAMPLE_RATE)
        frames += struct.pack("<h", int(max(-1.0, min(1.0, amplitude)) * 32767))
    return bytes(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="tests/audio/hello.wav")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--tone", type=float, default=220.0)
    parser.add_argument("--silence", type=float, default=0.4)
    args = parser.parse_args()

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(build(args.duration, args.tone, args.silence))

    print(f"wrote {path} ({path.stat().st_size} bytes, 16 kHz mono PCM16)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
