"""Quick-and-dirty energy-envelope probe for one or more audio files.

Usage:  python scripts/probe_audio.py FILE [FILE ...]

Prints duration, a 100 ms RMS envelope (as a text map) and the share of the
file that is "speech-ish" vs silence. Useful to tell apart a real pipeline
truncation bug from an audio file that is mostly silence.
"""

from __future__ import annotations

import array
import math
import subprocess
import sys


def decode(path: str, sr: int = 16000) -> array.array:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
        check=True,
        capture_output=True,
    ).stdout
    a = array.array("h")
    a.frombytes(out)
    return a


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: probe_audio.py FILE [FILE ...]", file=sys.stderr)
        return 2
    sr = 16000
    win = int(0.1 * sr)  # 100 ms
    for path in argv:
        a = decode(path, sr)
        dur = len(a) / sr
        rms: list[float] = []
        for i in range(0, len(a) - win, win):
            chunk = a[i : i + win]
            s = sum(v * v for v in chunk)
            rms.append(math.sqrt(s / len(chunk)))
        peak = max(rms) if rms else 1.0
        hi, lo = peak * 0.08, peak * 0.02
        speech = sum(1 for r in rms if r > hi)
        print(f"=== {path} ===")
        print(f"duration      : {dur:.2f}s  ({len(rms)} x 100ms windows)")
        print(f"peak RMS      : {peak:.0f}")
        print(f"speech-ish    : {speech} windows = {speech * 0.1:.2f}s ({speech / max(len(rms), 1):.0%})")
        print(f"silence-ish   : {len(rms) - speech} windows = {(len(rms) - speech) * 0.1:.2f}s")
        print("envelope      : # speech-ish   : quiet   . silence   (1 char = 100 ms)")
        print("".join("#" if r > hi else (":" if r > lo else ".") for r in rms))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
