"""Transcribe real files through the engine, with and without VAD.

This is the local end-to-end check: it loads the actual QUAN model inside the
container, runs the same code path the API uses (``TranscriptionEngine``), and
prints a per-segment breakdown so "the transcript looks truncated" can be told
apart from "the audio really is mostly silence".

A file is decoded twice: once with VAD off (the baseline) and once with VAD on,
which shows both the compute saved and any change in the text.

Usage (from the repository root, inside the image):

    python scripts/repro_local.py FILE [FILE ...]

Environment:
    DEVICE=cpu|cuda            (default cpu)
    USE_LM=true                beam-search with the loaded KenLM
    CHUNK_SECONDS=30
    VAD_TRIM_SILENCE=false     keep silence, only use VAD for cut points
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from app.audio import decode_audio_bytes, to_16k_mono
from app.config import Settings
from app.engine import TranscriptionEngine
from app.formats import transcript_text


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def run(engine: TranscriptionEngine, wav: np.ndarray, chunk_seconds: float):
    started = time.perf_counter()
    segments = engine.transcribe(wav, chunk_seconds=chunk_seconds)
    return segments, time.perf_counter() - started


def show(label: str, segments, elapsed: float, duration: float) -> None:
    text = transcript_text(segments)
    print(f"  {label}")
    for segment in segments:
        print(
            f"    [{segment.start:7.2f} - {segment.end:7.2f}] "
            f"{'speech' if segment.text else 'silent':6s} :: {segment.text!r}"
        )
    print(
        f"    -> {len(segments)} segment(s), {elapsed:.2f}s "
        f"(RTF {elapsed / max(duration, 1e-9):.3f})"
    )
    print(f"    -> {json.dumps({'text': text}, ensure_ascii=False)}")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.WARNING)

    chunk_seconds = float(os.getenv("CHUNK_SECONDS", "30"))
    base = Settings.from_env().merge(device=os.getenv("DEVICE", "cpu"))

    print("loading model ...", flush=True)
    engine = TranscriptionEngine(base.merge(use_vad=False))
    started = time.perf_counter()
    engine.load()
    print(f"loaded in {time.perf_counter() - started:.1f}s\n", flush=True)

    global segments_audio
    for path in argv:
        with open(path, "rb") as handle:
            wav, sr = decode_audio_bytes(handle.read())
        wav = to_16k_mono(wav, sr)
        segments_audio = wav
        duration = len(wav) / 16000

        engine.settings = base.merge(use_vad=False)
        plain, plain_s = run(engine, wav, chunk_seconds)

        engine.settings = base.merge(use_vad=True)
        vad, vad_s = run(engine, wav, chunk_seconds)

        print(f"=== {os.path.basename(path)} ({duration:.2f}s) ===")
        show(f"VAD off  ({duration:.2f}s decoded)", plain, plain_s, duration)
        stats = engine._vad.stats()
        print(
            f"  VAD on   ({stats['speech_seconds_total']:.2f}s speech kept, "
            f"backend={stats['backend']})"
        )
        for segment in vad:
            print(
                f"    [{segment.start:7.2f} - {segment.end:7.2f}] "
                f"{'speech' if segment.text else 'empty ':6s} :: {segment.text!r}"
            )
        print(f"    -> {len(vad)} segment(s), {vad_s:.2f}s "
              f"(RTF {vad_s / max(duration, 1e-9):.3f})")
        print(f"    -> {json.dumps({'text': transcript_text(vad)}, ensure_ascii=False)}")
        same = transcript_text(plain) == transcript_text(vad)
        saved = 1 - vad_s / plain_s if plain_s else 0.0
        print(f"  same transcript: {same}   decode time saved: {saved:.0%}\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
