"""Audio decoding, resampling and sentence-aware splitting.

The split logic mirrors ``quan_transcribe.audio.split_quiet`` cut-for-cut so the
segments reported by the API line up with the pieces the model actually decoded,
but this version also returns sample offsets so we can build timestamps.
"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MIN_PIECE_SECONDS = 0.1


class AudioDecodeError(ValueError):
    """Raised when the uploaded bytes are not decodable audio."""


@dataclass
class Segment:
    """One decoded piece of audio with its position in the source file."""

    text: str
    start: float
    end: float
    phonemes: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    try:
        import soxr

        return soxr.resample(np.asarray(x, dtype=np.float32), sr_in, sr_out).astype(
            np.float32
        )
    except ImportError:  # pragma: no cover - soxr ships with quan-transcribe
        logger.warning("soxr is unavailable; falling back to linear resampling")
        n_out = round(len(x) * sr_out / sr_in)
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        t_in = np.arange(len(x), dtype=np.float64)
        t_out = np.linspace(0.0, len(x) - 1, n_out) if len(x) > 1 else np.zeros(1)
        return np.interp(t_out, t_in, np.asarray(x, dtype=np.float64)).astype(
            np.float32
        )


def to_16k_mono(x: np.ndarray, sr: int) -> np.ndarray:
    """Down-mix to mono and resample to 16 kHz, matching the model's pipeline."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        # soundfile returns (samples, channels); also accept (channels, samples).
        x = x.mean(1) if x.shape[0] >= x.shape[1] else x.mean(0)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1).mean(1)
    return _resample(x, int(sr), SAMPLE_RATE)


def _decode_with_ffmpeg(data: bytes) -> tuple[np.ndarray, int]:
    """Decode anything ffmpeg understands (m4a, webm/opus, ...) to raw f32le."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioDecodeError(
            "unsupported audio container and ffmpeg is not installed in the image"
        )
    proc = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-vn",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "pipe:1",
        ],
        input=data,
        capture_output=True,
        check=False,  # the return code is inspected below
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise AudioDecodeError(
            "ffmpeg could not decode the audio"
            + (f": {detail[-1]}" if detail else "")
        )
    samples = np.frombuffer(proc.stdout, dtype="<f4")
    if samples.size == 0:
        raise AudioDecodeError("decoded audio is empty")
    return samples.astype(np.float32), SAMPLE_RATE


def decode_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode uploaded bytes into ``(mono float32, sample_rate)``.

    libsndfile (through soundfile) covers wav/flac/ogg/mp3. Anything else — most
    importantly m4a and webm/opus, which browsers produce — goes to ffmpeg.
    """
    if not data:
        raise AudioDecodeError("the uploaded file is empty")

    try:
        import soundfile as sf

        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:  # any libsndfile failure means "try ffmpeg"
        logger.debug("soundfile could not decode the upload: %s", exc)
        return _decode_with_ffmpeg(data)

    if x.size == 0:
        raise AudioDecodeError("the uploaded file contains no audio samples")
    return x, int(sr)


def _split_range(
    x: np.ndarray, start: int, end: int, span: int, search: int, win: int
) -> list[tuple[np.ndarray, int, int]]:
    """Cut ``x[start:end]`` into pieces of at most ``span`` samples.

    This is the exact ``quan_transcribe.audio.split_quiet`` rule, applied to a
    sub-range instead of the whole waveform so VAD regions can be split
    independently.
    """
    out: list[tuple[np.ndarray, int, int]] = []
    pos = start
    while end - pos > span:
        lo, hi = pos + span - search, pos + span
        energy = np.convolve(x[lo:hi] ** 2, np.ones(win), mode="valid")
        cut = lo + int(np.argmin(energy)) + win // 2
        out.append((x[pos:cut], pos, cut))
        pos = cut
    out.append((x[pos:end], pos, end))
    return out


def split_with_offsets(
    x: np.ndarray,
    chunk_s: float,
    search_s: float = 5.0,
    regions: list[tuple[int, int]] | None = None,
) -> list[tuple[np.ndarray, int, int]]:
    """Cut ``x`` into pieces of at most ``chunk_s`` at the quietest nearby point.

    Returns ``(samples, start_index, end_index)`` triples. With no ``regions``
    the cut points are identical to ``quan_transcribe.audio.split_quiet`` and
    the pieces tile the whole waveform; the same short-piece filter is applied,
    so piece *k* here is exactly what the model decoded.

    ``regions`` restricts — and splits — the work to those ``(start, end)``
    sample ranges. A VAD passes the speech ranges it found so silence between
    them is never cut up or decoded. Each region is split independently, so the
    sub-pieces of one region never span into another.
    """
    x = np.asarray(x, dtype=np.float32)
    span = int(chunk_s * SAMPLE_RATE)
    if span <= 0:
        raise ValueError("chunk_s must be positive")

    if regions is None:
        regions = [(0, len(x))]
    win = int(0.05 * SAMPLE_RATE)
    # Never search back further than the span itself: for a very small span a
    # 5 s search would reach before the range start.
    search = min(int(search_s * SAMPLE_RATE), span - win)

    pieces: list[tuple[np.ndarray, int, int]] = []
    for start, end in regions:
        start = max(0, int(start))
        end = min(len(x), int(end))
        if end - start <= 0:
            continue
        pieces.extend(_split_range(x, start, end, span, search, win))

    minimum = int(MIN_PIECE_SECONDS * SAMPLE_RATE)
    return [p for p in pieces if len(p[0]) > minimum]


def build_segments(
    texts: list[str], offsets: list[tuple[int, int]]
) -> list[Segment]:
    """Zip decoded texts with ``split_with_offsets`` offsets into segments."""
    segments = []
    for text, (start, end) in zip(texts, offsets, strict=False):
        start_s = start / SAMPLE_RATE
        end_s = end / SAMPLE_RATE
        segments.append(
            Segment(text=(text or "").strip(), start=start_s, end=end_s)
        )
    return segments


def splice_segments(
    head: Segment, tail: Segment, text: str
) -> Segment:
    """Merge two temporally adjacent segments into one carrying ``text``."""
    return Segment(text=text, start=head.start, end=tail.end)
