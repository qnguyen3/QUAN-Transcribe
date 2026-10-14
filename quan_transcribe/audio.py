"""Audio loading and splitting of long recordings."""

import numpy as np
import soundfile as sf
import soxr

SAMPLE_RATE = 16000


def load_audio(path: str) -> np.ndarray:
    """Any file libsndfile can read (wav, flac, ogg, mp3, ...) -> 16 kHz mono float32."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return to_16k_mono(x, sr)


def to_16k_mono(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        # (samples, channels) as returned by soundfile; also accept (channels, samples)
        x = x.mean(1) if x.shape[0] >= x.shape[1] else x.mean(0)
    return soxr.resample(x, sr, SAMPLE_RATE).astype(np.float32) if sr != SAMPLE_RATE else x


def split_quiet(x: np.ndarray, chunk_s: float, search_s: float = 5.0) -> list[np.ndarray]:
    """Split into pieces of at most chunk_s, cutting at the lowest-energy 50 ms window in the last search_s."""
    chunks, start, n = [], 0, int(chunk_s * SAMPLE_RATE)
    win = int(0.05 * SAMPLE_RATE)
    search = min(int(search_s * SAMPLE_RATE), n - win)
    while len(x) - start > n:
        lo, hi = start + n - search, start + n
        energy = np.convolve(x[lo:hi] ** 2, np.ones(win), mode="valid")
        cut = lo + int(np.argmin(energy)) + win // 2
        chunks.append(x[start:cut])
        start = cut
    chunks.append(x[start:])
    return [c for c in chunks if len(c) > SAMPLE_RATE // 10]
