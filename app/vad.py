"""Voice activity detection, used to stop decoding silence and ringback.

Why this exists: real call/voicemail audio is mostly *not* speech. A measured
90 s voicemail greeting had 78 s of digital silence and ringback tone and only
7 s of speech; a 72 s sales call had 59% speech. The CTC model correctly returns
nothing for the silent parts, but it still pays for them: a forward pass over
83 s of silence costs the same as 83 s of speech. Detecting speech first lets us
decode far less audio and cut pieces on real pauses instead of on the quietest
50 ms window.

Two things make this safe to add to a serving path:

* **Lazy and optional.** ``silero_vad`` (and its ``onnxruntime`` backend) are
  imported on first use, so the process still starts instantly and the package
  is not required at all when the VAD is off. A missing package degrades to
  energy-based splitting instead of failing the request.
* **No torch side effects.** ``silero_vad.model`` calls
  ``torch.set_num_threads(1)`` at import time. Left alone that would silently
  cripple CPU decoding for the whole process, so the thread count is captured
  before the import and restored afterwards.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Silero only supports 8 kHz and 16 kHz, and the engine works at 16 kHz.
VAD_SAMPLE_RATE = 16000
#: Silero consumes fixed windows: 512 samples (32 ms) at 16 kHz.
VAD_WINDOW_SAMPLES = 512


@dataclass(frozen=True)
class VadConfig:
    """Tuning for :class:`VoiceActivityDetector`."""

    enabled: bool = False
    #: Speech probability above which a 32 ms window counts as speech.
    threshold: float = 0.5
    #: Shortest run of speech worth keeping, and the silence that ends it (ms).
    min_speech_ms: int = 250
    min_silence_ms: int = 300
    #: Padding added around every speech region, in milliseconds. Silero
    #: defaults to 30 ms, which is too tight for this encoder: the first
    #: syllable of a region gets clipped ("quý khách" -> "cách"). 250 ms keeps
    #: it while still discarding long silences.
    speech_pad_ms: int = 250
    #: When true, silence-only pieces are dropped instead of decoded. The model
    #: returns "" for them, so this is a pure compute saving, not a quality
    #: trade-off.
    trim_silence: bool = True


class VoiceActivityDetector:
    """Lazily-loaded Silero VAD, safe to share across request threads.

    Silero keeps state between calls, so the model call is serialised with a
    lock. This is called from request threads (never the model thread, which
    stays free to decode), and ONNX inference on a full recording costs
    ~1% of realtime, so the serialisation is not a bottleneck.
    """

    def __init__(self, config: VadConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._load_error: str | None = None
        self._load_lock = threading.Lock()
        self._lock = threading.Lock()
        self.backend: str | None = None
        # Counters, mirrored into /metrics.
        self.calls = 0
        self.seconds = 0.0
        self.audio_seconds = 0.0
        self.kept_seconds = 0.0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def trim_silence(self) -> bool:
        return self.config.trim_silence

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> Any:
        """Load the VAD model on first use and return it."""
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                self._model = self._load()
        return self._model

    def _load(self) -> Any:
        started = time.perf_counter()
        try:
            import torch

            # ``silero_vad.model`` calls ``torch.set_num_threads(1)`` at import
            # time, which would cap CPU decoding for the whole process. Capture
            # and restore it around the import.
            threads = torch.get_num_threads()
            from silero_vad import load_silero_vad

            if torch.get_num_threads() != threads:
                torch.set_num_threads(threads)
                logger.debug(
                    "restored torch threads to %d after importing silero_vad",
                    threads,
                )

            # The ONNX session is ~2.5x faster than the TorchScript build on
            # CPU and needs no GPU, so prefer it and fall back when the
            # onnxruntime wheel is not installed.
            try:
                model = load_silero_vad(onnx=True)
                self.backend = "onnx"
            except Exception:  # pragma: no cover - depends on the install
                logger.debug("onnx backend unavailable, using torchscript", exc_info=True)
                model = load_silero_vad(onnx=False)
                self.backend = "jit"
        except ImportError as exc:
            self._load_error = (
                "silero-vad is not installed; install the 'vad' extra "
                "(pip install '.[vad]') or set USE_VAD=false"
            )
            logger.warning("%s", self._load_error)
            raise RuntimeError(self._load_error) from exc

        logger.info(
            "silero vad ready (%s) in %.2fs", self.backend, time.perf_counter() - started
        )
        return model

    # -- detection -----------------------------------------------------------

    def speech_regions(self, wav: np.ndarray) -> list[tuple[int, int]]:
        """Return ``(start, end)`` sample ranges that contain speech.

        The ranges are already expanded by ``speech_pad_ms`` and merged when the
        padding makes them touch, so they can be sliced straight out of ``wav``.
        """
        import torch
        from silero_vad import get_speech_timestamps

        model = self.load()
        audio = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))

        started = time.perf_counter()
        # get_speech_timestamps resets the model state, but it is still a
        # stateful call, so only one thread may be inside it at a time.
        with self._lock:
            stamps = get_speech_timestamps(
                audio,
                model,
                threshold=self.config.threshold,
                sampling_rate=VAD_SAMPLE_RATE,
                min_speech_duration_ms=self.config.min_speech_ms,
                min_silence_duration_ms=self.config.min_silence_ms,
                speech_pad_ms=self.config.speech_pad_ms,
                return_seconds=True,
            )
        elapsed = time.perf_counter() - started

        regions: list[tuple[int, int]] = []
        for stamp in stamps:
            start = max(0, round(stamp["start"] * VAD_SAMPLE_RATE))
            end = min(len(wav), round(stamp["end"] * VAD_SAMPLE_RATE))
            if end <= start:
                continue
            if regions and start <= regions[-1][1]:
                # Padding made two regions touch: merge them.
                regions[-1] = (regions[-1][0], max(regions[-1][1], end))
            else:
                regions.append((start, end))

        self.calls += 1
        self.seconds += elapsed
        self.audio_seconds += len(wav) / VAD_SAMPLE_RATE
        self.kept_seconds += sum(e - s for s, e in regions) / VAD_SAMPLE_RATE
        return regions

    def regions(self, wav: np.ndarray, *, keep_silence: bool) -> list[tuple[int, int]]:
        """Regions to decode, in the original timeline.

        ``keep_silence=False`` returns only the speech ranges, so the silence
        between them is never decoded. ``keep_silence=True`` returns the whole
        waveform cut at every speech boundary, which keeps the transcript
        identical to a VAD-less run while still giving the splitter real pauses
        to cut on.
        """
        speech = self.speech_regions(wav)
        if keep_silence:
            bounds = {0, len(wav)}
            for start, end in speech:
                bounds.add(start)
                bounds.add(end)
            ordered = sorted(bounds)
            return list(itertools.pairwise(ordered))
        return speech

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "loaded": self.loaded,
            "backend": self.backend,
            "trim_silence": self.trim_silence,
            "threshold": self.config.threshold,
            "speech_pad_ms": self.config.speech_pad_ms,
            "calls": self.calls,
            "seconds_total": round(self.seconds, 3),
            "audio_seconds_total": round(self.audio_seconds, 3),
            "speech_seconds_total": round(self.kept_seconds, 3),
            "load_error": self._load_error,
        }
