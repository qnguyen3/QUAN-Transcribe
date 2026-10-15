"""Model-owning wrapper around ``quan_transcribe``.

Two properties drive the design:

* **Loading is slow and expensive.** The weights plus Google's
  ``google/embeddinggemma-2`` audio tower must be downloaded and read once, then
  kept resident. Loading happens behind a lock, so a burst of concurrent
  requests triggers exactly one load.
* **The model is not re-entrant here.** Decoding is a plain forward pass, but it
  mutates shared buffers, so exactly one thread may touch the model at a time.
  Rather than a lock held for a whole decode, all access goes through
  :class:`~app.core.EngineCore`: a dedicated model thread with a priority,
  coalescing scheduler and bounded queues. See ``app/core.py`` for why.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from .audio import SAMPLE_RATE, Segment
from .config import Settings
from .core import EngineBusy, EngineCore
from .vad import VoiceActivityDetector

__all__ = [
    "EngineBusy",
    "EngineNotReady",
    "StreamEvent",
    "TranscriptionEngine",
]

logger = logging.getLogger(__name__)


class EngineNotReady(RuntimeError):
    """Raised when the model cannot be loaded."""


@dataclass
class StreamEvent:
    """One incremental update from a streaming decode."""

    kind: str  # "partial" | "final" | "done"
    text: str
    start: float
    end: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
        }


class TranscriptionEngine:
    """Loads QUAN-Transcribe once and serves transcription calls."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._asr: Any | None = None
        self._load_lock = threading.Lock()
        self._core_lock = threading.Lock()
        self._load_error: str | None = None
        self._load_seconds: float | None = None
        self._loaded_at: float | None = None
        self.requests = 0
        self.requests_success = 0
        self.requests_failed = 0
        self.audio_seconds = 0.0
        self.decode_seconds = 0.0
        self._is_loading: bool = False
        # The one thread that touches the model. Created on load, once the
        # sample-rate-derived budget can be trusted, and always present: even a
        # single request runs through it, so there is exactly one code path.
        self._core: EngineCore | None = None
        # Voice activity detection runs on request threads, never the model
        # thread, so it never blocks a decode. Created on first use.
        self._vad: VoiceActivityDetector | None = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._asr is not None

    @property
    def is_loading(self) -> bool:
        return self._is_loading

    @property
    def is_ready(self) -> bool:
        return self._asr is not None and self._load_error is None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def loaded_at(self) -> float | None:
        """Unix timestamp of the successful load, or ``None``."""
        return self._loaded_at

    def load(self) -> Any:
        """Load the model if needed and return the ``QuanTranscribe`` instance."""
        if self._asr is not None:
            self._ensure_core()
            return self._asr
        with self._load_lock:
            if self._asr is not None:
                self._ensure_core()
                return self._asr
            if self._load_error is not None:
                raise EngineNotReady(self._load_error)
            self._is_loading = True
            try:
                self._asr = self._load()
                return self._asr
            finally:
                self._is_loading = False

    def _ensure_core(self) -> EngineCore:
        """Return the model thread, starting it exactly once.

        Called whenever the model is known to be resident, so the invariant
        "loaded implies a core exists" holds no matter how ``_asr`` was set
        (a real load, or a test injecting a stub).
        """
        if self._core is not None:
            return self._core
        with self._core_lock:
            if self._core is None:
                self._start_core()
        return self._core

    def _load(self) -> Any:
        settings = self.settings
        import os

        if settings.hf_token:
            os.environ.setdefault("HF_TOKEN", settings.hf_token)

        try:
            from quan_transcribe import QuanTranscribe
        except ImportError as exc:  # pragma: no cover - image always has it
            raise EngineNotReady(
                "the 'quan-transcribe' package is not installed"
            ) from exc

        lm: bool | str = False
        if settings.lm_path:
            lm = settings.lm_path
        elif settings.use_lm:
            lm = True

        started = time.time()
        logger.info(
            "loading model %s (device=%s, lm=%s)",
            settings.model_path,
            settings.device or "auto",
            lm,
        )
        try:
            asr = QuanTranscribe.from_pretrained(
                settings.model_path, device=settings.device, lm=lm
            )
        except Exception as exc:
            self._load_error = f"failed to load {settings.model_path}: {exc}"
            logger.exception("model load failed")
            raise EngineNotReady(self._load_error) from exc

        # ``lm_config.json`` ships tuned values; an explicit setting wins.
        decoder = getattr(asr, "lm_decoder", None)
        if decoder is not None:
            if settings.lm_alpha is not None:
                decoder.alpha = settings.lm_alpha
            if settings.lm_beta is not None:
                decoder.beta = settings.lm_beta
            logger.info(
                "kenlm beam search ready (beam_width=%d, alpha=%s, beta=%s)",
                settings.beam_width,
                decoder.alpha,
                decoder.beta,
            )

        self._load_seconds = time.time() - started
        self._loaded_at = time.time()
        self._configure_torch(asr)
        self._ensure_core()
        logger.info(
            "model ready in %.1fs on %s (profile=%s)",
            self._load_seconds,
            asr.device,
            self.settings.profile,
        )
        return asr

    def _configure_torch(self, asr: Any) -> None:
        """Apply the throughput knobs that only take effect before the first pass.

        TF32 lets Ampere and later run FP32 matmuls on the tensor cores. For a
        CTC model the precision loss is irrelevant next to the speedup, so it is
        on by default; ``TF32=false`` restores exact FP32.
        """
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is in the image
            return

        if self.settings.tf32 and asr.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 matmuls enabled")

        # On CPU the model is a plain FP32 GEMM pipeline and scales with cores,
        # so an explicit thread count is the main lever there. Leaving it unset
        # keeps torch's own heuristic.
        threads = self.settings.cpu_threads
        if threads is not None and threads > 0:
            torch.set_num_threads(threads)
            logger.info("torch limited to %d CPU thread(s)", threads)

    def _start_core(self) -> None:
        """Bring up the model thread with the configured scheduling knobs."""
        self._core = EngineCore(
            self._run_batch,
            sample_rate=SAMPLE_RATE,
            window_ms=self.settings.microbatch_max_wait_ms,
            max_requests=self.settings.microbatch_max_requests,
            batch_seconds=self.settings.batch_seconds,
            subbatch_seconds=self.settings.subbatch_seconds,
            max_queued_pieces=self.settings.max_queue,
            max_queued_interactive=self.settings.max_interactive,
        )
        self._core.start()
        logger.info(
            "engine core on (%s)",
            self.settings.batching_summary,
        )

    def warmup(self) -> None:
        """Run one silent forward pass so the first real request is not slow."""
        try:
            asr = self.load()
        except EngineNotReady:
            return
        silent = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
        try:
            self._submit(lambda: asr._forward([silent], False, None))
        except Exception:
            logger.debug("warmup failed", exc_info=True)

    # -- introspection -------------------------------------------------------

    def info(self) -> dict[str, Any]:
        asr = self._asr
        device = str(getattr(asr, "device", self.settings.device or "unknown"))
        chunk_frames = None
        if asr is not None:
            chunk_frames = int(getattr(asr.model.cfg, "chunk_frames", 12))
        decoder = getattr(asr, "lm_decoder", None)
        return {
            "model": self.settings.served_model_name,
            "model_path": self.settings.model_path,
            "loaded": self.is_loaded,
            "loading": self._is_loading,
            "ready": self.is_ready,
            "device": device,
            "profile": self.settings.profile,
            "language": "vi",
            "lm": decoder is not None,
            "lm_path": self.settings.lm_path,
            "lm_alpha": getattr(decoder, "alpha", self.settings.lm_alpha),
            "lm_beta": getattr(decoder, "beta", self.settings.lm_beta),
            "beam_width": self.settings.beam_width,
            "chunk_seconds": (chunk_frames or 12) * 0.04,
            "stream_step_seconds": self.settings.stream_step_seconds,
            "left_chunks": self.settings.left_chunks,
            "microbatch_seconds": self.settings.microbatch_seconds,
            "microbatch_max_requests": self.settings.microbatch_max_requests,
            "microbatch_max_wait_ms": self.settings.microbatch_max_wait_ms,
            "max_queue": self.settings.max_queue,
            "max_interactive": self.settings.max_interactive,
            "subbatch_seconds": self.settings.subbatch_seconds,
            "tf32": self.settings.tf32,
            "vad": self._vad.stats() if self._vad is not None else {
                "enabled": self.settings.use_vad
            },
            "load_error": self._load_error,
            "load_seconds": self._load_seconds,
            "loaded_at": self._loaded_at,
        }

    def metrics(self) -> dict[str, Any]:
        rtf = (self.decode_seconds / self.audio_seconds) if self.audio_seconds else None
        core = self._core.stats() if self._core is not None else None
        return {
            "requests_total": self.requests,
            "requests_success": self.requests_success,
            "requests_failed": self.requests_failed,
            "audio_seconds_total": round(self.audio_seconds, 3),
            "total_audio_seconds": round(self.audio_seconds, 3),
            "decode_seconds_total": round(self.decode_seconds, 3),
            "realtime_factor": round(rtf, 4) if rtf else None,
            "rtf": round(rtf, 4) if rtf else None,
            "loaded": self.is_loaded,
            "core": core,
            "vad": self._vad.stats() if self._vad is not None else None,
        }

    def describe_device(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return f"cuda ({torch.cuda.get_device_name(0)})"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            # torch is absent only in a stripped-down test environment.
            pass
        return self.settings.device or "cpu"

    # -- decoding ------------------------------------------------------------

    def _decoder(self, use_lm: bool | None = None):
        """The KenLM decoder, or ``None`` to force greedy decoding.

        ``use_lm=True`` on a deployment without a loaded LM is an error rather
        than a silent fall back to greedy decoding: the caller asked for the
        accuracy the LM provides, and silently returning worse output would hide
        a misconfigured deployment.
        """
        asr = self._asr
        if use_lm is False or not self.settings.supports_lm:
            if use_lm:
                raise EngineNotReady(
                    "no KenLM model is loaded; start the server with USE_LM=true "
                    "(the lm/ folder is fetched from the model repo) or set "
                    "LM_PATH to a local KenLM model"
                )
            return None
        decoder = getattr(asr, "lm_decoder", None) if asr is not None else None
        if decoder is None and use_lm:
            raise EngineNotReady("the model loaded without a KenLM decoder")
        return decoder

    def _submit(self, call):
        """Run ``call`` on the model thread, or inline if the core is absent.

        The core only exists once the model is loaded. A few callers (``flush``
        of a session that never decoded a block) can fire before that, and one
        test constructs an engine without ever calling :meth:`load`; both fall
        back to running directly, which is safe because in those cases nothing
        else can be touching the model yet.
        """
        core = self._core
        if core is None:
            return call()
        return core.submit(call)

    def _run_batch(self, wavs: list[np.ndarray], context: Any) -> list[str]:
        """Model work for one bulk sub-batch. Called on the model thread only."""
        asr = self._asr
        if asr is None:  # pragma: no cover - guarded by load() before enqueue
            raise EngineNotReady("the model is not loaded")
        streaming, left_chunks, decoder, beam_width = context
        return asr._run(wavs, streaming, left_chunks, decoder, beam_width)

    def _decode_pieces(
        self,
        pieces: list[np.ndarray],
        streaming: bool,
        left_chunks: int | None,
        *,
        use_lm: bool | None = None,
        beam_width: int | None = None,
    ) -> list[str]:
        """Decode pieces, letting the core coalesce them with other work."""
        asr = self._asr
        if asr is None:
            raise EngineNotReady("the model is not loaded")
        core = self._ensure_core()
        decoder = self._decoder(use_lm)
        # Beam search is CPU-bound and dominates on a GPU, so an unsupported
        # request is worth warning about rather than ignoring.
        if decoder is not None and streaming:
            logger.debug(
                "beam search is applied per streaming utterance; on GPU this "
                "is usually the bottleneck"
            )
        beam_width = beam_width or self.settings.beam_width

        # The decoder has to be part of both the grouping key and the context,
        # so that greedy and beam pieces never share a launch.
        signature = (bool(streaming), left_chunks, id(decoder), beam_width)
        context = (bool(streaming), left_chunks, decoder, beam_width)
        return core.submit_bulk(pieces, signature, context)

    def _vad_detector(self) -> VoiceActivityDetector | None:
        """The shared VAD, or ``None`` when VAD is disabled.

        Created once and reused so the Silero model is loaded a single time.
        A load failure (the ``vad`` extra is not installed) is fatal here rather
        than silent: it is a configuration error, and the caller catches it and
        falls back to energy-based splitting for the request.
        """
        if not self.settings.use_vad:
            return None
        if self._vad is None:
            self._vad = VoiceActivityDetector(self.settings.vad)
        return self._vad

    def _regions_for(self, wav: np.ndarray) -> list[tuple[int, int]] | None:
        """VAD speech ranges to decode, or ``None`` to use the whole waveform.

        ``None`` is returned both when VAD is off and when it fails, so a broken
        VAD degrades to the pre-VAD behaviour instead of failing the request.
        When trimming is off the ranges still cover the whole waveform (cut at
        every speech boundary), which keeps the transcript identical to a
        VAD-less run while giving the splitter real pauses to cut on.
        """
        vad = self._vad_detector()
        if vad is None:
            return None
        try:
            return vad.regions(wav, keep_silence=not vad.trim_silence)
        except Exception:
            logger.warning("voice activity detection failed; decoding all audio", exc_info=True)
            return None

    def transcribe(
        self,
        wav: np.ndarray,
        *,
        streaming: bool = False,
        left_chunks: int | None = None,
        chunk_seconds: float | None = None,
        use_lm: bool | None = None,
        beam_width: int | None = None,
    ) -> list[Segment]:
        """Transcribe a full waveform and return timestamped segments.

        The waveform is cut at quiet points exactly like the upstream
        ``transcribe()`` does, but we keep the offsets so the response can carry
        real timestamps. When VAD is on, the cuts land on speech boundaries
        instead and silence is (optionally) never decoded.
        """
        from .audio import build_segments, split_with_offsets

        self.load()
        regions = self._regions_for(wav)
        pieces_and_offsets = split_with_offsets(
            wav, chunk_seconds or self.settings.chunk_seconds, regions=regions
        )
        if not pieces_and_offsets:
            return []

        pieces = [p[0] for p in pieces_and_offsets]
        offsets = [(p[1], p[2]) for p in pieces_and_offsets]

        started = time.time()
        texts = self._decode_pieces(
            pieces,
            streaming,
            left_chunks,
            use_lm=use_lm,
            beam_width=beam_width,
        )
        self._record(len(wav) / SAMPLE_RATE, time.time() - started)

        return build_segments(texts, offsets)

    def iter_stream(
        self,
        wav: np.ndarray,
        *,
        left_chunks: int | None = None,
        step_seconds: float | None = None,
        chunk_seconds: float | None = None,
        use_lm: bool | None = None,
        beam_width: int | None = None,
    ) -> Iterator[StreamEvent]:
        """Replay a whole file through the streaming decoder, step by step.

        This is what a streaming deployment would emit: an utterance in progress
        is re-decoded every step, finished utterances are decoded once more with
        full context, and only the newest words can still change.
        """
        self.load()
        step_seconds = step_seconds or self.settings.stream_step_seconds
        step = max(1, round(step_seconds * SAMPLE_RATE))
        step_seconds = step / SAMPLE_RATE

        session = self._new_live_session(
            left_chunks, chunk_seconds, use_lm, beam_width
        )
        total = len(wav)
        consumed = 0
        last_end = 0.0
        started = time.time()

        # Each step is scheduled independently on the model thread, so a long
        # SSE stream is a sequence of short interactive jobs: other requests
        # interleave between them instead of waiting for the whole stream.
        while consumed < total:
            block = wav[consumed : consumed + step]
            consumed += len(block)
            now = consumed / SAMPLE_RATE
            finals, partial = self._session_step(session, block)
            for final in finals:
                yield StreamEvent("final", final, last_end, now)
                last_end = now
            if partial:
                yield StreamEvent("partial", partial, last_end, now)

        tail = self._submit(session.flush)
        if tail:
            yield StreamEvent("final", tail, last_end, total / SAMPLE_RATE)

        self._record(total / SAMPLE_RATE, time.time() - started)
        yield StreamEvent("done", "", last_end, total / SAMPLE_RATE)

    def _session_step(self, session: Any, block: np.ndarray) -> tuple[list[str], str]:
        """Advance a live session by one block, as one interactive job.

        ``LiveSession.accept`` both decodes the partial hypothesis and may
        finalise a finished utterance, so the whole call is one atomic critical
        section on the model thread. It is submitted as *interactive* work, so
        it preempts queued bulk pieces rather than waiting behind them: a 0.48 s
        microphone step should never queue behind a ten-minute file.
        """
        def step() -> tuple[list[str], str]:
            finals = session.accept(block)
            return finals, session.partial

        return self._submit(step)

    def stream_segments(
        self,
        wav: np.ndarray,
        *,
        left_chunks: int | None = None,
        step_seconds: float | None = None,
        use_lm: bool | None = None,
        beam_width: int | None = None,
    ) -> list[Segment]:
        """Convenience wrapper: consume :meth:`iter_stream` into final segments."""
        segments: list[Segment] = []
        for event in self.iter_stream(
            wav,
            left_chunks=left_chunks,
            step_seconds=step_seconds,
            use_lm=use_lm,
            beam_width=beam_width,
        ):
            if event.kind == "final" and event.text:
                segments.append(
                    Segment(text=event.text, start=event.start, end=event.end)
                )
        return segments

    # -- live sessions -------------------------------------------------------

    def _new_live_session(
        self,
        left_chunks: int | None,
        chunk_seconds: float | None = None,
        use_lm: bool | None = None,
        beam_width: int | None = None,
    ):
        from quan_transcribe.live import LiveSession

        asr = self.load()
        decoder = self._decoder(use_lm)
        beam_width = beam_width or self.settings.beam_width
        session = LiveSession(
            asr,
            left_chunks if left_chunks is not None else self.settings.left_chunks,
            self.settings.endpoint_s,
            chunk_seconds or self.settings.max_utterance_s,
            beam_width,
        )
        if decoder is not getattr(asr, "lm_decoder", None):
            # LiveSession takes the decoder off the model rather than as an
            # argument, so a per-request override has to be applied here.
            # Rebinding the instance method keeps the change local to this
            # session instead of mutating state another request could see.
            session._final = lambda wav: asr._run(
                [wav], False, None, decoder, beam_width
            )[0]
        return session

    def new_live_session(
        self,
        left_chunks: int | None = None,
        chunk_seconds: float | None = None,
        use_lm: bool | None = None,
        beam_width: int | None = None,
    ) -> LiveStream:
        return LiveStream(self, left_chunks, chunk_seconds, use_lm, beam_width)

    # -- bookkeeping ---------------------------------------------------------

    def _record(
        self, audio_seconds: float, decode_seconds: float, success: bool = True
    ) -> None:
        self.requests += 1
        if success:
            self.requests_success += 1
        else:
            self.requests_failed += 1
        self.audio_seconds += audio_seconds
        self.decode_seconds += decode_seconds

    def record_failure(self) -> None:
        """Record an error/failed request."""
        self.requests += 1
        self.requests_failed += 1

    def close(self) -> None:
        core, self._core = self._core, None
        if core is not None:
            # Stop the model thread first: it is the only owner of the model, so
            # once it has drained there is no forward pass in flight to race the
            # teardown below.
            core.stop()
        self._asr = None
        try:  # pragma: no cover - only meaningful on CUDA
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            # torch is always present in the image; a missing import here just
            # means there is no CUDA cache to release.
            pass


class LiveStream:
    """A push-based live stream, used by the WebSocket endpoint.

    Feed 16 kHz mono float32 audio with :meth:`accept` and read back the events
    the server should forward. Every model call goes through the engine core as
    interactive work, so a live session preempts bulk transcription instead of
    waiting behind it.
    """

    def __init__(
        self,
        engine: TranscriptionEngine,
        left_chunks: int | None = None,
        chunk_seconds: float | None = None,
        use_lm: bool | None = None,
        beam_width: int | None = None,
    ):
        self.engine = engine
        self.left_chunks = (
            left_chunks if left_chunks is not None else engine.settings.left_chunks
        )
        self.chunk_seconds = chunk_seconds
        self.use_lm = use_lm
        self.beam_width = beam_width
        self._session = None
        self._pending = np.zeros(0, dtype=np.float32)
        self._consumed = 0
        self._last_end = 0.0
        self._closed = False

    @property
    def partial(self) -> str:
        return self._session.partial if self._session is not None else ""

    @property
    def consumed_seconds(self) -> float:
        return self._consumed / SAMPLE_RATE

    def accept(self, samples: np.ndarray) -> list[StreamEvent]:
        """Add audio and return the events produced by this step."""
        if self._closed:
            raise RuntimeError("the stream is closed")

        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return []

        self._pending = np.concatenate([self._pending, block])
        step = max(
            1,
            round(self.engine.settings.stream_step_seconds * SAMPLE_RATE),
        )
        if self._pending.size < step:
            return []

        if self._session is None:
            self._session = self.engine._new_live_session(
                self.left_chunks,
                self.chunk_seconds,
                self.use_lm,
                self.beam_width,
            )

        audio = self._pending
        self._consumed += audio.size
        now = self._consumed / SAMPLE_RATE

        try:
            finals, partial = self.engine._session_step(self._session, audio)
        except EngineBusy:
            # Shed rather than block: put the audio back so the next frame
            # retries it together with whatever arrives, instead of dropping a
            # chunk of the user's speech. The byte counter stays advanced, so
            # the cap on received audio is unaffected.
            self._pending = audio
            self._consumed -= audio.size
            raise
        self._pending = np.zeros(0, dtype=np.float32)

        events: list[StreamEvent] = []
        for final in finals:
            events.append(StreamEvent("final", final, self._last_end, now))
            self._last_end = now
        if partial:
            events.append(
                StreamEvent("partial", partial, self._last_end, now)
            )
        return events

    def flush(self) -> str:
        """Finish the utterance in progress and return it (may be empty)."""
        if self._closed:
            return ""
        if self._session is None:
            # Too short to have reached a decoding step; decode it directly.
            if self._pending.size == 0:
                self._closed = True
                return ""
            text = self.engine.transcribe(
                self._pending,
                streaming=False,
                chunk_seconds=self.chunk_seconds,
                use_lm=self.use_lm,
                beam_width=self.beam_width,
            )[0].text
            self._closed = True
            return text

        text = self.engine._submit(self._session.flush)
        self._pending = np.zeros(0, dtype=np.float32)
        self._closed = True
        return text

    def close(self) -> None:
        self._closed = True
        self._session = None
        self._pending = np.zeros(0, dtype=np.float32)
