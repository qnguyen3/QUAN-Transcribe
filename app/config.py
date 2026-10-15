"""Runtime configuration.

Everything is driven by environment variables so the same image works as a
GPUStack custom backend (which injects env vars) or as a plain ``docker run``.
Command line flags, when present, win over the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .vad import VadConfig

DEFAULT_MODEL = "qnguyen3/QUAN-Transcribe-1.0"
DEFAULT_SERVED_NAME = "QUAN-Transcribe-1.0"

# The model emits lowercase Vietnamese without punctuation. It has no language
# conditioning input, so these are the only accepted values.
SUPPORTED_LANGUAGES = ("vi", "vietnamese")


def _get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _flag(name: str, default: bool) -> bool:
    raw = _get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "y", "on")


def _maybe_int(name: str, default: int | None = None) -> int | None:
    raw = _get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _maybe_float(name: str, default: float | None = None) -> float | None:
    raw = _get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _maybe_str(name: str) -> str | None:
    return _get(name)


@dataclass(frozen=True)
class Settings:
    """Immutable server settings."""

    # --- model ---------------------------------------------------------------
    # Either a Hugging Face repo id or a local directory. GPUStack passes the
    # downloaded model directory through ``--model-path`` / ``{{model_path}}``.
    model_path: str = DEFAULT_MODEL
    served_model_name: str = DEFAULT_SERVED_NAME
    device: str | None = None
    hf_token: str | None = None
    preload: bool = True

    # --- decoding ------------------------------------------------------------
    #: Decode with the bundled 4-gram KenLM by default (CTC prefix beam search).
    use_lm: bool = False
    #: A local KenLM ``.bin``/``.arpa`` file, or a directory laid out like the
    #: model repo's ``lm/`` folder. Overrides ``use_lm`` when set.
    lm_path: str | None = None
    #: LM weight and word-insertion bonus. ``None`` (the default) keeps the
    #: values from the LM's own ``lm_config.json``, which ship at 0.3 / 1.0.
    lm_alpha: float | None = None
    lm_beta: float | None = None
    #: Beams kept by the beam search. Ignored when no LM is loaded.
    beam_width: int = 32
    #: How many past 0.48 s chunks the reader may attend to in streaming mode.
    #: ``None`` means "all of them" (best accuracy, unbounded context).
    left_chunks: int | None = None
    #: Long recordings are cut into pieces of at most this many seconds, at a
    #: quiet point, which also become the reported segments.
    chunk_seconds: float = 30.0

    # --- voice activity detection -------------------------------------------
    #: Skip decoding silence and cut pieces on real pauses instead of the
    #: quietest 50 ms window. Real call audio is mostly silence (a measured
    #: 90 s voicemail had 7 s of speech), so this is the single biggest win for
    #: long recordings. See ``app/vad.py``. Requires the ``vad`` extra; when the
    #: package is missing the server logs a warning and falls back to
    #: energy-based splitting.
    use_vad: bool = False
    #: Speech probability above which a 32 ms window counts as speech.
    vad_threshold: float = 0.5
    #: Shortest run of speech worth keeping, and the silence that ends it (ms).
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 300
    #: Padding around every speech region (ms). Silero defaults to 30 ms, which
    #: clips the first syllable of a region for this encoder; 250 ms keeps it.
    vad_speech_pad_ms: int = 250
    #: When true, silence is dropped before decoding (faster, model returns ""
    #: for it anyway). When false, silence is kept and VAD only supplies cut
    #: points, which is bit-for-bit the VAD-less transcript.
    vad_trim_silence: bool = True
    #: Padding budget per forward pass; lower it if you run out of VRAM.
    batch_seconds: float = 600.0
    #: Longest single forward pass, in audio seconds (the padded cost `k * n`).
    #: A large file becomes several launches, and the scheduler can serve a
    #: live request *between* them; without a cap one big file is one
    #: uninterruptible pass. ``0`` keeps `batch_seconds` as the only bound.
    #: On CPU batching is near-neutral for throughput (torch already uses every
    #: core per op) but a long pass visibly stalls live audio, so the CPU
    #: profiles set this; on GPU one 30 s piece is a few ms, so it can stay off.
    subbatch_seconds: float = 0.0
    #: Overlap (seconds) required between two concurrent requests before they are
    #: merged into one padded forward pass. Padding every piece in a batch to the
    #: longest one only pays off when the extra compute is less than the time a
    #: second launch would waste leaving the GPU idle, so this is the knob that
    #: trades throughput for latency. 0 disables cross-request batching entirely.
    microbatch_seconds: float = 0.15
    #: Cap on how long a request waits for companions to batch with. Bounds the
    #: added latency when there is no concurrent traffic.
    microbatch_max_wait_ms: float = 20.0
    #: Up to this many concurrent requests' pieces are packed into one batch.
    microbatch_max_requests: int = 8
    #: Backpressure for the decode queue: the most audio pieces that may be
    #: waiting at once. When full, new bulk requests are refused with a 503
    #: instead of piling up latency. ``0`` disables the cap (unbounded).
    max_queue: int = 256
    #: Backpressure for interactive (streaming) jobs. ``0`` disables the cap.
    max_interactive: int = 64
    #: Allow TF32 matmuls on Ampere and later. Faster, with a small precision
    #: cost that a CTC model tolerates. Set false for bit-exact FP32 math.
    tf32: bool = True
    #: CPU worker threads for the CPU profile. ``None`` lets torch pick, which
    #: it derives from the visible core count.
    cpu_threads: int | None = None

    # --- live streaming ------------------------------------------------------
    #: Silence, in seconds, that ends an utterance in streaming mode.
    endpoint_s: float = 0.8
    #: Hard cap on a single streaming utterance before it is cut at a quiet point.
    max_utterance_s: float = 30.0
    #: Audio fed per streaming step. 0.48 s is exactly one encoder chunk.
    stream_step_seconds: float = 0.48
    #: Reject streams longer than this, as a safety valve.
    max_stream_seconds: float = 3600.0

    # --- http & grpc ---------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    grpc_port: int | None = None
    api_key: str | None = None
    cors_origins: tuple[str, ...] = ("*",)
    max_upload_mb: int = 200
    log_level: str = "info"

    # --- enterprise postprocessing & multi-worker ----------------------------
    enable_punctuation: bool = False
    devices: tuple[str, ...] = ("auto",)

    @property
    def supports_lm(self) -> bool:
        """True when the deployment was asked to load a KenLM model."""
        return self.use_lm or bool(self.lm_path)

    @property
    def vad(self) -> VadConfig:
        """The VAD tuning as a frozen :class:`~app.vad.VadConfig`."""
        from .vad import VadConfig

        return VadConfig(
            enabled=self.use_vad,
            threshold=self.vad_threshold,
            min_speech_ms=self.vad_min_speech_ms,
            min_silence_ms=self.vad_min_silence_ms,
            speech_pad_ms=self.vad_speech_pad_ms,
            trim_silence=self.vad_trim_silence,
        )

    @property
    def lm_source(self) -> str:
        """A human-readable description of which LM is used."""
        if self.lm_path:
            return self.lm_path
        if self.use_lm:
            return "bundled lm/ from the model repo"
        return "none (greedy decoding)"

    @property
    def batching_summary(self) -> str:
        """A one-line description of the scheduler settings, for the startup log."""
        if self.microbatch_seconds <= 0:
            return "interactive priority, coalescing off"
        queue = "unbounded" if not self.max_queue else f"cap {self.max_queue}"
        return (
            f"coalescing window {self.microbatch_max_wait_ms:.0f}ms, "
            f"max_requests {self.microbatch_max_requests}, queue {queue}"
        )

    @property
    def profile(self) -> str:
        """The hardware profile this configuration looks tuned for.

        Derived rather than declared: ``cpu`` when a device was pinned to the
        CPU, ``cuda``/``mps`` when one was pinned explicitly, and ``auto`` when
        the device is left to the model's own detection.
        """
        device = (self.device or "auto").lower()
        if device.startswith(("cuda", "rocm")):
            return "cuda"
        if device.startswith(("cpu", "mps")):
            return device.split(":")[0]
        return "auto"

    @classmethod
    def from_env(cls) -> Settings:
        origins = _get("CORS_ORIGINS", "*")
        return cls(
            model_path=_get("MODEL_PATH") or _get("QUAN_MODEL") or DEFAULT_MODEL,
            served_model_name=(
                _get("SERVED_MODEL_NAME") or _get("MODEL_NAME") or DEFAULT_SERVED_NAME
            ),
            device=_maybe_str("DEVICE"),
            hf_token=_get("HF_TOKEN"),
            preload=_flag("PRELOAD", True),
            use_lm=_flag("USE_LM", bool(_get("LM_PATH"))),
            lm_path=_maybe_str("LM_PATH"),
            lm_alpha=_maybe_float("LM_ALPHA"),
            lm_beta=_maybe_float("LM_BETA"),
            beam_width=_maybe_int("BEAM_WIDTH", 32) or 32,
            left_chunks=_maybe_int("LEFT_CHUNKS"),
            chunk_seconds=_maybe_float("CHUNK_SECONDS", 30.0) or 30.0,
            use_vad=_flag("USE_VAD", False),
            vad_threshold=_maybe_float("VAD_THRESHOLD", 0.5) or 0.5,
            vad_min_speech_ms=_maybe_int("VAD_MIN_SPEECH_MS", 250) or 250,
            vad_min_silence_ms=_maybe_int("VAD_MIN_SILENCE_MS", 300) or 300,
            vad_speech_pad_ms=_maybe_int("VAD_SPEECH_PAD_MS", 250) or 0,
            vad_trim_silence=_flag("VAD_TRIM_SILENCE", True),
            batch_seconds=_maybe_float("BATCH_SECONDS", 600.0) or 600.0,
            subbatch_seconds=_maybe_float("SUBBATCH_SECONDS", 0.0) or 0.0,
            microbatch_seconds=_maybe_float("MICROBATCH_SECONDS", 0.15) or 0.0,
            # ``or 20.0`` would swallow a deliberate 0 (batching without wait),
            # so the wait uses the raw parsed value.
            microbatch_max_wait_ms=_maybe_float("MICROBATCH_MAX_WAIT_MS", 20.0),
            microbatch_max_requests=_maybe_int("MICROBATCH_MAX_REQUESTS", 8) or 1,
            max_queue=_maybe_int("MAX_QUEUE", 256) or 0,
            max_interactive=_maybe_int("MAX_INTERACTIVE", 64) or 0,
            tf32=_flag("TF32", True),
            cpu_threads=_maybe_int("CPU_THREADS"),
            endpoint_s=_maybe_float("ENDPOINT_SECONDS", 0.8) or 0.8,
            max_utterance_s=_maybe_float("MAX_UTTERANCE_SECONDS", 30.0) or 30.0,
            stream_step_seconds=_maybe_float("STREAM_STEP_SECONDS", 0.48) or 0.48,
            max_stream_seconds=_maybe_float("MAX_STREAM_SECONDS", 3600.0) or 3600.0,
            host=_get("HOST", "0.0.0.0"),
            port=_maybe_int("PORT", 8000) or 8000,
            grpc_port=_maybe_int("GRPC_PORT"),
            api_key=_get("API_KEY"),
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
            max_upload_mb=_maybe_int("MAX_UPLOAD_MB", 200) or 200,
            log_level=(_get("LOG_LEVEL", "info") or "info").lower(),
            enable_punctuation=_flag("PUNCTUATE", False),
            devices=(
                tuple(d.strip() for d in _get("DEVICES", "").split(",") if d.strip())
                or (_maybe_str("DEVICE") or "auto",)
            ),
        )

    def merge(self, **overrides: Any) -> Settings:
        """Return a copy with the non-``None`` overrides applied."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        known = {f.name for f in fields(self)}
        unknown = set(clean) - known
        if unknown:
            raise TypeError(f"unknown setting(s): {sorted(unknown)}")
        return replace(self, **clean)
