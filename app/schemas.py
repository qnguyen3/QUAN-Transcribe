"""OpenAI-compatible request parsing.

``/v1/audio/transcriptions`` is ``multipart/form-data``, so there is no Pydantic
model to declare. These helpers read and validate that form the way OpenAI's
server does and raise errors that render into OpenAI's error envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import SUPPORTED_LANGUAGES

#: Response formats OpenAI documents, plus ``vtt`` which some clients send.
RESPONSE_FORMATS = ("json", "text", "srt", "verbose_json", "vtt")
TIMESTAMP_GRANULARITIES = ("word", "segment")

#: Server-side bounds so a stray field cannot make the model do unbounded work.
MIN_CHUNK_SECONDS = 1.0
MAX_CHUNK_SECONDS = 300.0
MAX_BEAM_WIDTH = 256

_TRUTHY = {"1", "true", "yes", "y", "on", "t"}
_FALSY = {"0", "false", "no", "n", "off", "f", ""}

#: Form keys this server understands. Unknown fields are ignored rather than
#: rejected, matching how OpenAI's API tolerates extra fields.
_KNOWN_FIELDS = frozenset(
    {
        "file",
        "model",
        "language",
        "prompt",
        "response_format",
        "temperature",
        "stream",
        "stream_partials",
        "timestamp_granularities",
        "timestamp_granularities[]",
        "left_chunks",
        "use_lm",
        "beam_width",
        "chunk_seconds",
        "punctuate",
        "enable_punctuation",
        "phonemes",
        "include_phonemes",
        "g2p",
        "norm_mode",
    }
)


class APIError(Exception):
    """An error that renders in OpenAI's error envelope."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        code: str | None = None,
        param: str | None = None,
        error_type: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.param = param
        self.error_type = error_type

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


@dataclass
class TranscriptionParams:
    """Validated parameters of one transcription request."""

    filename: str
    language: str | None = None
    prompt: str | None = None
    response_format: str = "json"
    temperature: float = 0.0
    streaming: bool = False
    streaming_partials: bool = True
    timestamp_granularities: list[str] = field(default_factory=list)
    left_chunks: int | None = None
    use_lm: bool | None = None
    beam_width: int | None = None
    chunk_seconds: float | None = None
    punctuate: bool = False
    phonemes: bool = False
    norm_mode: str = "itn"

    @property
    def wants_timestamps(self) -> bool:
        return bool(self.timestamp_granularities)


def _bad(message: str, param: str) -> APIError:
    return APIError(message, param=param, code="invalid_value")


def _parse_bool(value: Any, param: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    raise _bad(f"'{value}' is not a valid boolean for '{param}'", param)


def _parse_float(value: Any, param: str) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError) as exc:
        raise _bad(f"'{value}' is not a valid number for '{param}'", param) from exc


def _parse_int(value: Any, param: str) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise _bad(f"'{value}' is not a valid integer for '{param}'", param) from exc


def _getlist(form: Any, key: str) -> list[Any]:
    """Read a repeated field, tolerating a form-like object without ``getlist``."""
    if hasattr(form, "getlist"):
        return [v for v in form.getlist(key) if v is not None]
    value = form.get(key)
    return [] if value is None else [value]


def parse_transcription_form(form: Any) -> TranscriptionParams:
    """Turn a parsed multipart form into validated transcription params.

    ``form`` only needs ``get`` (ideally also ``getlist``), so tests can pass a
    plain mapping.
    """
    upload = form.get("file")
    filename = getattr(upload, "filename", None) or "audio"
    _ = _KNOWN_FIELDS  # documented above; unknown keys fall through unused

    language: str | None = None
    raw_language = form.get("language")
    if raw_language not in (None, ""):
        normalized = str(raw_language).strip().lower()
        if normalized not in {"auto", "none"}:
            code = normalized.split("-")[0].split("_")[0]
            if code not in SUPPORTED_LANGUAGES:
                raise APIError(
                    f"language '{raw_language}' is not supported; this model only "
                    "transcribes Vietnamese (use 'vi', or omit the parameter)",
                    param="language",
                    code="unsupported_language",
                )
            language = "vi"

    response_format = str(form.get("response_format") or "json").strip().lower()
    if not response_format:
        response_format = "json"
    if response_format not in RESPONSE_FORMATS:
        raise _bad(
            f"'{response_format}' is not a valid response_format; choose one of "
            f"{', '.join(RESPONSE_FORMATS)}",
            "response_format",
        )

    temperature = 0.0
    raw_temperature = form.get("temperature")
    if raw_temperature not in (None, ""):
        temperature = _parse_float(raw_temperature, "temperature")
        if not 0.0 <= temperature <= 1.0:
            raise _bad("temperature must be between 0 and 1", "temperature")

    granularities: list[str] = []
    for value in _getlist(form, "timestamp_granularities[]") + _getlist(
        form, "timestamp_granularities"
    ):
        normalized = str(value).strip().lower()
        if not normalized:
            continue
        if normalized not in TIMESTAMP_GRANULARITIES:
            raise _bad(
                "timestamp_granularities must be a subset of "
                f"{', '.join(TIMESTAMP_GRANULARITIES)}",
                "timestamp_granularities[]",
            )
        if normalized not in granularities:
            granularities.append(normalized)

    if granularities and response_format not in ("verbose_json", "srt", "vtt"):
        raise _bad(
            "timestamp granularities require response_format=verbose_json, srt or vtt",
            "timestamp_granularities[]",
        )

    streaming = _parse_bool(form.get("stream", "false"), "stream")

    # ``stream_partials=false`` emits one event per finished utterance. It is an
    # extension: OpenAI's own API only ever streams deltas.
    streaming_partials = True
    if form.get("stream_partials") not in (None, ""):
        streaming_partials = _parse_bool(
            form.get("stream_partials"), "stream_partials"
        )

    left_chunks = None
    if form.get("left_chunks") not in (None, ""):
        left_chunks = _parse_int(form.get("left_chunks"), "left_chunks")
        if left_chunks < 0:
            raise _bad("left_chunks must be zero or greater", "left_chunks")

    use_lm = None
    if form.get("use_lm") not in (None, ""):
        use_lm = _parse_bool(form.get("use_lm"), "use_lm")

    beam_width = None
    if form.get("beam_width") not in (None, ""):
        beam_width = _parse_int(form.get("beam_width"), "beam_width")
        if not 1 <= beam_width <= MAX_BEAM_WIDTH:
            raise _bad(
                f"beam_width must be between 1 and {MAX_BEAM_WIDTH}", "beam_width"
            )

    chunk_seconds = None
    if form.get("chunk_seconds") not in (None, ""):
        chunk_seconds = _parse_float(form.get("chunk_seconds"), "chunk_seconds")
        if not MIN_CHUNK_SECONDS <= chunk_seconds <= MAX_CHUNK_SECONDS:
            raise _bad(
                f"chunk_seconds must be between {MIN_CHUNK_SECONDS} and "
                f"{MAX_CHUNK_SECONDS}",
                "chunk_seconds",
            )

    punctuate = False
    raw_punc = (
        form.get("punctuate")
        if form.get("punctuate") is not None
        else form.get("enable_punctuation")
    )
    if raw_punc not in (None, ""):
        punctuate = _parse_bool(raw_punc, "punctuate")

    phonemes = False
    raw_phonemes = (
        form.get("phonemes")
        if form.get("phonemes") is not None
        else (form.get("include_phonemes") if form.get("include_phonemes") is not None else form.get("g2p"))
    )
    if raw_phonemes not in (None, ""):
        phonemes = _parse_bool(raw_phonemes, "phonemes")

    norm_mode = "itn"
    raw_norm = form.get("norm_mode")
    if raw_norm not in (None, ""):
        norm_val = str(raw_norm).strip().lower()
        if norm_val not in ("itn", "tn", "none", "raw"):
            raise _bad("norm_mode must be 'itn', 'tn', or 'none'", "norm_mode")
        norm_mode = "none" if norm_val == "raw" else norm_val

    prompt = form.get("prompt")
    return TranscriptionParams(
        filename=filename,
        language=language,
        prompt=str(prompt) if prompt else None,
        response_format=response_format,
        temperature=temperature,
        streaming=streaming,
        streaming_partials=streaming_partials,
        timestamp_granularities=granularities,
        left_chunks=left_chunks,
        use_lm=use_lm,
        beam_width=beam_width,
        chunk_seconds=chunk_seconds,
        punctuate=punctuate,
        phonemes=phonemes,
        norm_mode=norm_mode,
    )
