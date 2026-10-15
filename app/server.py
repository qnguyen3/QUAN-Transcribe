"""OpenAI-compatible ASR server for QUAN-Transcribe.

Endpoints
---------
``POST /v1/audio/transcriptions``
    OpenAI-compatible transcription. Supports the ``text``, ``json``,
    ``verbose_json``, ``srt`` and ``vtt`` response formats, and streams
    Server-Sent Events when ``stream=true``.
``GET  /v1/models``, ``GET /v1/models/{model}``
    Model listing, so GPUStack's default ``/v1/models`` health check passes.
``GET  /health``
    Readiness probe reporting the real model state.
``GET  /metrics``
    Counters plus a realtime factor.
``WS   /v1/audio/transcriptions/stream``
    Raw audio in, JSON transcription events out.
``GET  /``
    A small playground for uploads and the microphone.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .audio import (
    SAMPLE_RATE,
    AudioDecodeError,
    Segment,
    decode_audio_bytes,
    to_16k_mono,
)
from .config import Settings
from .engine import EngineBusy, EngineNotReady, StreamEvent, TranscriptionEngine
from .formats import MEDIA_TYPES, merge_segments_by_sentence, render
from .postprocess import PostProcessor, default_post_processor, to_phonemes
from .schemas import APIError, TranscriptionParams, parse_transcription_form
from .ui import INDEX_HTML

logger = logging.getLogger("quan_asr")

settings = Settings.from_env()
engine = TranscriptionEngine(settings)

#: Extra model names OpenAI clients tend to send.
_ALIASES = {"whisper-1", "quan-transcribe", "quan-transcribe-1.0", "quan-asr", "quan-asr-server"}
_active_websockets = 0


# --------------------------------------------------------------------------- #
# lifespan
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info(
        "starting QUAN-ASR Server %s (model=%s, device=%s, preload=%s)",
        __version__,
        settings.model_path,
        settings.device or "auto",
        settings.preload,
    )
    preload_task: asyncio.Task[None] | None = None
    if settings.preload:
        async def _preload_model() -> None:
            try:
                await run_in_threadpool(engine.load)
                await run_in_threadpool(engine.warmup)
                logger.info("model preload and warmup complete; engine is ready")
            except EngineNotReady as exc:
                # Do not crash the container: /health reports the failure so an
                # operator can fix the cause (token, disk, GPU) and restart.
                logger.error("model preload failed: %s", exc)

        preload_task = asyncio.create_task(_preload_model())
    else:
        logger.info("preload disabled; the model loads on the first request")

    grpc_server = None
    if settings.grpc_port:
        try:
            from .grpc_server import create_grpc_server

            grpc_server = create_grpc_server(
                engine, host=settings.host, port=settings.grpc_port
            )
            grpc_server.start()
            logger.info(
                "gRPC speech service started on %s:%d",
                settings.host,
                settings.grpc_port,
            )
        except Exception as exc:
            logger.error("failed to start gRPC service: %s", exc)

    try:
        yield
    finally:
        if preload_task and not preload_task.done():
            preload_task.cancel()
        if grpc_server:
            grpc_server.stop(0)
            logger.info("gRPC service stopped")
        engine.close()
        logger.info("shutdown complete")


app = FastAPI(
    title="QUAN-ASR Server",
    version=__version__,
    description=(
        "OpenAI-compatible speech-to-text API for QUAN-Transcribe-1.0 "
        "(Vietnamese, offline and streaming)."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins) or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# error handling
# --------------------------------------------------------------------------- #


@app.exception_handler(APIError)
async def _api_error_handler(_: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload())


@app.exception_handler(StarletteHTTPException)
async def _http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    error = APIError(
        str(exc.detail),
        status_code=exc.status_code,
        code="not_found" if exc.status_code == 404 else None,
        error_type="invalid_request_error" if exc.status_code < 500 else "server_error",
    )
    return JSONResponse(status_code=exc.status_code, content=error.to_payload())


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    error = APIError(f"invalid request: {exc.errors()}", code="invalid_request_body")
    return JSONResponse(status_code=400, content=error.to_payload())


def require_api_key(request: Request) -> None:
    """Enforce ``API_KEY`` when one is configured, mirroring OpenAI bearer auth."""
    if not settings.api_key:
        return
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not token:
        token = request.headers.get("x-api-key", "").strip()
    if token != settings.api_key:
        raise APIError(
            "invalid API key",
            status_code=401,
            code="invalid_api_key",
            error_type="authentication_error",
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _model_payload() -> dict[str, Any]:
    return {
        "id": settings.served_model_name,
        "object": "model",
        "created": int(engine.loaded_at or time.time()),
        "owned_by": "quan-transcribe",
        "language": "vi",
    }


def _check_model(requested: str | None) -> None:
    """Reject a ``model`` value that is not this deployment.

    OpenAI clients always send a model name; GPUStack may send either the served
    name or the repo id, so both are accepted, alongside the Whisper alias.
    """
    if not requested or not requested.strip():
        return
    accepted = {settings.served_model_name.lower(), settings.model_path.lower()}
    accepted |= _ALIASES
    # Also accept the basename of a local model directory.
    accepted.add(
        settings.model_path.replace("\\", "/").rstrip("/").split("/")[-1].lower()
    )
    if requested.strip().lower() not in accepted:
        raise APIError(
            f"the model '{requested}' does not exist; this server serves "
            f"'{settings.served_model_name}'",
            status_code=404,
            code="model_not_found",
            param="model",
        )


def _decode_upload(data: bytes) -> np.ndarray:
    try:
        wav, sr = decode_audio_bytes(data)
    except AudioDecodeError as exc:
        raise APIError(str(exc), code="invalid_audio_file") from exc
    wav = to_16k_mono(wav, sr)
    if wav.size == 0:
        raise APIError("the uploaded file contains no audio", code="invalid_audio_file")
    return wav


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _shape_segments(
    segments: list[Segment], params: TranscriptionParams
) -> list[Segment]:
    """Merge model cut points that fell mid-sentence, for the timed formats."""
    if params.response_format in ("srt", "vtt", "verbose_json") or (
        params.wants_timestamps
    ):
        return merge_segments_by_sentence(segments)
    return segments


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #


_ROOT_DIR = Path(__file__).resolve().parent.parent
_SAMPLES_DIR = _ROOT_DIR / "samples"

_SCALAR_HTML = """<!doctype html>
<html lang="en">
  <head>
    <title>QUAN-ASR Server API Reference · Scalar</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%231a73e8'><path d='M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z'/></svg>" />
    <style>body { margin: 0; }</style>
  </head>
  <body>
    <script
      id="api-reference"
      data-url="/openapi.json"
      data-configuration='{"theme": "kepler", "showSidebar": true, "defaultHttpClient": {"targetKey": "python", "clientKey": "requests"}}'
      src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
  </body>
</html>"""


@app.get("/", include_in_schema=False)
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/scalar", include_in_schema=False)
async def scalar_docs() -> HTMLResponse:
    return HTMLResponse(_SCALAR_HTML)


@app.get("/samples", include_in_schema=False)
async def list_samples() -> list[dict[str, Any]]:
    samples = [
        {
            "id": "voicemail",
            "name": "Voicemail nhà mạng (90s)",
            "filename": "voicemail-90s.mp3",
            "duration": "90.2s",
            "size": "352 KB",
        },
        {
            "id": "sales-call",
            "name": "Cuộc gọi tư vấn (73s)",
            "filename": "sales-call-72s.mp3",
            "duration": "72.7s",
            "size": "568 KB",
        },
    ]
    return [s for s in samples if (_SAMPLES_DIR / s["filename"]).is_file()]


@app.get("/samples/{filename}", include_in_schema=False)
async def get_sample(filename: str) -> FileResponse:
    allowed = {
        "voicemail-90s.mp3",
        "sales-call-72s.mp3",
    }
    if filename not in allowed:
        raise StarletteHTTPException(status_code=404, detail="sample not found")
    path = _SAMPLES_DIR / filename
    if not path.is_file():
        raise StarletteHTTPException(status_code=404, detail="sample file not found")
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/health")
async def health(response: Response, ready: bool = False) -> dict[str, Any]:
    loaded = bool(engine.is_loaded)
    err = engine.load_error
    is_ready = loaded and not err
    # If caller specifically requested readiness check (e.g. ?ready=true) or there's a hard error:
    if (not loaded and err) or (ready and not is_ready):
        response.status_code = 503
    return {
        "status": "ok" if is_ready else ("error" if err else "loading"),
        "model": settings.served_model_name,
        "device": (
            engine.describe_device() if loaded else (settings.device or "unknown")
        ),
        "language": "vi",
        "loaded": loaded,
        "loading": bool(getattr(engine, "is_loading", False)),
        "ready": is_ready,
        "error": err,
    }


@app.get("/ready", include_in_schema=False)
@app.get("/readyz", include_in_schema=False)
async def ready_probe(response: Response) -> dict[str, Any]:
    """Readiness probe: returns 200 when model is loaded and ready, 503 when loading or failed."""
    loaded = bool(engine.is_loaded)
    err = engine.load_error
    is_ready = loaded and not err
    if not is_ready:
        response.status_code = 503
        return {
            "ready": False,
            "status": "error" if err else "loading",
            "error": err,
            "message": (
                f"model failed to load: {err}"
                if err
                else f"model '{settings.served_model_name}' is currently loading weights"
            ),
        }
    return {"ready": True, "status": "ok", "model": settings.served_model_name}


@app.get("/live", include_in_schema=False)
@app.get("/livez", include_in_schema=False)
async def live_probe() -> dict[str, Any]:
    """Liveness probe: returns 200 if the server process is responsive."""
    return {"live": True, "status": "ok"}


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    base = {**engine.metrics(), **engine.info()}
    base["active_streams"] = _active_websockets
    base["queue_size"] = (
        base.get("core", {}).get("queued_pieces", 0) if isinstance(base.get("core"), dict) else 0
    )
    return base


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {"object": "list", "data": [_model_payload()]}


@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str) -> dict[str, Any]:
    _check_model(model_id)
    return _model_payload()


@app.post("/v1/audio/transcriptions", dependencies=[Depends(require_api_key)])
async def transcribe(
    request: Request,
    file: UploadFile = File(..., description="The audio file to transcribe."),
    model: str = Form(default=""),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    stream: bool = Form(default=False),
    stream_partials: bool = Form(default=True),
) -> Response:
    """Transcribe an audio file.

    All documented OpenAI parameters are accepted. ``prompt`` and ``temperature``
    are validated but cannot influence a single-pass CTC model. ``stream=true``
    switches the response to Server-Sent Events. The extensions this server adds
    (``stream_partials``, ``left_chunks``, ``use_lm``, ``beam_width``,
    ``chunk_seconds``) are documented in the README.
    """
    # The File/Form declarations have already parsed the multipart body; the
    # shared validator turns it into OpenAI-shaped errors for everything else.
    form = await request.form()
    params = parse_transcription_form(form)
    # The typed declarations are authoritative for the fields they cover.
    params.streaming = bool(stream) or params.streaming
    params.streaming_partials = bool(stream_partials)
    params.response_format = (response_format or "json").strip().lower()
    if language:
        params.language = language.strip().lower().split("-")[0]

    _check_model(model or form.get("model"))
    if params.response_format not in MEDIA_TYPES:
        raise APIError(
            f"'{params.response_format}' is not a valid response_format; choose "
            f"one of {', '.join(MEDIA_TYPES)}",
            param="response_format",
            code="invalid_value",
        )

    # Readiness guard: reject immediately if model is not loaded
    if not engine.is_loaded:
        if engine.load_error:
            raise APIError(
                f"Model failed to load: {engine.load_error}",
                status_code=503,
                code="model_load_error",
                error_type="server_error",
            )
        raise APIError(
            f"Model '{settings.served_model_name}' is currently loading weights. Please wait a moment and retry.",
            status_code=503,
            code="model_loading",
            error_type="server_error",
        )

    data = await file.read()
    if not data:
        raise APIError("the uploaded file is empty", code="invalid_audio_file")
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise APIError(
            f"the audio file exceeds the {settings.max_upload_mb} MB limit",
            status_code=413,
            code="payload_too_large",
        )

    wav = _decode_upload(data)
    duration = wav.size / SAMPLE_RATE
    model_name = settings.served_model_name

    if params.streaming:
        return _streaming_response(wav, params, duration, model_name)

    try:
        segments = await run_in_threadpool(
            engine.transcribe,
            wav,
            streaming=False,
            left_chunks=params.left_chunks,
            chunk_seconds=params.chunk_seconds,
            use_lm=params.use_lm,
            beam_width=params.beam_width,
        )
    except EngineBusy as exc:
        # The decode queue is full. Refusing fast is the point of backpressure:
        # a 503 the client can retry beats a request that hangs for minutes.
        raise APIError(
            str(exc),
            status_code=503,
            code="server_busy",
            error_type="server_error",
        ) from exc
    except EngineNotReady as exc:
        raise APIError(str(exc), status_code=503, error_type="server_error") from exc

    segments = _shape_segments(segments, params)
    processor = (
        PostProcessor(norm_mode=params.norm_mode, enable_g2p=params.phonemes)
        if (params.norm_mode != "itn" or params.phonemes)
        else default_post_processor
    )
    if params.punctuate and settings.enable_punctuation:
        segments = processor.process_segments(segments, include_phonemes=params.phonemes)
    elif params.phonemes:
        for seg in segments:
            seg.phonemes = to_phonemes(seg.text)
    body = render(
        params.response_format,
        segments,
        language="vi",
        duration=duration,
        model=model_name,
    )
    media_type = MEDIA_TYPES[params.response_format]
    if params.response_format in ("text", "srt", "vtt"):
        return PlainTextResponse(body, media_type=media_type)
    return Response(content=body, media_type=media_type)


def _event_payload(event: StreamEvent) -> dict[str, Any]:
    """Map the engine's event kinds onto OpenAI-style event type names.

    The mapping has to be applied *after* the spread: ``as_dict`` also carries a
    ``type`` key, derived from the engine's internal kind, and spreading it last
    would overwrite the name the client is meant to see.
    """
    kind_to_type = {
        "partial": "transcript.text.delta",
        "final": "transcript.text.segment",
        "done": "transcript.text.completed",
    }
    payload = event.as_dict()
    payload["type"] = kind_to_type.get(event.kind, event.kind)
    return payload


def _streaming_response(
    wav: np.ndarray,
    params: TranscriptionParams,
    duration: float,
    model_name: str,
) -> Response:
    """Build the Server-Sent Events response for ``stream=true``."""

    async def events() -> AsyncIterator[str]:
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        stop = threading.Event()

        def produce() -> None:
            def emit(payload: dict[str, Any] | None) -> None:
                if stop.is_set():
                    return
                asyncio.run_coroutine_threadsafe(queue.put(payload), loop).result()

            try:
                for event in engine.iter_stream(
                    wav,
                    left_chunks=params.left_chunks,
                    chunk_seconds=params.chunk_seconds,
                    use_lm=params.use_lm,
                    beam_width=params.beam_width,
                ):
                    if stop.is_set():
                        break
                    if event.kind == "partial" and not params.streaming_partials:
                        continue
                    emit(_event_payload(event))
            except Exception as exc:
                logger.exception("streaming decode failed")
                emit({"type": "error", "error": {"message": str(exc)}})
            finally:
                emit(None)

        worker = loop.run_in_executor(None, produce)
        finals: list[str] = []
        try:
            while True:
                payload = await queue.get()
                if payload is None:
                    break
                if payload.get("type") == "transcript.text.segment":
                    seg_text = payload["text"]
                    if params.punctuate and settings.enable_punctuation:
                        seg_text = default_post_processor.process(seg_text, is_final=False)
                        payload["text"] = seg_text
                    if params.phonemes and seg_text:
                        payload["phonemes"] = to_phonemes(seg_text)
                    finals.append(seg_text)
                elif payload.get("type") == "transcript.text.delta":
                    if params.phonemes and payload.get("text"):
                        payload["phonemes"] = to_phonemes(payload["text"])
                yield _sse(payload)
        except asyncio.CancelledError:
            # The client went away: stop decoding instead of running to the end
            # of the file for a response nobody will read.
            stop.set()
            raise
        finally:
            stop.set()
            await worker

        # A final summary event carrying the whole transcript, so a client that
        # only looks at the last event still gets the text.
        full_transcript = " ".join(t for t in finals if t).strip()
        if params.punctuate and settings.enable_punctuation and full_transcript:
            full_transcript = default_post_processor.process(full_transcript, is_final=True)

        done_event: dict[str, Any] = {
            "type": "transcript.text.done",
            "text": full_transcript,
            "duration": round(duration, 3),
            "model": model_name,
            "language": "vi",
            "elapsed": round(time.perf_counter() - started, 3),
        }
        if params.phonemes and full_transcript:
            done_event["phonemes"] = to_phonemes(full_transcript)
        yield _sse(done_event)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Ask nginx and friends not to buffer, which would defeat streaming.
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------------------------------- #
# websocket
# --------------------------------------------------------------------------- #


@app.websocket("/v1/audio/transcriptions/stream")
async def transcribe_stream(websocket: WebSocket) -> None:
    """Live transcription over a WebSocket.

    The client sends binary audio frames and optional text control messages
    (``stop``, ``flush`` or ``eof`` to finish; ``ping``). Each binary frame goes
    to the streaming decoder, and every event it produces is returned as JSON.

    Query parameters:

    ``format``
        ``f32`` (default) for little-endian float32 samples in ``[-1, 1]``, or
        ``pcm16`` for signed 16-bit samples.
    ``left_chunks``
        Past 0.48 s chunks the reader may attend to. Higher is more accurate but
        less "live". Omit for the server default.
    ``use_lm``
        ``true`` to force KenLM beam search, ``false`` to force greedy decoding.
        Omit for the server default.
    ``beam_width``
        Beams kept by the beam search. Omit for the server default.
    ``api_key``
        Accepted as an alternative to the ``Authorization`` header.
    """
    await websocket.accept()

    if not engine.is_loaded:
        msg = (
            f"Model failed to load: {engine.load_error}"
            if engine.load_error
            else f"Model '{settings.served_model_name}' is currently loading weights. Please retry shortly."
        )
        await websocket.send_json(
            {
                "type": "error",
                "error": {
                    "message": msg,
                    "type": "server_error",
                    "code": "model_loading" if not engine.load_error else "model_load_error",
                },
            }
        )
        await websocket.close(code=1013)
        return

    if settings.api_key:
        header = websocket.headers.get("authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        token = token or websocket.query_params.get("api_key", "")
        if token != settings.api_key:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": {
                        "message": "invalid API key",
                        "type": "authentication_error",
                    },
                }
            )
            await websocket.close(code=4401)
            return

    sample_format = (websocket.query_params.get("format") or "f32").lower()
    if sample_format not in {"f32", "pcm16"}:
        await websocket.send_json(
            {
                "type": "error",
                "error": {
                    "message": "format must be 'f32' (default) or 'pcm16'",
                    "type": "invalid_request_error",
                },
            }
        )
        await websocket.close(code=4400)
        return

    left_chunks: int | None = settings.left_chunks
    raw_left = websocket.query_params.get("left_chunks")
    if raw_left not in (None, ""):
        try:
            left_chunks = max(0, int(raw_left))
        except ValueError:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": {
                        "message": "left_chunks must be an integer",
                        "type": "invalid_request_error",
                    },
                }
            )
            await websocket.close(code=4400)
            return

    raw_use_lm = websocket.query_params.get("use_lm")
    use_lm: bool | None = None
    if raw_use_lm not in (None, ""):
        use_lm = raw_use_lm.strip().lower() in {"1", "true", "yes", "on"}
    if use_lm and not settings.supports_lm:
        await websocket.send_json(
            {
                "type": "error",
                "error": {
                    "message": "no KenLM model is loaded; start the server with "
                    "USE_LM=true or set LM_PATH to a local KenLM model",
                    "type": "server_error",
                },
            }
        )
        await websocket.close(code=1011)
        return

    beam_width: int | None = None
    raw_beam = websocket.query_params.get("beam_width")
    if raw_beam not in (None, ""):
        try:
            beam_width = max(1, int(raw_beam))
        except ValueError:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": {
                        "message": "beam_width must be an integer",
                        "type": "invalid_request_error",
                    },
                }
            )
            await websocket.close(code=4400)
            return

    try:
        live = await run_in_threadpool(
            engine.new_live_session, left_chunks, None, use_lm, beam_width
        )
    except EngineNotReady as exc:
        await websocket.send_json(
            {"type": "error", "error": {"message": str(exc), "type": "server_error"}}
        )
        await websocket.close(code=1011)
        return

    global _active_websockets
    _active_websockets += 1

    raw_punc = websocket.query_params.get("punctuate") or websocket.query_params.get("enable_punctuation")
    should_punctuate = (
        raw_punc.strip().lower() in ("1", "true", "yes", "on")
        if raw_punc is not None
        else settings.enable_punctuation
    )
    raw_phonemes = websocket.query_params.get("phonemes") or websocket.query_params.get("g2p")
    include_phonemes = (
        raw_phonemes.strip().lower() in ("1", "true", "yes", "on")
        if raw_phonemes is not None
        else False
    )

    await websocket.send_json(
        {
            "type": "session.created",
            "model": settings.served_model_name,
            "language": "vi",
            "format": sample_format,
            "sample_rate": SAMPLE_RATE,
            "step_seconds": settings.stream_step_seconds,
            "left_chunks": left_chunks,
            "use_lm": use_lm if use_lm is not None else settings.supports_lm,
            "beam_width": beam_width or settings.beam_width,
            "punctuate": should_punctuate,
            "phonemes": include_phonemes,
        }
    )

    bytes_per_sample = 4 if sample_format == "f32" else 2
    max_bytes = int(settings.max_stream_seconds * SAMPLE_RATE * bytes_per_sample)
    received = 0
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            text = message.get("text")
            if text is not None:
                command = text.strip().lower()
                if command in {"stop", "flush", "eof"}:
                    final = await run_in_threadpool(live.flush)
                    if final:
                        final_text = (
                            default_post_processor.process(final, is_final=True)
                            if should_punctuate
                            else final
                        )
                        done_payload: dict[str, Any] = {
                            "type": "transcript.text.done",
                            "text": final_text,
                            "final": True,
                            "end": round(live.consumed_seconds, 3),
                        }
                        if include_phonemes and final_text:
                            done_payload["phonemes"] = to_phonemes(final_text)
                        await websocket.send_json(done_payload)
                    break
                if command == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": {
                            "message": f"unknown control message: {text!r}",
                            "type": "invalid_request_error",
                        },
                    }
                )
                continue

            payload = message.get("bytes")
            if not payload:
                continue

            received += len(payload)
            if received > max_bytes:
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": {
                            "message": (
                                "stream exceeded "
                                f"{settings.max_stream_seconds:.0f}s of audio"
                            ),
                            "type": "invalid_request_error",
                        },
                    }
                )
                break

            if sample_format == "pcm16":
                samples = np.frombuffer(payload, dtype="<i2").astype(np.float32)
                samples /= 32768.0
            else:
                # Tolerate a trailing partial sample from a chunked sender.
                usable = len(payload) - (len(payload) % 4)
                samples = np.frombuffer(payload[:usable], dtype="<f4").astype(
                    np.float32
                )

            try:
                events = await run_in_threadpool(live.accept, samples)
            except EngineBusy:
                # The engine shed this step. LiveStream kept the audio, so the
                # next frame retries it; tell the client it was throttled rather
                # than dropping a chunk of speech silently.
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": {
                            "message": "the server is busy; audio was buffered, "
                            "slow the stream down",
                            "type": "server_error",
                            "code": "server_busy",
                        },
                    }
                )
                continue
            for event in events:
                ev_payload = _event_payload(event)
                if (
                    event.kind in ("final", "done")
                    and ev_payload.get("text")
                    and should_punctuate
                ):
                    ev_payload["text"] = default_post_processor.process(
                        ev_payload["text"], is_final=True
                    )
                if include_phonemes and ev_payload.get("text"):
                    ev_payload["phonemes"] = to_phonemes(ev_payload["text"])
                await websocket.send_json(ev_payload)

        await websocket.send_json({"type": "session.closed"})
    except Exception:
        logger.debug("websocket session ended early", exc_info=True)
    finally:
        _active_websockets = max(0, _active_websockets - 1)
        await run_in_threadpool(live.close)
        with contextlib.suppress(RuntimeError):
            await websocket.close()
