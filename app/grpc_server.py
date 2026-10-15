"""Google Speech API v2 compatible gRPC Service for QUAN-ASR Server.

Provides bidirectional streaming (StreamingRecognize) over HTTP/2 multiplexing,
mirroring Google Cloud Speech-to-Text v2 semantics.
"""

from __future__ import annotations

import logging
from concurrent import futures
from typing import TYPE_CHECKING

import grpc
import numpy as np

from .audio import AudioDecodeError, decode_audio_bytes, to_16k_mono
from .postprocess import default_post_processor, to_phonemes
from .proto import speech_pb2, speech_pb2_grpc

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .engine import TranscriptionEngine

logger = logging.getLogger(__name__)


class SpeechServicer(speech_pb2_grpc.SpeechServicer):
    """gRPC servicer for synchronous and bidirectional streaming ASR."""

    def __init__(self, engine: TranscriptionEngine) -> None:
        self.engine = engine

    def Recognize(
        self, request: speech_pb2.RecognizeRequest, context: grpc.ServicerContext
    ) -> speech_pb2.RecognizeResponse:
        """Single-shot file recognition."""
        if not self.engine.is_loaded:
            err_msg = (
                f"model failed to load: {self.engine.load_error}"
                if self.engine.load_error
                else "model is currently loading weights, please retry shortly"
            )
            context.abort(grpc.StatusCode.UNAVAILABLE, err_msg)

        if not request.audio_content:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "audio_content is empty")

        try:
            wav, sr = decode_audio_bytes(request.audio_content)
            wav = to_16k_mono(wav, sr)
        except AudioDecodeError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"cannot decode audio: {exc}")

        enable_punc = True
        enable_phonemes = False
        left_chunks = None
        use_lm = None
        if request.config:
            if request.config.enable_punctuation is not None:
                enable_punc = request.config.enable_punctuation
            enable_phonemes = bool(request.config.enable_phonemes)
            if request.config.left_chunks > 0:
                left_chunks = request.config.left_chunks
            use_lm = request.config.use_lm or None

        segments = self.engine.transcribe(
            wav,
            streaming=False,
            left_chunks=left_chunks,
            use_lm=use_lm,
        )

        if enable_punc:
            segments = default_post_processor.process_segments(segments)

        full_text = " ".join(s.text for s in segments if s.text).strip()
        duration = len(wav) / 16000.0
        ph = to_phonemes(full_text) if enable_phonemes and full_text else ""

        return speech_pb2.RecognizeResponse(
            results=[
                speech_pb2.SpeechRecognitionAlternative(
                    transcript=full_text,
                    confidence=0.98,
                    phonemes=ph,
                )
            ],
            total_billed_time=duration,
        )

    def StreamingRecognize(
        self,
        request_iterator: Iterator[speech_pb2.StreamingRecognizeRequest],
        context: grpc.ServicerContext,
    ) -> Iterator[speech_pb2.StreamingRecognizeResponse]:
        """Google Cloud Speech-style bidirectional audio streaming."""
        if not self.engine.is_loaded:
            err_msg = (
                f"model failed to load: {self.engine.load_error}"
                if self.engine.load_error
                else "model is currently loading weights, please retry shortly"
            )
            context.abort(grpc.StatusCode.UNAVAILABLE, err_msg)

        config: speech_pb2.StreamingRecognitionConfig | None = None
        live = None
        enable_punc = True
        enable_phonemes = False
        sample_format = "pcm16"

        try:
            for req in request_iterator:
                # 1. First message may configure the stream
                if req.HasField("streaming_config"):
                    config = req.streaming_config
                    enable_punc = config.enable_punctuation
                    enable_phonemes = bool(config.enable_phonemes)
                    enc = (config.encoding or "LINEAR16").upper()
                    sample_format = "f32" if enc in ("FLOAT32", "F32") else "pcm16"

                    left_chunks = config.left_chunks if config.left_chunks > 0 else None
                    use_lm = config.use_lm or None
                    live = self.engine.create_live_session(
                        left_chunks=left_chunks,
                        use_lm=use_lm,
                    )
                    continue

                # Lazily initialize if no config message was sent
                if live is None:
                    live = self.engine.create_live_session()

                # 2. Audio chunks
                chunk = req.audio_content
                if not chunk:
                    continue

                if sample_format == "pcm16":
                    samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32)
                    samples /= 32768.0
                else:
                    usable = len(chunk) - (len(chunk) % 4)
                    samples = np.frombuffer(chunk[:usable], dtype="<f4").astype(np.float32)

                events = live.accept(samples)
                for ev in events:
                    if ev.kind == "partial":
                        text = ev.text
                        if enable_punc:
                            text = default_post_processor.process(text, is_final=False)
                        ph = to_phonemes(text) if enable_phonemes and text else ""
                        yield speech_pb2.StreamingRecognizeResponse(
                            results=[
                                speech_pb2.StreamingRecognitionResult(
                                    alternatives=[
                                        speech_pb2.SpeechRecognitionAlternative(
                                            transcript=text,
                                            confidence=0.85,
                                            phonemes=ph,
                                        )
                                    ],
                                    is_final=False,
                                    stability=0.7,
                                    result_end_time=round(ev.start + ev.duration, 2),
                                )
                            ]
                        )
                    elif ev.kind in ("final", "done"):
                        text = ev.text
                        if enable_punc:
                            text = default_post_processor.process(text, is_final=True)
                        ph = to_phonemes(text) if enable_phonemes and text else ""
                        yield speech_pb2.StreamingRecognizeResponse(
                            results=[
                                speech_pb2.StreamingRecognitionResult(
                                    alternatives=[
                                        speech_pb2.SpeechRecognitionAlternative(
                                            transcript=text,
                                            confidence=0.98,
                                            phonemes=ph,
                                        )
                                    ],
                                    is_final=True,
                                    stability=1.0,
                                    result_end_time=round(ev.start + ev.duration, 2),
                                )
                            ]
                        )

            # Stream finished by client
            if live is not None:
                for ev in live.flush():
                    if ev.text:
                        text = default_post_processor.process(ev.text, is_final=True) if enable_punc else ev.text
                        ph = to_phonemes(text) if enable_phonemes and text else ""
                        yield speech_pb2.StreamingRecognizeResponse(
                            results=[
                                speech_pb2.StreamingRecognitionResult(
                                    alternatives=[
                                        speech_pb2.SpeechRecognitionAlternative(
                                            transcript=text,
                                            confidence=0.98,
                                            phonemes=ph,
                                        )
                                    ],
                                    is_final=True,
                                    stability=1.0,
                                    result_end_time=round(ev.start + ev.duration, 2),
                                )
                            ]
                        )

        except Exception as exc:
            logger.exception("gRPC StreamingRecognize error")
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        finally:
            if live is not None:
                live.close()


def create_grpc_server(
    engine: TranscriptionEngine,
    host: str = "0.0.0.0",
    port: int = 50051,
    max_workers: int = 10,
) -> grpc.Server:
    """Instantiate a configured gRPC server."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ],
    )
    speech_pb2_grpc.add_SpeechServicer_to_server(SpeechServicer(engine), server)
    server.add_insecure_port(f"{host}:{port}")
    return server
