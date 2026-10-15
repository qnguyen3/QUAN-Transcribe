"""Command line entrypoint.

``quan-asr-server`` starts the server. The flags exist so the image can be
launched directly by GPUStack (which always appends ``--model-path`` and
``--port``) as well as by hand.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quan-asr-server",
        description="OpenAI-compatible ASR server for QUAN-Transcribe-1.0.",
    )
    parser.add_argument(
        "--version", action="version", version=f"quan-asr-server {__version__}"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="accepted (and ignored) so GPUStack commands can pass it; the "
        "server is always the action",
    )
    parser.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        default=None,
        help="Hugging Face repo id or local model directory",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--device", default=None, help="cuda, cuda:1, mps or cpu (default: auto)"
    )
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="name clients pass as 'model' (default: QUAN-Transcribe-1.0)",
    )
    parser.add_argument(
        "--lm",
        action="store_true",
        default=None,
        help="decode with the bundled 4-gram KenLM beam search "
        "(downloads the model repo's lm/ folder on first use)",
    )
    parser.add_argument(
        "--lm-path",
        default=None,
        help="your own KenLM model: a .bin/.arpa file, or a directory with "
        "vi_4gram.bin and optionally unigrams.txt/lm_config.json",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=None,
        help="beams kept by the LM beam search (default 32)",
    )
    parser.add_argument(
        "--lm-alpha",
        type=float,
        default=None,
        help="LM weight; defaults to the value in lm_config.json (0.3)",
    )
    parser.add_argument(
        "--lm-beta",
        type=float,
        default=None,
        help="word-insertion bonus; defaults to the value in lm_config.json (1.0)",
    )
    parser.add_argument(
        "--left-chunks",
        type=int,
        default=None,
        help="streaming: past 0.48s chunks the reader may attend to (default: all)",
    )
    parser.add_argument("--chunk-seconds", type=float, default=None)
    parser.add_argument(
        "--subbatch-seconds",
        type=float,
        default=None,
        help="longest single forward pass in audio seconds; a big file is "
        "split across launches so live audio can preempt between them "
        "(0 = no cap, default)",
    )
    parser.add_argument("--endpoint-seconds", type=float, default=None)
    parser.add_argument("--stream-step-seconds", type=float, default=None)
    parser.add_argument(
        "--microbatch-seconds",
        type=float,
        default=None,
        help="overlap between concurrent requests before they share a forward "
        "pass; 0 disables cross-request batching (default 0.15)",
    )
    parser.add_argument(
        "--microbatch-max-wait-ms",
        type=float,
        default=None,
        help="how long a request waits for companions to batch with (default 20)",
    )
    parser.add_argument(
        "--max-queue",
        type=int,
        default=None,
        help="most audio pieces that may wait to decode; 0 is unbounded "
        "(default 256). Full queue -> HTTP 503",
    )
    parser.add_argument(
        "--max-interactive",
        type=int,
        default=None,
        help="most streaming (interactive) jobs in flight; 0 is unbounded "
        "(default 64)",
    )
    parser.add_argument("--cpu-threads",
        type=int,
        default=None,
        help="torch CPU thread count for the CPU profile (default: auto)",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        default=None,
        help="trim silence and cut pieces on real pauses with Silero VAD "
        "(needs the 'vad' extra; big win on mostly-silent call audio)",
    )
    parser.add_argument(
        "--no-vad-trim",
        action="store_true",
        help="keep silence in the output and only use VAD for cut points",
    )
    parser.add_argument(
        "--vad-speech-pad-ms",
        type=int,
        default=None,
        help="padding around a speech region in ms (default 250; Silero's 30 "
        "clips the first syllable)",
    )
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="disable TF32 matmuls on Ampere+ (exact FP32, slower)",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="load the model on the first request instead of at startup",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Real environment variables win over flag defaults, so options only need to
    # be passed when they differ from the image's configuration.
    overrides: dict[str, object] = {
        "model_path": args.model_path,
        "host": args.host,
        "port": args.port,
        "subbatch_seconds": args.subbatch_seconds,
        "device": args.device,
        "served_model_name": args.served_model_name,
        "lm_path": args.lm_path,
        "lm_alpha": args.lm_alpha,
        "lm_beta": args.lm_beta,
        "beam_width": args.beam_width,
        "left_chunks": args.left_chunks,
        "chunk_seconds": args.chunk_seconds,
        "endpoint_s": args.endpoint_seconds,
        "stream_step_seconds": args.stream_step_seconds,
        "microbatch_seconds": args.microbatch_seconds,
        "microbatch_max_wait_ms": args.microbatch_max_wait_ms,
        "max_queue": args.max_queue,
        "max_interactive": args.max_interactive,
        "cpu_threads": args.cpu_threads,
        "vad_speech_pad_ms": args.vad_speech_pad_ms,
        "vad_trim_silence": False if args.no_vad_trim else None,
        "tf32": False if args.no_tf32 else None,
        "api_key": args.api_key,
        "log_level": args.log_level,
        "preload": False if args.no_preload else None,
    }
    if args.lm:
        overrides["use_lm"] = True
    if args.vad:
        overrides["use_vad"] = True

    os.environ.setdefault("QUAN_ASR_CLI", "1")
    for key, value in overrides.items():
        if value is None:
            continue
        env_name = _ENV_FOR_SETTING[key]
        os.environ[env_name] = str(value)

    # Import after the environment is finalised: app.config reads it at import.
    import uvicorn

    from .config import Settings

    settings = Settings.from_env()
    uvicorn.run(
        "app.server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        access_log=settings.log_level in ("debug", "trace"),
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
    return 0


_ENV_FOR_SETTING = {
    "model_path": "MODEL_PATH",
    "host": "HOST",
    "port": "PORT",
    "device": "DEVICE",
    "served_model_name": "SERVED_MODEL_NAME",
    "lm_path": "LM_PATH",
    "lm_alpha": "LM_ALPHA",
    "lm_beta": "LM_BETA",
    "beam_width": "BEAM_WIDTH",
    "left_chunks": "LEFT_CHUNKS",
    "subbatch_seconds": "SUBBATCH_SECONDS",
    "chunk_seconds": "CHUNK_SECONDS",
    "endpoint_s": "ENDPOINT_SECONDS",
    "stream_step_seconds": "STREAM_STEP_SECONDS",
    "max_queue": "MAX_QUEUE",
    "max_interactive": "MAX_INTERACTIVE",
    "microbatch_seconds": "MICROBATCH_SECONDS",
    "microbatch_max_wait_ms": "MICROBATCH_MAX_WAIT_MS",
    "cpu_threads": "CPU_THREADS",
    "vad_speech_pad_ms": "VAD_SPEECH_PAD_MS",
    "vad_trim_silence": "VAD_TRIM_SILENCE",
    "tf32": "TF32",
    "api_key": "API_KEY",
    "log_level": "LOG_LEVEL",
    "preload": "PRELOAD",
}


if __name__ == "__main__":
    sys.exit(main())
