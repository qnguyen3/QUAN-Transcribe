"""Command line: quan-transcribe audio1.wav audio2.mp3 ..."""

import argparse
import sys
import time

from .audio import SAMPLE_RATE, load_audio
from .transcriber import DEFAULT_MODEL, QuanTranscribe


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="quan-transcribe", description="Vietnamese speech recognition (QUAN-Transcribe).")
    ap.add_argument("audio", nargs="+", help="audio files (wav, flac, ogg, mp3, ...)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local directory")
    ap.add_argument("--device", default=None, help="cuda, cuda:1, cpu, ... (default: cuda if available)")
    ap.add_argument("--streaming", action="store_true", help="decode in streaming mode (0.48 s chunks)")
    ap.add_argument("--left-chunks", type=int, default=None, help="streaming: past chunks to attend to (default all)")
    ap.add_argument("--chunk-seconds", type=float, default=30.0, help="max piece length for long audio")
    ap.add_argument("--batch-seconds", type=float, default=600.0, help="padded audio per forward pass")
    ap.add_argument("--lm", action="store_true", help="beam search with the model's n-gram LM (pip install kenlm)")
    ap.add_argument("--lm-path", default=None, help="use your own KenLM model (.bin/.arpa or lm/ directory)")
    ap.add_argument("--beam-width", type=int, default=32, help="LM beam search: beams to keep")
    ap.add_argument("--lm-alpha", type=float, default=None, help="LM weight (default from lm_config.json)")
    ap.add_argument("--lm-beta", type=float, default=None, help="word insertion bonus (default from lm_config.json)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print duration and speed per file")
    args = ap.parse_args(argv)

    asr = QuanTranscribe.from_pretrained(args.model, device=args.device, lm=args.lm_path or args.lm)
    if asr.lm_decoder is not None and args.lm_alpha is not None:
        asr.lm_decoder.alpha = args.lm_alpha
    if asr.lm_decoder is not None and args.lm_beta is not None:
        asr.lm_decoder.beta = args.lm_beta
    for path in args.audio:
        t0 = time.time()
        wav = load_audio(path)
        text = asr.transcribe(
            wav, streaming=args.streaming, left_chunks=args.left_chunks,
            chunk_seconds=args.chunk_seconds, batch_seconds=args.batch_seconds, beam_width=args.beam_width,
        )
        dt = time.time() - t0
        if len(args.audio) > 1 or args.verbose:
            info = f"  ({len(wav) / SAMPLE_RATE:.1f}s audio, {dt:.2f}s, {len(wav) / SAMPLE_RATE / dt:.0f}x realtime)"
            print(f"==> {path}{info if args.verbose else ''}")
        print(text)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
