"""Command line: quan-transcribe audio1.wav audio2.mp3 ...  |  quan-transcribe --mic"""

import argparse
import sys
import time

from .audio import SAMPLE_RATE, load_audio
from .live import list_input_devices, transcribe_microphone
from .transcriber import DEFAULT_MODEL, QuanTranscribe


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="quan-transcribe", description="Vietnamese speech recognition (QUAN-Transcribe).")
    ap.add_argument("audio", nargs="*", help="audio files (wav, flac, ogg, mp3, ...)")
    ap.add_argument("--mic", action="store_true", help="transcribe the microphone live (Ctrl+C to stop)")
    ap.add_argument("--input-device", default=None, help="mic: input device index or name (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    ap.add_argument("--endpoint", type=float, default=0.8, help="mic: seconds of silence that end an utterance")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local directory")
    ap.add_argument("--device", default=None, help="cuda, cuda:1, mps, cpu, ... (default: cuda, then mps, then cpu)")
    ap.add_argument("--streaming", action="store_true", help="decode in streaming mode (0.48 s chunks)")
    ap.add_argument("--left-chunks", type=int, default=None, help="streaming: past chunks to attend to (default all)")
    ap.add_argument("--chunk-seconds", type=float, default=30.0, help="max piece length for long audio")
    ap.add_argument("--batch-seconds", type=float, default=600.0, help="padded audio per forward pass")
    ap.add_argument("-v", "--verbose", action="store_true", help="print duration and speed per file")
    args = ap.parse_args(argv)
    if args.list_devices:
        print(list_input_devices())
        return
    if args.mic == bool(args.audio):
        ap.error("give audio files or --mic (not both)")

    asr = QuanTranscribe.from_pretrained(args.model, device=args.device)
    if args.mic:
        device = int(args.input_device) if (args.input_device or "").isdigit() else args.input_device
        transcribe_microphone(asr, device, args.left_chunks, args.endpoint, args.chunk_seconds)
        return
    for path in args.audio:
        t0 = time.time()
        wav = load_audio(path)
        text = asr.transcribe(
            wav, streaming=args.streaming, left_chunks=args.left_chunks,
            chunk_seconds=args.chunk_seconds, batch_seconds=args.batch_seconds,
        )
        dt = time.time() - t0
        if len(args.audio) > 1 or args.verbose:
            info = f"  ({len(wav) / SAMPLE_RATE:.1f}s audio, {dt:.2f}s, {len(wav) / SAMPLE_RATE / dt:.0f}x realtime)"
            print(f"==> {path}{info if args.verbose else ''}")
        print(text)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
