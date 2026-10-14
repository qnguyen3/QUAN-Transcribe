"""Live transcription from a microphone: words appear while you speak.

The current utterance is re-decoded in streaming mode each time a new 0.48 s chunk of audio arrives. Streaming mode
never looks more than one chunk ahead, so the text from finished chunks stays put and only the newest words can still
change. An utterance ends after a pause with no new tokens; it is then decoded once more with full context (more
accurate, and with the n-gram LM if one is loaded), printed on its own line, and the buffer starts over.
"""

import queue
import shutil
import sys

import numpy as np
import soxr

from .audio import SAMPLE_RATE, split_quiet
from .model import BLANK, greedy_decode

FRAME_SAMPLES = 640  # one encoder frame (40 ms)


class LiveSession:
    """Incremental decoding of a 16 kHz mono stream. Feed audio with accept(); see the module docstring."""

    def __init__(
        self,
        asr,
        left_chunks: int | None = None,
        endpoint_s: float = 0.8,
        max_utterance_s: float = 30.0,
        beam_width: int = 32,
    ):
        self.asr = asr
        self.left_chunks = left_chunks
        self.beam_width = beam_width
        self.endpoint_frames = round(endpoint_s * SAMPLE_RATE / FRAME_SAMPLES)
        self.max_utterance_s = max_utterance_s
        self.buf = np.zeros(0, dtype=np.float32)
        self.partial = ""

    def accept(self, samples: np.ndarray) -> list[str]:
        """Add 16 kHz mono audio and decode. Returns the utterances that finished; see .partial for the rest."""
        self.buf = np.concatenate([self.buf, np.asarray(samples, dtype=np.float32)])
        finals = []
        if len(self.buf) > self.max_utterance_s * SAMPLE_RATE:
            # Long stretch without a pause: cut at the quietest point near the end.
            head = split_quiet(self.buf, self.max_utterance_s)[0]
            finals.append(self._final(head))
            self.buf = self.buf[len(head) :]

        logits, lens = self.asr._forward([self.buf], True, self.left_chunks)
        best = logits[0, : int(lens[0])].argmax(-1).cpu()
        self.partial = self.asr.decode_units(greedy_decode(logits, lens)[0])
        emitted = (best != BLANK).nonzero()
        if not self.partial:
            # Nothing said yet: drop old silence, keeping enough for a word that has started but not been emitted.
            if len(self.buf) > 3 * SAMPLE_RATE:
                self.buf = self.buf[-SAMPLE_RATE:]
        elif len(best) - 1 - int(emitted[-1]) >= self.endpoint_frames:
            finals.append(self._final(self.buf))
            self.buf = self.buf[:0]
            self.partial = ""
        return [t for t in finals if t]

    def flush(self) -> str:
        """Finish the current utterance (e.g. when the input stops)."""
        text = self._final(self.buf) if len(self.buf) > SAMPLE_RATE // 10 else ""
        self.buf = self.buf[:0]
        self.partial = ""
        return text

    def _final(self, wav: np.ndarray) -> str:
        return self.asr._run([wav], False, None, self.asr.lm_decoder, self.beam_width)[0]


def list_input_devices() -> str:
    import sounddevice as sd

    return str(sd.query_devices())


def transcribe_microphone(
    asr,
    device: int | str | None = None,
    left_chunks: int | None = None,
    endpoint_s: float = 0.8,
    max_utterance_s: float = 30.0,
    beam_width: int = 32,
    out=sys.stdout,
) -> None:
    """Transcribe the microphone until Ctrl+C. Finished utterances are printed one per line; on a terminal the
    utterance in progress is shown (dimmed) and updated in place."""
    import sounddevice as sd

    session = LiveSession(asr, left_chunks, endpoint_s, max_utterance_s, beam_width)
    step = asr.model.cfg.chunk_frames * FRAME_SAMPLES  # decode once per streaming chunk (0.48 s)
    tty = out.isatty()

    def show(partial: str = "", final: str | None = None) -> None:
        if not tty:
            if final:
                out.write(final + "\n")
        else:
            out.write("\r\033[K")
            if final:
                out.write(final + "\n")
            if partial:
                width = shutil.get_terminal_size().columns - 1
                if len(partial) > width:  # stay on one line so \r can redraw it
                    partial = "…" + partial[-(width - 1) :]
                out.write(f"\033[2m{partial}\033[0m")
        out.flush()

    blocks: queue.Queue[np.ndarray] = queue.Queue()
    sr = int(sd.query_devices(device, "input")["default_samplerate"])
    resampler = soxr.ResampleStream(sr, SAMPLE_RATE, 1, dtype="float32") if sr != SAMPLE_RATE else None

    def callback(indata, frames, time, status):
        blocks.put(indata[:, 0].copy())

    print("Listening... speak now (Ctrl+C to stop)", file=sys.stderr, flush=True)
    pending = []
    with sd.InputStream(device=device, channels=1, samplerate=sr, dtype="float32", blocksize=sr // 10, callback=callback):
        try:
            while True:
                # Wait for one chunk of new audio, then take everything queued (catches up if decoding fell behind).
                pending.append(blocks.get())
                while not blocks.empty():
                    pending.append(blocks.get())
                if sum(map(len, pending)) * SAMPLE_RATE < step * sr:
                    continue
                audio = np.concatenate(pending)
                pending = []
                if resampler is not None:
                    audio = resampler.resample_chunk(audio)
                for text in session.accept(audio):
                    show(final=text)
                show(session.partial)
        except KeyboardInterrupt:
            pass
    show(final=session.flush())
