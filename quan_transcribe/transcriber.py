"""High-level API: load a QUAN-Transcribe checkpoint and turn audio into text."""

import json
import os

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from .audio import SAMPLE_RATE, load_audio, split_quiet, to_16k_mono
from .model import ModelConfig, QuanCTC, greedy_decode, load_encoder

DEFAULT_MODEL = "qnguyen3/QUAN-Transcribe-1.0"
MODEL_FILES = ["config.json", "units.json", "model.safetensors"]

AudioInput = str | os.PathLike | np.ndarray | torch.Tensor


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class QuanTranscribe:
    def __init__(self, model: QuanCTC, tokenizer, gemma_ids: list[int], device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.gemma_ids = gemma_ids
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL,
        device: str | torch.device | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
    ) -> "QuanTranscribe":
        """Load from a Hugging Face repo id or a local directory with config.json, units.json, model.safetensors.

        The base encoder google/embeddinggemma-2 is downloaded on first use (accept its terms on the HF page).
        """
        if os.path.isdir(model_id):
            path = model_id
        else:
            path = snapshot_download(model_id, revision=revision, token=token, allow_patterns=MODEL_FILES)
        with open(os.path.join(path, "config.json")) as f:
            cfg = ModelConfig.from_dict(json.load(f))
        with open(os.path.join(path, "units.json")) as f:
            units = json.load(f)

        device = torch.device(device or default_device())
        model = QuanCTC(cfg, load_encoder(cfg.encoder_id))
        # The checkpoint holds the reader + the fine-tuned top encoder layers; the rest comes from the base model.
        state = load_file(os.path.join(path, "model.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("encoder.")]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint does not match the model: missing={missing} unexpected={unexpected}")
        model = model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(units.get("model_id", cfg.encoder_id))
        return cls(model, tokenizer, units["gemma_ids"], device)

    def decode_units(self, unit_ids: list[int]) -> str:
        # Unit ids: 0 = blank, 1 = <unk>, 2.. = Gemma tokens.
        return self.tokenizer.decode([self.gemma_ids[u - 2] for u in unit_ids if u >= 2]).strip()

    def _prepare(self, audio: AudioInput, sample_rate: int | None) -> np.ndarray:
        if isinstance(audio, (str, os.PathLike)):
            return load_audio(os.fspath(audio))
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        return to_16k_mono(audio, sample_rate or SAMPLE_RATE)

    def transcribe(
        self,
        audio: AudioInput | list[AudioInput],
        streaming: bool = False,
        left_chunks: int | None = None,
        sample_rate: int | None = None,
        chunk_seconds: float = 30.0,
        batch_seconds: float = 600.0,
    ) -> str | list[str]:
        """Transcribe one input (returns a string) or a list of inputs (returns a list of strings).

        audio: file path, or a waveform array (mono or (samples, channels)) at `sample_rate` (default 16 kHz).
        streaming: decode as the model would in streaming mode (0.48 s chunks, no lookahead beyond the chunk).
        left_chunks: in streaming mode, how many past chunks the reader may attend to (None = all).
        chunk_seconds: long recordings are cut into pieces of at most this length at quiet points.
        batch_seconds: padded audio per forward pass; lower it if you run out of memory.
        """
        single = not isinstance(audio, list)
        inputs = [audio] if single else audio
        pieces = []  # (input index, piece index, waveform)
        for i, a in enumerate(inputs):
            for j, p in enumerate(split_quiet(self._prepare(a, sample_rate), chunk_seconds)):
                pieces.append((i, j, p))

        texts: dict[tuple[int, int], str] = {}
        order = sorted(range(len(pieces)), key=lambda k: len(pieces[k][2]))
        batch: list[int] = []
        for k in order + [None]:
            if k is not None:
                longest = len(pieces[k][2]) / SAMPLE_RATE
                if not batch or longest * (len(batch) + 1) <= batch_seconds:
                    batch.append(k)
                    continue
            if batch:
                for kk, text in zip(batch, self._run([pieces[b][2] for b in batch], streaming, left_chunks)):
                    texts[pieces[kk][:2]] = text
            batch = [k] if k is not None else []

        out = []
        for i in range(len(inputs)):
            parts = [texts[(i, j)] for (ii, j, _) in pieces if ii == i]
            out.append(" ".join(t for t in parts if t))
        return out[0] if single else out

    def _run(self, wavs: list[np.ndarray], streaming: bool, left_chunks: int | None) -> list[str]:
        return [self.decode_units(u) for u in greedy_decode(*self._forward(wavs, streaming, left_chunks))]

    def _forward(
        self, wavs: list[np.ndarray], streaming: bool, left_chunks: int | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lens = torch.tensor([len(w) for w in wavs], dtype=torch.long)
        wav = torch.zeros(len(wavs), int(lens.max()), dtype=torch.float32)
        for i, w in enumerate(wavs):
            wav[i, : len(w)] = torch.from_numpy(w)
        on_cuda = self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=on_cuda):
            return self.model(wav.to(self.device), lens.to(self.device), streaming, left_chunks)
