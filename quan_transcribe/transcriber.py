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
LM_DIR = "lm"  # optional n-gram LM inside the model repo: lm/vi_4gram.bin, lm/unigrams.txt, lm/lm_config.json
LM_FILE, UNIGRAMS_FILE, LM_CONFIG_FILE = "vi_4gram.bin", "unigrams.txt", "lm_config.json"

AudioInput = str | os.PathLike | np.ndarray | torch.Tensor


class QuanTranscribe:
    def __init__(self, model: QuanCTC, tokenizer, gemma_ids: list[int], device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.gemma_ids = gemma_ids
        self.device = device
        self.lm_decoder = None

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL,
        device: str | torch.device | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        lm: bool | str = False,
    ) -> "QuanTranscribe":
        """Load from a Hugging Face repo id or a local directory with config.json, units.json, model.safetensors.

        The base encoder google/embeddinggemma-2 is downloaded on first use (accept its terms on the HF page).
        lm: True loads the n-gram LM shipped with the model (needs `pip install kenlm`); a path loads your own
            (a directory laid out like the model repo's lm/ folder, or a KenLM .bin/.arpa file).
        """
        if os.path.isdir(model_id):
            path = model_id
        else:
            path = snapshot_download(model_id, revision=revision, token=token, allow_patterns=MODEL_FILES)
        with open(os.path.join(path, "config.json")) as f:
            cfg = ModelConfig.from_dict(json.load(f))
        with open(os.path.join(path, "units.json")) as f:
            units = json.load(f)

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = QuanCTC(cfg, load_encoder(cfg.encoder_id))
        # The checkpoint holds the reader + the fine-tuned top encoder layers; the rest comes from the base model.
        state = load_file(os.path.join(path, "model.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("encoder.")]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint does not match the model: missing={missing} unexpected={unexpected}")
        model = model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(units.get("model_id", cfg.encoder_id))
        asr = cls(model, tokenizer, units["gemma_ids"], device)
        if lm is True:
            if os.path.isdir(model_id):
                lm = os.path.join(path, LM_DIR)
            else:
                lm_root = snapshot_download(model_id, revision=revision, token=token, allow_patterns=[f"{LM_DIR}/*"])
                lm = os.path.join(lm_root, LM_DIR)
        if lm:
            asr.load_lm(lm)
        return asr

    def load_lm(
        self, path: str, unigrams: str | None = None, alpha: float | None = None, beta: float | None = None
    ) -> None:
        """Enable beam-search decoding with a word-level KenLM model.

        path: a directory with vi_4gram.bin (+ optional unigrams.txt, lm_config.json), or a KenLM .bin/.arpa file.
        alpha (LM weight) and beta (word bonus) default to lm_config.json, else 0.3 / 1.0.
        """
        from .lm import KenLMDecoder

        cfg = {}
        if os.path.isdir(path):
            if unigrams is None and os.path.exists(os.path.join(path, UNIGRAMS_FILE)):
                unigrams = os.path.join(path, UNIGRAMS_FILE)
            if os.path.exists(os.path.join(path, LM_CONFIG_FILE)):
                with open(os.path.join(path, LM_CONFIG_FILE)) as f:
                    cfg = json.load(f)
            path = os.path.join(path, LM_FILE)
        words = None
        if unigrams:
            with open(unigrams, encoding="utf-8") as f:
                words = [w.strip() for w in f if w.strip()]
        # Unit ids: 0 = blank, 1 = <unk> (never emitted by the beam search), 2.. = Gemma tokens.
        pieces = ["", ""] + self.tokenizer.convert_ids_to_tokens(self.gemma_ids)
        self.lm_decoder = KenLMDecoder(
            path, pieces, words,
            alpha=cfg.get("alpha", 0.3) if alpha is None else alpha,
            beta=cfg.get("beta", 1.0) if beta is None else beta,
        )

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
        use_lm: bool | None = None,
        beam_width: int = 32,
    ) -> str | list[str]:
        """Transcribe one input (returns a string) or a list of inputs (returns a list of strings).

        audio: file path, or a waveform array (mono or (samples, channels)) at `sample_rate` (default 16 kHz).
        streaming: decode as the model would in streaming mode (0.48 s chunks, no lookahead beyond the chunk).
        left_chunks: in streaming mode, how many past chunks the reader may attend to (None = all).
        chunk_seconds: long recordings are cut into pieces of at most this length at quiet points.
        batch_seconds: padded audio per forward pass; lower it if you run out of memory.
        use_lm: beam search with the n-gram LM (default: whenever one is loaded); False forces greedy decoding.
        beam_width: beams kept by the LM beam search.
        """
        if use_lm and self.lm_decoder is None:
            raise ValueError("no LM loaded: use from_pretrained(..., lm=True) or load_lm(path)")
        decoder = self.lm_decoder if use_lm is not False else None
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
                wavs = [pieces[b][2] for b in batch]
                for kk, text in zip(batch, self._run(wavs, streaming, left_chunks, decoder, beam_width)):
                    texts[pieces[kk][:2]] = text
            batch = [k] if k is not None else []

        out = []
        for i in range(len(inputs)):
            parts = [texts[(i, j)] for (ii, j, _) in pieces if ii == i]
            out.append(" ".join(t for t in parts if t))
        return out[0] if single else out

    def _run(
        self, wavs: list[np.ndarray], streaming: bool, left_chunks: int | None, decoder, beam_width: int
    ) -> list[str]:
        lens = torch.tensor([len(w) for w in wavs], dtype=torch.long)
        wav = torch.zeros(len(wavs), int(lens.max()), dtype=torch.float32)
        for i, w in enumerate(wavs):
            wav[i, : len(w)] = torch.from_numpy(w)
        on_cuda = self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=on_cuda):
            logits, out_lens = self.model(wav.to(self.device), lens.to(self.device), streaming, left_chunks)
        if decoder is None:
            return [self.decode_units(u) for u in greedy_decode(logits, out_lens)]
        log_probs = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        return [
            self.decode_units(decoder.decode(log_probs[i, : int(n)], beam_width=beam_width))
            for i, n in enumerate(out_lens)
        ]
