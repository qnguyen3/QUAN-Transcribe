"""Batched log-mel features, numerically matching the EmbeddingGemma 2 audio feature extractor.

The HF extractor is a per-utterance numpy loop; this computes the exact same thing in torch so it can run batched
on the GPU (window and mel filters are copied from the extractor).
"""

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor


class LogMel(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        fe = AutoFeatureExtractor.from_pretrained(model_id)
        assert fe.preemphasis == 0.0 and fe.dither == 0.0 and fe.input_scale_factor == 1.0
        assert fe.per_bin_mean is None and fe.per_bin_stddev is None
        self.sampling_rate = fe.sampling_rate
        self.frame_length = fe.frame_length  # 320
        self.hop_length = fe.hop_length  # 160
        self.fft_length = fe.fft_length  # 512
        self.mel_floor = float(fe.mel_floor)
        self.register_buffer("window", torch.tensor(fe.window, dtype=torch.float32), persistent=False)
        self.register_buffer("mel_filters", torch.tensor(fe.mel_filters, dtype=torch.float32), persistent=False)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, wav_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """wav: (B, L) float32 in [-1, 1], zero padded. Returns features (B, T, 128) and mask (B, T)."""
        pad_left = self.frame_length // 2
        wav = torch.nn.functional.pad(wav.float(), (pad_left, 0))
        frames = wav.unfold(-1, self.frame_length + 1, self.hop_length)[..., :-1]  # (B, T, 320)
        spec = torch.fft.rfft(frames * self.window, n=self.fft_length, dim=-1).abs()
        feats = torch.log(spec @ self.mel_filters + self.mel_floor)
        num_frames = feats.shape[1]
        frame_end = torch.arange(num_frames, device=wav.device) * self.hop_length + self.frame_length
        mask = frame_end[None, :] < (wav_lens[:, None] + pad_left)
        return feats * mask[..., None], mask
