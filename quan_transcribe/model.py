"""QUAN-Transcribe model: EmbeddingGemma 2 audio encoder + CTC reader (inference only).

    wav --log-mel--> Gemma audio encoder (USM conformer, 25 frames/s, chunked local attention)
        --> softmax-weighted mix of its 13 hidden states
        --> Linear + causal depthwise conv + 6 pre-norm transformer layers (the reader)
        --> head over units in Gemma's text-embedding space
        --> greedy CTC

Streaming mode restricts the reader's self-attention to the current 12-frame (0.48 s) chunk and the chunks before
it. The encoder's own attention is already chunked (12 frames, left context only) and the reader's conv is causal,
so no part of the network looks more than one chunk ahead.
"""

from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .features import LogMel

BLANK = 0


@dataclass
class ModelConfig:
    encoder_id: str = "google/embeddinggemma-2"
    vocab_size: int = 0  # number of CTC units incl. blank and unk
    d_model: int = 768
    n_layers: int = 6
    n_heads: int = 12
    ff_mult: int = 4
    conv_kernel: int = 15
    layer_mix: bool = True  # weighted sum of all encoder hidden states (else: last layer only)
    unfreeze_top_k: int = 0  # top encoder layers that were fine-tuned (their weights are in the checkpoint)
    chunk_frames: int = 12  # streaming chunk size in encoder frames (40 ms each)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class TransformerBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, ff_mult: int):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        # Indices match the training-time Sequential(Linear, GELU, Dropout, Linear) so checkpoints load as-is.
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Identity(), nn.Linear(ff_mult * d, d))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ff(self.norm2(x))


def load_encoder(encoder_id: str) -> nn.Module:
    """The EmbeddingGemma 2 audio tower, without the text and vision parts."""
    # Loaded in bf16 and upcast, exactly as during training (the fine-tuned top layers are overwritten later).
    full = AutoModel.from_pretrained(encoder_id, dtype=torch.bfloat16, vision_config=None)
    encoder = full.audio_tower.float()
    del full
    return encoder


class QuanCTC(nn.Module):
    def __init__(self, cfg: ModelConfig, encoder: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.mel = LogMel(cfg.encoder_id)
        self.encoder = encoder
        n_enc_layers = encoder.config.num_hidden_layers
        enc_dim = encoder.config.hidden_size

        # Same requires_grad flags as in training. They change bf16 autocast numerics in the encoder; keeping them
        # identical reproduces the training-time evaluation results exactly.
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for layer in self.encoder.layers[n_enc_layers - cfg.unfreeze_top_k :]:
            for p in layer.parameters():
                p.requires_grad_(True)

        # Collect the subsampler output + every conformer layer output via hooks. (`output_hidden_states`
        # is not usable: it drops the last layer's raw output in favour of the 1536-d output_proj.)
        self._acts: list[torch.Tensor] = []
        grab = lambda mod, inp, out: self._acts.append(out[0] if isinstance(out, tuple) else out)  # noqa: E731
        self.encoder.subsample_conv_projection.register_forward_hook(grab)
        for layer in self.encoder.layers:
            layer.register_forward_hook(grab)

        d = cfg.d_model
        self.layer_logits = nn.Parameter(torch.zeros(n_enc_layers + 1)) if cfg.layer_mix else None
        self.in_norm = nn.LayerNorm(enc_dim)
        self.in_proj = nn.Linear(enc_dim, d)
        self.pos_conv = nn.Conv1d(d, d, cfg.conv_kernel, groups=d)  # causal: left-padded in forward()
        self.blocks = nn.ModuleList([TransformerBlock(d, cfg.n_heads, cfg.ff_mult) for _ in range(cfg.n_layers)])
        self.out_norm = nn.LayerNorm(d)

        text_dim = 512  # Gemma text-embedding dim; unit rows live in that space
        self.to_text = nn.Linear(d, text_dim)
        self.unit_rows = nn.Parameter(torch.zeros(cfg.vocab_size - 2, text_dim))
        self.special_rows = nn.Parameter(torch.zeros(2, text_dim))  # blank, unk
        self.head_bias = nn.Parameter(torch.zeros(cfg.vocab_size))

    def _encoder_hidden(self, feats: torch.Tensor, feat_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._acts.clear()
        out = self.encoder(feats, feat_mask, return_dict=True)
        acts, self._acts = self._acts, []
        assert len(acts) == self.encoder.config.num_hidden_layers + 1
        if self.cfg.layer_mix:
            w = torch.softmax(self.layer_logits.float(), dim=0)
            h = sum(w[i] * a.float() for i, a in enumerate(acts))
        else:
            h = acts[-1].float()
        return h, out.attention_mask

    def streaming_mask(self, T: int, left_chunks: int | None, device) -> torch.Tensor:
        """(T, T) bool: frame i may attend to frame j if j's chunk is i's chunk or up to `left_chunks` before it."""
        c = torch.arange(T, device=device) // self.cfg.chunk_frames
        allow = c[None, :] <= c[:, None]
        if left_chunks is not None:
            allow &= c[None, :] >= c[:, None] - left_chunks
        return allow

    @torch.no_grad()
    def forward(
        self, wav: torch.Tensor, wav_lens: torch.Tensor, streaming: bool = False, left_chunks: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns CTC logits (B, T, V) in float32 and output lengths (B,)."""
        feats, feat_mask = self.mel(wav, wav_lens)
        h, mask = self._encoder_hidden(feats, feat_mask)

        x = self.in_proj(self.in_norm(h))
        xc = (x * mask[..., None]).transpose(1, 2)
        xc = F.pad(xc, (self.cfg.conv_kernel - 1, 0))
        x = x + F.gelu(self.pos_conv(xc).transpose(1, 2))
        attn_mask = mask[:, None, None, :]  # (B, 1, 1, T) key padding
        if streaming:
            attn_mask = attn_mask & self.streaming_mask(x.shape[1], left_chunks, x.device)[None, None]
        for blk in self.blocks:
            x = blk(x, attn_mask)
        t = self.to_text(self.out_norm(x))
        rows = torch.cat([self.special_rows, self.unit_rows], dim=0)
        logits = t.float() @ rows.float().t() + self.head_bias.float()
        return logits, mask.sum(1)


def greedy_decode(logits: torch.Tensor, lens: torch.Tensor) -> list[list[int]]:
    """Best-path CTC decoding: argmax, collapse repeats, drop blanks."""
    best = logits.argmax(-1).cpu()
    out = []
    for b in range(best.shape[0]):
        seq = best[b, : int(lens[b])]
        keep = torch.ones_like(seq, dtype=torch.bool)
        keep[1:] = seq[1:] != seq[:-1]
        seq = seq[keep]
        out.append(seq[seq != BLANK].tolist())
    return out
