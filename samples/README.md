# Sample audio

Two short real-world Vietnamese recordings used to exercise the server and the
`scripts/` diagnostics. Both were chosen because they are *not* clean
single-speaker speech: they are the kind of audio a call-centre deployment
actually receives, where most of the file is silence, tone or hold music.

| File | Length | Speech (VAD) | What it is |
| --- | --- | --- | --- |
| `voicemail-90s.mp3` | 90.2 s | ~7 s (8%) | Carrier voicemail greeting, then ringback tone and silence |
| `sales-call-72s.mp3` | 72.7 s | ~60 s (83%) | Two-party outbound sales call over a phone line |

## Why these are useful

`voicemail-90s.mp3` is the motivating case for
[voice activity detection](../README.md#voice-activity-detection): a single
100 ms RMS envelope shows only the first ~35 s carrying any energy, and even
there the longest stretch is ringback tone (a steady ~1 kHz, near-zero spectral
flatness), not speech. Decoded without VAD the whole 90 s go through the model;
with VAD only ~7 s do. The transcript is the greeting and nothing else, because
that is genuinely all the speech in the file — a good file for telling a real
transcription regression apart from a mostly-silent input.

`sales-call-72s.mp3` is the opposite: dense, continuous, two speakers, with the
music/tone removed. It is the case where VAD does not save much and the model's
raw accuracy is what is being measured.

## Running the diagnostics

```bash
# Envelope only — needs ffmpeg, no model, instant.
python scripts/probe_audio.py samples/*.mp3

# Real decode, VAD on vs off, per-segment text and timing.
python scripts/repro_local.py samples/*.mp3
```

Inside the container:

```bash
docker run --rm --env-file .env -v "$PWD":/work -w /work \
  -v quan-hf:/cache/huggingface -e HF_HOME=/cache/huggingface \
  --entrypoint python quan-asr-server:1.0.0 scripts/repro_local.py samples/voicemail-90s.mp3
```

## Note on the content

These are excerpts of telephone audio. `sales-call-72s.mp3` contains a speaker
introducing themselves by name and a business hotline number; treat it as
personal data if you reuse it. They are included here as development fixtures
for a Vietnamese ASR server, not as freely reusable media.
