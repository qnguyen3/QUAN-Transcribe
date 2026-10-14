# QUAN-Transcribe

Inference code for [QUAN-Transcribe-1.0](https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0), a fast Vietnamese
speech recognition model built on Google's [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) audio
encoder with a CTC reader. One forward pass per utterance, no autoregressive decoder.

## Install

```bash
pip install git+https://github.com/qnguyen3/QUAN-Transcribe
```

For development, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/qnguyen3/QUAN-Transcribe && cd QUAN-Transcribe
uv sync
uv run quan-transcribe --mic
```

Install the PyTorch build that matches your CUDA driver first if needed (e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu126`).

The base encoder `google/embeddinggemma-2` is downloaded on first use. It is gated: accept its terms on the model page
and log in with `hf auth login` (or set `HF_TOKEN`).

## Usage

```bash
quan-transcribe audio.wav
quan-transcribe --streaming audio.wav
quan-transcribe -v *.mp3          # several files, with duration and speed
quan-transcribe --mic             # live from the microphone, Ctrl+C to stop
```

```python
from quan_transcribe import QuanTranscribe

asr = QuanTranscribe.from_pretrained("qnguyen3/QUAN-Transcribe-1.0")
print(asr.transcribe("audio.wav"))
print(asr.transcribe("audio.wav", streaming=True))

# waveforms and batches work too
texts = asr.transcribe([wav_a, wav_b], sample_rate=16000)
```

- **Input:** any file libsndfile can read (wav, flac, ogg, mp3, ...), or a numpy/torch waveform. Audio is converted
  to 16 kHz mono.
- **Output:** lowercase Vietnamese without punctuation, numbers written as words.
- **Long audio:** recordings are cut into pieces of at most 30 s at quiet points (`chunk_seconds`) and decoded in
  batches.
- **Device:** CUDA if available (bf16), then Apple Silicon GPU (`mps`, fp32), otherwise CPU (fp32). Choose with `device=` / `--device`.
- **Memory:** lower `batch_seconds` / `--batch-seconds` (default 600 s of padded audio per forward pass) if you run
  out of memory.

### Streaming mode

`streaming=True` decodes the way the model runs in streaming mode. The reader's attention covers only the current
0.48 s chunk and earlier ones. The encoder's attention is already chunked with left context only, and the reader's
convolution is causal, so nothing looks more than one chunk ahead. `left_chunks` / `--left-chunks` limits how many
past chunks the reader sees (default: all).

This mode runs over a complete file and returns the transcript a streaming deployment would produce.

### Live microphone

`quan-transcribe --mic` transcribes what you say while you say it. The sentence in progress is shown dimmed and
updated every 0.48 s. Streaming mode keeps the text from finished chunks fixed, so only the newest words can change.
After a pause (`--endpoint`, default 0.8 s) the sentence is decoded once more with full context and printed on its own
line. If you talk for longer than `--chunk-seconds` without pausing, the sentence is cut at a quiet point.

```bash
quan-transcribe --list-devices               # pick a microphone
quan-transcribe --mic --input-device 1       # by index or by name
quan-transcribe --mic > notes.txt            # when piped, only finished sentences are written
```

```python
from quan_transcribe.live import LiveSession

session = LiveSession(asr)
for block in blocks:                          # 16 kHz mono float32 audio, any block size
    for sentence in session.accept(block):    # finished sentences
        print(sentence)
    print(session.partial)                    # sentence in progress
print(session.flush())
```

Every update decodes the current sentence again from its start, so it needs a GPU (CUDA or Apple Silicon) to keep up
with real time. On CPU the display lags behind on long sentences.

## Accuracy check

This code reproduces the training-time evaluation of the released checkpoint. On the FLEURS vi_vn validation set
(317 utterances, greedy decoding):

| mode | WER |
|---|---|
| full context | 6.71% |
| streaming | 8.40% |

See the [model card](https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0) for the full benchmark.

## License

The model weights are a derivative of EmbeddingGemma 2 and are subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).
