# QUAN-Transcribe

Inference code for [QUAN-Transcribe-1.0](https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0), a fast Vietnamese
speech recognition model built on Google's [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) audio
encoder with a CTC reader. One forward pass per utterance, no autoregressive decoder.

## Install

```bash
pip install git+https://github.com/qnguyen3/QUAN-Transcribe
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
- **Device:** CUDA if available (bf16), otherwise CPU (fp32). Choose with `device=` / `--device`.
- **Memory:** lower `batch_seconds` / `--batch-seconds` (default 600 s of padded audio per forward pass) if you run
  out of memory.

### Streaming mode

`streaming=True` decodes the way the model runs in streaming mode. The reader's attention covers only the current
0.48 s chunk and earlier ones. The encoder's attention is already chunked with left context only, and the reader's
convolution is causal, so nothing looks more than one chunk ahead. `left_chunks` / `--left-chunks` limits how many
past chunks the reader sees (default: all).

This mode runs over a complete file and returns the transcript a streaming deployment would produce. It does not
include a live microphone API.

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
