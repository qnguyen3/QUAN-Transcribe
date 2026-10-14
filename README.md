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

### Optional n-gram language model

The default output comes from greedy decoding. For higher accuracy, add a 4-gram KenLM language model with CTC beam
search. The LM (91 MB) ships in the model repo's `lm/` folder and is downloaded the first time you enable it.

```bash
pip install "quan-transcribe[lm] @ git+https://github.com/qnguyen3/QUAN-Transcribe"   # adds kenlm (needs a C++ compiler)
quan-transcribe --lm audio.wav
```

```python
asr = QuanTranscribe.from_pretrained("qnguyen3/QUAN-Transcribe-1.0", lm=True)
asr.transcribe("audio.wav")                  # beam search with the LM
asr.transcribe("audio.wav", use_lm=False)    # greedy
```

- **Tuning:** `beam_width` / `--beam-width` (default 32), `--lm-alpha` (LM weight), `--lm-beta` (word bonus).
- **Your own LM:** `lm="path/to/model.bin"` or `--lm-path`. It must be a word-level KenLM model over text that is
  lowercase, without punctuation, and with numbers written as words.
- **Cost:** beam search runs on the CPU, one utterance at a time. It takes a few tens of milliseconds per utterance,
  which is small next to the model on CPU. On a GPU it becomes the main cost.
- **LM training text:** about 29M words of Vietnamese news and Wikipedia sentences from the
  [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download/Vietnamese) (news 2022, Wikipedia
  2021), plus training transcripts.

## Accuracy check

This code reproduces the training-time evaluation of the released checkpoint (FLEURS vi_vn valid, greedy: 6.71% full
context, 8.40% streaming). With the optional LM (beam 32, alpha 0.3, beta 1.0, tuned on the valid sets):

| set | greedy | + LM |
|---|---|---|
| FLEURS valid (317 utts) | 6.71% | 4.79% |
| FLEURS test (679 utts) | 7.10% | 5.03% |
| Common Voice 17 valid (641 utts) | 6.58% | 5.05% |
| Common Voice 17 test (1,274 utts) | 9.64% | 7.32% |

These subsets exclude utterances whose reference transcript contains digits. The LM adds about 15 ms per utterance
on one CPU core at beam 32 (7 ms at beam 8).

See the [model card](https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0) for the full benchmark.

## License

The model weights are a derivative of EmbeddingGemma 2 and are subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).
