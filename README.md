# QUAN-Transcribe

[![Model](https://img.shields.io/badge/%F0%9F%A4%97-QUAN--Transcribe--1.0-yellow)](https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Model license: Gemma](https://img.shields.io/badge/model%20license-Gemma%20Terms-black)](https://ai.google.dev/gemma/terms)
[![Server license: Apache-2.0](https://img.shields.io/badge/server%20license-Apache--2.0-blue)](https://www.apache.org/licenses/LICENSE-2.0)

Inference code **and** an OpenAI-compatible serving layer for
[QUAN-Transcribe-1.0](https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0), a fast Vietnamese speech recognition model
built on Google's [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) audio encoder with a CTC reader.
One forward pass per utterance, no autoregressive decoder.

This repository holds two parts that install together:

- **Library** (`quan_transcribe/`) — the inference package and the `quan-transcribe` CLI: offline, streaming and
  live-microphone decoding, plus an optional KenLM n-gram LM. Everything below up to *Server*.
- **Server** (`app/`) — `quan-asr-server`, an OpenAI-compatible HTTP / SSE / WebSocket API around the library, with a
  scheduler that keeps live audio responsive under load, a browser playground, and a container for GPUStack.

**Contents**

*Library*

- [Install](#install)
  - [From source (development)](#from-source-development)
  - [Model access](#model-access)
- [Quickstart](#quickstart)
- [Usage](#usage)
  - [Streaming mode](#streaming-mode)
  - [Live microphone](#live-microphone)
  - [Optional n-gram language model](#optional-n-gram-language-model)
- [Python API](#python-api)
- [Accuracy check](#accuracy-check)
- [Troubleshooting](#troubleshooting)
- [License](#license)

*Server* — [`quan-asr-server`](#quan-asr-server)

- [Architecture](#architecture)
- [Quick start](#quick-start)
- [API](#api)
- [Errors and limits](#errors-and-limits)
- [Observability](#observability)
- [Streaming](#streaming)
- [Optional n-gram language model (serving)](#optional-n-gram-language-model-serving)
- [Voice activity detection](#voice-activity-detection)
- [Configuration](#configuration)
- [Performance](#performance)
- [Docker](#docker)
- [Development](#development)
- [Licence](#licence)

## Install

Requires **Python 3.10+** and **PyTorch 2.6+**. Microphone mode also needs a working
[sounddevice](https://python-sounddevice.readthedocs.io/) / PortAudio backend.

```bash
pip install git+https://github.com/qnguyen3/QUAN-Transcribe
```

Install the PyTorch build that matches your CUDA driver first if needed (e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu126`).

### From source (development)

With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/qnguyen3/QUAN-Transcribe && cd QUAN-Transcribe
uv sync
uv run quan-transcribe --mic
```

### Model access

The base encoder `google/embeddinggemma-2` is downloaded on first use. It is gated: accept its terms on the model page
and log in with `hf auth login` (or set `HF_TOKEN`).

## Quickstart

```bash
quan-transcribe audio.wav
```

```python
from quan_transcribe import QuanTranscribe

asr = QuanTranscribe.from_pretrained("qnguyen3/QUAN-Transcribe-1.0")
print(asr.transcribe("audio.wav"))
```

## Usage

```bash
quan-transcribe audio.wav
quan-transcribe --streaming audio.wav
quan-transcribe -v *.mp3          # several files, with duration and speed
quan-transcribe --mic             # live from the microphone, Ctrl+C to stop
```

```python
# continuing from Quickstart, with the same `asr`
print(asr.transcribe("audio.wav", streaming=True))     # streaming decoding
print(asr.transcribe("audio.wav", use_lm=True))        # beam search (after loading an LM)

# waveforms and batches work too
texts = asr.transcribe([wav_a, wav_b], sample_rate=16000)   # -> list[str]
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
line. If you talk for longer than `--chunk-seconds` without pausing, the sentence is cut at a quiet point. With `--lm`
(see below), the final line is decoded with the n-gram LM.

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

## Python API

```python
from quan_transcribe import QuanTranscribe

asr = QuanTranscribe.from_pretrained(
    "qnguyen3/QUAN-Transcribe-1.0",  # HF repo id or a local directory
    device=None,                     # "cuda", "cuda:1", "mps", "cpu" (default: auto)
    revision=None,                   # HF revision / commit
    token=None,                      # HF token for gated downloads
    lm=False,                        # True loads the shipped LM; a path loads your own
)

asr.transcribe("audio.wav")                        # -> str
asr.transcribe(["a.wav", "b.wav"])                 # -> list[str]
asr.transcribe(wav, sample_rate=16000)             # numpy / torch waveform
asr.transcribe("audio.wav", streaming=True)        # streaming decoding
asr.transcribe("audio.wav", beam_width=8)          # smaller LM beam
```

`transcribe` arguments:

| argument | default | meaning |
|---|---|---|
| `audio` | — | file path, waveform, or a list of either (a list returns a list) |
| `sample_rate` | 16000 | sample rate of a waveform input |
| `streaming` | `False` | decode as a streaming deployment would (0.48 s chunks, no lookahead) |
| `left_chunks` | `None` | streaming: past chunks the reader may attend to (`None` = all) |
| `chunk_seconds` | 30.0 | long audio is cut into pieces of at most this length, at quiet points |
| `batch_seconds` | 600.0 | padded audio per forward pass; lower it if you run out of memory |
| `use_lm` | `None` | force beam search on (`True`) or greedy (`False`) regardless of the loaded LM |
| `beam_width` | 32 | beams kept by the LM beam search |

| helper | purpose |
|---|---|
| `QuanTranscribe.from_pretrained(...)` | build a model from an HF repo id or a local directory |
| `asr.load_lm(path, unigrams=None, alpha=None, beta=None)` | attach a KenLM model after construction |
| `asr.decode_units(unit_ids)` | map CTC unit ids to text |
| `quan_transcribe.live.LiveSession` | incremental decoding for a stream of audio blocks |
| `quan_transcribe.live.transcribe_microphone(asr, ...)` | the `--mic` loop as a function |

CLI flags: `--model`, `--device`, `--streaming`, `--left-chunks`, `--chunk-seconds`, `--batch-seconds`, `--lm`,
`--lm-path`, `--beam-width`, `--lm-alpha`, `--lm-beta`, `--mic`, `--input-device`, `--list-devices`, `--endpoint`,
`-v/--verbose`. Run `quan-transcribe --help` for the full list.

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

## Troubleshooting

| symptom | fix |
|---|---|
| `401` / `403` or "gated repo" while loading | accept the terms on the [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) page, then `hf auth login` or set `HF_TOKEN` |
| CUDA out of memory | lower `--batch-seconds` (e.g. `60`); long files are batched together by length |
| `no LM loaded` from `transcribe(use_lm=True)` | pass `lm=True` to `from_pretrained` or call `load_lm(path)` first |
| `ModuleNotFoundError: kenlm` with `--lm` | `pip install "quan-transcribe[lm]"` (KenLM builds from source, needs a C++ compiler) |
| no microphone / `--list-devices` prints nothing | install a PortAudio backend (macOS: `brew install portaudio`, Linux: `apt install libportaudio2`) and reinstall `sounddevice` |
| mp3/flac fails to load | `soundfile` needs libsndfile >= 1.1; upgrade it (`pip install -U soundfile`) |
| empty output on speech | the model expects 16 kHz mono; pass the correct `sample_rate` for raw waveforms |

## License

The model weights are a derivative of EmbeddingGemma 2 and are subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).

---

# QUAN-ASR Server

An OpenAI-compatible speech-to-text server for
[`qnguyen3/QUAN-Transcribe-1.0`](https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0),
packaged as a container that runs on [GPUStack](https://gpustack.ai) as a custom
backend.

Point any existing OpenAI client at this server and it keeps working. On top of
the compatible endpoint it adds real timestamps, Server-Sent Events streaming, a
WebSocket for live microphone audio, optional KenLM beam-search decoding, a
scheduler that keeps live audio responsive under load, and a small browser
playground.

The model behind it is a **356M-parameter, Vietnamese-only, single-pass CTC
network** built on Google's EmbeddingGemma 2 audio encoder. There is no
autoregressive decoder and no KV cache: one forward pass per piece of audio. That
makes it roughly **3,700× faster than realtime on a single H100** (upstream
benchmark) and makes its output lowercase Vietnamese without punctuation, with
numbers written as words.

### What this repository is

- A **serving layer**, not a training repo. It wraps the upstream
  [`quan_transcribe`](https://github.com/qnguyen3/QUAN-Transcribe) inference
  library and adds the HTTP/streaming/scheduling surface around it.
- **Production-shaped**: one model-owning scheduler thread, real backpressure
  (`503` instead of unbounded latency), lazy imports so the process starts
  without a GPU, `/health` and `/metrics`, structured OpenAI-style errors, and a
  container that is ready for GPUStack.
- **Single-model and Vietnamese-only.** It is not a multi-tenant, multi-model
  router, and it will not transcribe other languages correctly.

## Architecture

```
                     ┌──────────────── app/server.py (FastAPI) ───────────────┐
  HTTP  ────────────►│ POST /v1/audio/transcriptions   ── buffered or SSE    │
  (multipart)        │ GET  /health  /metrics  /v1/models  /  /docs          │
                     └───────────────┬───────────────────────┬───────────────┘
  WS ───────────────► /v1/audio/transcriptions/stream         │
  (16 kHz PCM)                  │                             │
                                ▼                             ▼
                     ┌── app/engine.py ─────────┐   ┌── app/audio.py ────────┐
                     │ TranscriptionEngine      │   │ decode → 16 kHz mono   │
                     │  · lazy torch / model    │   │ quiet-point splitting  │
                     │  · LiveStream sessions   │   └────────────────────────┘
                     └───────────┬──────────────┘
                                 │  submit() / submit_bulk()
                                 ▼
                     ┌── app/core.py ───────────────────────────────────────┐
                     │ EngineCore — ONE thread that owns the model          │
                     │  · interactive (live/SSE) preempts bulk (files)      │
                     │  · coalesces pieces into one padded forward pass     │
                     │  · bounded queues → EngineBusy → HTTP 503            │
                     └──────────────────┬───────────────────────────────────┘
                                        ▼
                          quan_transcribe (upstream library + torch)
```

Three invariants make the rest of the document make sense:

1. **The model is not re-entrant**, so exactly one thread may touch it. That
   thread is `EngineCore`; every model call goes through it, from any request
   thread. There is no global lock held across a request, which is why a live
   step does not queue behind a ten-minute file. See [Performance](#performance).
2. **Importing `app` never imports torch.** `torch` and `quan_transcribe` are
   imported lazily inside `TranscriptionEngine._load()`, so the process starts
   instantly, `/health` can report `loading`, and the test suite runs with no
   GPU and no model download.
3. **Streaming is honest by construction.** The encoder attention is chunked
   with left context only and the reader's convolution is causal, so nothing
   ever looks more than one chunk ahead — the partial text is what the model
   would have produced if the audio stopped there.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install .              # installs the library + the server from this repo
quan-asr-server --serve --port 8000
```

Then, from another shell:

```bash
curl -s -F "file=@sample.wav" http://127.0.0.1:8000/v1/audio/transcriptions
# {"text": "xin chào tôi tên là nam"}
```

The `samples/` directory holds two real recordings to try it on, with a short
note on what each one contains and what VAD does to it:

```bash
curl -s -F "file=@samples/voicemail-90s.mp3" http://127.0.0.1:8000/v1/audio/transcriptions
```

Open <http://127.0.0.1:8000/> for the playground: upload a file to watch the
streamed transcript, or hold the record button to transcribe from the mic.

### Requirements

- **Python 3.10+.** The container ships 3.11; this repository is developed and
  tested on 3.11 and 3.14.
- **PyTorch matching your CUDA driver.** The Dockerfile installs the cu126
  build; for a manual install pick the right index, e.g.
  `pip install torch --index-url https://download.pytorch.org/whl/cu124`.
  Apple Silicon (`mps`) and CPU work too — no CUDA required.
- **An `HF_TOKEN` with access to [`google/embeddinggemma-2`].** The audio
  encoder is a gated Gemma model, so you must accept its terms on Hugging Face
  and then `hf auth login` (or export `HF_TOKEN`) before the first run.
- **ffmpeg** if you want to transcribe containers libsndfile cannot read
  (m4a, webm/opus). WAV, FLAC, OGG and MP3 work without it.

[`google/embeddinggemma-2`]: https://huggingface.co/google/embeddinggemma-2

## API

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `POST` | `/v1/audio/transcriptions` | key | Transcribe an uploaded file (OpenAI-compatible); buffered or SSE |
| `WS` | `/v1/audio/transcriptions/stream` | key | Live transcription of 16 kHz PCM |
| `GET` | `/v1/models` | – | List the served model |
| `GET` | `/v1/models/{id}` | – | Fetch one model |
| `GET` | `/health` | – | Liveness plus real model state |
| `GET` | `/metrics` | – | Request counters and engine info |
| `GET` | `/` | – | Browser playground |
| `GET` | `/docs` | – | Swagger UI (FastAPI default) |
| `GET` | `/openapi.json` | – | OpenAPI schema (FastAPI default) |

"key" means the route requires `Authorization: Bearer <API_KEY>` **only when
`API_KEY` is set**; with no `API_KEY` every route is open. The read-only routes
above are always open, which is deliberate: an orchestrator must be able to probe
`/health` without credentials.

### `POST /v1/audio/transcriptions`

`multipart/form-data`. The endpoint accepts the documented OpenAI parameters:

| Parameter | Default | Notes |
| --- | --- | --- |
| `file` | *required* | Audio to transcribe. Empty file → `invalid_audio_file`; undecodable → same |
| `model` | served name | Validated; see [model aliases](#model-aliases) |
| `language` | `vi` | `vi`, `vietnamese`, `vi-VN`, `VI`… all normalise to `vi`. `auto`/`none` are accepted and mean "no language" |
| `response_format` | `json` | One of `json`, `text`, `srt`, `verbose_json`, `vtt` |
| `prompt` | – | Accepted for compatibility; a CTC model has no prompt channel |
| `temperature` | `0.0` | Accepted and validated (`0.0`–`1.0`); decoding is deterministic, so it has no effect |

Extensions this server adds:

| Parameter | Default | Bounds | Notes |
| --- | --- | --- | --- |
| `stream` | `false` | bool | Return Server-Sent Events instead of a buffered body |
| `stream_partials` | `true` | bool | Emit in-progress `transcript.text.delta` events while streaming |
| `punctuate` | `false` | bool | Intelligent punctuation, capitalization & Inverse Text Normalization (ITN) |
| `phonemes` | `false` | bool | Grapheme-to-Phoneme IPA output powered by [`sea-g2p`](https://github.com/pnnbao97/sea-g2p) |
| `norm_mode` | `itn` | `itn`/`tn`/`none` | Text norm mode: `itn` (spoken to digits/symbols), `tn` (`sea-g2p` spoken words), or `none` |
| `use_lm` | server setting | bool | Force beam search on/off for this request |
| `beam_width` | `32` | `1`–`256` | Beams kept by the beam search |
| `left_chunks` | server setting | `>= 0` | Past chunks the reader may attend to; `0` = lowest latency |
| `chunk_seconds` | `30` | `1`–`300` | Cut long audio into pieces of at most this many seconds |
| `timestamp_granularities[]` | – | `word`/`segment` | Only valid with `verbose_json`, `srt` or `vtt` |

Out-of-range values and unknown `response_format`/`language` values are rejected
with `400` — see [Errors and limits](#errors-and-limits). **Unknown form fields
are ignored**, matching OpenAI's forward-compatible behaviour, so a client that
sends a parameter this server does not know about still works.

Every response format maps to a media type:

| `response_format` | `Content-Type` |
| --- | --- |
| `json`, `verbose_json` | `application/json` |
| `text` | `text/plain; charset=utf-8` |
| `srt` | `application/x-subrip; charset=utf-8` |
| `vtt` | `text/vtt; charset=utf-8` |

```bash
# Plain text
curl -s -F "file=@sample.wav" -F "response_format=text" \
  http://127.0.0.1:8000/v1/audio/transcriptions
# xin chào tôi tên là nam

# Subtitles
curl -s -F "file=@sample.wav" -F "response_format=srt" \
  http://127.0.0.1:8000/v1/audio/transcriptions
# 1
# 00:00:00,000 --> 00:00:02,480
# xin chào tôi tên là nam

# Verbose JSON (segments + metadata)
curl -s -F "file=@sample.wav" -F "response_format=verbose_json" \
  http://127.0.0.1:8000/v1/audio/transcriptions
# {"task":"transcribe","language":"vi","duration":2.48,
#  "text":"xin chào tôi tên là nam",
#  "segments":[{"id":0,"seek":0,"start":0.0,"end":2.48,"text":" xin chào tôi tên là nam", ...}],
#  "words":[],"model":"QUAN-Transcribe-1.0"}

# Optional accuracy boost for this one request
curl -s -F "file=@sample.wav" -F "use_lm=true" -F "beam_width=32" \
  http://127.0.0.1:8000/v1/audio/transcriptions
```

`verbose_json` returns OpenAI-shaped segments whose `start`/`end` come from the
quiet points where the audio was actually cut, so the timestamps line up with
what the model decoded. Segments split mid-sentence are merged back together when
there is no terminal punctuation and the gap is under 0.5 s. Per-sentence
timestamps are the finest granularity available: `words` is always `[]` and the
`tokens` arrays are placeholders, because a single CTC pass does not produce
per-token alignment. `timestamp_granularities[]` is accepted for compatibility
and only affects whether that merge is suppressed.

#### Model aliases

`_check_model` accepts anything that identifies the served model, so clients do
not need to know the deployment's exact name:

- the configured `SERVED_MODEL_NAME` (default `QUAN-Transcribe-1.0`),
- the `MODEL_PATH` value or the basename of the model directory,
- and the aliases `whisper-1`, `quan-transcribe`, `quan-transcribe-1.0`,
  `quan-asr-server`.

Anything else is `404 model_not_found`. The alias list is what makes this a
drop-in replacement for code that hard-codes `model="whisper-1"`.

## Errors and limits

Every error from the HTTP endpoints uses OpenAI's envelope:

```json
{"error": {"message": "the uploaded file is empty",
           "type": "invalid_request_error",
           "param": null, "code": "invalid_audio_file"}}
```

The full set of `code` values, with the HTTP status and when they fire:

| Status | `code` | `type` | When |
| --- | --- | --- | --- |
| `400` | `invalid_value` | `invalid_request_error` | A numeric form field is out of range (`chunk_seconds`, `beam_width`, `temperature`, `left_chunks`) |
| `400` | `unsupported_language` | `invalid_request_error` | `language` is neither Vietnamese nor `auto`/`none` |
| `400` | `invalid_audio_file` | `invalid_request_error` | Empty upload, or bytes that libsndfile and ffmpeg both refuse |
| `400` | `invalid_request_body` | `invalid_request_error` | Malformed multipart body / missing `file` field |
| `401` | `invalid_api_key` | `authentication_error` | `API_KEY` set and the token is missing or wrong |
| `404` | `model_not_found` | `invalid_request_error` | `model` is not the served model or one of its aliases |
| `404` | `not_found` | `invalid_request_error` | Unknown route |
| `413` | `payload_too_large` | `invalid_request_error` | Upload larger than `MAX_UPLOAD_MB` (default `200`) |
| `503` | `server_busy` | `server_error` | The decode queue is full — retry after the backlog clears |
| `503` | – | `server_error` | The model is still loading, or a requested LM is not loaded |

Two caveats worth knowing:

- **`server_busy` is retryable and intentional.** It is the scheduler refusing
  work rather than parking another thread and letting latency grow without
  bound — see [Performance](#performance). Back off and retry.
- **A mid-stream failure is not an HTTP error.** Once an SSE response has
  started, the status is already `200`; a decode failure arrives as an `error`
  event in the body (see below). If you need a trustworthy status code, use the
  buffered (non-`stream`) mode.

Limits that are enforced, in one place: `MAX_UPLOAD_MB` (413), `MAX_STREAM_SECONDS`
for WebSocket sessions, `chunk_seconds` ∈ [1, 300], `beam_width` ∈ [1, 256],
`temperature` ∈ [0, 1], `left_chunks` ≥ 0, and the two queue caps `MAX_QUEUE` /
`MAX_INTERACTIVE` (503).

## Observability

### `GET /health`

Always `200` while the model is loading, and only `503` after a **hard** load
failure. That is deliberate: an orchestrator probe must not restart a container
that is merely downloading 356M parameters for the first time.

```json
{"status": "ok", "model": "QUAN-Transcribe-1.0", "device": "cuda:0",
 "language": "vi", "loaded": true, "error": null}
```

`status` is `loading` before the model is up, `ok` once it is, and `error` if it
failed — with the exception text in `error`. `device` reports the resolved device
(`cuda:0`, `mps`, `cpu`) once loaded, and the configured value or `unknown`
before that.

### `GET /metrics`

`{**engine.metrics(), **engine.info()}` — request counters, resolved
configuration, and the scheduler's own view:

| Group | Keys |
| --- | --- |
| Counters | `requests_total`, `audio_seconds_total`, `decode_seconds_total`, `realtime_factor` |
| Model | `model`, `model_path`, `loaded`, `device`, `profile`, `language`, `lm`, `lm_path`, `lm_alpha`, `lm_beta`, `beam_width`, `load_error`, `load_seconds`, `loaded_at` |
| Live streaming | `chunk_seconds`, `stream_step_seconds`, `left_chunks` |
| Scheduler config | `microbatch_seconds`, `microbatch_max_requests`, `microbatch_max_wait_ms`, `max_queue`, `max_interactive`, `subbatch_seconds`, `tf32` |
| `core` | `batches_run`, `pieces_run`, `jobs_run`, `queued_pieces`, `queued_interactive`, `shed`, `avg_batch_size`, `p50_latency_ms`, `p95_latency_ms`, `model_seconds` |

`realtime_factor` is `decode_seconds_total / audio_seconds_total` (lower is
better; `0.10` means ten times faster than realtime). The `core` block is the
direct answer to "is coalescing working and am I shedding load?":

- `batches_run` well below `jobs_run` means concurrent requests are sharing
  forward passes.
- `shed > 0` means clients are being refused with `503` — raise `MAX_QUEUE` or
  add capacity.
- `queued_pieces` / `queued_interactive` are the current backlogs.
- `p50_latency_ms` / `p95_latency_ms` are over a rolling window of the last 4096
  jobs (null until there is a sample).

## Streaming

### Server-Sent Events

Set `stream=true` and the response is `text/event-stream` with
`Cache-Control: no-cache, no-transform` and `X-Accel-Buffering: no` (so nginx
does not buffer it). Each frame is a single `data:` line of JSON followed by a
blank line:

```
data: {"type": "transcript.text.delta", "text": "xin", "start": 0.0, "end": 0.48}
data: {"type": "transcript.text.segment", "text": "xin chào", "start": 0.0, "end": 1.0}
data: {"type": "transcript.text.completed", "text": "", "start": 0.0, "end": 2.0}
data: {"type": "transcript.text.done", "text": "xin chào tôi tên là nam", "duration": 2.0, "model": "QUAN-Transcribe-1.0", "language": "vi", "elapsed": 2.31}
```

| Event `type` | Keys | Meaning |
| --- | --- | --- |
| `transcript.text.delta` | `type, text, start, end` | Unstable text, expected to change (`stream_partials=false` removes these) |
| `transcript.text.segment` | `type, text, start, end` | A finalized segment with its timestamps |
| `transcript.text.completed` | `type, text, start, end` | End-of-audio marker (`text` is always `""`) |
| `transcript.text.done` | `type, text, duration, model, language, elapsed` | Always last. `duration` is the file length, `elapsed` the wall time of the whole stream |
| `error` | `type, error.message` | Decode failed mid-stream |

If the client disconnects, the server sets a stop flag and stops decoding at the
next event boundary instead of finishing a file nobody is reading. `stop` is
checked *between* steps, so an in-flight forward pass still completes.

### WebSocket

`/v1/audio/transcriptions/stream` is for live microphone audio. All parameters
are **query-string** parameters — there is no handshake frame:

| Query param | Values | Default |
| --- | --- | --- |
| `format` | `f32`, `pcm16` | `f32` |
| `left_chunks` | integer ≥ 0 | server default |
| `use_lm` | truthy string | server default |
| `beam_width` | integer ≥ 1 | server `BEAM_WIDTH` |
| `api_key` | the API key | – (alternative to the `Authorization: Bearer` header) |

On connect the server immediately sends one frame describing the session, then
you send **binary** frames of raw mono audio:

```json
{"type": "session.created", "model": "QUAN-Transcribe-1.0", "language": "vi",
 "format": "f32", "sample_rate": 16000, "step_seconds": 0.48,
 "left_chunks": null, "use_lm": false, "beam_width": 32}
```

- **`sample_rate` is 16000 and the server does not resample this path.** You
  must send 16 kHz mono audio. This is the one thing to get right: 8 kHz or
  44.1 kHz input is decoded as if it were 16 kHz and produces garbage. (The
  browser playground resamples client-side with linear interpolation before it
  sends, for exactly this reason.)
- `format=f32` means **little-endian float32** in `[-1, 1]`. `format=pcm16`
  means **little-endian signed int16**, scaled by `/32768`. A trailing partial
  sample on the `f32` path is tolerated and dropped; keep `pcm16` frames an even
  number of bytes.
- Audio is buffered until at least one step (`0.48 s`) has arrived, then events
  start. Send `step_seconds` worth of audio per frame to keep latency low.

Text frames on the socket are **control messages** and must be the bare
lowercase strings, not JSON:

| You send | Server replies |
| --- | --- |
| `stop`, `flush`, `eof` | `{"type": "transcript.text.done", "text": "…", "final": true, "end": 2.4}` (omitted if nothing was decoded), then `session.closed` |
| `ping` | `{"type": "pong"}` |
| anything else | `error` with `unknown control message: '…'`; the session stays open |

`stop`, `flush` and `eof` are equivalent: all three flush the current utterance
and end the session. The server always finishes with
`{"type": "session.closed"}`.

Failure modes are signalled by a close code, after an `error` frame:

| Code | Reason |
| --- | --- |
| `4401` | `API_KEY` set and the token (header or `?api_key=`) is wrong |
| `4400` | Bad `format`, or `left_chunks`/`beam_width` is not an integer |
| `1011` | `use_lm` was requested but no KenLM model is loaded |
| `1000` | Normal close |

When the scheduler is saturated, a binary frame can come back as an `error` with
`code: "server_busy"`. The audio is **not lost** — the server re-buffers it — so
slow down and the next frame retries it. A session longer than
`MAX_STREAM_SECONDS` (default one hour) gets an `error` frame and closes
gracefully.

```python
# Minimal client: 16 kHz mono float32, 0.48 s per frame.
import asyncio
import websockets
import soundfile as sf

URL = "ws://127.0.0.1:8000/v1/audio/transcriptions/stream?format=f32"
STEP = int(0.48 * 16000)

async def main(path="mic16k.wav"):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == 16000, f"resample first: got {sr} Hz"
    samples = audio.mean(1)  # mono

    async with websockets.connect(URL) as ws:
        print(await ws.recv())  # session.created
        for i in range(0, len(samples), STEP):
            block = samples[i : i + STEP].astype("<f4")
            await ws.send(block.tobytes())
        await ws.send("stop")  # flush + end the session

        while True:
            message = await ws.recv()
            print(message)
            if '"session.closed"' in message:
                break

asyncio.run(main())
```

`scripts/ws_probe.py` is a fuller version of this that prints every event and
can point at any running server. It needs the `websockets` package, which
`uvicorn[standard]` already pulls in.

### How streaming stays honest

The encoder attention is chunked with left context only, the reader's
convolution is causal, and the reader attends to the current 0.48 s chunk plus
earlier ones. Nothing ever looks more than one chunk ahead, so the partial
output is the same text the model would produce if the audio simply stopped
there. `left_chunks` caps how far back the reader looks: `None` (the default)
uses all history for best accuracy, `0` gives the lowest latency.

Live utterances are also segmented: `ENDPOINT_SECONDS` of silence (`0.8`)
finalises a sentence, and one is force-cut at a quiet point after
`MAX_UTTERANCE_SECONDS` (`30`).

## Optional n-gram language model (serving)

By default the server decodes greedily. Turning on the language model swaps in
CTC prefix beam search against a 4-gram KenLM, which improves accuracy on
Vietnamese text:

| | WER (FLEURS valid) |
| --- | --- |
| Greedy | 6.71% |
| + KenLM, beam 32 | **4.79%** |

```bash
# Bundled LM: the model repo's lm/ folder (vi_4gram.bin, ~91 MB) downloads on
# first use.
USE_LM=true quan-asr-server --serve

# Your own KenLM model: a .bin/.arpa file, or a directory containing
# vi_4gram.bin and optionally unigrams.txt / lm_config.json.
LM_PATH=/models/vi_4gram.bin quan-asr-server --serve

# Tune the search.
USE_LM=true BEAM_WIDTH=16 LM_ALPHA=0.5 LM_BETA=1.5 quan-asr-server --serve
```

Or with flags: `--lm`, `--lm-path`, `--beam-width`, `--lm-alpha`, `--lm-beta`.

- **`LM_ALPHA`** (LM weight, default **0.3**) and **`LM_BETA`** (word-insertion
  bonus, default **1.0**) are the values the model was tuned with. Leaving them
  unset keeps whatever the LM's own `lm_config.json` specifies, which is the
  right choice unless you are tuning for your own domain.
- **`BEAM_WIDTH`** defaults to 32. Lower it to trade a little accuracy for
  latency.
- The LM expects **lowercase Vietnamese with numbers spelled out**, matching the
  model's own output format. Feed it text in another casing or script and the
  scores will be meaningless.
- **Beam search runs on the CPU**, one utterance at a time, and becomes the
  bottleneck on a fast GPU — roughly 15 ms per utterance at beam 32 on a single
  core. If you need maximum throughput, leave the LM off; if you need maximum
  accuracy, turn it on and give the worker spare CPU.
- The bundled LM was trained on about **29M words** of Vietnamese news
  (Leipzig corpora) and Wikipedia.
- A request can override the server default with `use_lm=false` (force greedy)
  or `use_lm=true`. Asking for an LM that is not loaded is refused, not silently
  downgraded: a buffered request gets `503`, and a streaming one gets an `error`
  event (the SSE status is already `200` by then).

Installing it needs a C++ compiler because `kenlm` has no prebuilt wheel:

```bash
pip install ".[lm]"
```

In Docker this is gated by `--build-arg INSTALL_LM=false` if you would rather
skip the compile entirely and ship a greedy-only image.

## Voice activity detection

Call and voicemail audio is mostly *not* speech. A measured 90 s voicemail
greeting in `samples/` is 78 s of digital silence and ringback tone and 7 s of
speech; a 72 s sales call is 59% speech. The model already returns nothing for
the silent parts, but it still pays for them — a forward pass over 83 s of
silence costs the same as 83 s of speech.

`USE_VAD=true` runs [Silero VAD](https://github.com/snakers4/silero-vad) over
the waveform first, then decodes only what it found:

- **Silence is never decoded.** Only the speech regions are cut into pieces.
  On the 90 s voicemail that means decoding ~7 s instead of 90 s.
- **Pieces are cut on real pauses,** not on the quietest 50 ms window, so a cut
  no longer lands in the middle of a word.
- **`VAD_TRIM_SILENCE=false` keeps the silence** and uses VAD only for cut
  points. The transcript is then byte-for-byte what a VAD-less run produces,
  while the decoder still gets clean boundaries.

What it costs is small: Silero ONNX runs at about **0.008 RTF** on one CPU core
(~0.7 s for the 90 s file), and it runs on the request thread, never the model
thread, so it cannot stall a decode. The ONNX session is ~2.5× faster than the
TorchScript one and is preferred automatically; the package also calls
`torch.set_num_threads(1)` at import, which this server captures and restores so
it cannot silently cap CPU decoding for the whole process.

**Tuning.** `VAD_SPEECH_PAD_MS` defaults to `250`, deliberately higher than
Silero's own `30`: this encoder needs the lead-in, and 30 ms clips the first
syllable of a region (`quý khách` degrades to `cách`). The other knobs are
`VAD_THRESHOLD` (0.5) and `VAD_MIN_SILENCE_MS` (300).

It is off by default. Turn it on for call-centre or voicemail workloads; leave
it off if your audio is already clean, clipped speech, where the extra pass buys
nothing.

```bash
pip install ".[vad]"     # silero-vad + onnxruntime; the image already ships it
```

If the package is missing while `USE_VAD=true`, the server logs a warning and
falls back to energy-based splitting for that request rather than failing it.

## Configuration

Every setting is an environment variable, so the same image works for
GPUStack, `docker run`, and the CLI. A CLI flag beats the environment, because
the CLI writes each non-default flag into the environment before the settings
are read.

### Model

| Variable | Default | Notes |
| --- | --- | --- |
| `MODEL_PATH` | `qnguyen3/QUAN-Transcribe-1.0` | HF repo id or local directory. Legacy alias `QUAN_MODEL` |
| `SERVED_MODEL_NAME` | `QUAN-Transcribe-1.0` | Name clients pass as `model`. Legacy alias `MODEL_NAME` |
| `DEVICE` | auto | `cuda`, `cuda:1`, `mps`, `cpu` |
| `HF_TOKEN` | – | For the gated audio encoder |
| `PRELOAD` | `true` | Load at startup rather than on the first request |

### Decoding

| Variable | Default | Notes |
| --- | --- | --- |
| `USE_LM` | `false` | Beam search with the bundled KenLM |
| `LM_PATH` | – | Your own KenLM model; implies `USE_LM` |
| `LM_ALPHA` | `lm_config.json` | LM weight (shipped value 0.3) |
| `LM_BETA` | `lm_config.json` | Word-insertion bonus (shipped value 1.0) |
| `BEAM_WIDTH` | `32` | Beams kept |
| `LEFT_CHUNKS` | all | Streaming lookback; `0` = lowest latency |
| `CHUNK_SECONDS` | `30.0` | Max seconds per decoded piece |
| `BATCH_SECONDS` | `600.0` | Padding budget per forward pass; lower it if VRAM is tight |
| `SUBBATCH_SECONDS` | `0` (off) | Hard cap on one forward pass, in audio seconds. Bounds how long a live step can wait behind a file; see [Performance](#performance). CPU profiles ship `30` |

### Voice activity detection

Requires the `vad` extra. See [Voice activity detection](#voice-activity-detection).

| Variable | Default | Notes |
| --- | --- | --- |
| `USE_VAD` | `false` | Silero VAD: skip silence, cut pieces on real pauses |
| `VAD_TRIM_SILENCE` | `true` | `false` keeps the silence and uses VAD only for cut points |
| `VAD_SPEECH_PAD_MS` | `250` | Padding around a speech region; Silero's `30` clips the first syllable |
| `VAD_THRESHOLD` | `0.5` | Speech probability that marks a 32 ms window as speech |
| `VAD_MIN_SPEECH_MS` | `250` | Shortest run of speech worth keeping |
| `VAD_MIN_SILENCE_MS` | `300` | Silence that ends a speech region |

### Throughput

These trade latency for throughput and are the knobs that matter when serving
more than one client at a time. See [Performance](#performance).

| Variable | Default | Notes |
| --- | --- | --- |
| `MICROBATCH_SECONDS` | `0.15` | Coalescing window: how long pieces wait for companions before a forward pass; `0` batches only what is already queued |
| `MICROBATCH_MAX_WAIT_MS` | `20.0` | Hard cap on that wait, in ms |
| `MICROBATCH_MAX_REQUESTS` | `8` | Cap on requests packed into one batch |
| `MAX_QUEUE` | `256` | Backpressure: most audio pieces that may wait to decode; a full queue returns `503`. `0` is unbounded |
| `MAX_INTERACTIVE` | `64` | Backpressure: most streaming jobs in flight; `0` is unbounded |
| `TF32` | `true` | Allow TF32 matmuls on Ampere+ (CUDA only); `false` for exact FP32 |
| `CPU_THREADS` | auto | torch thread count for the CPU profile; set it when the container has a cgroup quota |

### Streaming

| Variable | Default | Notes |
| --- | --- | --- |
| `ENDPOINT_SECONDS` | `0.8` | Silence that ends an utterance |
| `MAX_UTTERANCE_SECONDS` | `30.0` | Hard cap per utterance |
| `STREAM_STEP_SECONDS` | `0.48` | One encoder chunk; leave it alone |
| `MAX_STREAM_SECONDS` | `3600.0` | Reject longer streams |

### HTTP

| Variable | Default | Notes |
| --- | --- | --- |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | |
| `API_KEY` | – | When set, `POST /v1/audio/transcriptions` and the WebSocket need `Authorization: Bearer <key>` (REST also accepts `x-api-key`; the socket also accepts `?api_key=`) |
| `CORS_ORIGINS` | `*` | Comma-separated |
| `MAX_UPLOAD_MB` | `200` | Rejects larger uploads with `413` |
| `LOG_LEVEL` | `info` | |

`/health` reports `loading` while the weights download and only fails after a
hard error, so the GPUStack probe does not flap during a multi-minute first
start. See [Observability](#observability) for the full response.

## Performance

QUAN-Transcribe is a CTC model: one forward pass per piece of audio, no
autoregressive loop and no KV cache. So the useful optimisation is different
from an LLM server's, and there are four levers, in rough order of impact.

### 1. The engine core: one owner, priority, coalescing, backpressure

The model is not re-entrant, so exactly one thread may touch it at a time. That
thread is the *engine core* (`app/core.py`); every request, streaming or batch,
goes through it. One owner is what makes the rest possible:

- **A forward pass pads every piece of audio in it to the longest piece**, so
  batching *k* pieces of up to *n* samples costs `k * n`. The core gathers
  pieces from concurrent requests for up to `MICROBATCH_MAX_WAIT_MS`, groups
  them by decode signature, sorts ascending by length, and greedily packs them
  under the same padded-cost budget the library uses. Encoding is where almost
  all the compute goes and it is a pure GEMM pipeline, so packing more work into
  one launch is close to free throughput: two concurrent requests share one pass
  instead of paying for two.
- **Interactive work outranks bulk work.** A microphone step (`LiveSession`, the
  WebSocket, an SSE stream) is submitted as *interactive*; a file upload is
  *bulk*. Interactive jobs run immediately, never wait out a gather window, and
  preempt a bulk request's remaining sub-batches. A 0.48 s live step can no
  longer queue behind a ten-minute file.
- **A forward pass is bounded.** Preemption only helps *between* launches, so
  without a cap a single big file is one uninterruptible pass and the promise
  above is empty. `SUBBATCH_SECONDS` caps one pass at that many audio seconds
  (the padded cost `k * n`), turning a long file into several launches the
  scheduler can preempt between. Without it, a live step's worst-case wait is
  the whole file; with `SUBBATCH_SECONDS=10` it becomes about one chunk's
  compute (measured on CPU: a live step behind a 120 s file went from ~18 s to
  ~3.6 s with `CHUNK_SECONDS=10`). `0` keeps `BATCH_SECONDS` as the only bound.
- **Backpressure is real.** `MAX_QUEUE` bounds the number of pieces waiting to
  decode. When it is full, new uploads get a `503` (`"code": "server_busy"`)
  instead of parking another thread, so saturation shows up as a fast, retryable
  refusal rather than latency that grows without bound. Set `MAX_QUEUE=0` to
  disable the cap. `MAX_INTERACTIVE` does the same for streaming jobs, with the
  same retryable refusal.

The cost of coalescing is up to `MICROBATCH_MAX_WAIT_MS` of added latency, and
only on the first piece of a request. Set `MICROBATCH_SECONDS=0` to batch only
what happens to already be queued (no added latency at all, still helps under
load).

Pieces with different decode signatures — streaming vs not, a different
`left_chunks`, greedy vs beam search — are never mixed, because they need
different model behaviour.

> This is deliberately *not* an LLM-style continuous-batching scheduler. vLLM
> schedules tokens across steps with a KV cache; a CTC forward pass has neither,
> so there is nothing to interleave within a piece. The only reuse available is
> putting whole pieces into one padded pass, which is exactly what the bulk path
> does.

`GET /metrics` reports the scheduler under `core`: `batches_run`, `pieces_run`,
`jobs_run`, `queued_pieces`, `queued_interactive`, `shed`, `avg_batch_size` and
`p50_latency_ms` / `p95_latency_ms` — the direct way to see whether coalescing
is working and whether you are shedding load. See
[Observability](#observability) for the full key list.

Try `MICROBATCH_SECONDS=0` first if you serve one client at a time.

### 2. TF32 on Ampere and later

`TF32=true` (the default) lets FP32 matmuls run on the tensor cores. For a CTC
model the precision loss is immaterial next to the speedup, and it is a no-op
before Ampere and on CPU. Set `TF32=false` if you need bit-exact FP32.

The model already autocasts the encoder to bf16 on CUDA, so there is no bf16
knob to turn: it is on whenever the device is CUDA.

### 3. The language model is a CPU bottleneck

With `USE_LM=true` the beam search runs on the CPU, one utterance at a time,
*after* the GPU forward pass. At beam 32 it is roughly 15 ms per utterance on a
single core, which on a fast GPU becomes the wall-clock bottleneck — the GPU
finishes and waits. If throughput matters more than the last two WER points,
leave the LM off; if accuracy matters more, turn it on and give the worker spare
CPU. Raising or lowering `BEAM_WIDTH` moves this cost linearly.

### 4. CPU is a different machine

On CPU there is no fast device to hide behind, so the model is a plain FP32
GEMM pipeline that scales with cores and with batch width:

- `DEVICE=cpu` pins the profile, so a host that happens to expose a GPU does not
  silently switch paths.
- `CPU_THREADS` sets the torch thread count. Leave it unset to let torch size
  itself to the visible cores; set it to the container's quota when it is
  cgroup-limited, or the threads will oversubscribe and thrash.
- bf16 autocast does not apply on CPU, so the CUDA speedups above do not carry
  over. Expect realtime-factor numbers in the tens, not thousands.

Measured on an 8-core Apple laptop (Docker, 8 threads), 30 s of audio:

| Path | Time | Notes |
| --- | --- | --- |
| Offline transcribe | ~2.5–3.1 s | RTF ≈ 0.10, ≈ 10× realtime |
| SSE replay of the same file | ~27 s | re-decodes the utterance every 0.48 s step — inherent to the streaming decoder, not this server |
| 4 concurrent files (120 s audio) | ~11.8 s | one shared forward pass (`batches_run` +1 for 4 requests), but near-neutral in wall time: torch already saturates every core per launch |

The last row is the CPU-specific finding: **coalescing is nearly free on GPU
(idle SM time) but nearly worthless on CPU**, where the cores are the
bottleneck. CPU coalescing's real value is scheduling — it is what lets a live
step preempt a file — so on CPU set `SUBBATCH_SECONDS` and a small
`CHUNK_SECONDS` rather than chasing throughput with a wide batch window.

`GET /metrics` reports `realtime_factor` and, under `core`, everything the
scheduler saw. A quick health check: under steady concurrent load, `batches_run`
should be well below `jobs_run`, and `shed` should stay at zero.

## Docker

```bash
./scripts/build.sh                          # -> quan-asr-server:1.0.0
TAG=registry.example.com/asr:1.0.0 ./scripts/build.sh
INSTALL_LM=false ./scripts/build.sh         # skip the kenlm compile
CUDA=12.4.1 TORCH_INDEX=https://download.pytorch.org/whl/cu124 ./scripts/build.sh
```

Run it, keeping the Hugging Face cache on a volume so the weights survive a
restart:

```bash
docker run --gpus all -p 8000:8000 \
  -v hf-cache:/cache/huggingface \
  -e HF_TOKEN=hf_... \
  -e USE_LM=true \
  quan-asr-server:1.0.0
```

`docker/docker-compose.yaml` ships two profiles built from the same image:

```bash
docker compose --profile gpu up    # port 8000, DEVICE=cuda, TF32 on, reserves an nvidia device
docker compose --profile cpu up    # port 8001, DEVICE=cpu, TF32 off, a longer coalescing window
```

Both mount a named `hf-cache` volume at `/cache/huggingface`, so the weights
survive a restart, and both read `HF_TOKEN` from your shell or a `.env` file.
The CPU profile trades latency for throughput — a wider coalescing window
(`MICROBATCH_SECONDS=0.25`) with a shorter hard cap (`MICROBATCH_MAX_WAIT_MS=10`)
and a `SUBBATCH_SECONDS=30` cap so a live step can still preempt a file.

Two things the image deliberately does **not** do:

- **It has no `USER`**, so the process runs as root. That is what lets an
  orchestrator mount a model cache it owns without `chown` games, but if your
  policy forbids it, add a `USER` line and make `/cache/huggingface` writable.
- **It declares no `VOLUME`**, so nothing is persisted unless you mount
  `/cache/huggingface` yourself (the compose profiles do).

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"

pytest -q                     # 115 tests, no GPU or model download needed
ruff check app tests scripts
ruff format app tests scripts
```

The tests monkeypatch a scripted fake engine into `app.server`, so they exercise
the HTTP, SSE and WebSocket contracts without loading the model. They never
import `torch` or `quan_transcribe`, which is why CI can run them on a plain
Python image.

To drive the real endpoints without downloading 356M parameters:

```bash
python scripts/dev_server.py --port 8123              # stub engine, real ASGI stack
URL=http://127.0.0.1:8123 ./scripts/smoke.sh          # health + models + one transcribe + SSE
python scripts/ws_probe.py --url ws://127.0.0.1:8123/v1/audio/transcriptions/stream
```

`smoke.sh` takes the **audio file** as its first positional argument and the
**URL** from the `URL` environment variable; it checks `/health` and
`/v1/models` first so a failure points at the right layer.
`scripts/make_sample_audio.py` writes a small tone/quiet WAV if you need a
fixture. `scripts/ws_probe.py` streams that kind of audio over the live
WebSocket and prints every event; it needs `websockets`, which `uvicorn[standard]`
already pulls in.

Two scripts answer "is this file badly transcribed, or is it just mostly
silence?" — both run against the real model and the real recordings in
`samples/`:

```bash
python scripts/probe_audio.py samples/*.mp3    # 100 ms RMS envelope, % speech, no model
python scripts/repro_local.py samples/*.mp3    # per-segment text, VAD on vs off
```

`probe_audio.py` needs only ffmpeg, so it is the quick first look.
`repro_local.py` loads the model, transcribes each file twice (VAD off, then on),
and prints every segment with its timestamps, the joined transcript, the decode
time and the time VAD saved. Inside the container:

```bash
docker run --rm --env-file .env -v "$PWD":/work -w /work \
  -v quan-hf:/cache/huggingface -e HF_HOME=/cache/huggingface \
  --entrypoint python quan-asr-server:1.0.0 scripts/repro_local.py samples/voicemail-90s.mp3
```

### Layout

```
app/config.py     settings, all env-driven
app/audio.py      decode, resample, quiet-point and VAD-region splitting, segments
app/formats.py    json/text/srt/vtt/verbose_json rendering
app/vad.py        Silero voice activity detection (lazy, optional)
app/core.py       the engine core: model-owning thread, priority + coalescing
app/engine.py     model wrapper, streaming, live sessions, scheduler wiring
app/schemas.py    request validation, OpenAI-shaped errors
app/server.py     FastAPI routes, SSE, WebSocket
app/ui.py         browser playground
app/cli.py        `quan-asr-server` entrypoint
app/__main__.py   `python -m app` shim
```

Every model call goes through `app/core.py`, a single thread that owns the
model: interactive (streaming) work preempts bulk (file) work, near-simultaneous
bulk pieces are coalesced into one padded forward pass, and a bounded queue
sheds load with a `503` rather than growing latency. Loading happens behind its
own lock so concurrent first requests cannot race, and a load failure does not
kill the container — it surfaces through `/health` as `"error"`.

## Credits

Model and reference implementation: [QUAN-Transcribe](https://github.com/qnguyen3/QUAN-Transcribe)
by [qnguyen3](https://huggingface.co/qnguyen3). The base audio encoder is
Google's [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2).

## Licence

This server is released under **Apache-2.0** (see `pyproject.toml`).

The **model weights are not** Apache-2.0: `QUAN-Transcribe-1.0` is a derivative
of EmbeddingGemma 2 and is governed by the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms). Review those terms
before using the model in production, and note that the gated
`google/embeddinggemma-2` encoder requires accepting Google's licence before it
can be downloaded.
