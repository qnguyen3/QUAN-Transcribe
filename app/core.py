"""The engine core: the one thread that owns the model.

Why this exists
---------------
The model is not re-entrant. A forward pass mutates shared buffers, and on CUDA
concurrent kernels would fight for the same memory. So exactly one thing may be
talking to the model at a time. The question is *what* enforces that, and the
answer used to be a lock: every request grabbed a global ``RLock`` for the
duration of its decode. That is correct but it has three problems that only show
up under real load.

* **A long request holds the lock for its whole decode.** A ten-minute file is
  ~20 pieces; a microphone session that wants to decode one 0.48 s step has to
  wait for all of them. Head-of-line blocking, felt worst by the one caller who
  cares most about latency.
* **No priority.** A live microphone step and a bulk file upload queue behind
  each other in whatever order the OS happened to wake their threads. There is
  no way to say "this step is interactive, that piece is not".
* **No backpressure.** Threads pile up on the lock without bound. When the
  server is saturated the only feedback is that everything gets slow, and
  latency grows without limit until the client gives up.

This module replaces the lock with a *scheduler*. One dedicated thread runs a
loop, and only that thread ever touches the model. Callers from any thread hand
it work through :meth:`EngineCore.submit` (a single callable, run to completion
on the model thread) or :meth:`EngineCore.submit_bulk` (audio pieces that the
core is free to coalesce). Because the core is the sole owner, there is no lock
to hold across a request and therefore no head-of-line blocking.

Scheduling
----------
Two classes of work, in strict priority order:

* **interactive** — one callable, run immediately, never delayed to batch. This
  is a streaming step (`LiveSession.accept`) or a flush. Superseding a live
  decode by a few milliseconds is the whole point of the priority.
* **bulk** — audio pieces submitted by ``transcribe()``. The core waits a short
  window (``window_ms``) for near-simultaneous pieces, groups them by *decode
  signature*, sorts by length and greedily packs them into sub-batches under the
  same padded-cost budget the library uses. This is what makes N concurrent
  requests share one launch instead of paying for N.

Interactive work preempts bulk work at sub-batch granularity: the core checks
for interactive jobs before each sub-batch and, if any arrived, pushes the rest
back to the head of the queue and loops. At least one sub-batch always runs per
iteration, so gathered work can never be starved into livelock.

Backpressure
------------
The bulk queue is bounded (``max_queued_pieces``). When it is full,
:meth:`submit_bulk` raises :class:`EngineBusy` instead of parking another thread;
the HTTP layer turns that into a ``503``. That converts an unbounded latency
growth into a fast, honest refusal — the single most important property for
keeping a saturated server usable.

What this is *not*
------------------
vLLM's engine core and this share a shape — one owner of the model, a frontend
that submits and awaits — but not a strategy. vLLM schedules *tokens* across
steps with a KV cache so that an autoregressive decode can interleave many
sequences. QUAN-Transcribe is CTC: one forward pass per piece, no cache, no
incremental state across calls. So there is no continuous batching here, no
paged attention, no step-level iteration. The only reuse available is putting
several whole pieces into one padded forward pass, which is exactly what the
bulk path does.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Hashable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: A decode signature: everything that changes *how* a piece is decoded.
#: Pieces are only ever coalesced into one forward pass when their signatures
#: are equal, because they would otherwise need different model behaviour.
Signature = Hashable

#: ``run(wavs, context) -> list[str]``. Called on the model thread only.
RunBatch = Callable[[list[np.ndarray], Any], list[str]]

#: How many recent job latencies to keep for percentile reporting.
_LATENCY_WINDOW = 4096

class EngineBusy(RuntimeError):
    """Raised when the engine is saturated and shed the work.

    Callers should treat this as transient and retry, not as a bug. The HTTP
    layer maps it to a ``503`` with ``Retry-After``.
    """

class _Piece:
    """One audio piece waiting for its turn through the model."""

    __slots__ = ("job", "ordinal", "text", "wav")

    def __init__(self, job: _Job, wav: np.ndarray, ordinal: int) -> None:
        self.job = job
        self.wav = wav
        #: Position within the owning job, so results can be reassembled in
        #: submission order even though pieces are decoded out of order.
        self.ordinal = ordinal
        self.text: str = ""

class _Job:
    """A unit of work submitted by one caller.

    A *bulk* job carries several pieces and may be split across sub-batches; it
    completes when its last piece does. An *interactive* job carries a single
    callable and completes when that callable returns.
    """

    __slots__ = (
        "call",
        "context",
        "done",
        "enqueued",
        "error",
        "interactive",
        "remaining",
        "result",
        "signature",
        "texts",
    )

    def __init__(
        self,
        *,
        call: Callable[[], Any] | None = None,
        signature: Signature | None = None,
        context: Any = None,
        pieces: int = 0,
    ) -> None:
        self.call = call
        self.signature = signature
        self.context = context
        self.interactive = call is not None
        self.texts: list[str | None] = [None] * pieces
        self.remaining = pieces
        self.result: Any = None
        self.error: BaseException | None = None
        self.done = threading.Event()
        self.enqueued = time.monotonic()

    def complete(self) -> None:
        """Mark the job finished once, recording how long it was in flight."""
        if self.done.is_set():
            return
        self.done.set()

class EngineCore:
    """A single model-owning thread with a priority, coalescing scheduler."""

    def __init__(
        self,
        run_batch: RunBatch,
        *,
        sample_rate: int,
        window_ms: float,
        max_requests: int,
        batch_seconds: float,
        subbatch_seconds: float = 0.0,
        max_queued_pieces: int = 0,
        max_queued_interactive: int = 0,
        name: str = "quan-asr-server-core",
    ) -> None:
        self._run_batch = run_batch
        self._sample_rate = sample_rate
        #: Longest padded cost a single forward pass may cover, in audio
        #: seconds (``k`` pieces of up to ``n`` samples cost ``k * n``). 0 means
        #: "as large as ``batch_seconds`` allows". Lowering it caps how long the
        #: model thread can stay inside one launch, which bounds the worst-case
        #: time an interactive job can wait — the fairness/latency trade.
        self._subbatch_samples = max(0, int(subbatch_seconds * sample_rate))
        #: How long the core lingers after the first bulk piece, waiting for
        #: companions. This is the latency coalescing can add, and only to the
        #: first piece of a request. 0 means "batch only what is already queued".
        self._window_s = max(0.0, window_ms) / 1000.0
        self._max_requests = max(1, max_requests)
        self._max_samples = max(1, int(batch_seconds * sample_rate))
        #: 0 disables the cap (unbounded queue). A positive value sheds load.
        self._max_queued_pieces = max(0, max_queued_pieces)
        self._max_queued_interactive = max(0, max_queued_interactive)
        self._name = name

        self._cv = threading.Condition()
        self._bulk: deque[_Piece] = deque()
        self._interactive: deque[_Job] = deque()
        self._stopping = False
        self._thread: threading.Thread | None = None

        # Bookkeeping, read by /metrics. Counters are plain ints (the GIL makes
        # the increments atomic enough for reporting) and the latency window is
        # a bounded deque so memory stays flat under load.
        self.batches_run = 0
        self.pieces_run = 0
        self.jobs_run = 0
        self.shed = 0
        self.model_seconds = 0.0
        self._latencies: deque[float] = deque(maxlen=_LATENCY_WINDOW)
        self._batch_sizes: deque[int] = deque(maxlen=_LATENCY_WINDOW)

    # -- lifecycle -----------------------------------------------------------

    @property
    def started(self) -> bool:
        return self._thread is not None

    def start(self) -> None:
        """Start the model thread. Idempotent."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the model thread, failing anything still queued."""
        with self._cv:
            self._stopping = True
            pending = [*self._interactive]
            pending.extend({piece.job for piece in self._bulk})
            self._interactive.clear()
            self._bulk.clear()
            self._cv.notify_all()
        thread, self._thread = self._thread, None
        for job in pending:
            # Report as busy rather than a hard failure: from the caller's point
            # of view a shutdown and a shed are the same — retry elsewhere.
            job.error = EngineBusy("the engine is shutting down")
            job.complete()
        if thread is not None:
            thread.join(timeout=5.0)

    # -- submission ----------------------------------------------------------

    def submit(self, call: Callable[[], Any]) -> Any:
        """Run ``call`` on the model thread and return its result.

        This is the interactive path: it takes priority over every queued bulk
        piece and is never delayed to gather companions. ``call`` runs as one
        atomic critical section, so it may safely touch model state.
        """
        job = _Job(call=call)
        with self._cv:
            if self._stopping:
                raise EngineBusy("the engine is shutting down")
            if (
                self._max_queued_interactive
                and len(self._interactive) >= self._max_queued_interactive
            ):
                self.shed += 1
                raise EngineBusy(
                    "too many interactive requests in flight; retry shortly"
                )
            self._interactive.append(job)
            self._cv.notify_all()
        job.done.wait()
        self._latencies.append(time.monotonic() - job.enqueued)
        if job.error is not None:
            raise job.error
        return job.result

    def submit_bulk(
        self,
        wavs: list[np.ndarray],
        signature: Signature,
        context: Any = None,
    ) -> list[str]:
        """Decode ``wavs``, letting the core coalesce them with other work.

        Returns one transcript per input piece, in submission order. Raises
        :class:`EngineBusy` when the queue is full rather than parking the
        caller, and re-raises whatever the model raised otherwise.
        """
        if not wavs:
            return []

        job = _Job(signature=signature, context=context, pieces=len(wavs))
        pieces = [_Piece(job, wav, index) for index, wav in enumerate(wavs)]

        with self._cv:
            if self._stopping:
                raise EngineBusy("the engine is shutting down")
            if (
                self._max_queued_pieces
                and len(self._bulk) + len(pieces) > self._max_queued_pieces
            ):
                self.shed += 1
                raise EngineBusy(
                    "the decode queue is full; retry once the backlog clears"
                )
            self._bulk.extend(pieces)
            self._cv.notify_all()

        job.done.wait()
        self._latencies.append(time.monotonic() - job.enqueued)
        if job.error is not None:
            raise job.error
        return [text or "" for text in job.texts]

    # -- scheduler loop ------------------------------------------------------

    def _loop(self) -> None:
        while True:
            work = self._collect()
            if work is None:  # told to stop
                return
            interactive, bulk = work
            try:
                self._dispatch(interactive, bulk)
            except BaseException:  # pragma: no cover - defensive
                logger.exception("engine core iteration failed")

    def _collect(
        self,
    ) -> tuple[list[_Job], list[_Piece]] | None:
        """Block for work, then take a snapshot of what to run next.

        Interactive jobs are taken immediately. If none arrived, the core spends
        its coalescing window waiting for bulk companions; if some did, bulk is
        snapshotted without waiting so the interactive work is not delayed.
        """
        with self._cv:
            while (
                not self._interactive
                and not self._bulk
                and not self._stopping
            ):
                self._cv.wait()
            if self._stopping and not self._interactive and not self._bulk:
                return None

            interactive = list(self._interactive)
            self._interactive.clear()

            if not interactive and self._bulk:
                self._await_companions()
                # An interactive job that arrived mid-window aborts the gather.
                # Take it now so it is dispatched ahead of the bulk snapshot.
                if self._interactive:
                    interactive = list(self._interactive)
                    self._interactive.clear()

            bulk = list(self._bulk)
            self._bulk.clear()
            return interactive, bulk

    def _await_companions(self) -> None:
        """Wait out the gather window while the batch is still worth widening.

        Caller holds ``self._cv``. Returns as soon as the window closes, the
        queue holds pieces from ``max_requests`` distinct callers, or an
        interactive job arrives — interactive latency is exactly what priority
        exists to protect, so the gather never delays it.
        """
        deadline = time.monotonic() + self._window_s
        while True:
            if self._interactive:
                return
            callers = {piece.job for piece in self._bulk if not piece.job.done.is_set()}
            if len(callers) >= self._max_requests:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._cv.wait(timeout=remaining)

    def _dispatch(self, interactive: list[_Job], bulk: list[_Piece]) -> None:
        for job in interactive:
            self._run_interactive(job)

        if not bulk:
            return

        groups: dict[Signature, list[_Piece]] = {}
        for piece in bulk:
            groups.setdefault(piece.job.signature, []).append(piece)

        subs: list[list[_Piece]] = []
        for pieces in groups.values():
            subs.extend(self._pack(pieces))

        for index, sub in enumerate(subs):
            # Preempt between sub-batches, but always run the first one so
            # gathered work keeps making progress no matter how busy the
            # interactive stream is.
            if index and self._has_interactive():
                self._requeue([piece for rest in subs[index:] for piece in rest])
                return
            self._run_bulk(sub)

    def _has_interactive(self) -> bool:
        with self._cv:
            return bool(self._interactive)

    def _requeue(self, pieces: list[_Piece]) -> None:
        """Push pieces back to the head of the queue, preserving their order."""
        if not pieces:
            return
        with self._cv:
            for piece in reversed(pieces):
                self._bulk.appendleft(piece)
            self._cv.notify_all()

    # -- execution -----------------------------------------------------------

    def _run_interactive(self, job: _Job) -> None:
        started = time.monotonic()
        try:
            job.result = job.call()
            self.jobs_run += 1
        except BaseException as exc:
            job.error = exc
        finally:
            self.model_seconds += time.monotonic() - started
            job.complete()

    def _run_bulk(self, sub: list[_Piece]) -> None:
        wavs = [piece.wav for piece in sub]
        started = time.monotonic()
        try:
            texts = self._run_batch(wavs, sub[0].job.context)
            if len(texts) != len(sub):  # pragma: no cover - defensive
                raise RuntimeError(
                    f"expected {len(sub)} transcripts, got {len(texts)}"
                )
        except BaseException as exc:
            # Fail every job that owned a piece in this sub-batch. A job may
            # straddle sub-batches, but once it has an error the caller has
            # already been unblocked, so later pieces simply complete quietly.
            for job in {piece.job for piece in sub}:
                job.error = exc
                job.complete()
            for piece in sub:
                self._settle(piece, "")
            return
        finally:
            self.model_seconds += time.monotonic() - started

        self.batches_run += 1
        self.pieces_run += len(sub)
        self._batch_sizes.append(len(sub))
        for piece, text in zip(sub, texts, strict=True):
            self._settle(piece, text or "")

    def _settle(self, piece: _Piece, text: str) -> None:
        """Record one piece's transcript and complete its job if it was last."""
        job = piece.job
        piece.text = text
        if job.done.is_set():
            # The job already failed or was cancelled; nothing left to fill in.
            return
        if job.texts[piece.ordinal] is None:
            job.remaining -= 1
        job.texts[piece.ordinal] = text
        if job.remaining <= 0:
            job.result = job.texts
            self.jobs_run += 1
            job.complete()

    def _pack(self, pieces: list[_Piece]) -> list[list[_Piece]]:
        """Greedily pack pieces by padded cost, like the library's own batching.

        A forward pass pads every piece to the longest in the batch, so a batch
        of *k* pieces of at most *n* samples costs ``k * n``. Sorting ascending
        and admitting a piece while that product still fits the budget keeps
        short pieces from being dragged up to a long piece's length.

        The effective budget is the smaller of ``batch_seconds`` and
        ``subbatch_seconds``: the latter exists purely so a large file becomes
        several launches the scheduler can preempt between, instead of one
        uninterruptible pass that blocks live audio for its whole duration.
        """
        budget = self._max_samples
        if self._subbatch_samples:
            budget = min(budget, self._subbatch_samples)
        ordered = sorted(pieces, key=lambda piece: len(piece.wav))
        batches: list[list[_Piece]] = []
        current: list[_Piece] = []

        for piece in ordered:
            padded = len(piece.wav) * (len(current) + 1)
            if current and padded > budget:
                batches.append(current)
                current = [piece]
            else:
                current.append(piece)

        if current:
            batches.append(current)
        return batches

    # -- introspection -------------------------------------------------------

    @property
    def queued_pieces(self) -> int:
        return len(self._bulk)

    @property
    def queued_interactive(self) -> int:
        return len(self._interactive)

    def stats(self) -> dict[str, Any]:
        """A metrics-ready snapshot. Cheap enough to call on every scrape."""
        return {
            "batches_run": self.batches_run,
            "pieces_run": self.pieces_run,
            "jobs_run": self.jobs_run,
            "queued_pieces": len(self._bulk),
            "queued_interactive": len(self._interactive),
            "shed": self.shed,
            "avg_batch_size": (
                round(sum(self._batch_sizes) / len(self._batch_sizes), 3)
                if self._batch_sizes
                else None
            ),
            "p95_latency_ms": self._percentile(0.95),
            "p50_latency_ms": self._percentile(0.50),
            "model_seconds": round(self.model_seconds, 3),
        }

    def _percentile(self, quantile: float) -> float | None:
        if not self._latencies:
            return None
        ordered = sorted(self._latencies)
        index = min(len(ordered) - 1, int(quantile * len(ordered)))
        return round(ordered[index] * 1000.0, 3)
