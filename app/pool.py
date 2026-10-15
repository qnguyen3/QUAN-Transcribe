"""Multi-GPU & Multi-Worker Engine Pool for QUAN-ASR Server.

Allows a single QUAN-ASR Server process to orchestrate across multiple GPUs
(e.g. ``cuda:0``, ``cuda:1``, etc.) or multi-core CPU workers, achieving linear
throughput scaling and dynamic load balancing.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .core import EngineCore

logger = logging.getLogger(__name__)


class WorkerInstance:
    """Encapsulates a model instance and its dedicated EngineCore thread."""

    def __init__(
        self,
        device_id: str,
        core_factory: Callable[[str], EngineCore],
    ) -> None:
        self.device_id = device_id
        self.core: EngineCore = core_factory(device_id)

    @property
    def queue_load(self) -> int:
        """Heuristic load metric for scheduling: interactive jobs weighted 3x."""
        return (self.core.queued_interactive * 3) + self.core.queued_pieces

    def close(self) -> None:
        self.core.stop()


class EnginePool:
    """A pool of EngineCores distributed across multiple devices/threads."""

    def __init__(
        self,
        devices: list[str],
        core_factory: Callable[[str], EngineCore],
    ) -> None:
        self.devices = devices or ["auto"]
        self._core_factory = core_factory
        self.workers: list[WorkerInstance] = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, len(self.devices)),
            thread_name_prefix="pool-dispatch",
        )

        for dev in self.devices:
            worker = WorkerInstance(dev, core_factory)
            self.workers.append(worker)

        logger.info(
            "initialized EnginePool with %d worker(s): %s",
            len(self.workers),
            ", ".join(w.device_id for w in self.workers),
        )

    @property
    def size(self) -> int:
        return len(self.workers)

    def select_least_loaded_worker(self) -> WorkerInstance:
        """Route to the worker with the smallest current queue load."""
        with self._lock:
            return min(self.workers, key=lambda w: w.queue_load)

    def submit(self, call: Callable[[], Any]) -> Any:
        """Route an interactive/callable job to the least-loaded worker core."""
        worker = self.select_least_loaded_worker()
        return worker.core.submit(call)

    def submit_bulk(
        self,
        pieces: list[np.ndarray],
        signature: Any,
        context: Any,
    ) -> list[str]:
        """Distribute bulk audio pieces across workers for parallel multi-device execution."""
        if not pieces:
            return []

        # Single worker optimization (zero partitioning overhead)
        if len(self.workers) == 1:
            return self.workers[0].core.submit_bulk(pieces, signature, context)

        # Distribute pieces across workers
        n_workers = len(self.workers)
        chunks: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(n_workers)]
        for orig_idx, piece in enumerate(pieces):
            worker_idx = orig_idx % n_workers
            chunks[worker_idx].append((orig_idx, piece))

        futures = []
        for worker_idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            chunk_pieces = [c[1] for c in chunk]
            worker = self.workers[worker_idx]
            fut = self._executor.submit(
                worker.core.submit_bulk, chunk_pieces, signature, context
            )
            futures.append((chunk, fut))

        # Reassemble results in original ordinal order
        results: list[str] = [""] * len(pieces)
        for chunk, fut in futures:
            texts = fut.result()
            for (orig_idx, _), text in zip(chunk, texts, strict=False):
                results[orig_idx] = text

        return results

    def stats(self) -> dict[str, Any]:
        """Aggregate stats across all worker cores in the pool."""
        total_batches = 0
        total_pieces = 0
        total_jobs = 0
        queued_pieces = 0
        queued_interactive = 0
        total_shed = 0
        total_model_seconds = 0.0
        worker_stats = {}

        for w in self.workers:
            st = w.core.stats()
            worker_stats[w.device_id] = st
            total_batches += st["batches_run"]
            total_pieces += st["pieces_run"]
            total_jobs += st["jobs_run"]
            queued_pieces += st["queued_pieces"]
            queued_interactive += st["queued_interactive"]
            total_shed += st["shed"]
            total_model_seconds += st["model_seconds"]

        return {
            "workers_count": len(self.workers),
            "batches_run": total_batches,
            "pieces_run": total_pieces,
            "jobs_run": total_jobs,
            "queued_pieces": queued_pieces,
            "queued_interactive": queued_interactive,
            "shed": total_shed,
            "model_seconds": round(total_model_seconds, 3),
            "workers": worker_stats,
        }

    def close(self) -> None:
        """Shutdown all worker cores and executor."""
        self._executor.shutdown(wait=False)
        for w in self.workers:
            w.close()
