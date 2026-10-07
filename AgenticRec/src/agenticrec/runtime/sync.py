"""Bounded sync execution with observable best-effort timeout cancellation."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
import math
from threading import Lock

from .errors import SyncTaskTimeout


@dataclass(frozen=True)
class SyncRunnerSnapshot:
    max_workers: int
    submitted: int
    completed: int
    timed_out: int
    timed_out_running: int

    def to_dict(self):
        return {
            "max_workers": self.max_workers,
            "submitted": self.submitted,
            "completed": self.completed,
            "timed_out": self.timed_out,
            "timed_out_running": self.timed_out_running,
        }


class SyncTaskRunner:
    """Use a bounded pool; timeout cannot force-stop an already running thread."""

    def __init__(self, max_workers=2, *, thread_name_prefix="agenticrec-tool"):
        if type(max_workers) is not int or max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        if not isinstance(thread_name_prefix, str) or not thread_name_prefix.strip():
            raise ValueError("thread_name_prefix must be a nonempty string")
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=thread_name_prefix
        )
        self._lock = Lock()
        self._submitted = 0
        self._completed = 0
        self._timed_out = 0
        self._timed_out_running = 0
        self._closed = False

    def run(self, function, *args, timeout_seconds):
        if not callable(function):
            raise TypeError("function must be callable")
        if (type(timeout_seconds) not in (int, float)
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be finite and positive")
        with self._lock:
            if self._closed:
                raise RuntimeError("sync runner is closed")
            self._submitted += 1
        future = self._executor.submit(function, *args)
        future.add_done_callback(self._record_completion)
        try:
            return future.result(timeout=float(timeout_seconds))
        except FutureTimeout as error:
            # In Python 3.11 concurrent.futures.TimeoutError aliases the built-in
            # TimeoutError, which a completed tool may itself have raised.
            if future.done():
                return future.result()
            cancelled = future.cancel()
            with self._lock:
                self._timed_out += 1
                if not cancelled:
                    self._timed_out_running += 1
            raise SyncTaskTimeout(
                "sync_task_timeout",
                work_may_continue=not cancelled,
            ) from error

    def snapshot(self):
        with self._lock:
            return SyncRunnerSnapshot(
                max_workers=self.max_workers,
                submitted=self._submitted,
                completed=self._completed,
                timed_out=self._timed_out,
                timed_out_running=self._timed_out_running,
            ).to_dict()

    def shutdown(self, *, wait=True, cancel_futures=True):
        if type(wait) is not bool or type(cancel_futures) is not bool:
            raise ValueError("shutdown flags must be booleans")
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def _record_completion(self, _future):
        with self._lock:
            self._completed += 1
