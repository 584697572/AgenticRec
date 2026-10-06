"""Monotonic whole-round deadline with per-attempt timeout derivation."""

import math
import time

from .errors import DeadlineExceeded


class RoundDeadline:
    def __init__(self, duration_seconds, *, clock=time.monotonic):
        if (type(duration_seconds) not in (int, float)
                or not math.isfinite(duration_seconds) or duration_seconds <= 0):
            raise ValueError("duration_seconds must be finite and positive")
        self.duration_seconds = float(duration_seconds)
        self._clock = clock
        self.started_at = float(clock())
        self.expires_at = self.started_at + self.duration_seconds

    def remaining_seconds(self):
        return max(0.0, self.expires_at - float(self._clock()))

    def timeout_for_attempt(self, per_request_timeout_seconds):
        if (type(per_request_timeout_seconds) not in (int, float)
                or not math.isfinite(per_request_timeout_seconds)
                or per_request_timeout_seconds <= 0):
            raise ValueError("per_request_timeout_seconds must be finite and positive")
        remaining = self.remaining_seconds()
        if remaining <= 0:
            raise DeadlineExceeded("round_deadline_exceeded_before_attempt")
        return min(float(per_request_timeout_seconds), remaining)

    def require_backoff_window(self, delay_seconds):
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be nonnegative")
        if delay_seconds >= self.remaining_seconds() and delay_seconds > 0:
            raise DeadlineExceeded("round_deadline_exceeded_before_retry")
