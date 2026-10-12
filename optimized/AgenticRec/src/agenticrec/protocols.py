"""Minimal scoring and tool-result contracts; model implementations follow T06-T10."""
from dataclasses import dataclass
from typing import Any, Protocol, Sequence


class Recommender(Protocol):
    def score(self, user_id: int, candidate_ids: Sequence[int]) -> Sequence[float]: ...
    def recommend(self, user_id: int, candidate_ids: Sequence[int], k: int) -> Sequence[int]: ...


class SessionScorer(Protocol):
    def score_from_seeds(self, liked_ids: Sequence[int], disliked_ids: Sequence[int],
                         candidate_ids: Sequence[int]) -> Sequence[float]: ...


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: Any = None
    error_code: str | None = None
    retryable: bool = False
    elapsed_ms: float | None = None
    provenance: str | None = None

    def __post_init__(self):
        if type(self.ok) is not bool or type(self.retryable) is not bool:
            raise ValueError("ok/retryable must be booleans")
        if self.ok and (self.error_code is not None or self.retryable):
            raise ValueError("successful result cannot carry an error or request retry")
        if not self.ok and not self.error_code:
            raise ValueError("failure requires an error_code")
