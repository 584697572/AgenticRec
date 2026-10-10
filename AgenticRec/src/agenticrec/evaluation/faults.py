"""Evaluator-owned execution flags, isolated from planner-visible inputs."""

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from types import MappingProxyType


class FaultSchedule:
    __slots__ = ("_events", "sha256")

    def __init__(self, events):
        if type(events) is not dict or any(
            not isinstance(key, str) or value not in (None, "ranking_timeout")
            for key, value in events.items()
        ):
            raise ValueError("unsupported evaluator fault schedule")
        self._events = MappingProxyType(dict(events))
        encoded = json.dumps(events, sort_keys=True, separators=(",", ":")).encode()
        self.sha256 = hashlib.sha256(encoded).hexdigest()

    def __call__(self, episode_id):
        return self._events.get(episode_id)

    def __repr__(self):
        return "FaultSchedule(sha256=" + repr(self.sha256) + ")"


class FaultAwarePipeline:
    """Inject a deterministic local ranking failure, then run the safe fallback.

    No timeout traffic is sent to a provider. The failure is an execution fixture,
    not a measured remote timeout; the fallback keeps the same hard constraints.
    """

    def __init__(self, pipeline, fallback):
        self.pipeline = pipeline
        self.fallback = fallback
        self.retriever = pipeline.retriever
        self.router = pipeline.router
        self.invocation_count = 0
        self.fault_count = 0
        self._active = None

    @contextmanager
    def activate(self, flag):
        if self._active is not None:
            raise RuntimeError("fault pipeline is not reentrant")
        self._active = flag
        try:
            yield
        finally:
            self._active = None

    def recommend(self, request):
        self.invocation_count += 1
        if self._active == "ranking_timeout":
            self.fault_count += 1
            self.invocation_count += 1
            response = self.fallback.recommend(request)
            return replace(response, fallback_reason="fixed_pipeline")
        return self.pipeline.recommend(request)


class FaultBoundExecutor:
    def __init__(self, executor, schedule, pipeline):
        self.executor, self.schedule, self.pipeline = executor, schedule, pipeline
        self.feedback_sha256 = executor.feedback_sha256
        self.fault_sha256 = schedule.sha256

    def __call__(self, episode):
        before = self.pipeline.invocation_count
        faults_before = self.pipeline.fault_count
        with self.pipeline.activate(self.schedule(episode.episode_id)):
            attempt = self.executor(episode)
        calls = self.pipeline.invocation_count - before
        injected = self.pipeline.fault_count - faults_before
        # U1 also executes original Ranking and Mapping tool actions.
        calls = max(calls, attempt.tool_calls + injected)
        self.executor.last_trace["fault_execution"] = {
            "injected_ranking_failures": injected,
            "pipeline_invocations": self.pipeline.invocation_count - before,
            "provider_fault_traffic": 0,
        }
        return replace(attempt, tool_calls=calls)
