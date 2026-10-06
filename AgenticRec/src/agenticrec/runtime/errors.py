"""Typed runtime failures used to make retry decisions explicit."""


class LLMRuntimeError(RuntimeError):
    """Base class for controlled LLM runtime failures."""


class LiveCallsDisabled(LLMRuntimeError):
    """Raised before transport when paid/live calls are not authorized."""


class BudgetExceeded(LLMRuntimeError):
    """Raised before transport when a configured budget would be exceeded."""


class DeadlineExceeded(LLMRuntimeError):
    """Raised before a new attempt when the shared round deadline expired."""


class ResponseValidationError(LLMRuntimeError):
    """Raised for malformed JSON or a response that violates its schema."""


class LLMTransportError(LLMRuntimeError):
    """Provider/connection failure with auditable retry classification."""

    def __init__(self, message, *, status_code=None, error_type=None, retryable=None):
        super().__init__(message)
        if status_code is not None and (type(status_code) is not int or status_code < 100):
            raise ValueError("status_code must be an HTTP status integer or null")
        if error_type is not None and not isinstance(error_type, str):
            raise ValueError("error_type must be a string or null")
        if retryable is not None and type(retryable) is not bool:
            raise ValueError("retryable must be boolean or null")
        self.status_code = status_code
        self.error_type = error_type
        self.retryable = retryable

    def should_retry(self):
        if self.retryable is not None:
            return self.retryable
        if self.status_code is not None:
            return self.status_code in {408, 425, 429, 500, 502, 503, 504}
        return self.error_type in {"connection", "timeout"}
