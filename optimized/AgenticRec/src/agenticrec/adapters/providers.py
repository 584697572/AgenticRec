"""Allowlisted construction of live provider adapters."""

from .llm import ChatAdapter
from .openai_compatible import OpenAICompatibleTransport
from ..config import LLMBudget
from ..runtime.budget import BudgetLedger
from ..runtime.secrets import LocalKeyError, read_api_key


_PROVIDER_BASES = {"deepseek": "https://api.deepseek.com"}


def build_live_chat_adapter(config, *, api_key=None, ledger=None, sender=None, seed=None):
    """Build the configured provider without making a network request."""
    if not isinstance(config, LLMBudget):
        raise TypeError("config must be LLMBudget")
    if config.provider not in _PROVIDER_BASES:
        raise ValueError("unsupported live provider")
    api_key = read_api_key() if api_key is None else api_key
    if not isinstance(api_key, str) or not api_key.strip():
        raise LocalKeyError("OPENAI_API_KEY is missing")
    if ledger is not None and not isinstance(ledger, BudgetLedger):
        raise TypeError("ledger must be BudgetLedger or null")
    transport = OpenAICompatibleTransport(
        api_base=_PROVIDER_BASES[config.provider],
        api_key=api_key.strip(),
        sender=sender,
        seed=seed,
    )
    return ChatAdapter(transport, config, ledger=ledger)
