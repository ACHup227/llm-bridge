"""llm_bridge — provider-agnostic LLM port, adapters and factory. See README.md.

Public surface: the ``LLMClient`` Protocol, the 5-exception failure taxonomy, and ``make_llm()``.
Adapter classes are importable from ``llm_bridge.adapters.*`` directly for a caller that wants to
construct one explicitly instead of going through the env-driven factory (e.g. a smoke test).
"""

from __future__ import annotations

from llm_bridge.factory import make_llm
from llm_bridge.port import (
    LLMAuthError,
    LLMClient,
    LLMError,
    LLMParseError,
    LLMQuotaError,
    LLMTimeoutError,
)

__all__ = [
    "LLMAuthError",
    "LLMClient",
    "LLMError",
    "LLMParseError",
    "LLMQuotaError",
    "LLMTimeoutError",
    "make_llm",
]
