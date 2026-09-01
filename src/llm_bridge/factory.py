"""Factory: ``make_llm()`` — the one place a caller picks a concrete adapter by name.

Reads ``LLM_PROVIDER`` from the environment (default ``"claude-cli"``, the subscription-only,
zero-marginal-cost default every project in this workspace should stay on unless a caller
opts into something else on purpose — see the plan's Alignment Summary). ``vertex-gemini`` is
never picked by a missing env var; it is only ever reached by setting ``LLM_PROVIDER`` to it
explicitly, and its own constructor separately refuses without ``GOOGLE_CLOUD_PROJECT`` (belt
and suspenders — a factory bug here is not the last line of defence).

``LLM_MODEL``, when set, overrides the CHOSEN adapter's own default model — passed through as
that adapter's constructor ``model=`` kwarg, never a bridge-wide default (see ``port.py`` and
``README.md``'s migration contract). This only moves what the adapter's *default* is; a caller
that passes its own explicit ``model=`` on a ``complete()``/``complete_json()`` call still wins,
because that argument is resolved per-call, after construction, by the adapter itself.
"""

from __future__ import annotations

import os
from typing import Final

from llm_bridge.adapters.claude_cli import ClaudeCliAdapter
from llm_bridge.adapters.codex_cli import CodexCliAdapter
from llm_bridge.adapters.vertex_gemini import VertexGeminiAdapter
from llm_bridge.port import LLMClient

DEFAULT_PROVIDER: Final = "claude-cli"

_KNOWN_PROVIDERS: Final = ("claude-cli", "codex-cli", "vertex-gemini")


def make_llm() -> LLMClient:
    """Build the ``LLMClient`` adapter selected by ``LLM_PROVIDER`` (default ``"claude-cli"``).

    Raises ``ValueError`` on an ``LLM_PROVIDER`` value that names none of the three known
    adapters — fail loud at construction, not with a confusing failure on the first call.
    """
    provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
    model = os.environ.get("LLM_MODEL") or None  # "" counts as unset, same as absent

    if provider == "claude-cli":
        return ClaudeCliAdapter() if model is None else ClaudeCliAdapter(model=model)
    if provider == "codex-cli":
        return CodexCliAdapter() if model is None else CodexCliAdapter(model=model)
    if provider == "vertex-gemini":
        return VertexGeminiAdapter() if model is None else VertexGeminiAdapter(model=model)

    raise ValueError(f"unknown LLM_PROVIDER {provider!r} — expected one of {_KNOWN_PROVIDERS}")
