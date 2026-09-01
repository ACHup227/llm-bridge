"""Port: the provider-agnostic LLM contract every adapter implements and every caller depends on.

``LLMClient`` is a structural ``Protocol`` — adapters (claude-cli, codex-cli, vertex-gemini) satisfy
it by shape, not by inheritance. ``complete`` returns the raw text reply with no de-fencing (that
stays the caller's job, e.g. ``_strip_wrapping_fence`` in meeting-summerizer); ``complete_json``
does the de-fencing plus defensive parsing and one retry on a parse failure, per adapter.

``model=None`` and ``timeout_s=None`` mean "the adapter's own default", never a global default
picked by this package — see the migration-contract note in README.md: every existing call site
keeps naming its own model explicitly, so wiring in this package changes no caller's behaviour by
itself. ``effort=None`` means "don't pass a reasoning-effort hint"; how a non-``None`` value maps
onto each backend (``--effort``, ``model_reasoning_effort``, a no-op) is documented on the adapter
that receives it, not here — the port only fixes the shape of the call, not what each value means.

The exception taxonomy lives here on the port, not on any one adapter, so callers depend on the
port for the failure contract they must handle (e.g. pausing on ``LLMQuotaError``) rather than on a
concrete adapter. Every adapter classifies its own failures into these five names — status-code
first, substring fallback, generic ``LLMError`` only as the last resort.
"""

from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        effort: str | None = None,
        timeout_s: int | None = None,
    ) -> str:
        """Run ``prompt``; return the model's raw text reply, unparsed and un-de-fenced."""
        ...

    def complete_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        effort: str | None = None,
        timeout_s: int | None = None,
    ) -> Any:
        """Run ``prompt``; return the model's reply parsed as a JSON value (genuinely ``Any``)."""
        ...


class LLMError(Exception):
    """Base for every ``LLMClient`` failure — the port's failure contract."""


class LLMAuthError(LLMError):
    """Auth failed — token/credential missing, invalid, or expired. Not retryable; human-owned."""


class LLMQuotaError(LLMError):
    """The rate/quota window is exhausted. Callers should back off and resume later, not retry
    inline."""


class LLMParseError(LLMError):
    """The CLI envelope or the model reply was not valid JSON, even after the one allowed retry."""


class LLMTimeoutError(LLMError):
    """The completion exceeded its timeout."""
