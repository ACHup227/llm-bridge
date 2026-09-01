"""Adapter: Gemini on Vertex AI, via the ``google-genai`` SDK's Vertex transport.

Verified against ``google-genai==1.75.0`` (installed 2026-09-01 for this file) — every SDK
shape referenced below (``errors.APIError``/``ClientError``/``ServerError``,
``HttpOptions.timeout`` in milliseconds, ``GenerateContentConfig.http_options`` as a per-call
override, ``ThinkingConfig.thinking_budget``) was read from the installed package's source,
not assumed. Only ``DEFAULT_MODEL`` below is unverified — it needs a live Vertex project.

Construction, not the first call, enforces ``GOOGLE_CLOUD_PROJECT`` — the adapter-level
guarantee that Vertex is never a silent default backend (see llm-bridge README and the
tldr-filter / hots_replay_coach invariant amendments this package's plan makes).

The constructor also accepts ``model=`` (mirrors the other two adapters' constructor default,
so ``make_llm()``'s ``LLM_MODEL`` env override reaches this adapter the same way) and
``client=`` (the test seam — same role as ``runner`` on the CLI adapters: inject a
``google.genai.Client``-shaped fake so tests never construct the real SDK client, which would
otherwise try to resolve real GCP credentials).

**Effort mapping — documented no-op, not an oversight.** ``ThinkingConfig.thinking_budget``
is an integer token count with model-dependent defaults and ranges; there is no non-lossy way
to turn a level string (``"low"``/``"medium"``/``"high"``) into a budget without inventing a
mapping nobody asked for. Per the plan's decision, a non-``None`` ``effort`` is accepted for
``LLMClient`` Protocol compliance and silently dropped, with a DEBUG-level log so a caller who
expected it to matter can find out why it didn't.

  Flag for whoever revisits this: the same installed SDK also exposes
  ``ThinkingConfig.thinking_level``, a ``ThinkingLevel`` enum with members ``LOW``/``MEDIUM``/
  ``HIGH`` (plus ``MINIMAL`` and ``THINKING_LEVEL_UNSPECIFIED``) — a genuinely non-lossy string
  target for 3 of the 4 accepted ``effort`` values (``codex_cli``'s provisional set is
  ``none|low|medium|high``; there is no ``NONE`` member here, so the 4th value would still need
  a decision). The plan's no-op call was made without knowing this field existed. It is left
  unimplemented here anyway, because doing it changes the port's cross-adapter effort contract
  (``test_port_contract.py`` asserts uniform behaviour) and that is a plan-level decision, not
  an adapter-file one — raise it against the plan before wiring it.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Final

from google import genai
from google.genai import types
from google.genai.errors import APIError

from llm_bridge.port import (
    LLMAuthError,
    LLMError,
    LLMParseError,
    LLMQuotaError,
    LLMTimeoutError,
)

log = logging.getLogger(__name__)

# Best current knowledge only (training cutoff 2026-01; this file was written 2026-09-01, so a
# newer GA model may exist by now). Not verified against a live Vertex project — no GCP
# credentials were available while writing this adapter. Confirm gemini-2.5-flash is still GA
# in the target project/region before relying on it in production, and update if it has moved.
DEFAULT_MODEL: Final = "gemini-2.5-flash"

# No measured floor exists yet (unlike codex_cli's measured 180s — see the plan). Vertex is an
# HTTP API, not a subprocess with its own retry warm-up, so there is no equivalent floor to
# measure the same way; 120s is a starting guess, not a measurement. Revisit once this adapter
# has a real smoke run.
DEFAULT_TIMEOUT_S: Final = 120


class VertexGeminiAdapter:
    """``LLMClient`` over Vertex AI Gemini, via ``google.genai.Client(vertexai=True, ...)``.

    Never a default provider for any caller in this workspace — always opt-in, gated on
    ``GOOGLE_CLOUD_PROJECT`` being set explicitly (constructor guard below, not a first-call
    check), because it is the one adapter with a real per-token bill behind it.
    """

    def __init__(
        self,
        *,
        project: str | None = None,
        location: str | None = None,
        model: str | None = None,
        client: genai.Client | None = None,
    ) -> None:
        resolved_project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not resolved_project:
            # LLMAuthError, not a bare RuntimeError/ValueError: it subclasses LLMError, so a
            # caller that already catches the port's taxonomy around construction-and-first-call
            # (a common pattern for a factory like make_llm()) catches this too. Its docstring —
            # "credential missing... not retryable; human-owned" — is exactly the situation: a
            # human must set the env var, no retry fixes it.
            #
            # Checked even when `client` is injected: this guard is the adapter-level guarantee
            # that vertex is never a silent default (module docstring), not a real-client-only
            # concern — a test exercising the fake client still exercises this invariant.
            raise LLMAuthError(
                "GOOGLE_CLOUD_PROJECT is not set. vertex-gemini requires an explicit GCP "
                "project and is never a silent default backend for llm-bridge — set the env "
                "var or pass project= explicitly."
            )
        resolved_location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        # `model` mirrors the other two adapters' constructor default (make_llm()'s LLM_MODEL
        # seam); `client` is the test seam — same role as `runner` on the CLI adapters, just
        # named for what it replaces here (an HTTP-backed SDK client, not a subprocess runner).
        # Typed as the concrete `genai.Client`, not a Protocol: a fake passed in tests satisfies
        # it structurally at runtime but needs `typing.cast` at the call site to type-check,
        # which keeps `self._client`'s static type exact and avoids an `Any` leak into every
        # method below that reads from it (mypy strict's `warn_return_any`).
        self._model = model
        self._client = (
            client
            if client is not None
            else genai.Client(vertexai=True, project=resolved_project, location=resolved_location)
        )

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        effort: str | None = None,
        timeout_s: int | None = None,
    ) -> str:
        """Run ``prompt``; return the model's raw text reply, unparsed and un-de-fenced."""
        response = self._generate(prompt, model=model, effort=effort, timeout_s=timeout_s)
        return _require_text(response)

    def complete_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        effort: str | None = None,
        timeout_s: int | None = None,
    ) -> Any:
        """Run ``prompt``; return the reply parsed as JSON, with one retry on parse failure.

        Same policy as claude_cli/codex_cli: de-fence, parse, and on failure retry the whole
        call once more before giving up as ``LLMParseError``. Auth/quota/timeout failures are
        not retried here — they propagate immediately from ``_generate``, same as ``complete``.
        """
        last_error: Exception | None = None
        for _ in range(2):
            response = self._generate(prompt, model=model, effort=effort, timeout_s=timeout_s)
            text = _require_text(response)
            try:
                return json.loads(_strip_code_fence(text))
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
        raise LLMParseError(f"vertex-gemini reply was not valid JSON after 1 retry: {last_error}")

    def _generate(
        self,
        prompt: str,
        *,
        model: str | None,
        effort: str | None,
        timeout_s: int | None,
    ) -> types.GenerateContentResponse:
        if effort is not None:
            log.debug(
                "vertex-gemini: effort=%r ignored (documented no-op) — thinking_budget is an "
                "integer token count, not a level string; no mapping without real need. See "
                "this adapter's module docstring for the thinking_level field this could use "
                "if the plan is amended.",
                effort,
            )
        resolved_timeout_s = timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S
        # Config is built outside the try below: a wiring mistake here (e.g. a bad field name)
        # is our bug, not an SDK/API failure, and must not be relabelled as a generic LLMError.
        config = types.GenerateContentConfig(
            http_options=types.HttpOptions(timeout=resolved_timeout_s * 1000),
        )
        try:
            return self._client.models.generate_content(
                model=model or self._model or DEFAULT_MODEL,
                contents=prompt,
                config=config,
            )
        except APIError as exc:
            raise _map_api_error(exc) from exc
        except Exception as exc:
            # The SDK's default transport is httpx; a connection/read timeout raises
            # httpx.TimeoutException (or a subclass) at the transport layer, before any HTTP
            # response exists — so it never becomes an APIError and must be caught here instead.
            # Matched by class name, not `except httpx.TimeoutException`, because httpx is an
            # indirect dependency (pulled in by google-genai) and pinning to its exception type
            # would couple this adapter to a transport google-genai is free to swap.
            if _looks_like_timeout(exc):
                raise LLMTimeoutError(str(exc)) from exc
            raise LLMError(str(exc)) from exc


def _require_text(response: types.GenerateContentResponse) -> str:
    """``response.text`` is ``None`` (not an exception) on a safety block or empty candidates —
    verified against ``GenerateContentResponse._get_text`` in the installed SDK. Treating that
    as an empty string would silently swallow a blocked completion in ``complete()``, and would
    mislabel it as a JSON parse failure in ``complete_json()``. Raise it as what it is instead.
    """
    text = response.text
    if text is not None:
        return text
    finish_reason = None
    if response.candidates:
        finish_reason = response.candidates[0].finish_reason
    raise LLMError(
        "vertex-gemini returned no text content "
        f"(finish_reason={finish_reason!r}, prompt_feedback={response.prompt_feedback!r})"
    )


def _strip_code_fence(text: str) -> str:
    """De-fence a ```` ```json ... ``` ```` (or bare ``` ``` ```) wrapped reply. Adapter-local,
    same pattern as tldr's ``_strip_wrapping_fence`` — no shared util exists yet in this
    package, per the plan (``complete_json`` de-fences per-adapter, not on the port).
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped
    body = lines[1:]  # drop the opening ``` or ```json line
    if body and body[-1].strip() == "```":
        body = body[:-1]
    return "\n".join(body).strip()


def _map_api_error(exc: APIError) -> LLMError:
    """Status-code first, ``status`` string fallback — same discipline as the other two
    adapters. Field names (``code``, ``status``) verified against ``errors.APIError.__init__``
    in the installed SDK, not assumed.
    """
    if exc.code in (401, 403):
        return LLMAuthError(str(exc))
    if exc.code == 429 or exc.status == "RESOURCE_EXHAUSTED":
        return LLMQuotaError(str(exc))
    if exc.code == 504 or exc.status == "DEADLINE_EXCEEDED":
        return LLMTimeoutError(str(exc))
    return LLMError(str(exc))


def _looks_like_timeout(exc: Exception) -> bool:
    name = type(exc).__name__
    return isinstance(exc, TimeoutError) or "Timeout" in name or "DeadlineExceeded" in name
