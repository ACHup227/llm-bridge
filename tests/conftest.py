"""Shared test fakes: a scripted ``CommandRunner`` for the two CLI adapters, and a
``google-genai`` ``Client``-shaped stub for the Vertex adapter — the only two transport seams
every adapter accepts for injection (``runner=`` on ``claude_cli``/``codex_cli``, ``client=`` on
``vertex_gemini``), so one small set of fakes here backs every test file plus the port-contract
suite. No real subprocess and no real HTTP call happens anywhere in this test suite.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, cast

import pytest
from google.genai import types


class FakeCommandRunner:
    """A scripted ``CommandRunner``: returns queued items in order, records every call.

    Shared by ``claude_cli`` and ``codex_cli`` — both declare the identical
    ``Callable[[Sequence[str], str, int], CommandResult]`` shape (their module docstrings call
    this out explicitly as deliberate), so one fake fakes both; pass either module's own
    ``CommandResult`` instances, it does not matter which — only the attributes are read.

    Each queued item is either a result object (returned) or an ``Exception`` instance
    (raised) — the latter is how a test simulates the runner's own timeout path: the real
    ``_subprocess_runner`` raises ``LLMTimeoutError`` itself on ``subprocess.TimeoutExpired``,
    it never returns a result for that case, so the fake mirrors that by raising too.
    """

    def __init__(self, results: Sequence[Any]) -> None:
        self._results: list[Any] = list(results)
        self.calls: list[tuple[list[str], str, int]] = []

    def __call__(self, argv: Sequence[str], stdin_text: str, timeout_s: int) -> Any:
        self.calls.append((list(argv), stdin_text, timeout_s))
        if not self._results:
            # pytest.fail, not `raise AssertionError`: it raises `Failed`, which derives from
            # `BaseException` rather than `Exception` — so an adapter's own `except Exception`
            # (both CLI adapters' generic-failure paths) cannot swallow an under-queued test and
            # relabel it as a plausible-looking LLMError.
            pytest.fail("FakeCommandRunner exhausted: no more queued results")
        item = self._results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def fake_command_runner() -> type[FakeCommandRunner]:
    """Factory fixture: ``runner = fake_command_runner([result_or_exception, ...])``."""
    return FakeCommandRunner


class _FakeModels:
    """Stub for ``genai.Client.models`` — the only attribute ``VertexGeminiAdapter`` reads off
    the client (verified against ``vertex_gemini.py``'s ``_generate``, not assumed)."""

    def __init__(self, results: Sequence[Any]) -> None:
        self._results: list[Any] = list(results)
        self.calls: list[dict[str, Any]] = []

    def generate_content(
        self, *, model: str, contents: str, config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        if not self._results:
            # pytest.fail, not `raise AssertionError`: `VertexGeminiAdapter._generate`'s own
            # `except Exception` (its generic-failure fallback) WOULD catch a plain
            # AssertionError and relabel an under-queued test as a plausible LLMError — pytest's
            # `Failed` derives from `BaseException`, so it passes straight through instead.
            pytest.fail("FakeGenaiClient exhausted: no more queued results")
        item = self._results.pop(0)
        if isinstance(item, Exception):
            raise item
        # `item` is queued as `Any` (either a real response object or an exception) — the cast
        # keeps this method's declared return type exact instead of leaking `Any` into every
        # caller that reads a `GenerateContentResponse` back from it (mypy strict's
        # `warn_return_any`). Runtime behaviour is unaffected either way.
        return cast(types.GenerateContentResponse, item)


class FakeGenaiClient:
    """``google.genai.Client``-shaped stub: implements only ``.models.generate_content``, the
    sole surface the adapter calls. Same queued-result/queued-exception + call-recording pattern
    as ``FakeCommandRunner``, adapted to an attribute (``.models``) instead of a direct call —
    that shape difference is exactly why Vertex needs its own fake rather than reusing that one.
    """

    def __init__(self, results: Sequence[Any]) -> None:
        self.models = _FakeModels(results)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.models.calls


@pytest.fixture
def fake_genai_client() -> type[FakeGenaiClient]:
    """Factory fixture: ``client = fake_genai_client([response_or_exception, ...])``."""
    return FakeGenaiClient


@pytest.fixture
def claude_envelope() -> Callable[..., str]:
    """Factory fixture building a ``claude -p --output-format json`` envelope JSON string —
    the measured shape from ``claude_cli.py``'s module docstring (``type``/``is_error``/
    ``result``, plus ``api_error_status`` when the caller wants status-code classification)."""

    def _build(
        *, result: str = "", is_error: bool = False, api_error_status: int | None = None
    ) -> str:
        envelope: dict[str, Any] = {"type": "result", "is_error": is_error, "result": result}
        if api_error_status is not None:
            envelope["api_error_status"] = api_error_status
        return json.dumps(envelope)

    return _build


@pytest.fixture
def codex_jsonl() -> Callable[..., str]:
    """Factory fixture joining event dicts into codex exec's newline-delimited JSON stdout."""

    def _build(*events: dict[str, Any]) -> str:
        return "\n".join(json.dumps(event) for event in events)

    return _build
