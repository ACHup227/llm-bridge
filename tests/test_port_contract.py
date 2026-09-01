"""The port contract, proven identically against all 3 adapters — not adapter-internal detail.

Each of the 5 ``LLMClient`` exceptions is raised for its corresponding injected failure kind
(auth / quota / parse / timeout / generic-unclassified), and only ``LLMParseError`` ever
triggers the one allowed retry. Argv shape, JSONL event shapes, and other adapter-specific
plumbing stay in each adapter's own test file — this file only asserts the shape every caller of
``LLMClient`` is entitled to rely on, regardless of which provider is behind it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
from google import genai
from google.genai import types
from google.genai.errors import APIError

from llm_bridge.adapters.claude_cli import ClaudeCliAdapter
from llm_bridge.adapters.claude_cli import CommandResult as ClaudeResult
from llm_bridge.adapters.codex_cli import CodexCliAdapter
from llm_bridge.adapters.codex_cli import CommandResult as CodexResult
from llm_bridge.adapters.vertex_gemini import VertexGeminiAdapter
from llm_bridge.port import (
    LLMAuthError,
    LLMClient,
    LLMError,
    LLMParseError,
    LLMQuotaError,
    LLMTimeoutError,
)
from tests.conftest import FakeCommandRunner, FakeGenaiClient

# A message with no substring any adapter's marker list recognises (auth/quota lists checked in
# every adapter's own test file) — the deliberate "nothing matches" case for the generic kind.
_UNRECOGNISED_MESSAGE = "a wholly unrecognised failure sentence"

FAILURE_KINDS = ("auth", "quota", "parse", "timeout", "generic")

_EXPECTED_EXCEPTION: dict[str, type[LLMError]] = {
    "auth": LLMAuthError,
    "quota": LLMQuotaError,
    "parse": LLMParseError,
    "timeout": LLMTimeoutError,
    "generic": LLMError,
}


def _claude_case(
    kind: str,
    fake_command_runner: type[FakeCommandRunner],
    claude_envelope: Callable[..., str],
) -> tuple[LLMClient, FakeCommandRunner]:
    queue: list[object]
    if kind == "auth":
        env = claude_envelope(result="not authenticated", api_error_status=401)
        queue = [ClaudeResult(1, env, "")]
    elif kind == "quota":
        env = claude_envelope(result="usage limit reached", api_error_status=429)
        queue = [ClaudeResult(1, env, "")]
    elif kind == "parse":
        bad = claude_envelope(result="not json")
        queue = [ClaudeResult(0, bad, ""), ClaudeResult(0, bad, "")]
    elif kind == "timeout":
        queue = [LLMTimeoutError("claude -p timed out")]
    else:  # generic
        queue = [ClaudeResult(1, claude_envelope(result=_UNRECOGNISED_MESSAGE, is_error=True), "")]
    runner = fake_command_runner(queue)
    return ClaudeCliAdapter(runner=runner), runner


def _codex_case(
    kind: str,
    fake_command_runner: type[FakeCommandRunner],
    codex_jsonl: Callable[..., str],
) -> tuple[LLMClient, FakeCommandRunner]:
    def failure(message: str) -> str:
        return codex_jsonl(
            {"type": "thread.started"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"type": "error", "message": message}},
            {"type": "turn.failed"},
        )

    queue: list[object]
    if kind == "auth":
        queue = [CodexResult(1, failure("401 Unauthorized"), "")]
    elif kind == "quota":
        queue = [CodexResult(1, failure("rate limit exceeded, try later"), "")]
    elif kind == "parse":
        # Well-formed JSONL, exit 0, but no recognizable agent_message item — the success-path
        # shape is unmeasured (module docstring), so this IS the "reply failed to parse" case.
        no_reply = codex_jsonl({"type": "thread.started"}, {"type": "turn.completed"})
        queue = [CodexResult(0, no_reply, ""), CodexResult(0, no_reply, "")]
    elif kind == "timeout":
        queue = [LLMTimeoutError("codex exec timed out")]
    else:  # generic
        queue = [CodexResult(1, failure(_UNRECOGNISED_MESSAGE), "")]
    runner = fake_command_runner(queue)
    return CodexCliAdapter(runner=runner), runner


def _vertex_case(
    kind: str,
    fake_genai_client: type[FakeGenaiClient],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LLMClient, FakeGenaiClient]:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")

    def text_response(text: str) -> types.GenerateContentResponse:
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=[types.Part(text=text)]))]
        )

    if kind == "auth":
        queue: list[object] = [APIError(401, {"message": "no credentials"})]
    elif kind == "quota":
        queue = [APIError(429, {"message": "quota exceeded"})]
    elif kind == "parse":
        bad = text_response("not json")
        queue = [bad, bad]
    elif kind == "timeout":
        queue = [TimeoutError("deadline exceeded")]
    else:  # generic
        queue = [RuntimeError(_UNRECOGNISED_MESSAGE)]
    client = fake_genai_client(queue)
    adapter = VertexGeminiAdapter(client=cast(genai.Client, client))
    return adapter, client


@pytest.mark.parametrize("adapter_name", ["claude-cli", "codex-cli", "vertex-gemini"])
@pytest.mark.parametrize("kind", FAILURE_KINDS)
def test_exception_taxonomy_and_retry_policy(
    adapter_name: str,
    kind: str,
    fake_command_runner: type[FakeCommandRunner],
    fake_genai_client: type[FakeGenaiClient],
    claude_envelope: Callable[..., str],
    codex_jsonl: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client: LLMClient
    fake: FakeCommandRunner | FakeGenaiClient
    if adapter_name == "claude-cli":
        client, fake = _claude_case(kind, fake_command_runner, claude_envelope)
    elif adapter_name == "codex-cli":
        client, fake = _codex_case(kind, fake_command_runner, codex_jsonl)
    else:
        client, fake = _vertex_case(kind, fake_genai_client, monkeypatch)

    expected = _EXPECTED_EXCEPTION[kind]
    with pytest.raises(expected) as exc_info:
        client.complete_json("prompt")
    if kind == "generic":
        # LLMError is also the base of the other 4 — pin the EXACT type here, else a bug that
        # mis-raises e.g. LLMAuthError for an unrecognised failure would slip past `pytest.raises`.
        assert type(exc_info.value) is LLMError

    expected_calls = 2 if kind == "parse" else 1
    assert len(fake.calls) == expected_calls
