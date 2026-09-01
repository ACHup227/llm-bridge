"""``VertexGeminiAdapter``-specific detail: the construction-time ``GOOGLE_CLOUD_PROJECT`` guard
(the adapter-level "never a silent default" invariant) and a round trip against the fake
``google-genai`` client. The shared 5-exception/retry contract lives in
``test_port_contract.py`` — this file only covers what is unique here.
"""

from __future__ import annotations

from typing import cast

import pytest
from google import genai
from google.genai import types

from llm_bridge.adapters.vertex_gemini import VertexGeminiAdapter
from llm_bridge.port import LLMAuthError, LLMError
from tests.conftest import FakeGenaiClient


def _text_response(text: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(text=text)]))]
    )


# --- construction guard ---------------------------------------------------


def test_construction_raises_without_google_cloud_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(LLMAuthError, match="GOOGLE_CLOUD_PROJECT"):
        VertexGeminiAdapter()


def test_construction_raises_even_with_a_client_injected(
    monkeypatch: pytest.MonkeyPatch, fake_genai_client: type[FakeGenaiClient]
) -> None:
    # "Never a silent default backend" is checked at construction regardless of transport — an
    # injected fake must not bypass the guard, only the real network call.
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    client = fake_genai_client([_text_response("hi")])
    with pytest.raises(LLMAuthError):
        VertexGeminiAdapter(client=cast(genai.Client, client))


def test_construction_succeeds_with_env_var_set_against_the_fake_client(
    monkeypatch: pytest.MonkeyPatch, fake_genai_client: type[FakeGenaiClient]
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    client = fake_genai_client([_text_response("hi")])
    adapter = VertexGeminiAdapter(client=cast(genai.Client, client))
    assert adapter.complete("x") == "hi"


def test_construction_succeeds_with_explicit_project_kwarg_and_no_env_var(
    monkeypatch: pytest.MonkeyPatch, fake_genai_client: type[FakeGenaiClient]
) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    client = fake_genai_client([_text_response("hi")])
    adapter = VertexGeminiAdapter(project="explicit-project", client=cast(genai.Client, client))
    assert adapter.complete("x") == "hi"


# --- model resolution (constructor default / LLM_MODEL seam / call-level) ---


def test_default_model_used_when_nothing_overrides_it(
    monkeypatch: pytest.MonkeyPatch, fake_genai_client: type[FakeGenaiClient]
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    client = fake_genai_client([_text_response("hi")])
    VertexGeminiAdapter(client=cast(genai.Client, client)).complete("x")
    assert client.calls[0]["model"] == "gemini-2.5-flash"  # DEFAULT_MODEL


def test_constructor_model_kwarg_overrides_the_default(
    monkeypatch: pytest.MonkeyPatch, fake_genai_client: type[FakeGenaiClient]
) -> None:
    # This is the seam make_llm() uses to apply LLM_MODEL to this adapter.
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    client = fake_genai_client([_text_response("hi")])
    VertexGeminiAdapter(model="gemini-1.5-pro", client=cast(genai.Client, client)).complete("x")
    assert client.calls[0]["model"] == "gemini-1.5-pro"


def test_call_level_model_overrides_the_constructor_model(
    monkeypatch: pytest.MonkeyPatch, fake_genai_client: type[FakeGenaiClient]
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    client = fake_genai_client([_text_response("hi")])
    VertexGeminiAdapter(model="gemini-1.5-pro", client=cast(genai.Client, client)).complete(
        "x", model="gemini-2.0-flash"
    )
    assert client.calls[0]["model"] == "gemini-2.0-flash"


# --- effort no-op + missing-text handling ---------------------------------


def test_effort_is_accepted_and_silently_ignored(
    monkeypatch: pytest.MonkeyPatch, fake_genai_client: type[FakeGenaiClient]
) -> None:
    # Documented no-op (module docstring): accepted for Protocol compliance, never crashes.
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    client = fake_genai_client([_text_response("hi")])
    result = VertexGeminiAdapter(client=cast(genai.Client, client)).complete("x", effort="high")
    assert result == "hi"


def test_no_text_content_raises_llm_error_not_a_silent_empty_string(
    monkeypatch: pytest.MonkeyPatch, fake_genai_client: type[FakeGenaiClient]
) -> None:
    # response.text is None (not an exception) on a safety block — must surface as a real
    # failure, never as "".
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    blocked = types.GenerateContentResponse(
        candidates=[types.Candidate(finish_reason=types.FinishReason.SAFETY)]
    )
    client = fake_genai_client([blocked])
    with pytest.raises(LLMError):
        VertexGeminiAdapter(client=cast(genai.Client, client)).complete("x")
