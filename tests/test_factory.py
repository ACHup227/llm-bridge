"""``make_llm()``: ``LLM_PROVIDER`` selects the adapter class, ``LLM_MODEL`` reaches the chosen
adapter's constructor default, and an unknown provider value raises loudly at construction
rather than failing confusingly on the first real call.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_bridge.adapters.claude_cli import ClaudeCliAdapter
from llm_bridge.adapters.codex_cli import CodexCliAdapter
from llm_bridge.adapters.vertex_gemini import VertexGeminiAdapter
from llm_bridge.factory import make_llm


class _DummyGenaiClient:
    """Stands in for ``google.genai.Client``'s constructor in these tests — never invoked for
    real, so construction never tries to resolve real GCP credentials."""

    def __init__(self, **kwargs: Any) -> None:
        del kwargs


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every test starts from a clean slate — a leaked LLM_PROVIDER/LLM_MODEL from the real
    # environment must never silently change which adapter a test observes.
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)


# --- provider selection ----------------------------------------------------


def test_defaults_to_claude_cli_when_llm_provider_is_unset() -> None:
    assert isinstance(make_llm(), ClaudeCliAdapter)


def test_llm_provider_claude_cli_selects_the_claude_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    assert isinstance(make_llm(), ClaudeCliAdapter)


def test_llm_provider_codex_cli_selects_the_codex_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "codex-cli")
    assert isinstance(make_llm(), CodexCliAdapter)


def test_llm_provider_vertex_gemini_selects_the_vertex_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "vertex-gemini")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setattr("llm_bridge.adapters.vertex_gemini.genai.Client", _DummyGenaiClient)
    assert isinstance(make_llm(), VertexGeminiAdapter)


def test_unknown_provider_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai-direct")
    with pytest.raises(ValueError, match="openai-direct"):
        make_llm()


# --- LLM_MODEL reaches the chosen adapter's constructor default ------------


def test_llm_model_override_reaches_claude_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "claude-opus-4-8")
    client = make_llm()
    assert isinstance(client, ClaudeCliAdapter)
    assert client._model == "claude-opus-4-8"


def test_llm_model_override_reaches_codex_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "codex-cli")
    monkeypatch.setenv("LLM_MODEL", "gpt-5-codex")
    client = make_llm()
    assert isinstance(client, CodexCliAdapter)
    assert client._model == "gpt-5-codex"


def test_llm_model_override_reaches_vertex_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "vertex-gemini")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("LLM_MODEL", "gemini-1.5-pro")
    monkeypatch.setattr("llm_bridge.adapters.vertex_gemini.genai.Client", _DummyGenaiClient)
    client = make_llm()
    assert isinstance(client, VertexGeminiAdapter)
    assert client._model == "gemini-1.5-pro"


def test_no_llm_model_leaves_the_adapter_on_its_own_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_llm()
    assert isinstance(client, ClaudeCliAdapter)
    assert client._model == "sonnet"  # ClaudeCliAdapter.DEFAULT_MODEL, untouched


def test_empty_llm_model_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # "" is what os.environ.get returns for a variable set-but-empty — must not shadow the
    # adapter's own default with an empty model string.
    monkeypatch.setenv("LLM_MODEL", "")
    client = make_llm()
    assert isinstance(client, ClaudeCliAdapter)
    assert client._model == "sonnet"
