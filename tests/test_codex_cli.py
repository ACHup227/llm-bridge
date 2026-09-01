"""``CodexCliAdapter``-specific detail: argv shape (incl. ``-c model_reasoning_effort=``) and
JSONL parsing, exercised against the MEASURED failure-path event shape from the plan
(``thread.started`` -> ``turn.started`` -> optional reconnect-noise ``error`` events ->
``item.completed{item.type=="error"}`` -> ``turn.failed``). The shared 5-exception/retry
contract lives in ``test_port_contract.py`` — this file only covers what is unique here.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from llm_bridge.adapters.codex_cli import CodexCliAdapter, CommandResult
from llm_bridge.port import LLMAuthError, LLMParseError
from tests.conftest import FakeCommandRunner


def _ok(stdout: str) -> CommandResult:
    return CommandResult(returncode=0, stdout=stdout, stderr="")


def _agent_message_event(*, text: str = "hi") -> dict[str, object]:
    return {"type": "item.completed", "item": {"type": "agent_message", "text": text}}


# --- argv shape + stdin wiring ------------------------------------------


def test_argv_shape_and_default_model(
    fake_command_runner: type[FakeCommandRunner], codex_jsonl: Callable[..., str]
) -> None:
    success = codex_jsonl(_agent_message_event())
    runner = fake_command_runner([_ok(success)])
    CodexCliAdapter(runner=runner).complete("x")
    argv, stdin_text, _timeout = runner.calls[0]
    assert argv == [
        "codex",
        "exec",
        "-s",
        "read-only",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--json",
        "-m",
        "sonnet",  # DEFAULT_MODEL
        "-",
    ]
    assert stdin_text == "x"


def test_effort_appends_model_reasoning_effort_flag(
    fake_command_runner: type[FakeCommandRunner], codex_jsonl: Callable[..., str]
) -> None:
    success = codex_jsonl(_agent_message_event())
    runner = fake_command_runner([_ok(success)])
    CodexCliAdapter(runner=runner).complete("x", effort="high")
    argv, _stdin, _timeout = runner.calls[0]
    assert "-c" in argv
    assert argv[argv.index("-c") + 1] == "model_reasoning_effort=high"
    # The prompt marker stays the LAST element even with the extra pair inserted before it.
    assert argv[-1] == "-"


def test_effort_flag_absent_when_not_given(
    fake_command_runner: type[FakeCommandRunner], codex_jsonl: Callable[..., str]
) -> None:
    success = codex_jsonl(_agent_message_event())
    runner = fake_command_runner([_ok(success)])
    CodexCliAdapter(runner=runner).complete("x")
    argv, _stdin, _timeout = runner.calls[0]
    assert "-c" not in argv


def test_explicit_model_overrides_constructor_default(
    fake_command_runner: type[FakeCommandRunner], codex_jsonl: Callable[..., str]
) -> None:
    success = codex_jsonl(_agent_message_event())
    runner = fake_command_runner([_ok(success)])
    CodexCliAdapter(runner=runner, model="gpt-5-codex").complete("x", model="o3")
    argv, _stdin, _timeout = runner.calls[0]
    assert argv[argv.index("-m") + 1] == "o3"


# --- success-path JSONL parsing (PROVISIONAL shape, per module docstring) ----


def test_success_path_reads_agent_message_text(
    fake_command_runner: type[FakeCommandRunner], codex_jsonl: Callable[..., str]
) -> None:
    stdout = codex_jsonl(
        {"type": "thread.started"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "PONG"}},
        {"type": "turn.completed"},
    )
    runner = fake_command_runner([_ok(stdout)])
    assert CodexCliAdapter(runner=runner).complete("x") == "PONG"


def test_success_path_falls_back_to_content_field(
    fake_command_runner: type[FakeCommandRunner], codex_jsonl: Callable[..., str]
) -> None:
    # `text` missing, `content` present — the fallback guess (module docstring).
    stdout = codex_jsonl(
        {"type": "item.completed", "item": {"type": "agent_message", "content": "PONG"}}
    )
    runner = fake_command_runner([_ok(stdout)])
    assert CodexCliAdapter(runner=runner).complete("x") == "PONG"


def test_non_json_stdout_lines_are_skipped_not_fatal(
    fake_command_runner: type[FakeCommandRunner], codex_jsonl: Callable[..., str]
) -> None:
    events = codex_jsonl(_agent_message_event(text="PONG"))
    stdout = "not json at all\n" + events + "\nalso not json"
    runner = fake_command_runner([_ok(stdout)])
    assert CodexCliAdapter(runner=runner).complete("x") == "PONG"


def test_complete_json_parses_and_de_fences_the_reply(
    fake_command_runner: type[FakeCommandRunner], codex_jsonl: Callable[..., str]
) -> None:
    stdout = codex_jsonl(_agent_message_event(text='```json\n{"a": 1}\n```'))
    runner = fake_command_runner([_ok(stdout)])
    assert CodexCliAdapter(runner=runner).complete_json("x") == {"a": 1}


# --- measured failure-path JSONL shape (plan Phase 1 §2) -----------------


def test_measured_failure_shape_with_reconnect_noise_classifies_as_auth(
    fake_command_runner: type[FakeCommandRunner], codex_jsonl: Callable[..., str]
) -> None:
    # thread.started -> turn.started -> [error]* (reconnect noise, ignorable) ->
    # item.completed{item.type=="error"} -> turn.failed — the sequence measured against two real
    # 401s (codex was unauthenticated during the build; module docstring).
    stdout = codex_jsonl(
        {"type": "thread.started"},
        {"type": "turn.started"},
        {"type": "error", "error": {"message": "reconnecting, attempt 1"}},
        {"type": "error", "error": {"message": "reconnecting, attempt 2"}},
        {"type": "item.completed", "item": {"type": "error", "message": "401 Unauthorized"}},
        {"type": "turn.failed"},
    )
    runner = fake_command_runner([_ok(stdout)])
    with pytest.raises(LLMAuthError) as exc_info:
        CodexCliAdapter(runner=runner).complete("x")
    assert "401" in str(exc_info.value)
    assert len(runner.calls) == 1


def test_error_item_message_preferred_over_a_thin_turn_failed(
    fake_command_runner: type[FakeCommandRunner], codex_jsonl: Callable[..., str]
) -> None:
    # turn.failed carries no message of its own here — the real "401" text lives on the error
    # item, and that is the one the message must come from (module docstring rationale).
    stdout = codex_jsonl(
        {"type": "item.completed", "item": {"type": "error", "message": "401 Unauthorized"}},
        {"type": "turn.failed"},
    )
    runner = fake_command_runner([_ok(stdout)])
    with pytest.raises(LLMAuthError) as exc_info:
        CodexCliAdapter(runner=runner).complete("x")
    assert "401" in str(exc_info.value)


def test_no_recognizable_success_event_is_a_parse_error(
    fake_command_runner: type[FakeCommandRunner], codex_jsonl: Callable[..., str]
) -> None:
    # Exit 0, well-formed JSONL, but no failure event AND no agent_message item — the success
    # shape is unmeasured (module docstring), so this is a parse failure, not a silent empty ok.
    # complete() never retries (only complete_json() does), so one queued result is consumed.
    stdout = codex_jsonl({"type": "thread.started"}, {"type": "turn.completed"})
    runner = fake_command_runner([_ok(stdout)])
    with pytest.raises(LLMParseError):
        CodexCliAdapter(runner=runner).complete("x")
    assert len(runner.calls) == 1
