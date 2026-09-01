"""``ClaudeCliAdapter``-specific detail: argv shape, stdin wiring, the isolation flags,
``_json_after_noise`` tolerance and fence-stripping. The shared 5-exception/retry contract lives
in ``test_port_contract.py`` — this file only covers what is unique to this adapter.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from llm_bridge.adapters.claude_cli import ClaudeCliAdapter, CommandResult
from llm_bridge.port import LLMParseError
from tests.conftest import FakeCommandRunner


def _ok(stdout: str) -> CommandResult:
    return CommandResult(returncode=0, stdout=stdout, stderr="")


# --- argv shape + stdin wiring ------------------------------------------


def test_argv_shape_and_default_model(
    fake_command_runner: type[FakeCommandRunner], claude_envelope: Callable[..., str]
) -> None:
    runner = fake_command_runner([_ok(claude_envelope(result="hi"))])
    ClaudeCliAdapter(runner=runner).complete("x")
    argv, _stdin, _timeout = runner.calls[0]
    assert argv[:2] == ["claude", "-p"]
    assert "--output-format" in argv
    assert argv[argv.index("--output-format") + 1] == "json"
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "sonnet"  # DEFAULT_MODEL


def test_prompt_is_fed_on_stdin_not_argv(
    fake_command_runner: type[FakeCommandRunner], claude_envelope: Callable[..., str]
) -> None:
    # The whole reason for this shape: stdin is capped at 10MB, argv at ~128KB on Windows.
    runner = fake_command_runner([_ok(claude_envelope(result="hi"))])
    ClaudeCliAdapter(runner=runner).complete("a very long prompt body")
    argv, stdin_text, _timeout = runner.calls[0]
    assert stdin_text == "a very long prompt body"
    assert not any("a very long prompt body" in arg for arg in argv)


def test_safe_mode_and_empty_tools_are_always_present(
    fake_command_runner: type[FakeCommandRunner], claude_envelope: Callable[..., str]
) -> None:
    # The flagged runtime-behaviour change (module docstring): every completion is tool-free.
    runner = fake_command_runner([_ok(claude_envelope(result="hi"))])
    ClaudeCliAdapter(runner=runner).complete("x")
    argv, _stdin, _timeout = runner.calls[0]
    assert "--safe-mode" in argv
    assert argv[argv.index("--tools") + 1] == ""


def test_explicit_model_overrides_constructor_default(
    fake_command_runner: type[FakeCommandRunner], claude_envelope: Callable[..., str]
) -> None:
    runner = fake_command_runner([_ok(claude_envelope(result="hi"))])
    ClaudeCliAdapter(runner=runner, model="opus").complete("x", model="claude-opus-4-8")
    argv, _stdin, _timeout = runner.calls[0]
    assert argv[argv.index("--model") + 1] == "claude-opus-4-8"


def test_effort_flag_passed_through_when_given(
    fake_command_runner: type[FakeCommandRunner], claude_envelope: Callable[..., str]
) -> None:
    runner = fake_command_runner([_ok(claude_envelope(result="hi"))])
    ClaudeCliAdapter(runner=runner).complete("x", effort="high")
    argv, _stdin, _timeout = runner.calls[0]
    assert argv[argv.index("--effort") + 1] == "high"


def test_effort_flag_absent_when_not_given(
    fake_command_runner: type[FakeCommandRunner], claude_envelope: Callable[..., str]
) -> None:
    runner = fake_command_runner([_ok(claude_envelope(result="hi"))])
    ClaudeCliAdapter(runner=runner).complete("x")
    argv, _stdin, _timeout = runner.calls[0]
    assert "--effort" not in argv


def test_call_level_timeout_overrides_constructor_default(
    fake_command_runner: type[FakeCommandRunner], claude_envelope: Callable[..., str]
) -> None:
    runner = fake_command_runner([_ok(claude_envelope(result="hi"))])
    ClaudeCliAdapter(runner=runner, timeout_s=180).complete("x", timeout_s=30)
    _argv, _stdin, timeout_s = runner.calls[0]
    assert timeout_s == 30


# --- envelope unwrapping -------------------------------------------------


def test_complete_returns_raw_result_unparsed_and_unfenced(
    fake_command_runner: type[FakeCommandRunner], claude_envelope: Callable[..., str]
) -> None:
    fenced = '```json\n{"kept": true}\n```'
    runner = fake_command_runner([_ok(claude_envelope(result=fenced))])
    result = ClaudeCliAdapter(runner=runner).complete("x")
    assert result == fenced  # complete() never de-fences — that stays the caller's job


def test_complete_json_strips_markdown_code_fence(
    fake_command_runner: type[FakeCommandRunner], claude_envelope: Callable[..., str]
) -> None:
    fenced = '```json\n{"kept": true}\n```'
    runner = fake_command_runner([_ok(claude_envelope(result=fenced))])
    assert ClaudeCliAdapter(runner=runner).complete_json("x") == {"kept": True}


def test_complete_json_strips_code_fence_with_no_closing_line(
    fake_command_runner: type[FakeCommandRunner], claude_envelope: Callable[..., str]
) -> None:
    # A truncated model reply can open a ```json fence and never close it.
    fenced = '```json\n{"kept": true}'
    runner = fake_command_runner([_ok(claude_envelope(result=fenced))])
    assert ClaudeCliAdapter(runner=runner).complete_json("x") == {"kept": True}


# --- _json_after_noise tolerance (hots S23d merge) -----------------------


def test_banner_before_the_envelope_is_tolerated(
    fake_command_runner: type[FakeCommandRunner], claude_envelope: Callable[..., str]
) -> None:
    # An untrusted workspace makes `claude -p` print a warning to stdout AHEAD of the JSON
    # envelope. A strict whole-string parse would choke on it and misreport a parse failure.
    banner = "Ignoring 5 permissions.allow entries — this workspace has not been trusted\n"
    noisy = banner + claude_envelope(result="[1, 2, 3]")
    runner = fake_command_runner([_ok(noisy)])
    assert ClaudeCliAdapter(runner=runner).complete_json("x") == [1, 2, 3]


def test_genuinely_non_json_stdout_still_raises_after_the_noise_tolerance(
    fake_command_runner: type[FakeCommandRunner],
) -> None:
    # Tolerance is "skip leading noise", not "accept anything". complete() never retries (only
    # complete_json() does), so exactly one queued result is consumed.
    runner = fake_command_runner([_ok("Ignoring 5 permissions.allow entries\nno json here at all")])
    with pytest.raises(LLMParseError):
        ClaudeCliAdapter(runner=runner).complete("x")
    assert len(runner.calls) == 1


# --- the model's OWN reply is parsed strictly, never noise-scanned -------
#
# The tolerant scanner (_json_after_noise) exists for CLI-produced bytes ahead of the outer
# envelope — see the two tests above. It must NEVER be applied to the model's own reply text
# (the envelope's `result` string): that is untrusted content the model wrote, and scanning it
# for "a plausible JSON substring" would let an incidental JSON-looking fragment inside a prose
# refusal or explanation get silently returned as a successful structured result.


def test_banner_text_inside_the_inner_reply_is_no_longer_tolerated(
    fake_command_runner: type[FakeCommandRunner], claude_envelope: Callable[..., str]
) -> None:
    # Superseded case: this used to assert the opposite (tolerance at both parse layers) before
    # the model's own reply was moved to strict parsing. CLI chatter never actually lands inside
    # the model's generated text in practice — the banner prints before the model is even
    # invoked — so this is not real-world coverage lost, only the incorrect generalisation of it.
    banner = "Ignoring 5 permissions.allow entries — this workspace has not been trusted\n"
    inner = banner + "[1, 2, 3]"
    bad = claude_envelope(result=inner)
    runner = fake_command_runner([_ok(bad), _ok(bad)])
    with pytest.raises(LLMParseError):
        ClaudeCliAdapter(runner=runner).complete_json("x")
    assert len(runner.calls) == 2  # the one allowed retry was spent, and still failed


def test_incidental_json_fragment_in_a_prose_reply_raises_parse_error_not_a_fabricated_success(
    fake_command_runner: type[FakeCommandRunner], claude_envelope: Callable[..., str]
) -> None:
    # The adversarial-judge case: the model's actual answer is a prose refusal/explanation that
    # happens to CONTAIN a JSON-looking fragment. The old `_json_after_noise`-based `_parse_reply`
    # would scan past the prose, find the fragment, and return it as a successful structured
    # result — silently discarding what should have been a parse failure. Strict parsing must
    # reject the whole string instead, on both the first attempt and the one allowed retry.
    prose = (
        "I can't complete that request as structured data. "
        'For reference, here is an example schema: {"foo": "bar"} — but that is not my answer.'
    )
    bad = claude_envelope(result=prose)
    runner = fake_command_runner([_ok(bad), _ok(bad)])
    with pytest.raises(LLMParseError):
        ClaudeCliAdapter(runner=runner).complete_json("x")
    assert len(runner.calls) == 2  # one retry spent, per the retry-once-on-parse-failure policy
