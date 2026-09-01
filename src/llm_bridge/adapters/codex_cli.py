"""CodexCliAdapter — the ``LLMClient`` adapter over ``codex exec --json`` (headless Codex CLI).

Argv, built fresh per call from the requested model/effort::

    ["codex", "exec", "-s", "read-only", "--skip-git-repo-check", "--ignore-user-config",
     "--json", "-m", <model>, "-c", "model_reasoning_effort=<effort>", "-"]   # prompt on stdin

The ``-c model_reasoning_effort=<effort>`` pair is appended only when ``effort`` is not ``None`` —
unlike ``claude_cli``, codex has no native ``--effort`` flag; the run banner reflecting
"reasoning effort: high"/"none" when this flag is passed is what was actually observed. The
accepted value set (``none|low|medium|high``) is **inferred, not measured** — PROVISIONAL until
validated against a real authenticated run.

``-s read-only`` is documented isolation, not equivalent isolation: codex is intrinsically
agentic and has no flag that zeroes its tool surface the way claude's ``--tools ""`` does, so this
adapter is weaker-sandboxed than ``claude_cli`` by construction, not by an oversight here.
``--ignore-user-config`` is a measured necessity — without it, a locally configured MCP server
(observed: a Figma server) pollutes the JSONL stream before the real failure is ever reached.
``--skip-git-repo-check`` — this bridge is not guaranteed to run from inside a git worktree.

Output channel: this adapter parses **stdout**, not ``-o/--output-last-message <FILE>``. That is a
deliberate choice, not an oversight — a file-based output channel is incompatible with the shared
``CommandRunner`` injection seam (the same ``Callable[[argv, stdin_text, timeout_s],
CommandResult]`` shape as ``claude_cli``) and its fake-runner test pattern, which both CLI
adapters are built to share.

JSONL shape, MEASURED on the failure path only (codex was unauthenticated during this build;
two real 401s were reproduced)::

    thread.started -> turn.started -> [error]* (reconnect noise, ignorable)
        -> item.completed{item.type=="error"} -> turn.failed

The success path (``turn.completed`` / the assistant-message ``item.completed`` event) is
**UNMEASURED** — everything under ``_extract_success_text`` is a best-effort PROVISIONAL guess at
the real shape, deliberately logged at DEBUG on every unrecognized event so the first authenticated
run (the ``codex exec ... "reply with exactly PONG"`` smoke test) can lock the real parser from
that log rather than from guesswork.

Quota markers (``insufficient_quota`` / ``rate limit`` / ``429``) are likewise PROVISIONAL —
reaching a real codex rate limit was never observed here; an unmatched failure message still fails
safe as a generic ``LLMError`` rather than a silently swallowed one.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from llm_bridge.port import (
    LLMAuthError,
    LLMError,
    LLMParseError,
    LLMQuotaError,
    LLMTimeoutError,
)

# "sonnet" is the bridge-wide package default picked in port.py's migration contract (3 of the 4
# current claude call sites already use it) — it is NOT a measured or even plausible codex model
# id. `codex exec -m sonnet` at this default will most likely be rejected by the codex CLI as an
# unrecognized model. That rejection message matches neither the auth nor the quota markers below,
# so it surfaces as an unclassified generic LLMError, not a special-cased one. Every real call
# site is expected to pass its own explicit codex model id (README migration-contract note), same
# as every other adapter — this default exists only so the Protocol has *a* default, not a working
# one.
DEFAULT_MODEL: Final = "sonnet"

# The codex binary's own internal retry floor took ~30s just to surface the measured 401 failure —
# timeout_s below 60 is documented as unsafe for that reason alone, before any real model latency.
DEFAULT_TIMEOUT_S: Final = 180

log = logging.getLogger(__name__)

_AUTH_MARKERS: Final = ("401", "unauthorized")
# PROVISIONAL — see module docstring. Never observed against a real codex rate limit.
_QUOTA_MARKERS: Final = ("insufficient_quota", "rate limit", "429")


@dataclass(frozen=True)
class CommandResult:
    """Captured result of one subprocess run (decouples the adapter from ``subprocess``)."""

    returncode: int
    stdout: str
    stderr: str


# (argv, stdin_text, timeout_s) -> result. Injected so tests fake the CLI with no real LLM call.
# Same shape as claude_cli's CommandRunner by design — both CLI adapters share one fake-runner
# test pattern (test_port_contract.py parametrizes over both).
CommandRunner = Callable[[Sequence[str], str, int], CommandResult]


def _subprocess_runner(argv: Sequence[str], stdin_text: str, timeout_s: int) -> CommandResult:
    try:
        completed = subprocess.run(
            list(argv),
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise LLMTimeoutError(f"codex exec timed out after {timeout_s}s") from exc
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _parse_jsonl(stdout: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse ``stdout`` as one JSON object per line, tolerating noise.

    ``--ignore-user-config`` is what keeps unrelated JSONL out of this stream in the first place
    (module docstring); this parser is the second line of defense — a stray non-JSON or non-object
    line is logged at DEBUG and skipped, never fatal on its own. Zero *parsable* events, though, is
    not silently tolerated: it falls through to ``_extract_success_text`` finding nothing to return
    and raising ``LLMParseError``, which is exactly the signal ``complete_json``'s single retry is
    for.
    """
    events: list[dict[str, Any]] = []
    unparsed: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            unparsed.append(line)
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
        else:
            unparsed.append(line)
    return events, unparsed


def _event_message(event: dict[str, Any]) -> str | None:
    """Best-effort text out of one event, trying the shapes a failure event has been seen to use."""
    item = event.get("item")
    if isinstance(item, dict):
        message = item.get("message")
        if isinstance(message, str) and message:
            return message
    error = event.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    message = event.get("message")
    if isinstance(message, str) and message:
        return message
    return None


def _select_failure_event(events: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick whichever failure event actually carries the failure text.

    The measured sequence emits BOTH an ``item.completed{item.type=="error"}`` event and a later
    ``turn.failed`` event. Taking "the last failure event" would always pick ``turn.failed``,
    which may just be a thin turn-level summary — the error item is where the real "401
    Unauthorized" text was actually observed. So: collect both candidates, prefer whichever has a
    non-empty message, and tie-break to the error item when both (or neither) do, since it sits
    closer to the actual failure than the turn-level wrapper.
    """
    error_item: dict[str, Any] | None = None
    turn_failed: dict[str, Any] | None = None
    for event in events:
        event_type = event.get("type")
        if event_type == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "error":
                error_item = event
        elif event_type == "turn.failed":
            turn_failed = event
    for candidate in (error_item, turn_failed):
        if candidate is not None and _event_message(candidate):
            return candidate
    return error_item or turn_failed


def _classify_message(text: str) -> type[LLMError] | None:
    """Route a failure message to an error class, or None if unrecognised (-> generic LLMError)."""
    low = text.lower()
    if any(marker in low for marker in _AUTH_MARKERS):
        return LLMAuthError
    if any(marker in low for marker in _QUOTA_MARKERS):  # PROVISIONAL, see module docstring
        return LLMQuotaError
    return None


def _strip_code_fence(text: str) -> str:
    """Drop a wrapping ```json … ``` fence if the model added one, else return text unchanged."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):  # opening fence line: ``` or ```json
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":  # closing fence line
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_success_text(events: Sequence[dict[str, Any]]) -> str:
    """Best-effort pull of the assistant's reply text out of the success-path events.

    **PROVISIONAL — entirely unmeasured** (module docstring): codex was never authenticated during
    this build, so nothing here has been checked against a real ``turn.completed`` run. The primary
    guess is an ``item.completed`` event whose ``item.type == "agent_message"``, text in
    ``item.text`` — named as a guess, not presented as a measurement. A ``content`` field is tried
    as a fallback and, if it fires, is logged at DEBUG so it gets noticed and tightened rather than
    silently trusted forever. Every event is logged at DEBUG when nothing recognizable is found at
    all, which is exactly the raw material the plan's smoke test (``... "reply with exactly
    PONG"``) is meant to lock this parser against.
    """
    for event in events:
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            continue
        text = item.get("text")
        if isinstance(text, str) and text:
            return text
        content = item.get("content")  # fallback guess, not primary — logged below when it fires
        if isinstance(content, str) and content:
            log.debug(
                "codex exec success path used the 'content' fallback field, not 'text': %r",
                event,
            )
            return content
    for event in events:
        log.debug("codex exec success path — unrecognized event shape: %r", event)
    raise LLMParseError(
        "codex exec produced no recognizable assistant-message event "
        "(see DEBUG log for the raw JSONL — success-path shape is unmeasured, see module "
        "docstring)"
    )


class CodexCliAdapter:
    """``LLMClient`` implemented over the Codex CLI subprocess (see module docstring)."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        runner: CommandRunner = _subprocess_runner,
    ) -> None:
        self._model = model
        self._timeout_s = timeout_s
        self._runner = runner

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        effort: str | None = None,
        timeout_s: int | None = None,
    ) -> str:
        argv = self._build_argv(model, effort)
        events = self._run_and_classify(argv, prompt, timeout_s)
        return _extract_success_text(events)

    def complete_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        effort: str | None = None,
        timeout_s: int | None = None,
    ) -> Any:
        argv = self._build_argv(model, effort)
        try:
            return self._attempt_json(argv, prompt, timeout_s)
        except LLMParseError as exc:
            # Exactly one retry, same policy as claude_cli: a one-off malformed reply is common
            # with an unconstrained CLI and usually transient. Auth, quota and timeout are never
            # LLMParseError, so they never reach this retry.
            log.warning("codex reply failed to parse (%s) — retrying once", exc)
            return self._attempt_json(argv, prompt, timeout_s)

    def _attempt_json(self, argv: Sequence[str], prompt: str, timeout_s: int | None) -> Any:
        events = self._run_and_classify(argv, prompt, timeout_s)
        text = _extract_success_text(events)
        try:
            return json.loads(_strip_code_fence(text))
        except json.JSONDecodeError as exc:
            raise LLMParseError(f"model reply was not valid JSON: {text[:200]!r}") from exc

    def _build_argv(self, model: str | None, effort: str | None) -> list[str]:
        effective_model = model or self._model
        argv = [
            "codex",
            "exec",
            "-s",
            "read-only",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--json",
            "-m",
            effective_model,
        ]
        if effort:
            argv += ["-c", f"model_reasoning_effort={effort}"]
        argv.append("-")  # prompt on stdin, not argv — see module docstring
        return argv

    def _run_and_classify(
        self, argv: Sequence[str], prompt: str, timeout_s: int | None
    ) -> list[dict[str, Any]]:
        """Run one ``codex exec`` invocation; raise the fail-safe taxonomy, else return its events.

        Fail-safe invariant (never a false success, Phase 7 judge check): a recognized failure
        event wins even over a zero exit code, since the success/failure exit-code contract itself
        is unmeasured; a nonzero exit with no recognized failure event still raises loud rather than
        returning an empty or partial reply.
        """
        effective_timeout = timeout_s if timeout_s is not None else self._timeout_s
        result = self._runner(argv, prompt, effective_timeout)
        events, unparsed_lines = _parse_jsonl(result.stdout)
        for line in unparsed_lines:
            log.debug("codex exec emitted a non-JSON stdout line, skipped: %r", line)

        failure_event = _select_failure_event(events)
        if failure_event is not None:
            message = (
                _event_message(failure_event) or "codex exec reported a failure with no message"
            )
            error_cls = _classify_message(message)
            log.error(
                "codex exec failure classified as %s — raw event: %r",
                error_cls.__name__ if error_cls is not None else "UNCLASSIFIED (generic LLMError)",
                failure_event,
            )
            if error_cls is not None:
                raise error_cls(message)
            raise LLMError(message)

        if result.returncode != 0:
            raw = (result.stderr or result.stdout).strip()
            log.error(
                "codex exec exited %s with no recognized failure event — raw: %r",
                result.returncode,
                raw,
            )
            raise LLMError(f"codex exec exited {result.returncode}: {raw[:500] or '<no output>'}")

        return events
