"""``ClaudeCliAdapter`` — the ``LLMClient`` adapter over headless ``claude -p``.

Fusion of two adapters that had already diverged from a shared ancestor: tldr-filter's
``adapters/claude/cli.py`` (the mature base — ``CommandRunner`` injection, status-code-first
classification measured against a real 401, ``_strip_code_fence``, retry-once-on-parse-failure)
and hots_replay_coach's ``coach/claude_cli.py`` (``_json_after_noise`` — tolerant parsing that
survives an untrusted-workspace banner ahead of the JSON envelope, and the
``--safe-mode --tools ""`` isolation flags). See ``ok-du-coup-j-ai-curried-gosling.md`` Phase 1
§2 "Adaptateurs" for the merge rationale this module implements.

**Runtime-behaviour change, flagged here rather than left to slide silently** (plan Phase 1 §3
point 3): tldr-filter's prior adapter ran ``claude -p`` with neither ``--safe-mode`` nor
``--tools`` — an agent session with ambient access to this machine's ``CLAUDE.md``, skills, and
``Write``/``Bash``, identical to an interactive one. This adapter always passes
``--safe-mode --tools ""``, so every completion is a pure text call with zero tool access. Any
caller migrating from the old tldr adapter onto this one is picking up an isolation change, not
just a new import path. Subscription auth (``CLAUDE_CODE_OAUTH_TOKEN``, inherited from the process
environment) is unaffected, and this module does not check ``ANTHROPIC_API_KEY`` itself —
that guard (``guard_subscription_only()``) stays a hots_replay_coach-local function, not moved
into the bridge, because ``coach/chat.py`` (out of scope, Agent SDK) depends on it too; callers
that need the guard call it themselves before reaching this adapter.

The prompt goes on **stdin**, never as an argv element — a large prompt (a full email's HTML, a
game context pack) can run past Windows' ~128 KB per-argument cap; stdin is capped at 10 MB
instead (tldr-filter S6).

``model=None``/``timeout_s=None`` on a call mean "this adapter instance's own default" per the
port contract — the constructor default, not a bridge-wide one. ``effort``, when not ``None``, is
passed through as-is via ``--effort <value>`` (the CLI's own flag; no bridge-side mapping).

``complete_json`` unwraps the CLI envelope — tolerating a banner or other non-JSON noise ahead of
the OUTER envelope only (``_json_after_noise``, merged in from hots S23d; that tolerance exists
for CLI chatter, e.g. an untrusted-workspace warning, never for the model's own words) — then
strips a wrapping ```` ```json ```` fence off the model's reply and parses it with exactly one
strict ``json.loads``. A reply that is prose with an incidental JSON-looking fragment inside it
(e.g. an example schema mentioned in an explanatory sentence) is a parse FAILURE, not a lucky
extraction: scanning the model's own reply text for "a plausible JSON substring" would silently
turn a prose refusal or error message into a fabricated successful result, defeating the point of
raising ``LLMParseError`` in the first place. Retrying happens **exactly once**, and only on
``LLMParseError``: auth, quota and timeout failures are never retried.
``complete`` is new — neither source adapter has it as a public method today — and returns the
envelope's unwrapped ``result`` string as-is, with no JSON parsing and no fence-stripping; turning
that raw text into something structured stays the caller's job (e.g.
``_strip_wrapping_fence`` in meeting-summerizer).

Classification is status-code-first: the CLI's ``api_error_status`` field (401/403 ->
``LLMAuthError``, 429 -> ``LLMQuotaError``) decides when present — measured against a real 401
(tldr-filter, invalid ``CLAUDE_CODE_OAUTH_TOKEN``). Only when no usable status is available does
classification fall back to substring matching on the failure sentence, and anything unmatched
stays a generic ``LLMError`` — fail loud, never a false success.
"""

from __future__ import annotations

import contextlib
import json
import logging
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final, NoReturn

from llm_bridge.port import (
    LLMAuthError,
    LLMError,
    LLMParseError,
    LLMQuotaError,
    LLMTimeoutError,
)

DEFAULT_MODEL: Final = "sonnet"  # Package-level default — every real caller overrides this
# explicitly at its own call site (see README.md "Migration contract"); tldr-filter keeps
# "claude-opus-4-8", hots/meeting/video keep "sonnet". This constant only backstops a caller
# that genuinely has no opinion.
DEFAULT_TIMEOUT_S: Final = 180

log = logging.getLogger(__name__)

# HTTP status -> error class, read from the envelope's `api_error_status`. This is the PRIMARY
# classifier and it is measured, not guessed: an auth failure was reproduced by running the real
# CLI in the real image with a deliberately invalid CLAUDE_CODE_OAUTH_TOKEN, which returned
#
#   {"type":"result","is_error":true,"api_error_status":401,
#    "result":"Failed to authenticate. API Error: 401 OAuth access token is invalid.", ...}
#
# on **stdout** with an EMPTY stderr and exit code 1 (tldr-filter). That measurement is also why
# the status field is preferred over the substring markers below: `detail` is `stderr or stdout`,
# and stderr was empty, so a substring matcher scans the WHOLE envelope — session_id and UUIDs
# included — and a stray hex triple can misroute an auth failure to LLMQuotaError.
#
# 429 stays mapped here for symmetry, but note it is UNMEASURED — see _QUOTA_MARKERS below.
_STATUS_ERRORS: Final[dict[int, type[LLMError]]] = {
    401: LLMAuthError,
    403: LLMAuthError,
    429: LLMQuotaError,
}

# Fallback substrings, used only when no `api_error_status` is available (a non-JSON failure, or a
# CLI version that drops the field). Ordered quota-then-auth, matched against the envelope's
# `result` sentence when there is one — never the raw envelope, for the UUID-collision reason
# above. Union of both source adapters' marker lists; "authenticate" is the measured real wording
# and would NOT have matched hots' narrower "authentication"-only list.
#
# The QUOTA markers remain **PROVISIONAL** in both source adapters: triggering a real rate-limit
# means deliberately exhausting the subscription, so unlike the auth path these have never been
# seen live. Anything unmatched stays a generic LLMError, so a wrong guess still fails safe — loud,
# never a false success.
_QUOTA_MARKERS: Final = ("usage limit", "rate limit", "quota", "429", "too many requests")
_AUTH_MARKERS: Final = (
    "invalid api key",
    "authenticate",
    "authentication",
    "unauthorized",
    "not logged in",
    "oauth",
    "log in",
    "login",
    "please log in",
    "expired",
)


@dataclass(frozen=True)
class CommandResult:
    """Captured result of one subprocess run (decouples the adapter from ``subprocess``)."""

    returncode: int
    stdout: str
    stderr: str


# (argv, stdin_text, timeout_s) -> result. Injected so tests fake the CLI with no real LLM call.
# Same shape as tldr-filter's `CommandRunner`, deliberately, so a fake written for one adapter
# fakes the other identically.
CommandRunner = Callable[[Sequence[str], str, int], CommandResult]


def _subprocess_runner(argv: Sequence[str], stdin_text: str, timeout_s: int) -> CommandResult:
    try:
        completed = subprocess.run(
            list(argv),
            input=stdin_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise LLMTimeoutError(f"claude -p timed out after {timeout_s}s") from exc
    except FileNotFoundError as exc:
        raise LLMError("`claude` is not on PATH — install Claude Code") from exc
    return CommandResult(completed.returncode, completed.stdout or "", completed.stderr or "")


def _classify_text(text: str) -> type[LLMError] | None:
    """Route a failure SENTENCE to an error class, or None if unrecognised (-> generic)."""
    low = text.lower()
    if any(marker in low for marker in _QUOTA_MARKERS):
        return LLMQuotaError
    if any(marker in low for marker in _AUTH_MARKERS):
        return LLMAuthError
    return None


def _json_after_noise(text: str) -> Any:
    """Parse the first JSON value in ``text``, tolerating leading non-JSON noise.

    Merged in from hots_replay_coach (S23d): the CLI is free to print warnings before its real
    output — an untrusted workspace prints "Ignoring N permissions.allow entries … this workspace
    has not been trusted" ahead of the JSON envelope, and a strict whole-string parse dies on that
    banner. Scans to the first ``{`` or ``[`` and decodes from there; ``raw_decode`` stops at the
    end of that one value, so trailing noise is ignored too. A ``{``/``[`` inside the noise that
    does not begin a valid value is skipped, not fatal. Raises ``json.JSONDecodeError`` if no JSON
    value is found anywhere in ``text``.

    Scope, deliberately: this is for the OUTER stdout envelope only — ``_parse_envelope``,
    ``_classify`` and ``_sentence`` below all scan CLI-produced bytes, where tolerating a banner
    ahead of the real envelope is the measured, legitimate need this function exists for. It must
    never be reused to parse the model's own reply text (the envelope's ``result`` string): that
    text is untrusted content the model wrote, and "skip past noise to find a plausible JSON
    value" would let an incidental JSON-looking fragment inside a prose refusal or explanation get
    silently returned as a successful structured result. ``_parse_reply`` parses the reply
    strictly instead (fence-strip, then one ``json.loads`` — no scanning).
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            return value
    raise json.JSONDecodeError("no JSON value found in reply", text or "", 0)


def _classify(text: str) -> type[LLMError] | None:
    """Route CLI failure output to an error class: by status code first, by wording second.

    ``text`` is whatever the CLI emitted — usually its JSON envelope on stdout, since a failing
    ``claude -p`` writes nothing to stderr (measured). When that envelope carries an
    ``api_error_status`` the code decides, full stop. Only if there is no usable status does this
    fall back to matching wording, and then against the envelope's ``result`` sentence rather than
    the whole envelope. Envelope parsing goes through ``_json_after_noise`` so a leading banner
    does not defeat classification either.
    """
    envelope: object = None
    with contextlib.suppress(json.JSONDecodeError):
        envelope = _json_after_noise(text)

    if isinstance(envelope, dict):
        status = envelope.get("api_error_status")
        if isinstance(status, int) and status in _STATUS_ERRORS:
            return _STATUS_ERRORS[status]
        sentence = envelope.get("result")
        if isinstance(sentence, str) and sentence:
            return _classify_text(sentence)
    return _classify_text(text)


def _sentence(raw: str) -> str:
    """The human-readable half of CLI failure output — the envelope's ``result``, else ``raw``.

    Kept separate from the classification source because they want opposite things:
    classification wants every byte the CLI produced, the message wants one sentence a person can
    act on. Before this split (tldr-filter), an auth failure surfaced to a caller as the entire
    ~1 KB JSON envelope, UUIDs and token counts included.
    """
    envelope: object = None
    with contextlib.suppress(json.JSONDecodeError):
        envelope = _json_after_noise(raw)
    if isinstance(envelope, dict):
        result = envelope.get("result")
        if isinstance(result, str) and result:
            return result
    return raw


def _raise_classified(source: str, fallback: str) -> NoReturn:
    """Raise the auth/quota class ``source`` classifies to, else a generic LLMError.

    ``source`` is the full CLI output (classified, never shown); ``fallback`` is the sentence a
    person reads.
    """
    error_cls = _classify(source)
    detail = _sentence(source)
    # Log the RAW text verbatim. The auth path is measured; the QUOTA markers are still guesses —
    # this line is the evidence needed to reconcile them against real wording the first time a
    # rate limit fires. An UNCLASSIFIED failure is the one to look at: it means neither the status
    # code nor any marker matched.
    log.error(
        "claude CLI failure classified as %s — raw text: %r",
        error_cls.__name__ if error_cls is not None else "UNCLASSIFIED (generic LLMError)",
        detail or fallback,
    )
    if error_cls is not None:
        raise error_cls(detail or fallback)
    raise LLMError(fallback)


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


class ClaudeCliAdapter:
    """``LLMClient`` implemented over the Claude Code CLI subprocess (see module docstring)."""

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
        """Run ``prompt``; return the envelope's raw ``result`` string, un-parsed, un-de-fenced."""
        argv = self._argv(model, effort)
        result = self._runner(argv, prompt, self._resolve_timeout(timeout_s))
        if result.returncode != 0:
            self._raise_for_exit(result)
        envelope = self._parse_envelope(result.stdout)
        return self._extract_reply(envelope)

    def complete_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        effort: str | None = None,
        timeout_s: int | None = None,
    ) -> Any:
        """Run ``prompt``; return the reply parsed as JSON, retrying exactly once on parse failure.

        Only ``LLMParseError`` is retried: a one-off malformed envelope/reply is common with an
        unconstrained CLI and usually transient. Auth, quota and timeout are NOT parse failures, so
        they never reach this retry.
        """
        argv = self._argv(model, effort)
        resolved_timeout = self._resolve_timeout(timeout_s)
        try:
            return self._attempt_json(argv, prompt, resolved_timeout)
        except LLMParseError as exc:
            log.warning("claude reply failed to parse (%s) — retrying once", exc)
            return self._attempt_json(argv, prompt, resolved_timeout)

    def _attempt_json(self, argv: Sequence[str], prompt: str, timeout_s: int) -> Any:
        result = self._runner(argv, prompt, timeout_s)
        if result.returncode != 0:
            self._raise_for_exit(result)
        envelope = self._parse_envelope(result.stdout)
        reply = self._extract_reply(envelope)
        return self._parse_reply(reply)

    def _argv(self, model: str | None, effort: str | None) -> list[str]:
        argv = [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--model",
            model if model is not None else self._model,
            "--safe-mode",
            "--tools",
            "",
        ]
        if effort is not None:
            argv += ["--effort", effort]
        return argv

    def _resolve_timeout(self, timeout_s: int | None) -> int:
        return timeout_s if timeout_s is not None else self._timeout_s

    @staticmethod
    def _raise_for_exit(result: CommandResult) -> NoReturn:
        # stderr first, but a failing `claude -p --output-format json` writes NOTHING to stderr and
        # puts its error envelope on stdout (measured against the real CLI) — so stdout is the
        # normal path here, not the fallback it reads like.
        source = (result.stderr or result.stdout).strip()
        fallback = f"claude -p exited {result.returncode}: {_sentence(source) or '<no output>'}"
        _raise_classified(source, fallback)

    @staticmethod
    def _parse_envelope(stdout: str) -> dict[str, Any]:
        try:
            envelope = _json_after_noise(stdout)
        except json.JSONDecodeError as exc:
            raise LLMParseError(f"CLI envelope was not valid JSON: {stdout[:200]!r}") from exc
        if not isinstance(envelope, dict):
            raise LLMParseError(f"CLI envelope was not a JSON object: {type(envelope).__name__}")
        if envelope.get("is_error"):
            # A well-formed error envelope is a real failure, not a parse hiccup — do not retry.
            # Classify from the WHOLE envelope (`stdout`), not the sentence: `api_error_status` is
            # the reliable signal and it lives beside `result`, not inside it.
            detail = str(envelope.get("result") or envelope.get("subtype") or "unknown error")
            _raise_classified(stdout, f"claude reported an error: {detail}")
        return envelope

    @staticmethod
    def _extract_reply(envelope: dict[str, Any]) -> str:
        result = envelope.get("result")
        if not isinstance(result, str):
            raise LLMParseError(f"CLI envelope missing string 'result': {envelope!r}")
        return result

    @staticmethod
    def _parse_reply(reply_text: str) -> Any:
        """Parse the MODEL'S OWN reply strictly: fence-strip, then exactly one ``json.loads``.

        Deliberately NOT ``_json_after_noise``. That scanner tolerates CLI chatter ahead of the
        outer envelope — a legitimate need — but the reply text here is the model's own words,
        already unwrapped from the envelope. A tolerant scan would extract an incidental
        JSON-looking fragment out of a prose refusal or explanation (e.g. "here is an example
        schema: {\"foo\": \"bar\"}") and return it as a successful structured result, silently
        discarding what should have been a parse failure. A strict single-shot parse is the only
        way this method can tell "the model returned JSON" from "the model returned prose that
        happens to contain something JSON-shaped" — and only the former is a real success.
        """
        try:
            return json.loads(_strip_code_fence(reply_text))
        except json.JSONDecodeError as exc:
            raise LLMParseError(f"model reply was not valid JSON: {reply_text[:200]!r}") from exc
