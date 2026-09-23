"""The provider seam — M1 step 6, spec `agent/M1_SPEC.md` §Q11, DL-024.

One function is the only place in omega that knows an LLM exists::

    complete(role, messages) -> Response

Nothing above this line names OpenAI, a model string, or a request shape. That
is the memory seam's discipline applied for the memory seam's reason: **a
boundary crossed in exactly one place is a boundary you can move.** Swapping
providers, or running ``judge`` locally while ``act`` stays remote, is an edit
to this file and nothing else.

**Why the seam takes a role.** ``judge`` and ``act`` have opposite cost
profiles. ``judge`` fires on *every* event including every idle tick and its
answer is almost always "nothing", so it wants a small fast model. ``act``
fires rarely and does the work, so it wants a capable one. Splitting them now
costs nothing; unpicking a hardcoded model later touches every call site.

**What this deliberately does not do.** No retries, no backoff, no fallback
model, no response caching. Tool error/retry policy is explicitly deferred in
`CLAUDE.md`, and a seam that quietly retried would be *deciding* that deferred
question by implementation — the exact failure DL-006 records, where a stub's
shape nearly became the contract by default. Errors surface as
:class:`ProviderError`. The loop decides what to do about them, once we have
decided what the loop should do.

**The seam carries tool calls, and it was widened additively** (M1 step 7,
DL-028). ``complete(role, messages)`` had no field in which a model could say
*call this tool with these arguments*, so it grew an optional ``tools``
parameter and :class:`Response` grew ``tool_calls``. Every caller that passes
no tools produces exactly the request it produced before — the ``tools`` key is
not even present — so the widening cannot regress the one property M1 exists to
measure.

*Why not parse tool calls out of the text.* ``judge`` already parses model text
and survives only because its whole grammar is one word from a closed set of
three. Tool arguments are structured and adversarial in the way a verdict is
not: a path with a space, a command with a quote, a URL with a ``)``. §2.4
forbids automatic retry, so every parse failure would be a *failed turn* rather
than a retried one. Native tool calling moves that failure into the provider's
typed channel, where it is the provider's problem and not a new class of lost
turn.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

__all__ = [
    "JUDGE",
    "ACT",
    "ROLES",
    "Message",
    "ToolCall",
    "Response",
    "Provider",
    "ProviderError",
    "ProviderNotConfigured",
    "FakeProvider",
    "OpenAIProvider",
    "load_env",
    "provider_from_env",
    "system",
    "user",
    "assistant",
    "assistant_tool_calls",
    "tool_result",
]

#: Decide what this event deserves — including deciding it deserves nothing.
#: Runs on every event, so it is the cost-sensitive one.
JUDGE = "judge"

#: Do the work. Runs rarely; wants the capable model.
ACT = "act"

ROLES = frozenset({JUDGE, ACT})

#: Default models if `.env` does not name one. Deliberately explicit rather than
#: "whatever the client library defaults to" — an invisible default is a
#: decision nobody made, and cost is the thing we are splitting roles over.
_DEFAULT_MODELS = {
    JUDGE: "gpt-4o-mini",
    ACT: "gpt-4o",
}

#: Default sampling temperature per role, and ``None`` means *do not send the
#: parameter at all*.
#:
#: ``judge`` is pinned to 0. It is a one-word classifier over a three-way
#: choice, and at the API's default of 1.0 it was measured returning different
#: verdicts for byte-identical input — the same request judged ``act`` three
#: times in isolation and ``silent`` inside a turn. `CLAUDE.md` grades
#: reliability as pass^k rather than pass@1, and a router that samples cannot
#: clear that bar no matter how good the prompt is.
#:
#: ``act`` stays ``None``: composing a reply is not a classification, and the
#: role's model is the one most likely to be a reasoning model that rejects the
#: parameter outright. Omitting it keeps that request byte-for-byte what it was.
_DEFAULT_TEMPERATURES: dict[str, Optional[float]] = {
    JUDGE: 0.0,
    ACT: None,
}


class ProviderError(RuntimeError):
    """The model call did not produce a usable answer.

    One class for every remote failure — network, auth, rate limit, refusal,
    malformed response. Not split finer *yet*, because splitting it is only
    useful once something reacts differently to each, and what reacts is the
    retry policy `CLAUDE.md` still defers. ``__cause__`` carries the original.
    """


class ProviderNotConfigured(ProviderError):
    """No usable credentials. Raised at construction, not at first call.

    Separate because the response is different in kind: every other
    ``ProviderError`` is something that might work next time, and this one
    never will until a human edits `.env`.
    """


Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


def system(content: str) -> Message:
    return {"role": "system", "content": content}


def user(content: str) -> Message:
    return {"role": "user", "content": content}


def assistant(content: str) -> Message:
    return {"role": "assistant", "content": content}


@dataclass(frozen=True)
class ToolCall:
    """One request from the model to run one tool.

    ``arguments`` is a **decoded** object, not the JSON string the wire
    carries. Decoding it here is the point of native tool calling: the one
    place that can be handed a malformed argument blob is the one place that
    knows what the provider promised, and a decode failure is a
    :class:`ProviderError` rather than a new parser in the loop (DL-028).

    ``id`` matters as much as the name. A pass may request several tools, and
    the result of each has to be handed back against the call it answers — a
    result matched by position would silently pair the wrong output with the
    wrong request the first time a model reordered them.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


def assistant_tool_calls(
    tool_calls: Sequence[ToolCall], content: str = ""
) -> Message:
    """The assistant turn that *asked* for tools, ready to send back.

    The next pass has to see the model's own request in its history or it is
    being asked to interpret results to questions it cannot see — and the API
    refuses a ``tool`` message that answers no call. ``content`` is carried
    because a model may reason aloud *and* call a tool in one turn.
    """
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, sort_keys=True),
                },
            }
            for call in tool_calls
        ],
    }


def tool_result(call_id: str, content: str) -> Message:
    """What one tool returned, against the call id that asked for it.

    This is **data, never instruction** (DL-014). A fetched page is the first
    thing omega reads that someone else wrote, and it arrives in a role the
    model is told nothing by: the approval gate that decides what may run is
    static code in the execution layer (`omega.tools`), so nothing said inside
    this content can widen what the next pass is allowed to do.
    """
    return {"role": "tool", "tool_call_id": call_id, "content": content}


@dataclass(frozen=True)
class Response:
    """What came back.

    ``text`` is the whole answer and is never ``None`` — a provider that
    returned nothing raises instead, because an empty string flowing into
    ``judge`` would be indistinguishable from a model choosing silence, and
    that is the one distinction M1 exists to measure (DL-024).

    ``model`` is recorded rather than assumed: it is what the provider *says*
    it used, which is the only honest answer when a name can alias to a
    dated snapshot.

    ``tool_calls`` is empty for every call that offered no tools, which is
    every call `judge` and `reply` make.
    """

    text: str
    model: str
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def total_tokens(self) -> Optional[int]:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


class Provider(Protocol):
    """Anything that can answer a role with text. The whole contract.

    ``tools`` is optional and defaults to none, so a provider written against
    the pre-DL-028 seam still satisfies every call `judge` and `reply` make.
    """

    def complete(
        self,
        role: str,
        messages: Sequence[Message],
        tools: Optional[Sequence[dict[str, Any]]] = None,
    ) -> Response: ...


# --- configuration ----------------------------------------------------------


def load_env(path: Optional[Path] = None, *, override: bool = False) -> dict[str, str]:
    """Read a ``.env`` file into ``os.environ`` and return what it set.

    A deliberately small parser rather than a dependency: ``KEY=value``, ``#``
    comments, blank lines, optional surrounding quotes, and an optional
    ``export`` prefix. It does **not** do variable expansion, multi-line values,
    or shell escapes — if a value ever needs those, that is the signal to take
    the dependency, not to grow this.

    A missing file is not an error: the key may legitimately come from the real
    environment instead, and failing here would make that impossible.

    By default an existing environment variable **wins** over the file, because
    the shell is the more deliberate act — `override=True` when a test needs the
    file to be authoritative.
    """
    path = path or Path.cwd() / ".env"
    if not path.exists():
        return {}

    set_values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
        set_values[key] = value
    return set_values


def model_for(role: str) -> str:
    """Which model serves ``role``, from the environment.

    ``OMEGA_MODEL_JUDGE`` / ``OMEGA_MODEL_ACT``, falling back to ``OMEGA_MODEL``
    for both, then to the defaults. They live in `.env` beside the key on
    purpose: **changing which model judges is a config edit, not a commit.**

    The bare ``OMEGA_MODEL`` exists because the two-role split is ours, not
    something a reader of `.env` can be expected to infer, and the failure mode
    without it is the worst kind: setting it did nothing, said nothing, and
    billed the defaults. A variable that is obviously meant to choose the model
    should choose the model. Per-role names still win, so the cheap-judge design
    (DL-024) is one line away rather than overridden.
    """
    _check_role(role)
    return (
        os.environ.get(f"OMEGA_MODEL_{role.upper()}")
        or os.environ.get("OMEGA_MODEL")
        or _DEFAULT_MODELS[role]
    )


def temperature_for(role: str) -> Optional[float]:
    """The sampling temperature for ``role``, or ``None`` to omit it.

    ``OMEGA_TEMPERATURE_JUDGE`` / ``OMEGA_TEMPERATURE_ACT`` override the
    defaults, and the literal ``none`` means *send no temperature* — the escape
    hatch that matters, because a reasoning model rejects the parameter with a
    400 rather than ignoring it, and pinning the judge to 0 must never be the
    reason a model cannot be used at all.

    There is deliberately no bare ``OMEGA_TEMPERATURE``. The roles want
    genuinely different values (see :data:`_DEFAULT_TEMPERATURES`), so one
    variable for both would be a footgun of exactly the shape that put ``act``
    on a model that cannot call tools.
    """
    _check_role(role)
    raw = os.environ.get(f"OMEGA_TEMPERATURE_{role.upper()}")
    if raw is None or not raw.strip():
        return _DEFAULT_TEMPERATURES[role]
    if raw.strip().lower() == "none":
        return None
    try:
        return float(raw)
    except ValueError:
        raise ProviderNotConfigured(
            f"OMEGA_TEMPERATURE_{role.upper()}={raw!r} is not a number or 'none'"
        ) from None


def _tool_hint(
    role: str,
    model: str,
    tools: Optional[Sequence[dict[str, Any]]],
    exc: BaseException,
) -> str:
    """A sentence naming the fix when a model cannot do function tools.

    This exists because of a real, silent trap. ``act`` is the **only** role
    offered tools, and a bare ``OMEGA_MODEL`` applies to both roles — so
    pointing ``OMEGA_MODEL`` at a model that rejects function tools on
    ``/v1/chat/completions`` disables every tool omega has, and the only symptom
    is a raw 400 arriving one layer below anything that knows what a tool is.
    Measured on a real store: every act pass failed and the turn was recorded
    ``failed`` with the provider's own wording, which names neither the role nor
    the variable that chose the model.

    Kept to a hint on an existing error rather than a preflight check. omega
    cannot know what a model supports without asking, and asking on startup
    would spend a call on every boot to answer a question that only matters when
    something has already gone wrong.
    """
    if not tools:
        return ""
    text = str(exc).lower()
    if "tool" not in text and "function" not in text:
        return ""
    return (
        f"\n\n{model} was offered {len(tools)} tools and rejected them. The "
        f"{role} role is the one that calls tools, so it needs a model that "
        f"supports function calling on chat completions. Set "
        f"OMEGA_MODEL_{role.upper()} to one — a bare OMEGA_MODEL applies to "
        f"every role, including this one."
    )


def provider_from_env(*, env_path: Optional[Path] = None) -> "OpenAIProvider":
    """Build the real provider from `.env` + the environment.

    Raises :class:`ProviderNotConfigured` *here* rather than at the first call,
    so a missing key is a startup failure with a stack that points at
    configuration — not a mystery an hour later in the middle of a turn.
    """
    load_env(env_path)
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ProviderNotConfigured(
            "OPENAI_API_KEY is not set. Put it in .env (gitignored) as "
            "OPENAI_API_KEY=sk-... , or export it in the shell."
        )
    return OpenAIProvider(api_key=key)


def _check_role(role: str) -> None:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; roles are {sorted(ROLES)}")


# --- the real provider ------------------------------------------------------


class OpenAIProvider:
    """The one place OpenAI is named.

    The client is imported **lazily**, inside the constructor, so that importing
    this module — which every test does — never requires the package to be
    installed or a key to exist. That is not a convenience: it is what keeps
    "the judge chose silence" assertable with no network and no spend.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        models: Optional[dict[str, str]] = None,
        client: object = None,
    ) -> None:
        """``client`` overrides the constructed one.

        It exists so the response *mapping* below can be tested without a key
        or a network. That mapping contains the branch that turns a
        content-less response into an error rather than an empty string, and a
        branch that can only be exercised against the live API is a branch that
        ships untested — which is the thing this whole milestone is about.
        """
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise ProviderNotConfigured(
                    "the `openai` package is not installed; `pip install openai`"
                ) from exc

            if not api_key:
                raise ProviderNotConfigured("api_key must not be empty")

            # max_retries=0 on purpose: retry policy is deferred in CLAUDE.md,
            # and a client retrying under the seam would settle that deferred
            # question invisibly, which is how a default becomes a decision.
            client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)

        self._client = client
        self._models = models or {}

    def model_for(self, role: str) -> str:
        _check_role(role)
        return self._models.get(role) or model_for(role)

    def complete(
        self,
        role: str,
        messages: Sequence[Message],
        tools: Optional[Sequence[dict[str, Any]]] = None,
    ) -> Response:
        _check_role(role)
        if not messages:
            raise ValueError("messages must not be empty")

        model = self.model_for(role)
        request: dict[str, Any] = {"model": model, "messages": list(messages)}
        temperature = temperature_for(role)
        if temperature is not None:
            request["temperature"] = temperature
        if tools:
            # Added only when there are tools to offer. A caller that passes
            # none sends the request it sent before the seam was widened, key
            # for key — which is what makes DL-028's "bit-for-bit unchanged"
            # a fact about the wire rather than an intention.
            request["tools"] = list(tools)
        try:
            raw = self._client.chat.completions.create(**request)
        except Exception as exc:  # noqa: BLE001 - every remote failure is one class
            raise ProviderError(
                f"{role} call to {model} failed: {exc}{_tool_hint(role, model, tools, exc)}"
            ) from exc

        try:
            choice = raw.choices[0]
            text = choice.message.content
            raw_calls = getattr(choice.message, "tool_calls", None) or ()
        except (AttributeError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"{role} call to {model} returned an unreadable response"
            ) from exc

        tool_calls = tuple(_read_tool_call(role, model, c) for c in raw_calls)

        if text is None:
            if not tool_calls:
                # Never let this become an empty string: `judge` reading "" would
                # be indistinguishable from a model choosing to stay silent, and
                # that distinction is the whole point of measuring silence.
                raise ProviderError(
                    f"{role} call to {model} returned no content "
                    f"(finish_reason={getattr(choice, 'finish_reason', None)!r})"
                )
            # ...but the API returns `content: null` *exactly when* the model
            # answered with tool calls instead of words, and that is not a model
            # that said nothing — it is a model that said "run these". The
            # reasoning above is untouched and the guard is narrowed, not
            # removed: with no tool calls either, it still raises. Nothing that
            # reaches `judge` can take this branch, because `judge` offers no
            # tools and so can never come back with any.
            text = ""

        usage = getattr(raw, "usage", None)
        return Response(
            text=text,
            model=getattr(raw, "model", model),
            finish_reason=getattr(choice, "finish_reason", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            tool_calls=tool_calls,
        )


def _read_tool_call(role: str, model: str, raw: Any) -> ToolCall:
    """One wire tool call as a :class:`ToolCall`, or a :class:`ProviderError`.

    Strict on purpose. Arguments arrive as a JSON *string*, and the three ways
    that can be wrong — unreadable object, invalid JSON, valid JSON that is not
    an object — are each a failed call rather than a guess. Guessing here would
    put a half-decoded argument in front of the approval classifier, which is
    the one place in omega that must never be handed something it cannot read.
    """
    try:
        call_id = raw.id
        function = raw.function
        name = function.name
        arguments = function.arguments
    except AttributeError as exc:
        raise ProviderError(
            f"{role} call to {model} returned an unreadable tool call"
        ) from exc

    if isinstance(arguments, dict):
        decoded: Any = arguments
    elif isinstance(arguments, str):
        try:
            decoded = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"{role} call to {model} asked for {name!r} with arguments that "
                f"are not JSON: {exc}"
            ) from exc
    else:
        raise ProviderError(
            f"{role} call to {model} asked for {name!r} with arguments of type "
            f"{type(arguments).__name__}"
        )

    if not isinstance(decoded, dict):
        raise ProviderError(
            f"{role} call to {model} asked for {name!r} with arguments that "
            f"decoded to a {type(decoded).__name__}, not an object"
        )
    if not isinstance(name, str) or not name:
        raise ProviderError(f"{role} call to {model} returned a nameless tool call")
    return ToolCall(id=str(call_id), name=name, arguments=decoded)


# --- the test provider ------------------------------------------------------


class FakeProvider:
    """A scripted provider. The only one the test suite ever uses.

    It exists so every assertion about the loop — including "the judge chose
    silence" — runs with no network, no key, and no spend. It also *records*
    calls, because the interesting M1 assertions are about what was asked and
    how often, not only about what came back.

    ``answers`` maps a role to either a list of replies consumed in order, or a
    callable taking ``(role, messages)``. Running out of scripted answers is an
    error rather than a repeat of the last one: a test that silently reuses an
    answer passes for a reason it did not state.

    **An answer may be a tool call**, so the `act` sub-loop is exercisable with
    no network and no key: a scripted answer that is a :class:`ToolCall`, a
    sequence of them, or a whole :class:`Response` becomes exactly that. A
    string stays a string, so every case written before the seam widened means
    what it meant.
    """

    def __init__(
        self,
        answers: Optional[dict[str, object]] = None,
        *,
        model: str = "fake-model",
    ) -> None:
        # Copy the *lists*, not only the dict: answers are consumed with
        # `pop(0)`, so a shallow copy would drain the caller's own script and
        # two tests sharing a fixture would silently interfere.
        self._answers = {
            role: (list(value) if isinstance(value, list) else value)
            for role, value in (answers or {}).items()
        }
        self._model = model
        self.calls: list[tuple[str, list[Message]]] = []
        #: What was *offered* on each call, in lockstep with ``calls``. Kept
        #: beside them rather than inside them so the tuple every existing case
        #: unpacks keeps its two fields.
        self.offers: list[Optional[list[dict[str, Any]]]] = []

    def calls_for(self, role: str) -> list[list[Message]]:
        return [m for r, m in self.calls if r == role]

    def offers_for(self, role: str) -> list[Optional[list[dict[str, Any]]]]:
        """What tools were offered to ``role``, call by call.

        The assertion this exists for is the one DL-028 rests on: `judge` and
        `reply` must offer **nothing**, and a widened seam that quietly started
        offering them tools would change the property M1 measures without
        changing a single test that reads only text.
        """
        return [o for (r, _), o in zip(self.calls, self.offers) if r == role]

    def complete(
        self,
        role: str,
        messages: Sequence[Message],
        tools: Optional[Sequence[dict[str, Any]]] = None,
    ) -> Response:
        _check_role(role)
        if not messages:
            raise ValueError("messages must not be empty")
        self.calls.append((role, list(messages)))
        self.offers.append(None if tools is None else list(tools))

        scripted = self._answers.get(role)
        if scripted is None:
            raise ProviderError(f"FakeProvider has no answers scripted for {role!r}")

        if callable(scripted):
            answer = scripted(role, list(messages))
        elif isinstance(scripted, list):
            if not scripted:
                raise ProviderError(
                    f"FakeProvider ran out of scripted {role!r} answers "
                    f"after {len(self.calls_for(role)) - 1}"
                )
            answer = scripted.pop(0)
        else:
            answer = scripted

        return self._as_response(role, answer)

    def _as_response(self, role: str, answer: object) -> Response:
        if isinstance(answer, Response):
            return answer
        if isinstance(answer, str):
            return Response(text=answer, model=self._model, finish_reason="stop")
        if isinstance(answer, ToolCall):
            answer = [answer]
        if (
            isinstance(answer, (list, tuple))
            and answer
            # Non-empty on purpose: an empty sequence would become a response
            # with no tool calls *and* no text, which is a scripted answer that
            # says nothing while looking deliberate.
            and all(isinstance(c, ToolCall) for c in answer)
        ):
            return Response(
                # Empty text beside tool calls, mirroring the live API's
                # `content: null` — the shape the seam now has to survive.
                text="",
                model=self._model,
                finish_reason="tool_calls",
                tool_calls=tuple(answer),
            )
        raise ProviderError(
            f"scripted {role!r} answer must be a string, a ToolCall, a sequence "
            f"of ToolCalls or a Response, got {type(answer).__name__}"
        )


# --- the module-level seam --------------------------------------------------

_provider: Optional[Provider] = None


def set_provider(provider: Optional[Provider]) -> Optional[Provider]:
    """Install the process-wide provider and return the previous one.

    One provider per process, matching the singleton rule: the alternative is
    two halves of one turn answered by different models, which would make the
    log's ``model`` field a lie.
    """
    global _provider
    previous, _provider = _provider, provider
    return previous


def complete(
    role: str,
    messages: Sequence[Message],
    tools: Optional[Sequence[dict[str, Any]]] = None,
) -> Response:
    """The seam. Everything above this line speaks roles, messages and tools.

    ``tools`` is passed straight through and defaults to none, so the two
    callers that predate DL-028 — `judge` and `reply` — reach the provider with
    the argument list they always had.
    """
    if _provider is None:
        raise ProviderNotConfigured(
            "no provider installed; call set_provider(provider_from_env()) at startup"
        )
    return _provider.complete(role, messages, tools)


def completer() -> Callable[..., Response]:
    """The seam as a value, for injecting into a loop rather than importing it."""
    return complete
