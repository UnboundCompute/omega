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

**The transport is the Responses API, for every role.** Chat completions cannot
carry function tools for a reasoning model at all — it answers ``Function tools
with reasoning_effort are not supported ... use /v1/responses`` — and ``act`` is
both the role that calls tools *and* the role DL-024 reserves for the capable
model. Those two facts together mean chat completions cannot serve omega's core
loop, so the choice was never "which endpoint per model": it was to demote
``act`` to a weaker model, or to speak the endpoint that does the job. Demoting
the loop to keep the transport is the tail wagging the dog.

One transport rather than two, chosen per model, on purpose. A seam that picked
an endpoint by sniffing the model name would be wrong the day a name changed,
and one that tried chat first and fell back on a 400 would be the silent retry
this file refuses everywhere else. The Responses API serves the non-reasoning
models too, so the second path buys nothing and costs a branch that only the
live API can test.

**Messages stay in the shape callers already write.** :func:`system`,
:func:`assistant_tool_calls` and :func:`tool_result` are unchanged, and the
translation into Responses' item vocabulary happens here — which is the seam
doing exactly its job: the wire format moved and nothing above this line was
edited.

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
    "LEARN",
    "ROLES",
    "Message",
    "Part",
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
    "text_part",
    "image_part",
    "assistant_tool_calls",
    "tool_result",
]

#: Decide what this event deserves — including deciding it deserves nothing.
#: Runs on every event, so it is the cost-sensitive one.
JUDGE = "judge"

#: Do the work. Runs rarely; wants the capable model.
ACT = "act"

#: Turn a teaching note into claims (DL-043). Its own role rather than a second
#: consumer of ``judge``: the two want the same *kind* of model, but a model
#: change made for extraction must not silently re-calibrate the router that
#: runs on every event.
LEARN = "learn"

ROLES = frozenset({JUDGE, ACT, LEARN})

#: Default models if `.env` does not name one. Deliberately explicit rather than
#: "whatever the client library defaults to" — an invisible default is a
#: decision nobody made, and cost is the thing we are splitting roles over.
_DEFAULT_MODELS = {
    JUDGE: "gpt-4o-mini",
    ACT: "gpt-4o",
    LEARN: "gpt-4o-mini",
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
#:
#: ``learn`` is pinned to 0 for the judge's reason rather than the judge's job:
#: it reads one note and returns a fixed shape, and a sampled extraction would
#: file different claims from the same sentence on two runs — which makes the
#: receipt in DL-043 a check on a coin flip instead of on omega.
_DEFAULT_TEMPERATURES: dict[str, Optional[float]] = {
    JUDGE: 0.0,
    ACT: None,
    LEARN: 0.0,
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


Message = dict  # {"role": "system"|"user"|"assistant", "content": str | list}

#: One piece of a multimodal user turn. Responses calls these content blocks
#: and takes a list of them wherever a plain string would go.
Part = dict


def system(content: str) -> Message:
    return {"role": "system", "content": content}


def user(content: "str | Sequence[Part]") -> Message:
    """A user turn: either plain text, or a list of parts (DL-031).

    The list form is how an image reaches the model. It stays a *list of
    parts* rather than a dedicated ``images=`` argument because that is the
    shape the wire already has — a caller that wants text, then a picture,
    then more text about the picture writes exactly that, and the seam has no
    opinion about the order.
    """
    return {"role": "user", "content": content}


def text_part(text: str) -> Part:
    """Text inside a multimodal user turn.

    Only valid on an *input* item. Assistant text comes back as
    ``output_text``, which is why this is not the same thing as
    :func:`assistant` and is not reusable there.
    """
    return {"type": "input_text", "text": text}


def image_part(image_url: str) -> Part:
    """An image inside a multimodal user turn.

    ``image_url`` is either a URL the API can reach or a ``data:`` URI
    carrying the bytes. omega sends the second: the blobs are on this machine
    and DL-014 does not hand out a fetchable address for local bytes just to
    save an upload.
    """
    return {"type": "input_image", "image_url": image_url}


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


def reasoning_effort_for(role: str) -> Optional[str]:
    """``reasoning.effort`` for ``role``, or ``None`` to send no ``reasoning``.

    ``OMEGA_REASONING_EFFORT_JUDGE`` / ``_ACT``. Unset means omit the parameter
    entirely and let the model use its own default, because the alternative —
    picking one here — would silently apply to non-reasoning models that reject
    the field, and an invisible default is a decision nobody made.

    This is the reasoning models' replacement for :func:`temperature_for`: they
    reject ``temperature`` outright rather than ignoring it, and effort is the
    knob that actually moves cost. It is the lever DL-024's cheap/capable split
    reaches for once *both* roles are reasoning models and the model name is no
    longer the only thing separating them.
    """
    _check_role(role)
    raw = os.environ.get(f"OMEGA_REASONING_EFFORT_{role.upper()}", "").strip()
    return raw or None


def _as_input(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Canonical messages as Responses input items.

    Three shapes go in and three come out. A plain ``{role, content}`` passes
    through untouched — Responses takes it verbatim, and that is true whether
    ``content`` is a string or a list of parts (DL-031), which is the whole
    reason images cost this function nothing. The other two are the ones
    chat completions expresses as *fields on a message* and Responses expresses
    as *items in a list*:

    * an assistant turn carrying ``tool_calls`` becomes its text (when it said
      anything) followed by one ``function_call`` item per call;
    * a ``tool`` message becomes a ``function_call_output``.

    ``call_id`` is the join in both directions and is why :class:`ToolCall`
    keeps the id the wire gave it. Responses also hands back an item ``id``
    that is *not* the same string; pairing results by that one would look right
    and answer every call with the wrong output.
    """
    items: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message.get("content") or "",
                }
            )
            continue

        calls = message.get("tool_calls")
        if not calls:
            items.append({"role": message["role"], "content": message.get("content") or ""})
            continue

        # A model may reason aloud *and* call a tool in one turn; dropping the
        # text would delete the only record of why it chose those calls.
        if message.get("content"):
            items.append({"role": message["role"], "content": message["content"]})
        for call in calls:
            function = call.get("function", {})
            items.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": function.get("name", ""),
                    "arguments": function.get("arguments", "{}"),
                }
            )
    return items


def _as_tool_schemas(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool declarations flattened for Responses.

    Chat completions nests the declaration under a ``function`` key; Responses
    puts ``name``, ``description`` and ``parameters`` at the top level beside
    ``type``. Sending the nested shape is not an error the API explains well —
    it reports a missing ``name`` on a payload that plainly has one — so the
    flattening happens here rather than being spelled into every schema in
    `omega.tools`, which has no business knowing which endpoint is in use.
    """
    flattened: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function")
        if function is None:
            flattened.append(dict(tool))
            continue
        flattened.append({"type": tool.get("type", "function"), **function})
    return flattened


def _config_hint(
    role: str,
    model: str,
    tools: Optional[Sequence[dict[str, Any]]],
    exc: BaseException,
) -> str:
    """A sentence naming the variable to edit, when a 400 is really a config bug.

    Model choice is config, so a model rejecting a parameter is a config error —
    but it arrives as a raw 400 one layer below anything that knows a role
    exists, naming neither the role nor the variable that chose the model.
    Measured on a real store: every act pass failed and the turn was recorded
    ``failed`` with the provider's own wording, which is unactionable.

    Two traps, both from the same root: **a bare ``OMEGA_MODEL`` applies to
    every role**, so one edit silently re-points a role with different needs.

    Kept to a hint on an existing error rather than a preflight check. omega
    cannot know what a model supports without asking, and asking on startup
    would spend a call on every boot to answer a question that only matters
    when something has already gone wrong.
    """
    text = str(exc).lower()

    if "temperature" in text:
        # Reasoning models reject `temperature` outright instead of ignoring
        # it, and the judge is *pinned* to 0 for determinism — so the pin and
        # the model are in direct conflict and only a human can say which one
        # gives. `reasoning.effort` is that model's equivalent knob.
        return (
            f"\n\n{model} rejects the temperature parameter — reasoning models "
            f"do. The {role} role sends {temperature_for(role)!r}. Either set "
            f"OMEGA_TEMPERATURE_{role.upper()}=none to stop sending it (and "
            f"OMEGA_REASONING_EFFORT_{role.upper()} instead), or point "
            f"OMEGA_MODEL_{role.upper()} at a non-reasoning model. A bare "
            f"OMEGA_MODEL applies to every role, including this one."
        )

    if tools and ("tool" in text or "function" in text):
        return (
            f"\n\n{model} was offered {len(tools)} tools and rejected them. The "
            f"{role} role is the one that calls tools, so it needs a model that "
            f"supports function calling. Set OMEGA_MODEL_{role.upper()} to one "
            f"— a bare OMEGA_MODEL applies to every role, including this one."
        )

    return ""


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
        request: dict[str, Any] = {"model": model, "input": _as_input(messages)}
        temperature = temperature_for(role)
        if temperature is not None:
            request["temperature"] = temperature
        effort = reasoning_effort_for(role)
        if effort is not None:
            request["reasoning"] = {"effort": effort}
        if tools:
            # Added only when there are tools to offer, so a caller that passes
            # none sends no `tools` key at all — the property DL-028 rests on,
            # kept true across the change of transport.
            request["tools"] = _as_tool_schemas(tools)
        try:
            raw = self._client.responses.create(**request)
        except Exception as exc:  # noqa: BLE001 - every remote failure is one class
            raise ProviderError(
                f"{role} call to {model} failed: {exc}"
                f"{_config_hint(role, model, tools, exc)}"
            ) from exc

        text, tool_calls = _read_output(role, model, raw)

        if text is None and not tool_calls:
            # Never let this become an empty string: `judge` reading "" would be
            # indistinguishable from a model choosing to stay silent, and that
            # distinction is the whole point of measuring silence.
            #
            # A model that answered with tool calls and no words *has* said
            # something — "run these" — so the guard is narrowed to "nothing at
            # all came back" rather than "no text". Nothing reaching `judge` can
            # take that escape, because `judge` offers no tools and so can never
            # come back with any.
            raise ProviderError(
                f"{role} call to {model} returned no content "
                f"(status={getattr(raw, 'status', None)!r})"
            )

        usage = getattr(raw, "usage", None)
        return Response(
            # Only reachable as `None` when tool calls came back instead of
            # words, which is the model saying "run these", not saying nothing.
            text=text or "",
            model=getattr(raw, "model", model),
            finish_reason=getattr(raw, "status", None),
            # Responses counts the same two things under different names.
            prompt_tokens=getattr(usage, "input_tokens", None),
            completion_tokens=getattr(usage, "output_tokens", None),
            tool_calls=tool_calls,
        )


def _read_output(
    role: str, model: str, raw: Any
) -> tuple[Optional[str], tuple[ToolCall, ...]]:
    """The response's ``output`` list as text plus tool calls.

    Responses returns a *list of items* where chat completions returned one
    message with optional fields, so reading it is a walk rather than two
    attribute lookups. Three item types matter and the rest are skipped by
    design — a reasoning model emits ``reasoning`` items omega has no use for,
    and a transport that raised on an item type it did not recognise would
    break on the next one the API adds.

    Text is joined across message items rather than taking the first, because
    nothing promises there is only one, and dropping the others would lose part
    of an answer silently — the failure mode that is hardest to notice.
    """
    items = getattr(raw, "output", None)
    if items is None:
        raise ProviderError(
            f"{role} call to {model} returned an unreadable response"
        )

    parts: list[str] = []
    calls: list[ToolCall] = []
    spoke = False
    for item in items:
        kind = getattr(item, "type", None)
        if kind == "function_call":
            calls.append(_read_tool_call(role, model, item))
        elif kind == "message":
            # Seen at all, even carrying no text. That is the distinction the
            # caller's guard turns on: a model that produced a message and put
            # nothing in it *answered*, emptily; a model that produced no
            # message item at all did not answer. Collapsing those two makes an
            # outage read as a deliberate silence, which DL-024 exists to count
            # separately.
            spoke = True
            for block in getattr(item, "content", None) or ():
                chunk = getattr(block, "text", None)
                if chunk:
                    parts.append(chunk)
    return ("".join(parts) if spoke else None), tuple(calls)


def _read_tool_call(role: str, model: str, raw: Any) -> ToolCall:
    """One wire tool call as a :class:`ToolCall`, or a :class:`ProviderError`.

    Strict on purpose. Arguments arrive as a JSON *string*, and the three ways
    that can be wrong — unreadable object, invalid JSON, valid JSON that is not
    an object — are each a failed call rather than a guess. Guessing here would
    put a half-decoded argument in front of the approval classifier, which is
    the one place in omega that must never be handed something it cannot read.

    The id taken is ``call_id``, **not** ``id``. Responses carries both and they
    are different strings: ``id`` identifies the output item, ``call_id`` is
    what a ``function_call_output`` must quote to answer it. Taking the wrong
    one produces a request the API rejects as answering no call — or, worse, a
    pass where results and calls are paired by something that is not their join.
    """
    try:
        call_id = raw.call_id
        name = raw.name
        arguments = raw.arguments
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
