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
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence

__all__ = [
    "JUDGE",
    "ACT",
    "ROLES",
    "Message",
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
class Response:
    """What came back.

    ``text`` is the whole answer and is never ``None`` — a provider that
    returned nothing raises instead, because an empty string flowing into
    ``judge`` would be indistinguishable from a model choosing silence, and
    that is the one distinction M1 exists to measure (DL-024).

    ``model`` is recorded rather than assumed: it is what the provider *says*
    it used, which is the only honest answer when a name can alias to a
    dated snapshot.
    """

    text: str
    model: str
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    @property
    def total_tokens(self) -> Optional[int]:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


class Provider(Protocol):
    """Anything that can answer a role with text. The whole contract."""

    def complete(self, role: str, messages: Sequence[Message]) -> Response: ...


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

    def complete(self, role: str, messages: Sequence[Message]) -> Response:
        _check_role(role)
        if not messages:
            raise ValueError("messages must not be empty")

        model = self.model_for(role)
        try:
            raw = self._client.chat.completions.create(
                model=model, messages=list(messages)
            )
        except Exception as exc:  # noqa: BLE001 - every remote failure is one class
            raise ProviderError(f"{role} call to {model} failed: {exc}") from exc

        try:
            choice = raw.choices[0]
            text = choice.message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"{role} call to {model} returned an unreadable response"
            ) from exc

        if text is None:
            # Never let this become an empty string: `judge` reading "" would be
            # indistinguishable from a model choosing to stay silent, and that
            # distinction is the whole point of measuring silence.
            raise ProviderError(
                f"{role} call to {model} returned no content "
                f"(finish_reason={getattr(choice, 'finish_reason', None)!r})"
            )

        usage = getattr(raw, "usage", None)
        return Response(
            text=text,
            model=getattr(raw, "model", model),
            finish_reason=getattr(choice, "finish_reason", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )


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

    def calls_for(self, role: str) -> list[list[Message]]:
        return [m for r, m in self.calls if r == role]

    def complete(self, role: str, messages: Sequence[Message]) -> Response:
        _check_role(role)
        if not messages:
            raise ValueError("messages must not be empty")
        self.calls.append((role, list(messages)))

        scripted = self._answers.get(role)
        if scripted is None:
            raise ProviderError(f"FakeProvider has no answers scripted for {role!r}")

        if callable(scripted):
            text = scripted(role, list(messages))
        elif isinstance(scripted, list):
            if not scripted:
                raise ProviderError(
                    f"FakeProvider ran out of scripted {role!r} answers "
                    f"after {len(self.calls_for(role)) - 1}"
                )
            text = scripted.pop(0)
        else:
            text = scripted

        if not isinstance(text, str):
            raise ProviderError(
                f"scripted {role!r} answer must be a string, got {type(text).__name__}"
            )
        return Response(text=text, model=self._model, finish_reason="stop")


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


def complete(role: str, messages: Sequence[Message]) -> Response:
    """The seam. Everything above this line speaks roles and messages only."""
    if _provider is None:
        raise ProviderNotConfigured(
            "no provider installed; call set_provider(provider_from_env()) at startup"
        )
    return _provider.complete(role, messages)


def completer() -> Callable[[str, Sequence[Message]], Response]:
    """The seam as a value, for injecting into a loop rather than importing it."""
    return complete
