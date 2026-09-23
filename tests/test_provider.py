"""M1 step 6 — the provider seam (`agent/M1_SPEC.md` §Q11, DL-024).

Green / red / yellow, as the rest of the suite.

**Nothing here touches the network or needs a key**, which is the property the
seam exists to give us: if asserting "the judge chose silence" required a live
call, it would be asserted rarely, and the one behaviour M1 adds a real model
for would go effectively untested.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from omega import provider as pv


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch, tmp_path):
    """Every test starts with no omega/OpenAI environment and no installed
    provider, so nothing passes because of the developer's own shell."""
    for key in ("OPENAI_API_KEY", "OMEGA_MODEL_JUDGE", "OMEGA_MODEL_ACT"):
        monkeypatch.delenv(key, raising=False)
    previous = pv.set_provider(None)
    monkeypatch.chdir(tmp_path)
    yield
    pv.set_provider(previous)


def _fake_openai_response(content, *, model="m", finish="stop", usage=(3, 5)):
    """The shape the real client returns, reproduced from the spec of the API
    rather than imported — so this stays a test of *our* mapping."""
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish
            )
        ],
        usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]),
    )


class _StubClient:
    def __init__(self, response=None, error=None):
        self._response, self._error = response, error
        self.seen = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.seen.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


# --- green ------------------------------------------------------------------


def test_the_seam_routes_to_the_installed_provider():
    fake = pv.FakeProvider({pv.JUDGE: ["stay_silent"]})
    pv.set_provider(fake)
    response = pv.complete(pv.JUDGE, [pv.user("anything happen?")])
    assert response.text == "stay_silent"
    assert fake.calls_for(pv.JUDGE) == [[pv.user("anything happen?")]]


def test_roles_map_to_different_models():
    """The whole reason the seam takes a role: judge fires on every tick and
    act fires rarely, so they must be able to differ."""
    os.environ["OMEGA_MODEL_JUDGE"] = "small-one"
    os.environ["OMEGA_MODEL_ACT"] = "big-one"
    assert pv.model_for(pv.JUDGE) == "small-one"
    assert pv.model_for(pv.ACT) == "big-one"
    assert pv.model_for(pv.JUDGE) != pv.model_for(pv.ACT)


def test_the_real_provider_maps_a_response():
    client = _StubClient(_fake_openai_response("hello", model="gpt-4o-2024"))
    provider = pv.OpenAIProvider(client=client, models={pv.ACT: "gpt-4o"})
    response = provider.complete(pv.ACT, [pv.user("hi")])

    assert response.text == "hello"
    assert response.model == "gpt-4o-2024", "reports what was used, not what was asked"
    assert response.finish_reason == "stop"
    assert response.total_tokens == 8
    assert client.seen[0]["model"] == "gpt-4o"


def test_load_env_reads_a_dotenv_file():
    Path(".env").write_text(
        "# a comment\n"
        "\n"
        "OPENAI_API_KEY=sk-test-123\n"
        "export OMEGA_MODEL_JUDGE='small-one'\n"
        'OMEGA_MODEL_ACT="big-one"\n',
        encoding="utf-8",
    )
    got = pv.load_env()
    assert got["OPENAI_API_KEY"] == "sk-test-123"
    assert os.environ["OMEGA_MODEL_JUDGE"] == "small-one", "quotes stripped"
    assert os.environ["OMEGA_MODEL_ACT"] == "big-one"


def test_provider_from_env_builds_when_configured():
    Path(".env").write_text("OPENAI_API_KEY=sk-test-123\n", encoding="utf-8")
    pv.load_env()
    assert os.environ["OPENAI_API_KEY"] == "sk-test-123"


def test_set_provider_returns_the_previous_one():
    a, b = pv.FakeProvider(), pv.FakeProvider()
    assert pv.set_provider(a) is None
    assert pv.set_provider(b) is a
    assert pv.set_provider(None) is b


# --- red --------------------------------------------------------------------


def test_calling_the_seam_with_no_provider_is_an_error():
    with pytest.raises(pv.ProviderNotConfigured):
        pv.complete(pv.JUDGE, [pv.user("x")])


def test_a_missing_key_fails_at_startup_not_at_first_call():
    """The stack should point at configuration, not at the middle of a turn an
    hour later."""
    with pytest.raises(pv.ProviderNotConfigured) as exc:
        pv.provider_from_env()
    assert "OPENAI_API_KEY" in str(exc.value)
    assert ".env" in str(exc.value), "the error should say where to put it"


def test_an_unknown_role_is_refused_everywhere():
    pv.set_provider(pv.FakeProvider({pv.JUDGE: ["x"]}))
    for call in (
        lambda: pv.complete("oracle", [pv.user("x")]),
        lambda: pv.model_for("oracle"),
        lambda: pv.FakeProvider().complete("oracle", [pv.user("x")]),
        lambda: pv.OpenAIProvider(client=_StubClient()).complete(
            "oracle", [pv.user("x")]
        ),
    ):
        with pytest.raises(ValueError):
            call()


def test_empty_messages_are_refused():
    for provider in (pv.FakeProvider({pv.JUDGE: ["x"]}),
                     pv.OpenAIProvider(client=_StubClient())):
        with pytest.raises(ValueError):
            provider.complete(pv.JUDGE, [])


def test_a_remote_failure_becomes_a_provider_error():
    client = _StubClient(error=ConnectionError("no route to host"))
    provider = pv.OpenAIProvider(client=client)
    with pytest.raises(pv.ProviderError) as exc:
        provider.complete(pv.JUDGE, [pv.user("x")])
    assert "no route to host" in str(exc.value)
    assert isinstance(exc.value.__cause__, ConnectionError), "original preserved"


def test_an_unreadable_response_is_a_provider_error():
    for broken in (SimpleNamespace(choices=[]), SimpleNamespace(), None):
        provider = pv.OpenAIProvider(client=_StubClient(broken))
        with pytest.raises(pv.ProviderError):
            provider.complete(pv.JUDGE, [pv.user("x")])


def test_the_fake_refuses_an_unscripted_role():
    """A fake that invents an answer makes a test pass for a reason it did not
    state."""
    with pytest.raises(pv.ProviderError):
        pv.FakeProvider({pv.JUDGE: ["x"]}).complete(pv.ACT, [pv.user("x")])


def test_the_fake_refuses_to_reuse_its_last_answer():
    fake = pv.FakeProvider({pv.JUDGE: ["one"]})
    assert fake.complete(pv.JUDGE, [pv.user("a")]).text == "one"
    with pytest.raises(pv.ProviderError) as exc:
        fake.complete(pv.JUDGE, [pv.user("b")])
    assert "ran out" in str(exc.value)


# --- yellow -----------------------------------------------------------------


def test_no_content_is_an_error_and_never_an_empty_string():
    """The load-bearing branch. A model that returns nothing is a *failure*; a
    model that decides to stay quiet is a *success*. If the first arrives as
    `""` the two become one row, and DL-024's silence metric starts counting
    outages as judgement."""
    client = _StubClient(_fake_openai_response(None, finish="content_filter"))
    provider = pv.OpenAIProvider(client=client)
    with pytest.raises(pv.ProviderError) as exc:
        provider.complete(pv.JUDGE, [pv.user("x")])
    assert "content_filter" in str(exc.value), "says why, not just that"

    # ...whereas a genuinely empty answer from a working call is not an error.
    ok = pv.OpenAIProvider(client=_StubClient(_fake_openai_response("")))
    assert ok.complete(pv.JUDGE, [pv.user("x")]).text == ""


def test_not_configured_is_a_provider_error_but_distinguishable():
    """Every other ProviderError might work next time; this one never will
    until a human edits .env. A caller that only wants "the call failed" should
    not have to name both."""
    assert issubclass(pv.ProviderNotConfigured, pv.ProviderError)
    with pytest.raises(pv.ProviderError):
        pv.complete(pv.JUDGE, [pv.user("x")])


def test_the_shell_beats_the_file_unless_overridden():
    """Exporting a key is the more deliberate act, so it wins. `override=True`
    exists because a test needs the file to be authoritative."""
    os.environ["OPENAI_API_KEY"] = "from-shell"
    Path(".env").write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")

    pv.load_env()
    assert os.environ["OPENAI_API_KEY"] == "from-shell"
    pv.load_env(override=True)
    assert os.environ["OPENAI_API_KEY"] == "from-file"


def test_a_missing_dotenv_is_not_an_error():
    """The key may legitimately come from the real environment; failing here
    would make that impossible."""
    assert not Path(".env").exists()
    assert pv.load_env() == {}


def test_load_env_ignores_junk_rather_than_crashing():
    """A malformed line should not stop a valid key on the next one from
    loading — the failure mode of a strict parser here is "omega will not
    start", for a stray line in a file nobody validates."""
    Path(".env").write_text(
        "this line has no equals sign\n"
        "=novalue\n"
        "OPENAI_API_KEY=sk-good\n",
        encoding="utf-8",
    )
    got = pv.load_env()
    assert got == {"OPENAI_API_KEY": "sk-good"}


def test_a_value_containing_equals_survives():
    """Keys and URLs routinely contain `=`; splitting on every one truncates
    them silently, which shows up as a baffling auth failure."""
    Path(".env").write_text("OPENAI_API_KEY=sk-a=b=c\n", encoding="utf-8")
    assert pv.load_env()["OPENAI_API_KEY"] == "sk-a=b=c"


def test_defaults_are_explicit_rather_than_the_library_s():
    """An invisible default is a decision nobody made — and the thing being
    defaulted is cost."""
    assert pv.model_for(pv.JUDGE) == pv._DEFAULT_MODELS[pv.JUDGE]
    assert pv.model_for(pv.ACT) == pv._DEFAULT_MODELS[pv.ACT]
    assert pv._DEFAULT_MODELS[pv.JUDGE] != pv._DEFAULT_MODELS[pv.ACT]
    assert set(pv._DEFAULT_MODELS) == pv.ROLES, "a role with no default"


def test_a_bare_omega_model_serves_both_roles():
    """Setting the one variable whose name says "model" must choose the model.

    The two-role split is ours and a reader of `.env` cannot infer it, so the
    failure this prevents is the quiet one: `OMEGA_MODEL=` was accepted by the
    file, read by nothing, and the defaults were billed instead.
    """
    os.environ["OMEGA_MODEL"] = "one-model"
    assert pv.model_for(pv.JUDGE) == "one-model"
    assert pv.model_for(pv.ACT) == "one-model"


def test_a_per_role_model_outranks_the_bare_one():
    """Otherwise the convenience would cost the cheap-judge design (DL-024):
    one variable would make every idle tick pay act-model prices."""
    os.environ["OMEGA_MODEL"] = "one-model"
    os.environ["OMEGA_MODEL_JUDGE"] = "small-one"
    assert pv.model_for(pv.JUDGE) == "small-one"
    assert pv.model_for(pv.ACT) == "one-model", "act still falls back to the bare one"


def test_an_empty_bare_model_is_not_a_model():
    """Same rule as the key: `OMEGA_MODEL=` is someone clearing it, and an
    empty model name reaches the API as a 404 about nothing."""
    os.environ["OMEGA_MODEL"] = ""
    assert pv.model_for(pv.JUDGE) == pv._DEFAULT_MODELS[pv.JUDGE]
    assert pv.model_for(pv.ACT) == pv._DEFAULT_MODELS[pv.ACT]


def test_an_empty_env_value_is_treated_as_unset():
    """`OPENAI_API_KEY=` in a .env is someone clearing it, not setting it to
    the empty string — and an empty key produces an opaque 401 rather than a
    configuration error."""
    Path(".env").write_text("OPENAI_API_KEY=   \n", encoding="utf-8")
    with pytest.raises(pv.ProviderNotConfigured) as exc:
        pv.provider_from_env()
    assert "OPENAI_API_KEY" in str(exc.value)


def test_a_whitespace_key_from_the_shell_is_also_unset():
    """The shell path skips `load_env`'s own stripping, so `provider_from_env`
    has to strip too. Without it, `export OPENAI_API_KEY="   "` reads as
    configured and surfaces as an opaque 401 from the API instead of a
    configuration error naming the variable.

    The assertion is on the *message*, not just the type: when `openai` is not
    installed the constructor raises `ProviderNotConfigured` for an unrelated
    reason, and a bare `pytest.raises` would pass either way — vacuously.
    """
    os.environ["OPENAI_API_KEY"] = "   \t "
    with pytest.raises(pv.ProviderNotConfigured) as exc:
        pv.provider_from_env()
    assert "OPENAI_API_KEY" in str(exc.value)
    assert "not installed" not in str(exc.value), "passed for the wrong reason"


def test_the_fake_records_what_was_asked_not_only_what_came_back():
    """The interesting M1 assertions are about how often judge ran and what it
    saw, which a provider that only returns text cannot answer."""
    fake = pv.FakeProvider({pv.JUDGE: ["a", "b"], pv.ACT: ["done"]})
    pv.set_provider(fake)
    pv.complete(pv.JUDGE, [pv.system("rules"), pv.user("one")])
    pv.complete(pv.ACT, [pv.user("do it")])
    pv.complete(pv.JUDGE, [pv.user("two")])

    assert [r for r, _ in fake.calls] == [pv.JUDGE, pv.ACT, pv.JUDGE]
    assert len(fake.calls_for(pv.JUDGE)) == 2
    assert fake.calls_for(pv.JUDGE)[0][0] == {"role": "system", "content": "rules"}


def test_the_fake_can_answer_from_a_callable():
    fake = pv.FakeProvider(
        {pv.JUDGE: lambda role, messages: f"saw:{messages[-1]['content']}"}
    )
    assert fake.complete(pv.JUDGE, [pv.user("ping")]).text == "saw:ping"


def test_the_fake_does_not_alias_the_caller_s_script():
    """Consuming answers must not mutate the dict the test handed in, or two
    tests sharing a fixture would interfere."""
    script = {pv.JUDGE: ["one", "two"]}
    fake = pv.FakeProvider(script)
    fake.complete(pv.JUDGE, [pv.user("x")])
    assert len(script[pv.JUDGE]) == 2, "the caller's list was consumed in place"


def test_recorded_messages_are_not_aliased_to_the_caller_s_list():
    """A caller reusing a list must not retroactively rewrite what a past call
    recorded — the same aliasing rule the episode codec enforces."""
    fake = pv.FakeProvider({pv.JUDGE: ["x"]})
    messages = [pv.user("original")]
    fake.complete(pv.JUDGE, messages)
    messages.append(pv.user("added later"))
    assert len(fake.calls[0][1]) == 1


# --- the widened seam: tool calls (M1 step 7, DL-028) ------------------------


def _tool_call(name="read_file", args='{"path": "/tmp/x"}', call_id="call_1"):
    """The wire shape of one tool call, reproduced from the API's spec rather
    than imported — so this stays a test of *our* mapping."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=args),
    )


def _with_tool_calls(calls, content=None, model="m", finish="tool_calls"):
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=calls),
                finish_reason=finish,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5),
    )


def test_a_caller_offering_no_tools_sends_the_request_it_always_sent():
    """The property DL-028 rests on: the widening is additive, so `judge` and
    `reply` cannot be regressed by it. Asserted on the *request keys*, because
    an extra `tools: None` on the wire is a change even when it is ignored."""
    client = _StubClient(_fake_openai_response("SILENT"))
    pv.OpenAIProvider(client=client).complete(pv.JUDGE, [pv.user("x")])
    assert set(client.seen[0]) == {"model", "messages"}
    assert "tools" not in client.seen[0]


def test_offered_tools_reach_the_request_untouched():
    schema = {"type": "function", "function": {"name": "read_file"}}
    client = _StubClient(_fake_openai_response("ok"))
    pv.OpenAIProvider(client=client).complete(pv.ACT, [pv.user("x")], [schema])
    assert client.seen[0]["tools"] == [schema]


def test_tool_calls_come_back_decoded_with_their_ids():
    client = _StubClient(
        _with_tool_calls(
            [
                _tool_call("read_file", '{"path": "/tmp/a"}', "call_a"),
                _tool_call("fetch", '{"url": "https://example.com/"}', "call_b"),
            ]
        )
    )
    response = pv.OpenAIProvider(client=client).complete(pv.ACT, [pv.user("x")])

    assert [c.id for c in response.tool_calls] == ["call_a", "call_b"]
    assert [c.name for c in response.tool_calls] == ["read_file", "fetch"]
    assert response.tool_calls[0].arguments == {"path": "/tmp/a"}
    assert response.tool_calls[1].arguments == {"url": "https://example.com/"}


def test_null_content_beside_tool_calls_is_not_an_error():
    """The collision DL-028 names. The API returns `content: null` *exactly
    when* the model answered with tool calls, and a model that said "run these"
    is not a model that said nothing."""
    client = _StubClient(_with_tool_calls([_tool_call()], content=None))
    response = pv.OpenAIProvider(client=client).complete(pv.ACT, [pv.user("x")])
    assert response.text == ""
    assert len(response.tool_calls) == 1


def test_null_content_with_no_tool_calls_still_raises():
    """The guard was narrowed, not removed. Without this the fix would have
    quietly handed `judge` an empty string, which is the one thing DL-024 says
    must never happen."""
    client = _StubClient(_with_tool_calls([], content=None, finish="length"))
    with pytest.raises(pv.ProviderError) as exc:
        pv.OpenAIProvider(client=client).complete(pv.JUDGE, [pv.user("x")])
    assert "length" in str(exc.value), "says why, not just that"


def test_a_response_with_no_tools_offered_carries_no_tool_calls():
    client = _StubClient(_fake_openai_response("SPEAK"))
    response = pv.OpenAIProvider(client=client).complete(pv.JUDGE, [pv.user("x")])
    assert response.tool_calls == ()


def test_unparseable_tool_arguments_are_a_provider_error_not_a_guess():
    """A half-read argument must never reach the approval classifier — the one
    place in omega that has to be handed exactly what the model asked for."""
    client = _StubClient(_with_tool_calls([_tool_call(args="{not json")]))
    with pytest.raises(pv.ProviderError) as exc:
        pv.OpenAIProvider(client=client).complete(pv.ACT, [pv.user("x")])
    assert "read_file" in str(exc.value)


def test_tool_arguments_that_are_not_an_object_are_refused():
    client = _StubClient(_with_tool_calls([_tool_call(args="[1, 2]")]))
    with pytest.raises(pv.ProviderError):
        pv.OpenAIProvider(client=client).complete(pv.ACT, [pv.user("x")])


def test_empty_tool_arguments_decode_to_an_empty_object():
    """A tool that takes nothing is sent `""`, and that is not malformed."""
    client = _StubClient(_with_tool_calls([_tool_call(args="")]))
    response = pv.OpenAIProvider(client=client).complete(pv.ACT, [pv.user("x")])
    assert response.tool_calls[0].arguments == {}


def test_the_fake_can_script_a_tool_call():
    """No network and no key, which is what makes the act sub-loop testable at
    all."""
    call = pv.ToolCall(id="c1", name="read_file", arguments={"path": "/tmp/a"})
    fake = pv.FakeProvider({pv.ACT: [[call], "all done"]})

    first = fake.complete(pv.ACT, [pv.user("x")], [{"name": "read_file"}])
    assert first.tool_calls == (call,)
    assert first.text == "", "mirrors the live API's content: null"

    second = fake.complete(pv.ACT, [pv.user("x")])
    assert second.tool_calls == () and second.text == "all done"


def test_the_fake_records_what_was_offered_and_to_whom():
    """DL-028's invariant, made assertable: judge is never offered a tool."""
    fake = pv.FakeProvider({pv.JUDGE: ["SPEAK"], pv.ACT: ["done"]})
    fake.complete(pv.JUDGE, [pv.user("x")])
    fake.complete(pv.ACT, [pv.user("x")], [{"name": "read_file"}])

    assert fake.offers_for(pv.JUDGE) == [None]
    assert fake.offers_for(pv.ACT) == [[{"name": "read_file"}]]
    assert [r for r, _ in fake.calls] == [pv.JUDGE, pv.ACT], "calls kept its shape"


def test_a_scripted_answer_that_is_neither_text_nor_tool_calls_is_refused():
    fake = pv.FakeProvider({pv.ACT: [42]})
    with pytest.raises(pv.ProviderError):
        fake.complete(pv.ACT, [pv.user("x")])


def test_the_assistant_tool_call_message_carries_ids_and_json_arguments():
    """The next pass must see the model's own request, and the API refuses a
    tool result that answers no call."""
    call = pv.ToolCall(id="c1", name="run_code", arguments={"argv": ["ls"]})
    message = pv.assistant_tool_calls([call], "let me look")

    assert message["role"] == "assistant" and message["content"] == "let me look"
    assert message["tool_calls"][0]["id"] == "c1"
    assert message["tool_calls"][0]["function"]["name"] == "run_code"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"argv": ["ls"]}'


def test_a_tool_result_message_names_the_call_it_answers():
    message = pv.tool_result("c1", "total 0")
    assert message == {"role": "tool", "tool_call_id": "c1", "content": "total 0"}
