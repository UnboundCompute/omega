"""Ring 1's remainder and the gate in front of it — M1 step 7; DL-014, DL-028.

**Nothing here reaches the network or reads a key.** The address rule is tested
by calling the classifier on resolved addresses directly and by injecting a
resolver, which is the only honest way to test it anyway: a real request would
test whatever the network happened to be doing that minute, and the property
under test is what omega *refuses*, which is unobservable from a request that
was never sent.

One case is a deliberate exception and says so at the point of use: the pinned
transport binds a loopback server and serves itself one request. No DNS, no
external host, no key — but a socket, because a wiring fault under the address
check would leave every decision test green and `fetch` broken in practice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omega import tools
from omega.tools import (
    EXPLORATION,
    EXTERNAL,
    LOCAL,
    Decision,
    ToolBox,
    ToolError,
    ToolRejected,
)


@pytest.fixture
def box(tmp_path: Path) -> ToolBox:
    store = tmp_path / "memory"
    store.mkdir()
    return ToolBox(store_root=store)


# --- the set does not grow --------------------------------------------------


def test_ring_one_is_four_tools_and_the_registry_agrees() -> None:
    """DL-014: the tool set is closed. A fifth entry is a design decision, not
    an import."""
    assert tools.TOOL_NAMES == {"read_file", "write_file", "run_code", "fetch"}
    assert {s["function"]["name"] for s in tools.schemas()} == tools.TOOL_NAMES


def test_an_unknown_tool_is_refused_not_guessed(box: ToolBox) -> None:
    with pytest.raises(ToolRejected) as caught:
        box.classify("delete_everything", {})
    assert "does not grow" in str(caught.value)


# --- the classification table, whole ----------------------------------------


def test_read_file_is_exploration(box: ToolBox, tmp_path: Path) -> None:
    decision = box.classify("read_file", {"path": str(tmp_path / "anything.txt")})
    assert decision.tier == EXPLORATION
    assert decision.dispatches


def test_read_file_is_exploration_even_outside_the_store(box: ToolBox) -> None:
    """Reading is always allowed; what closes the exfiltration chain is `fetch`
    being external, not a path rule on reads."""
    assert box.classify("read_file", {"path": "/etc/hosts"}).tier == EXPLORATION


def test_a_write_inside_the_store_is_local(box: ToolBox) -> None:
    decision = box.classify(
        "write_file", {"path": str(box.store_root / "notes" / "x.md"), "content": "hi"}
    )
    assert decision.tier == LOCAL
    assert decision.dispatches


def test_a_write_outside_the_store_is_external(box: ToolBox, tmp_path: Path) -> None:
    decision = box.classify(
        "write_file", {"path": str(tmp_path / "elsewhere.txt"), "content": "hi"}
    )
    assert decision.tier == EXTERNAL
    assert not decision.dispatches
    assert "outside omega's own store" in decision.why


def test_a_write_escaping_the_store_by_dotdot_is_external(box: ToolBox) -> None:
    """The tier follows the *resolved* path, so `store/../secrets` does not
    inherit the store's tier from its spelling."""
    escaping = str(box.store_root / ".." / "secrets.txt")
    assert box.classify("write_file", {"path": escaping, "content": "x"}).tier == EXTERNAL


def test_a_write_through_a_symlink_out_of_the_store_is_external(
    box: ToolBox, tmp_path: Path
) -> None:
    """The reason paths are resolved rather than compared as text: a symlink
    inside the store must not launder a write to outside it."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (box.store_root / "door").symlink_to(outside)
    decision = box.classify(
        "write_file", {"path": str(box.store_root / "door" / "loot.txt"), "content": "x"}
    )
    assert decision.tier == EXTERNAL


def test_a_read_only_command_is_exploration(box: ToolBox, tmp_path: Path) -> None:
    decision = box.classify("run_code", {"argv": ["ls", "-l"], "cwd": str(tmp_path)})
    assert decision.tier == EXPLORATION


@pytest.mark.parametrize("program", ["rm", "python3", "sed", "git", "curl", "find"])
def test_anything_off_the_allowlist_is_external(
    box: ToolBox, tmp_path: Path, program: str
) -> None:
    """An allowlist, not a denylist: not being on it is the whole test, and
    each of these is here because it *would* be on a careless allowlist."""
    decision = box.classify("run_code", {"argv": [program], "cwd": str(tmp_path)})
    assert decision.tier == EXTERNAL
    assert program in decision.why


def test_the_allowlist_matches_argv0_exactly_and_not_its_basename(
    box: ToolBox, tmp_path: Path
) -> None:
    """`/tmp/evil/ls` must not inherit `ls`'s tier — basename matching would
    hand any attacker-writable directory an exploration tier."""
    decision = box.classify(
        "run_code", {"argv": [str(tmp_path / "evil" / "ls")], "cwd": str(tmp_path)}
    )
    assert decision.tier == EXTERNAL


def test_fetch_is_always_external(box: ToolBox) -> None:
    """Even for a plainly harmless URL. It is what stops read-then-send from
    running end to end with nobody in the loop."""
    decision = box.classify("fetch", {"url": "https://example.com/docs"})
    assert decision.tier == EXTERNAL
    assert not decision.dispatches


# --- arguments that cannot be read are refused, never guessed ---------------


def test_a_shell_string_for_argv_is_refused_not_split(box: ToolBox, tmp_path: Path) -> None:
    """Splitting it here would re-import the shell-text parsing that argv
    exists to avoid."""
    with pytest.raises(ToolRejected) as caught:
        box.classify("run_code", {"argv": "ls -l", "cwd": str(tmp_path)})
    assert "never runs a shell" in str(caught.value)


@pytest.mark.parametrize(
    "name,args",
    [
        ("read_file", {}),
        ("read_file", {"path": ""}),
        ("read_file", {"path": 7}),
        ("write_file", {"path": "/tmp/x", "content": 7}),
        ("run_code", {"argv": [], "cwd": "/tmp"}),
        ("run_code", {"argv": ["ls", 3], "cwd": "/tmp"}),
        ("run_code", {"argv": ["ls"]}),
        ("fetch", {}),
    ],
)
def test_unreadable_arguments_are_rejected(box: ToolBox, name: str, args: dict) -> None:
    with pytest.raises(ToolRejected):
        box.classify(name, args)


def test_non_object_arguments_are_rejected(box: ToolBox) -> None:
    with pytest.raises(ToolRejected):
        box.classify("read_file", ["/etc/hosts"])  # type: ignore[arg-type]


# --- the gate is the invariant, not advice ----------------------------------


def test_dispatch_refuses_an_external_decision(box: ToolBox) -> None:
    """A dispatch path reachable with an external decision would make the gate
    advisory, and an advisory gate is one refactor from no gate."""
    decision = box.classify("fetch", {"url": "https://example.com"})
    with pytest.raises(ToolRejected):
        box.dispatch(decision)


def test_a_decision_carries_the_arguments_it_was_made_about(box: ToolBox) -> None:
    """The thing classified has to be the thing that runs; re-reading the
    arguments at dispatch is how a gate approves one command and runs another."""
    decision = box.classify("read_file", {"path": "~/notes.txt"})
    assert decision.args["path"] == str(Path("~/notes.txt").expanduser().resolve())


# --- read_file --------------------------------------------------------------


def test_read_file_returns_the_contents(box: ToolBox, tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    target.write_text("remember the milk", encoding="utf-8")
    assert box.dispatch(box.classify("read_file", {"path": str(target)})) == (
        "remember the milk"
    )


def test_read_file_on_a_missing_file_is_a_tool_error(box: ToolBox, tmp_path: Path) -> None:
    with pytest.raises(ToolError):
        box.dispatch(box.classify("read_file", {"path": str(tmp_path / "nope")}))


def test_read_file_on_a_directory_is_a_tool_error(box: ToolBox, tmp_path: Path) -> None:
    with pytest.raises(ToolError):
        box.dispatch(box.classify("read_file", {"path": str(tmp_path)}))


def test_read_file_truncates_rather_than_returning_everything(tmp_path: Path) -> None:
    target = tmp_path / "big.txt"
    target.write_text("x" * 5000, encoding="utf-8")
    out = tools.read_file(str(target), max_bytes=100)
    assert out.startswith("x" * 100)
    assert "truncated" in out


def test_read_file_does_not_choke_on_bad_bytes(tmp_path: Path) -> None:
    target = tmp_path / "binary.bin"
    target.write_bytes(b"\xff\xfe ok")
    assert "ok" in tools.read_file(str(target))


# --- write_file -------------------------------------------------------------


def test_write_file_writes_and_makes_parents(box: ToolBox) -> None:
    target = box.store_root / "deep" / "er" / "note.md"
    box.dispatch(box.classify("write_file", {"path": str(target), "content": "hello"}))
    assert target.read_text(encoding="utf-8") == "hello"


# --- run_code ---------------------------------------------------------------


def test_run_code_reports_the_exit_status_and_output(box: ToolBox, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("", encoding="utf-8")
    out = box.dispatch(
        box.classify("run_code", {"argv": ["ls"], "cwd": str(tmp_path)})
    )
    assert "exit status 0" in out
    assert "a.txt" in out


def test_run_code_does_not_use_a_shell(tmp_path: Path) -> None:
    """The load-bearing property: argv[0] *is* the program, so shell grammar is
    inert data. If this ever passes through a shell, `ls; touch pwned` would
    create the file and the classifier's soundness argument collapses."""
    marker = tmp_path / "pwned"
    out = tools.run_code(["echo", "hi; touch pwned"], str(tmp_path))
    assert not marker.exists()
    assert "hi; touch pwned" in out


def test_run_code_reports_a_nonzero_exit_rather_than_hiding_it(tmp_path: Path) -> None:
    out = tools.run_code(["ls", "no-such-thing-here"], str(tmp_path))
    assert "exit status 0" not in out
    assert "stderr" in out


def test_run_code_times_out_rather_than_hanging(tmp_path: Path) -> None:
    with pytest.raises(ToolError) as caught:
        # `cat` with no argument would read stdin forever if stdin were a
        # terminal or a pipe; it is DEVNULL, so this asserts the *timeout* with
        # a sleep instead and the DEVNULL behaviour separately below.
        tools.run_code(["sleep", "5"], str(tmp_path), timeout=0.3)
    assert "did not finish" in str(caught.value)


def test_run_code_gives_the_child_no_stdin(tmp_path: Path) -> None:
    """A command that reads stdin must end, not wait for a person who is not
    there. This is the most common accident the bounds exist for."""
    out = tools.run_code(["cat"], str(tmp_path), timeout=5)
    assert "exit status 0" in out


def test_run_code_keeps_omegas_secrets_out_of_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fake value, never a real key: the point is that the name does not
    reach the child, and a made-up value proves that as well as a real one."""
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key-abc")
    monkeypatch.setenv("OMEGA_SOMETHING", "also-secret")
    out = tools.run_code(["env"], str(tmp_path))
    assert "not-a-real-key-abc" not in out
    assert "also-secret" not in out


def test_run_code_on_a_missing_program_is_a_tool_error(tmp_path: Path) -> None:
    with pytest.raises(ToolError):
        tools.run_code(["definitely-not-a-program-xyz"], str(tmp_path))


def test_run_code_refuses_a_cwd_that_is_not_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ToolError):
        tools.run_code(["ls"], str(tmp_path / "nowhere"))


# --- fetch: refusal by resolved address -------------------------------------


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "127.5.5.5",
        "0.0.0.0",
        "169.254.169.254",  # cloud instance metadata
        "10.0.0.7",
        "192.168.1.1",
        "172.16.0.1",
        "100.64.0.1",  # carrier-grade NAT: neither private nor global
        "224.0.0.1",
        "240.0.0.1",
        "::1",
        "fe80::1",
        "fc00::1",
        "::ffff:127.0.0.1",  # IPv4-mapped loopback
        "::",
    ],
)
def test_these_addresses_are_refused(address: str) -> None:
    assert tools.refusal_for_address(address) is not None


@pytest.mark.parametrize("address", ["93.184.216.34", "8.8.8.8", "2606:2800:220:1::1"])
def test_public_addresses_are_allowed(address: str) -> None:
    """The control. Without it the refusal test above would pass on a function
    that refuses everything, which is not a check."""
    assert tools.refusal_for_address(address) is None


def test_the_loopback_refusal_names_omegas_own_channel() -> None:
    """The concrete local reason, not an abstract one: omega's channel listens
    on 127.0.0.1:7717, so a fetch tool without this rule reaches into it."""
    why = tools.refusal_for_address("127.0.0.1")
    assert why is not None and "7717" in why


def test_an_unparseable_address_is_refused() -> None:
    assert tools.refusal_for_address("not-an-address") is not None


def test_a_host_is_judged_by_what_it_resolves_to_not_what_it_is_called() -> None:
    """DNS is the rebinding seam: the hostname string is the attacker's to
    choose, so a friendly name resolving to loopback must still be refused."""
    why = tools.refusal_for_host(
        "totally-fine.example", 443, resolve=lambda h, p: ["127.0.0.1"]
    )
    assert why is not None and "127.0.0.1" in why


def test_every_resolved_address_is_checked_not_only_the_first() -> None:
    """A name with a public *and* a loopback record would otherwise pass on the
    ordering of a DNS answer, which the attacker also chooses."""
    why = tools.refusal_for_host(
        "split.example", 443, resolve=lambda h, p: ["93.184.216.34", "127.0.0.1"]
    )
    assert why is not None


def test_a_host_that_resolves_to_nothing_is_refused() -> None:
    """Fails closed: resolving to nothing is not resolving to something safe."""
    assert tools.refusal_for_host("void.example", 443, resolve=lambda h, p: []) is not None


def test_a_public_host_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert (
        tools.refusal_for_host("example.com", 443, resolve=lambda h, p: ["93.184.216.34"])
        is None
    )


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://example.com/x", "gopher://example.com", "/x"]
)
def test_only_http_and_https_are_fetchable(url: str) -> None:
    """Every other scheme is a different capability wearing fetch's name."""
    with pytest.raises(ToolError):
        tools.check_url(url, resolve=lambda h, p: ["93.184.216.34"])


def test_check_url_refuses_a_loopback_url() -> None:
    with pytest.raises(ToolError) as caught:
        tools.check_url("http://localhost:7717/", resolve=lambda h, p: ["127.0.0.1"])
    assert "refusing to fetch" in str(caught.value)


def test_check_url_returns_the_host_when_it_is_allowed() -> None:
    assert (
        tools.check_url("https://example.com/a", resolve=lambda h, p: ["93.184.216.34"])
        == "example.com"
    )


def test_a_redirect_hop_onto_a_refused_address_is_stopped() -> None:
    """The ordinary shape of this attack: a public page answering
    `302 -> http://169.254.169.254/`. A first-hop-only check walks right past
    it, so the handler re-checks every hop. Driven directly — no request is
    made — because what is under test is the decision, not the transport.
    """
    handler = tools._CheckedRedirects()
    with pytest.raises(ToolError) as caught:
        handler.redirect_request(
            None, None, 302, "Found", {}, "http://169.254.169.254/latest/meta-data/"
        )
    assert "169.254.169.254" in str(caught.value)


def test_fetch_refuses_before_it_opens_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the refusal ever moved after the open, this would notice: the opener
    is replaced with one that fails loudly if it is ever used."""

    def explode(*args, **kwargs):
        raise AssertionError("fetch opened a connection to a refused address")

    monkeypatch.setattr(tools.urllib.request, "build_opener", explode)
    with pytest.raises(ToolError):
        tools.fetch("http://127.0.0.1:7717/", resolve=lambda h, p: ["127.0.0.1"])


# --- the check and the connection are about the same address ----------------


class _DeadSocket:
    """Enough of a socket for ``connect()`` to finish. Nothing is sent."""

    def setsockopt(self, *args: object) -> None:
        pass

    def close(self) -> None:
        pass


def test_the_address_that_was_vetted_is_the_address_dialled() -> None:
    """The whole mechanism in one assertion: the socket goes to the checked
    address, and the *name* stays on the connection for ``Host`` and for TLS.

    Carrying the address in ``host`` instead would have been the shorter fix
    and a worse one — the certificate would then be validated against an IP,
    which fails, and the tempting repair for that failure is to stop verifying.
    """
    dialled: list[tuple[str, int]] = []

    conn = tools._PinnedHTTPConnection(
        "example.com", address="93.184.216.34", port=80
    )
    conn._create_connection = lambda addr, *a, **kw: (  # type: ignore[assignment]
        dialled.append(addr),
        _DeadSocket(),
    )[1]
    conn.connect()

    assert dialled == [("93.184.216.34", 80)]
    assert conn.host == "example.com"


def test_tls_is_validated_against_the_name_not_the_pinned_address() -> None:
    """The claim the whole design rests on, and the one that would rot
    silently: if the address were carried in ``host``, the handshake would be
    validated against `93.184.216.34`, fail, and the obvious repair for *that*
    failure is to stop verifying — pinning would have bought an SSRF fix and
    sold a TLS one.
    """
    wrapped: dict[str, object] = {}

    class _Context:
        def wrap_socket(self, sock: object, server_hostname: str = "") -> object:
            wrapped["server_hostname"] = server_hostname
            return sock

    conn = tools._PinnedHTTPSConnection(
        "example.com", address="93.184.216.34", port=443
    )
    conn._create_connection = lambda addr, *a, **kw: _DeadSocket()  # type: ignore[assignment]
    conn._context = _Context()  # type: ignore[assignment]
    conn.connect()

    assert wrapped["server_hostname"] == "example.com"


def test_a_name_that_changes_its_answer_between_check_and_connect_is_refused() -> None:
    """**The regression test for the TOCTOU.** This resolver answers publicly
    the first time and loopback the second — DNS rebinding, exactly as it is
    done.

    The old code resolved once to decide and let urllib resolve again to
    connect, so the second answer was never judged by anything: the check
    passed on an address the connection did not use. Now the address that will
    be dialled is itself vetted, so the second answer is refused and the reason
    names it.
    """
    answers = iter([["93.184.216.34"], ["127.0.0.1"]])

    with pytest.raises(ToolError) as caught:
        tools.fetch("http://rebind.example/", resolve=lambda h, p: next(answers))
    assert "127.0.0.1" in str(caught.value)


def test_the_pin_vets_every_record_of_the_resolution_it_pins_from() -> None:
    handler = tools._PinnedHTTPHandler(lambda h, p: ["93.184.216.34", "127.0.0.1"])
    request = tools.urllib.request.Request("http://split.example/")
    with pytest.raises(ToolError):
        handler._address_for(request)


def test_the_pin_returns_the_vetted_address_for_a_good_name() -> None:
    handler = tools._PinnedHTTPHandler(lambda h, p: ["93.184.216.34"])
    request = tools.urllib.request.Request("http://fine.example/")
    assert handler._address_for(request) == "93.184.216.34"


def test_the_https_handler_builds_a_pinned_connection() -> None:
    """``https_open`` is the one wiring path no end-to-end case reaches, since
    a real TLS server would need a certificate — and a signature fault there
    would break every https fetch with all of these still green.

    So drive the factory ``do_open`` would call, with the arguments ``do_open``
    passes it. That covers the address lookup, the factory's signature, the
    forwarding of ``context`` and ``check_hostname``, and the construction
    itself.
    """
    handler = tools._PinnedHTTPSHandler(lambda h, p: ["93.184.216.34"])
    captured: dict[str, object] = {}

    def fake_do_open(factory, req, **kw):
        captured["conn"] = factory("example.com", timeout=5, **kw)
        return "opened"

    handler.do_open = fake_do_open  # type: ignore[assignment]
    request = tools.urllib.request.Request("https://example.com/a")
    assert handler.https_open(request) == "opened"

    conn = captured["conn"]
    assert isinstance(conn, tools._PinnedHTTPSConnection)
    assert conn._address == "93.184.216.34"
    assert conn.host == "example.com"


def test_vetted_address_refuses_rather_than_returning_a_verdict() -> None:
    """It hands back an address precisely so the caller cannot forget to look
    at a boolean — the failure mode the predicate form invites."""
    with pytest.raises(ToolError):
        tools.vetted_address("bad.example", 443, resolve=lambda h, p: ["169.254.169.254"])


def test_a_proxy_in_the_environment_is_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """A proxy would send every fetch to a host the address check never saw and
    the pin never covered: the guard would still pass and mean nothing.

    The environment variable is set on purpose. Without one in scope, an empty
    proxy map and *the system's* proxy map are the same empty dict, so the
    assertion would hold just as well against a handler that honours whatever
    it finds — a check that passes because nothing was there to catch.
    """
    monkeypatch.setenv("http_proxy", "http://someone-elses-host.example:8080")
    seen: list[object] = []

    def capture(*handlers: object):
        seen.extend(handlers)
        raise ToolError("stop here")

    monkeypatch.setattr(tools.urllib.request, "build_opener", capture)
    with pytest.raises(ToolError):
        tools.fetch("http://fine.example/", resolve=lambda h, p: ["93.184.216.34"])

    proxies = [h for h in seen if isinstance(h, tools.urllib.request.ProxyHandler)]
    assert proxies and proxies[0].proxies == {}


def test_the_pinned_transport_actually_connects(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one case that binds a socket, and it earns it.

    Everything above tests the *decision*; a wiring mistake in the opener chain
    would leave every one of them green and `fetch` broken for real, because
    nothing else drives ``do_open``. So this serves one request over loopback
    and checks both halves of the pin at once: the connection arrived, and the
    ``Host`` header carries the *name* rather than the address it dialled.

    ``refusal_for_address`` is neutralised for this case only — loopback is
    exactly what the rule exists to refuse, and the rule has its own tests
    directly above. What is under test here is the transport beneath it.
    """
    import http.server
    import threading

    seen: dict[str, str] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - the stdlib's spelling
            seen["host"] = self.headers.get("Host", "")
            body = b"hello from the pin"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(tools, "refusal_for_address", lambda address: None)
        text = tools.fetch(
            f"http://pinned.example:{port}/",
            resolve=lambda h, p: ["127.0.0.1"],
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert "hello from the pin" in text
    assert seen["host"] == f"pinned.example:{port}"


# --- fetch returns pages, not binaries --------------------------------------


@pytest.mark.parametrize(
    "kind", ["text/html", "text/plain", "application/json", "application/ld+json",
             "image/svg+xml", "TEXT/HTML"]
)
def test_a_textual_type_is_readable(kind: str) -> None:
    assert tools.refusal_for_content_type(kind) is None


@pytest.mark.parametrize(
    "kind", ["application/pdf", "image/png", "video/mp4", "font/woff2",
             "application/octet-stream", "application/zip", ""]
)
def test_a_binary_type_is_refused(kind: str) -> None:
    """An allowlist because the failure is open-ended: the binary types cannot
    be enumerated, the textual ones nearly can."""
    assert tools.refusal_for_content_type(kind) is not None


class _FakeResponse:
    def __init__(self, content_type: str, body: bytes) -> None:
        from email.message import Message

        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self._body = body
        self.status = 200

    def read(self, n: int) -> bytes:
        return self._body[:n]

    def geturl(self) -> str:
        return "http://fine.example/"

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        pass


def _opener_returning(response: _FakeResponse):
    class _Opener:
        def open(self, request: object, timeout: float = 0) -> _FakeResponse:
            return response

    return lambda *handlers: _Opener()


def test_fetch_refuses_a_binary_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refused on the header, before the body is read: `max_bytes` of a video
    is still a download, and a megabyte of replacement characters is not a
    page the model can even report as broken."""
    response = _FakeResponse("application/pdf", b"%PDF-1.7 binary junk")
    monkeypatch.setattr(
        tools.urllib.request, "build_opener", _opener_returning(response)
    )
    with pytest.raises(ToolError) as caught:
        tools.fetch("http://fine.example/x.pdf", resolve=lambda h, p: ["93.184.216.34"])
    assert "not text" in str(caught.value)


def test_fetch_decodes_with_the_charset_the_server_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Latin-1 text decoded as utf-8 is mojibake, and mojibake in context is
    indistinguishable from a page that was written that way."""
    response = _FakeResponse("text/plain; charset=latin-1", "café".encode("latin-1"))
    monkeypatch.setattr(
        tools.urllib.request, "build_opener", _opener_returning(response)
    )
    assert "café" in tools.fetch(
        "http://fine.example/", resolve=lambda h, p: ["93.184.216.34"]
    )


def test_an_unknown_charset_does_not_fail_the_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse("text/plain; charset=nonesuch-9", b"hello")
    monkeypatch.setattr(
        tools.urllib.request, "build_opener", _opener_returning(response)
    )
    assert "hello" in tools.fetch(
        "http://fine.example/", resolve=lambda h, p: ["93.184.216.34"]
    )


# --- results are capped -----------------------------------------------------


def test_a_result_is_capped_so_the_log_and_the_model_saw_the_same_thing() -> None:
    capped = tools._cap("y" * (tools.MAX_RESULT_CHARS + 500))
    assert len(capped) < tools.MAX_RESULT_CHARS + 200
    assert "truncated" in capped


def test_a_decision_is_frozen() -> None:
    """A gate whose verdict can be edited after the fact is not a gate."""
    decision = Decision(tool="fetch", tier=EXTERNAL, why="x")
    with pytest.raises(Exception):
        decision.tier = EXPLORATION  # type: ignore[misc]
