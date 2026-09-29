"""HTML becomes prose before the cap, and only HTML does (DL-065).

The capability: what `fetch` hands the model is the page's text, not its
markup. The violation that must not regress: nothing else is touched — a JSON
body, a plain-text body, and the address/scheme/type guards are all exactly as
they were, and the extractor never turns a fetch that succeeded into an error.
"""

from __future__ import annotations

import pytest

from omega import readable, tools


# --- what is dropped, and what is emphatically not --------------------------


def test_script_and_style_contents_are_gone_not_just_their_tags() -> None:
    """The live failure this module was written for: a search URL came back
    8,035 characters of `please enable JavaScript` with zero results in it."""
    out = readable.to_text(
        "<html><body><script>var x = 'SECRETJS';</script>"
        "<style>.a{color:red}</style><p>Real sentence.</p></body></html>"
    )
    assert "Real sentence." in out
    assert "SECRETJS" not in out
    assert "color:red" not in out


def test_a_void_element_does_not_swallow_the_rest_of_the_document() -> None:
    """The bug that cost a Wikipedia article, kept as a test.

    `input` is in DROPPED and is *void*: `</input>` never arrives. The first
    version opened a drop region on it that never closed, so a page's search box
    discarded every paragraph after it — 119 characters came back for an article
    of thousands of words. Any DROPPED tag that is also void has this shape.
    """
    for void in sorted(readable.DROPPED & readable.VOID):
        out = readable.to_text(f"<body><{void}><p>Survives {void}.</p></body>")
        assert f"Survives {void}." in out, void


def test_a_nested_script_does_not_end_the_drop_early() -> None:
    """The mirror of the void case: nesting must be counted, or the *outer*
    region reopens the document halfway through boilerplate."""
    out = readable.to_text(
        "<body><nav>menu<script>junk</script>still menu</nav><p>Content.</p></body>"
    )
    assert "Content." in out
    assert "still menu" not in out


def test_header_is_kept_even_though_nav_and_footer_are_not() -> None:
    """Sites use `header` for both a site menu and an article's title block, so
    dropping it would sometimes delete the headline. The asymmetry is deliberate
    and this is where it is written down."""
    out = readable.to_text(
        "<body><nav>NAVJUNK</nav><header><h1>The Headline</h1></header>"
        "<footer>FOOTJUNK</footer></body>"
    )
    assert "The Headline" in out
    assert "NAVJUNK" not in out
    assert "FOOTJUNK" not in out


# --- structure that carries meaning survives --------------------------------


def test_headings_and_lists_come_through_as_markdown() -> None:
    out = readable.to_text("<h2>Title</h2><ul><li>one</li><li>two</li></ul>")
    assert "## Title" in out
    assert "- one" in out
    assert "- two" in out


def test_a_link_keeps_its_text_and_an_absolute_url() -> None:
    """Relative, so that what comes back is something omega could pass to
    `fetch` again rather than a fragment it would have to reassemble."""
    out = readable.to_text(
        '<a href="/story/1">Read this</a>', base_url="https://news.example/tv/"
    )
    assert "[Read this](https://news.example/story/1)" in out


@pytest.mark.parametrize("href", ["javascript:evil()", "mailto:a@b.c", "#section"])
def test_a_link_that_is_not_a_page_keeps_its_text_and_loses_its_href(
    href: str,
) -> None:
    out = readable.to_text(f'<a href="{href}">Click</a>', base_url="https://e.example/")
    assert "Click" in out
    assert href not in out


def test_entities_arrive_as_characters() -> None:
    assert "Tom & Jerry's" in readable.to_text("<p>Tom &amp; Jerry&#39;s</p>")


# --- link URLs are half the output, so they are kept short ------------------


def test_a_percent_encoded_url_is_decoded_because_it_is_mostly_padding() -> None:
    """Measured: link URLs are about half of what this module emits, and a
    non-Latin one is inflated roughly threefold by `%XX`."""
    encoded = "https://hi.wikipedia.org/wiki/%E0%A4%AC%E0%A4%BF%E0%A4%97"
    out = readable.to_text(f'<a href="{encoded}">x</a>')
    assert "बिग" in out
    assert "%E0%A4" not in out


def test_a_url_is_left_encoded_when_decoding_it_would_break_the_link() -> None:
    """A space or a bracket in the decoded form would end the `](...)` early
    and turn one link into corrupted prose. Cheaper to leave it encoded."""
    out = readable.to_text('<a href="https://e.example/a%20b%28c%29">x</a>')
    assert "https://e.example/a%20b%28c%29" in out


# --- it never turns a successful fetch into a failure -----------------------


@pytest.mark.parametrize(
    "broken",
    [
        "<p>unclosed",
        "<<>><p>angle</p>",
        "<body><script>if (a < b) { }</script><p>after</p>",
        "",
        "<a href='http://e.example'>ends inside a link",
    ],
)
def test_malformed_html_returns_text_and_never_raises(broken: str) -> None:
    """Malformed HTML is the normal case on the web, not the exception, and the
    caller already has the page — failing here would fail a fetch that worked."""
    assert isinstance(readable.to_text(broken), str)


# --- and only HTML is touched ------------------------------------------------


@pytest.mark.parametrize(
    "content_type,expected",
    [
        ("text/html", True),
        ("text/html; charset=utf-8", True),
        ("application/xhtml+xml", True),
        ("application/json", False),
        ("text/plain", False),
        ("text/markdown", False),
        ("application/x-ndjson", False),
    ],
)
def test_only_declared_html_is_extracted(content_type: str, expected: bool) -> None:
    """By the declared type and never by sniffing: a JSON body that happens to
    contain a `<` must come back as the JSON it is."""
    assert readable.looks_like_html(content_type) is expected


def test_a_json_body_survives_a_fetch_unmangled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The violation control for the whole change. An API answer run through an
    HTML extractor is silently destroyed, and the model cannot tell."""
    from tests.test_tools import _FakeResponse, _opener_returning

    payload = b'{"items": [{"title": "a < b", "n": 3}]}'
    monkeypatch.setattr(
        tools.urllib.request,
        "build_opener",
        _opener_returning(_FakeResponse("application/json", payload)),
    )
    out = tools.fetch("http://fine.example/api", resolve=lambda h, p: ["93.184.216.34"])
    assert '{"items": [{"title": "a < b", "n": 3}]}' in out


def test_an_html_body_is_extracted_on_the_way_through_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The capability control, at the seam rather than in the extractor: it is
    `fetch` that has to call this, and a wiring mistake would leave every unit
    test above passing."""
    from tests.test_tools import _FakeResponse, _opener_returning

    page = b"<html><head><title>T</title></head><body><script>JUNKJS</script>"
    page += b"<h1>Headline</h1><p>The body.</p></body></html>"
    monkeypatch.setattr(
        tools.urllib.request,
        "build_opener",
        _opener_returning(_FakeResponse("text/html", page)),
    )
    out = tools.fetch("http://fine.example/", resolve=lambda h, p: ["93.184.216.34"])
    assert "# Headline" in out
    assert "The body." in out
    assert "JUNKJS" not in out


def test_a_page_gets_a_larger_cap_than_a_command_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prose earns the room that raw markup did not (DL-065): 8,000 characters
    of an extracted article is two paragraphs, where 8,000 of HTML was `<head>`
    either way."""
    from tests.test_tools import _FakeResponse, _opener_returning

    assert tools.MAX_FETCH_CHARS > tools.MAX_RESULT_CHARS
    page = b"<body><p>" + (b"word " * 12_000) + b"</p></body>"
    monkeypatch.setattr(
        tools.urllib.request,
        "build_opener",
        _opener_returning(_FakeResponse("text/html", page)),
    )
    out = tools.fetch("http://fine.example/", resolve=lambda h, p: ["93.184.216.34"])
    assert tools.MAX_RESULT_CHARS < len(out) <= tools.MAX_FETCH_CHARS + 100
