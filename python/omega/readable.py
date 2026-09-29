"""HTML to readable text, so what `fetch` hands the model is prose (DL-065).

`fetch` used to return the bytes the server sent. For a modern page that is
`<head>`, a few hundred lines of inline script, and a navigation menu — and
since the result is capped, the cap was spent before any sentence of the page
appeared. Measured on the live failure: a search URL came back 200 OK as 8,035
characters of "please enable JavaScript" containing zero results, and a
Wikipedia article spent most of its budget on `<script>` before the first
paragraph.

The job here is the one a browser's reader mode does and nothing more: drop what
is not content, keep the structure that carries meaning (headings, lists, links),
and emit text. It is deliberately **not** a readability scorer — no counting
words per node, no guessing which `<div>` is the article. Those heuristics are
where this class of code stops being legible, and a wrong guess silently deletes
the page. Dropping a closed set of tags is a rule you can read and predict.

Not a security boundary. What comes back is still untrusted input (DL-014) and
is still returned as data; this only changes how much of it is worth reading.
"""

from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import unquote, urljoin, urlsplit

__all__ = ["to_text", "looks_like_html", "DROPPED", "BLOCK", "VOID"]

#: Tags whose *contents* are discarded, not just their markup.
#:
#: Two kinds, and the distinction is worth keeping in mind when editing: script
#: and style are not prose in any rendering, so dropping them loses nothing.
#: nav, footer and aside are chrome — they are content, but they are the same
#: content on every page of a site, so they are repetition that crowds out the
#: part that differs.
#:
#: `header` is deliberately absent. It is the ambiguous one: sites use it both
#: for a site-wide menu and for an article's own title block, so dropping it
#: would sometimes delete the headline. Keeping it costs a menu; dropping it
#: could cost the thing the page is about.
DROPPED = frozenset(
    {
        "script", "style", "noscript", "template", "svg", "math", "canvas",
        "iframe", "object", "embed", "applet", "form", "select", "textarea",
        "button", "input", "label", "nav", "footer", "aside",
    }
)

#: Tags that end the current line. Everything else is inline.
BLOCK = frozenset(
    {
        "p", "div", "section", "article", "main", "header", "br", "hr",
        "ul", "ol", "li", "table", "tr", "td", "th", "thead", "tbody",
        "blockquote", "pre", "figure", "figcaption", "dl", "dt", "dd",
        "h1", "h2", "h3", "h4", "h5", "h6", "address", "details", "summary",
    }
)

#: Elements with no closing tag. They matter here because a DROPPED tag opens a
#: region that the matching end tag closes — and a void element has no end tag,
#: so treating one as a region swallows the rest of the document.
#:
#: This is not hypothetical: it is how the first version of this module lost a
#: Wikipedia article. The page's search box is an `<input>`, `</input>` never
#: arrives, and every paragraph after it was discarded. 119 characters came
#: back for a page with thousands of words.
VOID = frozenset(
    {
        "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr",
    }
)

_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def looks_like_html(content_type: str) -> bool:
    """Whether a body should be run through :func:`to_text`.

    By declared type and never by sniffing the body: a JSON API that happens to
    contain a `<` must come back as the JSON it is, and markdown that mentions a
    tag must not be reduced to its own text. The caller knows the header; that
    is the answer.
    """
    base = content_type.split(";", 1)[0].strip().lower()
    return base in ("text/html", "application/xhtml+xml")


class _Reader(HTMLParser):
    """Collects text, headings, list bullets and links; drops the rest."""

    def __init__(self, base_url: str = "") -> None:
        # convert_charrefs so `&amp;` and `&#8217;` arrive as characters. The
        # model should read an apostrophe, not an entity.
        super().__init__(convert_charrefs=True)
        self._base = base_url
        self._out: list[str] = []
        self._depth = 0          # how deep inside a DROPPED subtree we are
        self._title = ""
        self._in_title = False
        self._href: str | None = None
        self._link_text: list[str] = []

    # -- structure ----------------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in DROPPED and tag in VOID:
            # Nothing to drop: it has no contents and no end tag. Opening a
            # region here would never close. See VOID.
            return
        if self._depth:
            # Already inside dropped content. Track nesting so that the closing
            # tag of an *inner* `<script>` does not end the outer drop early.
            if tag in DROPPED:
                self._depth += 1
            return
        if tag in DROPPED:
            self._depth = 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag == "a":
            href = dict(attrs).get("href") or ""
            self._href = self._absolute(href)
            self._link_text = []
            return
        if tag in _HEADINGS:
            self._out.append(f"\n\n{'#' * _HEADINGS[tag]} ")
            return
        if tag == "li":
            self._out.append("\n- ")
            return
        if tag in BLOCK:
            self._out.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._depth:
            if tag in DROPPED:
                self._depth -= 1
            return
        if tag == "title":
            self._in_title = False
            return
        if tag == "a":
            self._close_link()
            return
        if tag in _HEADINGS or tag in BLOCK:
            self._out.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Self-closing, so it opens no region either — `<svg/>` must not
        # swallow the document any more than `<input>` may.
        if not self._depth and tag in ("br", "hr"):
            self._out.append("\n")

    def handle_data(self, data: str) -> None:
        if self._depth:
            return
        if self._in_title:
            self._title += data
            return
        if self._href is not None:
            self._link_text.append(data)
            return
        self._out.append(data)

    def close(self) -> None:  # noqa: D102 - a page that ends inside <a>
        self._close_link()
        super().close()

    # -- links ---------------------------------------------------------------

    def _absolute(self, href: str) -> str | None:
        """An absolute http(s) URL, or ``None`` for anything not worth keeping.

        `javascript:` and `mailto:` are not pages, and a bare `#anchor` points
        back at the page already being read. Relative links are resolved so the
        model gets something it could actually pass back to `fetch`.
        """
        href = href.strip()
        if not href or href.startswith("#"):
            return None
        resolved = urljoin(self._base, href) if self._base else href
        if urlsplit(resolved).scheme not in ("http", "https"):
            return None
        return _shortened(resolved)

    def _close_link(self) -> None:
        if self._href is None:
            return
        text = " ".join("".join(self._link_text).split())
        href, self._href, self._link_text = self._href, None, []
        if not text:
            return
        # A link whose text is already the URL is written once, not twice.
        self._out.append(text if text == href or not href else f"[{text}]({href})")

    # -- result --------------------------------------------------------------

    @property
    def title(self) -> str:
        return " ".join(self._title.split())

    def text(self) -> str:
        return "".join(self._out)


def _shortened(url: str) -> str:
    """The same URL, percent-decoding undone where that is safe.

    Measured, not guessed: link URLs are roughly half the text this module
    emits, and a non-Latin one is inflated about threefold by `%XX` — a
    Wikipedia interwiki link spends ~200 characters saying a word that is eight
    characters long. Undoing the encoding is free budget, and the URL still
    resolves, because a browser re-encodes on the way out.

    It is skipped when the decoded form contains whitespace or brackets, which
    would break the `[text](url)` around it and turn one link into corrupted
    prose. Cheaper to leave those encoded than to mangle the line.
    """
    decoded = unquote(url)
    if decoded == url:
        return url
    if any(c in decoded for c in " \t\n()[]<>\""):
        return url
    return decoded


def _tidy(text: str) -> str:
    """Collapse the whitespace the tags left behind.

    Line by line so that indentation inside a line is dropped but the line
    structure survives, then runs of blank lines are squeezed to one. Without
    this the output is mostly the whitespace that made the HTML readable.
    """
    lines = [" ".join(line.split()) for line in text.splitlines()]
    kept: list[str] = []
    for line in lines:
        if line or (kept and kept[-1]):
            kept.append(line)
    while kept and not kept[-1]:
        kept.pop()
    return "\n".join(kept)


def to_text(html: str, *, base_url: str = "") -> str:
    """Render `html` as readable text. Never raises.

    A parse failure returns the tags-stripped fallback rather than an error:
    the caller already has the page, and handing back *something* readable beats
    failing a fetch that succeeded. Malformed HTML is the normal case on the web,
    not the exception.
    """
    reader = _Reader(base_url)
    try:
        reader.feed(html)
        reader.close()
    except Exception:  # noqa: BLE001 - see the docstring: never fail a fetch
        pass
    body = _tidy(reader.text())
    title = reader.title
    if title and title not in body[:200]:
        return f"{title}\n\n{body}" if body else title
    return body
