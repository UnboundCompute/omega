"""The final handoff: a link, or a document, and never a payment (DL-097).

omega does the work up to the point where money or a commitment would move,
and then hands it over. This module is that hand: it **builds strings and
stores documents, and that is all it can do**. It holds no tool, no client and
no socket, so no path through it can reach a checkout, a payment, an order or
a booking. That is the money hard rule kept in one place by construction, and
``tests/test_handoff.py`` holds it there.

**Links.** Plain ``https`` URLs to the item or the search page, which open the
service's app on a phone that has it and the site everywhere else:

* directions on Google Maps (the documented ``maps/dir/?api=1`` form);
* a search on Amazon (amazon.in), Zomato, District and ixigo (flights);
* an item page they already have, kept only when it is ``https`` on that
  service's own domain.

**Documents.** Text omega wrote for them (a plan, a list, a summary) goes into
the blob store, named by its digest, and comes back as a marker,
``[attached: plan.md sha256:<hex>]``. A channel that can deliver a file (the
Discord DM, :mod:`omega.discord`) swaps the marker for the file; any other
reads the marker as it stands.

A requested handoff whose name reads as moving money (``pay``, ``checkout``,
``book``, ``order``) is refused with :func:`omega.mcp.money_refusal`'s own
sentence: the same rule, worded once.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlencode, urlsplit

from omega import blobs

__all__ = [
    "KINDS",
    "MODES",
    "MAX_DOCUMENT_BYTES",
    "MAX_QUERY",
    "HandoffError",
    "Attachment",
    "directions",
    "search",
    "item",
    "page",
    "open_line",
    "document",
    "MARKER",
    "markers",
    "build",
]

#: What :func:`build` hands over.
KINDS = ("directions", "amazon", "zomato", "district", "ixigo", "link", "document")
#: Google Maps' travel modes.
MODES = ("driving", "walking", "bicycling", "transit")
MAX_QUERY = 200
MAX_NAME = 80
#: A document is a page of text, not an archive: a Discord DM takes it whole.
MAX_DOCUMENT_BYTES = 512 * 1024

#: Each service's own domains: an item link is kept only on one of these.
_DOMAINS = {
    "amazon": ("amazon.in", "amazon.com", "amzn.in", "amzn.to"),
    "zomato": ("zomato.com",),
    "district": ("district.in",),
    "ixigo": ("ixigo.com",),
    "directions": ("google.com", "maps.app.goo.gl", "goo.gl"),
}

#: ``[attached: <name> sha256:<64 hex>]``, as :func:`document` writes it.
MARKER = re.compile(r"\[attached: ([A-Za-z0-9._ -]{1,80}) (sha256:[0-9a-f]{64})\]")


class HandoffError(ValueError):
    """What was asked for cannot be handed over; the message says why."""


def _text(value: Any, what: str, limit: int = MAX_QUERY) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HandoffError(f"{what} must be a non-empty string")
    words = " ".join(value.split())
    if len(words) > limit:
        raise HandoffError(f"{what} is longer than {limit} characters")
    return words


def directions(destination: str, origin: Optional[str] = None, mode: str = "driving") -> str:
    """Google Maps directions to ``destination``, from ``origin`` or from
    wherever the phone is when they open it."""
    if mode not in MODES:
        raise HandoffError(f"mode must be one of {list(MODES)}, not {mode!r}")
    query = {"api": "1"}
    if origin is not None:
        query["origin"] = _text(origin, "origin")
    query["destination"] = _text(destination, "destination")
    query["travelmode"] = mode
    return "https://www.google.com/maps/dir/?" + urlencode(query, quote_via=quote)


def _day(value: Any) -> date:
    try:
        return date.fromisoformat(_text(value, "date", 10))
    except ValueError as exc:
        raise HandoffError(f"date must be YYYY-MM-DD: {exc}") from None


def _airport(value: Any, what: str) -> str:
    code = _text(value, what, 3).upper()
    if not re.fullmatch(r"[A-Z]{3}", code):
        raise HandoffError(f"{what} must be a three-letter airport code (BLR), not {value!r}")
    return code


def search(service: str, query: str = "", *, city: Optional[str] = None,
           origin: Optional[str] = None, destination: Optional[str] = None,
           when: Optional[str] = None) -> str:
    """The service's own search page for ``query`` (for ixigo: the flights
    from ``origin`` to ``destination`` on ``when``)."""
    if service == "amazon":
        return "https://www.amazon.in/s?" + urlencode({"k": _text(query, "query")}, quote_via=quote)
    if service == "zomato":
        # Always Zomato's own search, never a city page built from ``city``:
        # its city slugs are irregular (``bangalore`` but not ``bengaluru``,
        # ``ncr`` but not ``delhi``), so a guessed slug is often a 404, and
        # Zomato finds the city from wherever the link is opened. ``city`` is
        # still accepted so a caller passing it is not refused.
        return "https://www.zomato.com/search?" + urlencode({"q": _text(query, "query")}, quote_via=quote)
    if service == "district":
        return "https://www.district.in/search?" + urlencode({"q": _text(query, "query")}, quote_via=quote)
    if service == "ixigo":
        day = _day(when)
        return "https://www.ixigo.com/search/result/flight?" + urlencode({
            "from": _airport(origin, "from"), "to": _airport(destination, "to"),
            "date": day.strftime("%d%m%Y"), "adults": "1", "children": "0",
            "infants": "0", "class": "e",
        })
    raise HandoffError(f"there is no search for {service!r}; there is {sorted(_DOMAINS)}")


def item(service: str, url: str) -> str:
    """``url`` itself, when it is ``https`` on ``service``'s own domain."""
    domains = _DOMAINS.get(service)
    if domains is None:
        raise HandoffError(f"there is no service {service!r}; there is {sorted(_DOMAINS)}")
    link = _text(url, "url", 2000)
    parts = urlsplit(link)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not any(host == d or host.endswith("." + d) for d in domains):
        raise HandoffError(f"{link!r} is not an https link on {service} ({', '.join(domains)})")
    return link


def page(url: Any, limit: int = 500) -> Optional[str]:
    """A page they asked about, handed back as it is, or ``None`` when it is
    not a plain http(s) link. The page watcher's alert link (DL-096)."""
    if isinstance(url, str) and re.match(r"^https?://\S+$", url) and len(url) <= limit:
        return url
    return None


def open_line(link: str) -> str:
    """The line that hands a link over at the end of a message."""
    return f"\nOpen it: {link}"


@dataclass(frozen=True)
class Attachment:
    """A document in the blob store, and the marker that delivers it."""

    name: str
    digest: str
    bytes: int

    @property
    def marker(self) -> str:
        return f"[attached: {self.name} {self.digest}]"


def _name(value: Any) -> str:
    name = _text(value, "name", MAX_NAME)
    name = re.sub(r"[^A-Za-z0-9._ -]+", "-", Path(name).name).strip(" .-")
    if not name:
        raise HandoffError("name must have a letter or a digit in it")
    return name if "." in name else name + ".md"


def document(store_root: Any, name: Any, text: Any) -> Attachment:
    """Store ``text`` as the document ``name`` and return it."""
    title = _name(name)
    if not isinstance(text, str) or not text.strip():
        raise HandoffError("a document needs its text")
    data = text.encode("utf-8")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise HandoffError(f"a document is at most {MAX_DOCUMENT_BYTES} bytes, not {len(data)}")
    store = blobs.BlobStore.open(store_root)
    # Through a file, as the executor stores derived text: the blob store
    # hashes a stream it opened itself, and has no second entry point.
    with tempfile.TemporaryDirectory(prefix="omega-handoff-") as tmp:
        scratch = Path(tmp) / title
        scratch.write_bytes(data)
        ref = store.put(scratch)
    return Attachment(name=title, digest=ref.digest, bytes=ref.bytes)


def markers(text: str) -> list[tuple[str, str]]:
    """The ``(name, digest)`` of every document marker in ``text``, in order."""
    return [(m.group(1), m.group(2)) for m in MARKER.finditer(text or "")]


def build(kind: str, args: dict[str, Any], *, store_root: Any = None) -> str:
    """One handoff from a tool's arguments: the words that carry it."""
    if not isinstance(kind, str) or kind not in KINDS:
        # Imported here: :mod:`omega.mcp` imports :mod:`omega.tools`, which
        # imports this module, and the refusal is only read on this path.
        from omega.mcp import money_refusal

        why = money_refusal(kind) if isinstance(kind, str) else None
        raise HandoffError(why or f"kind must be one of {list(KINDS)}, not {kind!r}")
    if kind == "document":
        if store_root is None:
            raise HandoffError("there is nowhere to keep a document in this turn")
        made = document(store_root, args.get("name"), args.get("text"))
        return (f"Stored {made.name} ({made.bytes} bytes). Put this marker in your "
                f"reply exactly as it is, and the document goes with it: {made.marker}")
    if kind == "link":
        link = page(args.get("url"))
        if link is None or not link.startswith("https://"):
            raise HandoffError("link needs an https url")
        return link
    if args.get("url") is not None:
        return item(kind, args["url"])
    if kind == "directions":
        return directions(args.get("destination"), args.get("origin"),
                          args.get("mode") or "driving")
    return search(kind, args.get("query", ""), city=args.get("city"),
                  origin=args.get("from"), destination=args.get("to"),
                  when=args.get("date"))
