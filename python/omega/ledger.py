"""The told-ledger (C1, DL-109): what omega saw, and what it has told them.

One Python fold over the log, droppable and rebuilt like every other view.
It replaces "what did I last say, in prose" as the guard against telling
them the same thing twice: a thread has a **key**, set by code from fields
that already exist, and every unprompted message names the keys it spoke for.

**Keys come from code only.** A mail is ``mail:<Message-ID>`` (or
``mail:<mailbox>#<uid>`` when it has none); a CI failure mail is
``ci:<repo>/<workflow>``, parsed from the subject; a Slack notification mail
``slack:<sender>:<ts>``, from the message link in its body; a calendar event
``cal:<uid>@<start>``; a reminder they set ``reminder:<id>``. No model ever
names a key, so no model can collide two threads or split one.

**Admission** (what opens a key): mail from a person, a CI failure, travel
mail, a non-recurring calendar event, and their own words (a reminder).
Bulk mail (List-Unsubscribe, Precedence, Auto-Submitted) is one aggregate
observation per read and never a key. CI, Slack and travel mail are checked
before bulk, because GitHub and airlines send with list headers.

**What the fold answers**, per key: when it was last observed and how often
(``recurrence``: distinct items on the one thread, so a second failure of the
same workflow is 2), when and with what words it was last told, whether it is
open, and whether there is anything new since it was told or last
considered. It also counts unprompted sends with no key, which is the
coverage the M1a gate reads.

**Reminders absorb.** An item a reminder they set will cover within
:data:`REMINDER_ABSORB_SECONDS` is left to the reminder (DL-075 keeps the
reminder never-silent; nothing else repeats it). Whether a reminder covers an
item is a code join (a shared flight or booking code, or a deadline just
after the reminder and a shared word): a heuristic, starting guess [A].
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

from omega import episodes

if TYPE_CHECKING:  # pragma: no cover
    from omega.memory import MemoryStore

__all__ = [
    "REMINDER_ABSORB_SECONDS",
    "FRESH_SECONDS",
    "Item",
    "State",
    "Ledger",
    "mail_key",
    "ci_of",
    "slack_ts",
    "is_travel",
    "calendar_key",
    "mail_observations",
    "event_observations",
    "record_mails",
    "record_events",
    "normalized",
    "covers",
    "omega_authored",
]

#: A reminder due this soon takes the items it covers (C3) [A].
REMINDER_ABSORB_SECONDS = 6 * 3600
#: How old an item may be and still be offered to the composer: the look reads
#: a few days of mail, and a mail from three days ago is not news.
FRESH_SECONDS = 24 * 3600
#: Whose words an observation's ``what`` repeats, when it repeats a mail's.
INBOX_SOURCE = "their inbox"
_REMINDER_LEAD = "Reminder you promised them: "
_PROMISE_REMINDER = re.compile(r"^pr[1-9][0-9]*$")
_MODEL_REMINDER = re.compile(r"^r([1-9][0-9]*)-")
_UNPROMPTED = frozenset({"self", "schedule", "task"})

#: Their answer to the one question about routine meetings (DL-110), in their
#: own words on a channel they write on. The latest one stands.
_ROUTINE = r"(?:routine|recurring|regular|standups?|stand-ups?)"
ROUTINE_ON = re.compile(rf"\bwarn me (?:before|about) (?:my |the )?{_ROUTINE}", re.I)
ROUTINE_OFF = re.compile(
    rf"\b(?:stop|no more|don'?t send|do not send) (?:the |my )?{_ROUTINE}"
    r"(?: meetings?)? warnings?\b", re.I)

# --- keys, from code ------------------------------------------------------------

_PRINTABLE = re.compile(r"^[!-~]{1,280}$")


def _safe(part: str) -> str:
    """A key part that is printable ASCII with no space, or its digest."""
    part = part.strip().strip("<>")
    if _PRINTABLE.match(part):
        return part
    return "h" + hashlib.sha256(part.encode("utf-8", "replace")).hexdigest()[:16]


def key(prefix: str, part: Any) -> str:
    """A key for something omega itself tells (``outage:look``,
    ``access:google``, ``watch:w12-ab``): a fixed prefix and a safe part."""
    return f"{prefix}:{_safe(str(part)) or 'x'}"


def mail_key(mail: Any, box: str = "personal") -> str:
    """``mail:<Message-ID>``, or ``mail:<box>#<uid>`` when it has none."""
    mid = str(getattr(mail, "message_id", "") or "").strip()
    if mid:
        return f"mail:{_safe(mid)}"
    return f"mail:{_safe(box)}#{int(getattr(mail, 'uid', 0))}"


#: GitHub Actions failure subjects, in the two shapes GitHub has sent:
#: "[owner/repo] Run failed: CI - main (abc1234)" and
#: "[owner/repo] CI workflow run failed for main branch".
_NAME = r"[A-Za-z0-9_.-]{1,100}"
_WORKFLOW = r"[A-Za-z0-9 _.()/-]{1,60}?"
_CI_SUBJECTS = (
    re.compile(rf"^\[(?P<owner>{_NAME})/(?P<repo>{_NAME})\]\s+Run failed:\s+(?P<wf>{_WORKFLOW})\s+-\s+\S"),
    re.compile(rf"^\[(?P<owner>{_NAME})/(?P<repo>{_NAME})\]\s+(?P<wf>{_WORKFLOW})\s+workflow run failed\b"),
)


@dataclass(frozen=True)
class CI:
    """A CI failure, as code read it from the subject and body."""

    owner: str
    repo: str
    workflow: str
    #: The run's page, only when the body links a run of this very repo.
    url: Optional[str] = None

    @property
    def key(self) -> str:
        slug = "-".join(self.workflow.lower().split())
        return f"ci:{_safe(self.repo)}/{_safe(slug)}"


def ci_of(mail: Any) -> Optional[CI]:
    """The CI failure a mail reports, or ``None``. Code only: the subject's
    shape, with names held to a narrow character set, and a run link taken
    only when it is exactly ``https://github.com/<owner>/<repo>/actions/runs/<n>``
    for the repo the subject names."""
    sender = parseaddr(str(getattr(mail, "sender", "") or ""))[1].lower()
    if not (sender.endswith("@github.com") or sender.endswith(".github.com")):
        return None
    subject = " ".join(str(getattr(mail, "subject", "") or "").split())
    for pattern in _CI_SUBJECTS:
        m = pattern.match(subject)
        if m is None:
            continue
        owner, repo, wf = m.group("owner"), m.group("repo"), m.group("wf").strip()
        if not wf:
            return None
        body = f"{getattr(mail, 'snippet', '') or ''} {getattr(mail, 'text', '') or ''}"
        link = re.search(
            rf"https://github\.com/{re.escape(owner)}/{re.escape(repo)}/actions/runs/([0-9]{{1,20}})(?![0-9A-Za-z/])",
            body,
        )
        url = (
            f"https://github.com/{owner}/{repo}/actions/runs/{link.group(1)}" if link else None
        )
        return CI(owner=owner, repo=repo, workflow=wf, url=url)
    return None


_SLACK_LINK = re.compile(r"https://[a-z0-9-]{1,63}\.slack\.com/archives/[A-Z0-9]{1,20}/p([0-9]{10})([0-9]{6})")


def slack_ts(mail: Any) -> Optional[str]:
    """The Slack message ts a notification mail links to, or ``None``."""
    sender = parseaddr(str(getattr(mail, "sender", "") or ""))[1].lower()
    if not sender.endswith("slack.com"):
        return None
    body = f"{getattr(mail, 'snippet', '') or ''} {getattr(mail, 'text', '') or ''}"
    m = _SLACK_LINK.search(body)
    return f"{m.group(1)}.{m.group(2)}" if m else None


_TRAVEL = re.compile(
    r"\b(PNR|booking (?:is )?confirmed|e-?ticket|boarding pass|itinerary|web check-?in|flight (?:is )?(?:confirmed|delayed|cancelled|rescheduled))\b",
    re.I,
)


def is_travel(mail: Any) -> bool:
    return bool(_TRAVEL.search(str(getattr(mail, "subject", "") or "")))


def calendar_key(occurrence: Any) -> str:
    uid = str(getattr(occurrence, "uid", "") or "") or str(getattr(occurrence, "summary", ""))
    return f"cal:{_safe(uid)}@{occurrence.start.astimezone(timezone.utc).isoformat()}"


def _clip(text: str, cap: int = 300) -> str:
    flat = " ".join("".join(c if c.isprintable() else " " for c in str(text)).split())
    return flat if len(flat) <= cap else flat[: cap - 1].rstrip() + "…"


def _who(sender: str) -> str:
    name, addr = parseaddr(sender)
    return _clip(name or addr or "unknown sender", 80)


# --- what the senses write ------------------------------------------------------


def mail_observations(
    mails: Sequence[Any], seen: "State", now: datetime, *, box: str = "personal"
) -> list[dict[str, Any]]:
    """The observations one inbox reading adds: one per newly seen admitted
    mail, and one aggregate for newly seen bulk mail. Held mail (DL-082) is
    never read. Every ``what`` repeats a sender's words, so every one is
    marked :data:`INBOX_SOURCE`."""
    out: list[dict[str, Any]] = []
    bulk: list[str] = []
    at = now.isoformat()
    for m in mails:
        if getattr(m, "held", False):
            continue
        ref = mail_key(m, box)
        when = m.date.isoformat() if getattr(m, "date", None) is not None else None
        ci = ci_of(m)
        ts = None if ci else slack_ts(m)
        if ci is not None:
            key, source = ci.key, "ci"
            what = f"CI failed: {ci.workflow} on {ci.owner}/{ci.repo}"
        elif ts is not None:
            addr = parseaddr(str(m.sender))[1].lower() or "slack"
            key, source = f"slack:{_safe(addr)}:{ts}", "slack"
            what = f"Slack, via mail: {m.subject or '(no subject)'}"
        elif is_travel(m):
            key, source = ref, "travel"
            what = f"travel mail from {_who(m.sender)}: {m.subject or '(no subject)'}"
        elif getattr(m, "bulk", False):
            if not seen.has_ref(None, ref):
                bulk.append(ref)
            continue
        else:
            key, source = ref, "mail"
            what = f"mail from {_who(m.sender)}: {m.subject or '(no subject)'}"
        if seen.has_ref(key, ref):
            continue
        out.append(episodes.observation_recorded(
            key=key, source=source, what=_clip(what), ref=ref, untrusted=INBOX_SOURCE,
            url=ci.url if ci is not None else None, when=when, at=at,
        ))
    if bulk:
        bulk = bulk[: 200]
        out.append(episodes.observation_recorded(
            key=None, source="bulk", what=f"{len(bulk)} newsletters/notifications",
            refs=bulk, at=at,
        ))
    return out


def event_observations(occurrences: Sequence[Any], seen: "State", now: datetime) -> list[dict[str, Any]]:
    """One observation per newly seen non-recurring calendar event (C1: a
    routine one is not news; M1b warns on its change). Its deadline is its
    start."""
    out = []
    for o in occurrences:
        if getattr(o, "recurring", False):
            continue
        key = calendar_key(o)
        if seen.has_ref(key, key):
            continue
        when = o.start.astimezone().strftime("%a %d %b %H:%M") if not o.all_day else (
            o.start.astimezone().strftime("%a %d %b, all day"))
        what = f"{o.summary} ({when})" + (f" at {o.location}" if o.location else "")
        out.append(episodes.observation_recorded(
            key=key, source="calendar", what=_clip(what), ref=key,
            deadline=o.start.isoformat(), when=o.start.isoformat(), at=now.isoformat(),
        ))
    return out


def record_mails(queue: Any, mails: Sequence[Any], now: datetime) -> int:
    """Append what one inbox reading adds to the ledger; the count appended."""
    found = mail_observations(mails, Ledger.rebuild(queue.store).state(), now)
    for payload in found:
        queue.append(payload)
    return len(found)


def record_events(queue: Any, occurrences: Sequence[Any], now: datetime) -> int:
    found = event_observations(occurrences, Ledger.rebuild(queue.store).state(), now)
    for payload in found:
        queue.append(payload)
    return len(found)


# --- the fold -------------------------------------------------------------------


def normalized(text: Optional[str]) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", str(text or "").lower()).split())


def _parse(at: Any) -> Optional[datetime]:
    try:
        moment = datetime.fromisoformat(str(at))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


@dataclass
class Item:
    """One thread, as the fold knows it."""

    key: str
    source: str
    what: str
    first_seq: int
    last_seq: int
    last_at: str
    when: Optional[str] = None
    deadline: Optional[str] = None
    url: Optional[str] = None
    refs: set = field(default_factory=set)
    observations: int = 0
    untrusted: tuple = ()
    told_seq: int = 0
    told_at: Optional[str] = None
    told_text: Optional[str] = None
    considered_seq: int = 0
    closed: bool = False

    @property
    def recurrence(self) -> int:
        return max(len(self.refs), self.observations)

    @property
    def fresh(self) -> bool:
        """Open, with something observed since it was last told or weighed."""
        return not self.closed and self.last_seq > max(self.told_seq, self.considered_seq)

    @property
    def reminder(self) -> bool:
        return self.key.startswith("reminder:")


@dataclass(frozen=True)
class State:
    through: int = 0
    items: dict = field(default_factory=dict)
    bulk_refs: frozenset = frozenset()
    #: Unprompted messages that reached them, and how many named no key.
    sends: int = 0
    unkeyed: int = 0
    #: Keys told again with nothing observed since they were last told.
    repeats: int = 0
    #: Every key an event has named, told or still waiting to be: what a sense
    #: that tells once (C2's T-minus, its one question) checks before telling.
    named: frozenset = frozenset()
    #: Whether they asked to be warned before routine meetings too (DL-110).
    routine_warnings: bool = False

    def has_ref(self, key: Optional[str], ref: str) -> bool:
        if key is None:
            return ref in self.bulk_refs
        item = self.items.get(key)
        return item is not None and ref in item.refs

    def get(self, key: str) -> Optional[Item]:
        return self.items.get(key)

    def reminders(self, now: datetime, *, within: int = REMINDER_ABSORB_SECONDS) -> list[Item]:
        """Their open reminders due in the next ``within`` seconds."""
        out = []
        for item in self.items.values():
            due = _parse(item.deadline) if item.reminder and not item.closed else None
            if due is not None and 0 <= (due - now).total_seconds() <= within:
                out.append(item)
        return out

    def absorbed(self, item: Item, now: datetime, *, within: int = REMINDER_ABSORB_SECONDS) -> Optional[Item]:
        """The reminder due soon that will cover ``item``, if any."""
        if item.reminder:
            return None
        for r in self.reminders(now, within=within):
            if covers(r, item):
                return r
        return None

    def covered_by(self, reminder: Item) -> list[str]:
        """The open keys a reminder's delivery speaks for."""
        return [
            k for k, i in sorted(self.items.items())
            if not i.closed and not i.reminder and covers(reminder, i)
        ]

    def waiting(self, now: datetime, *, fresh_seconds: int = FRESH_SECONDS,
                within: int = REMINDER_ABSORB_SECONDS) -> list[Item]:
        """What the composer may tell them: open, new since told or weighed,
        recent, not a reminder of theirs (it tells itself), and not absorbed
        by one due soon. Oldest first."""
        out = []
        for item in self.items.values():
            if item.reminder or not item.fresh:
                continue
            seen = _parse(item.when) or _parse(item.last_at)
            if seen is None or (now - seen).total_seconds() > fresh_seconds:
                continue
            if self.absorbed(item, now, within=within) is not None:
                continue
            out.append(item)
        return sorted(out, key=lambda i: i.last_seq)

    def repeats_told(self, keys: Sequence[str], text: str) -> bool:
        """Would saying ``text`` for ``keys`` repeat what was told, with
        nothing new? True only when every key was told before, nothing was
        observed on it since, and the words are the same."""
        if not keys:
            return False
        words = normalized(text)
        for k in keys:
            item = self.items.get(k)
            if item is None or item.told_seq == 0 or item.last_seq > item.told_seq:
                return False
            if normalized(item.told_text) != words:
                return False
        return True

    def stale(self, keys: Sequence[str], since_seq: int) -> Optional[str]:
        """Why a draft written at ``since_seq`` for ``keys`` should not go,
        or ``None``: every key was told after it was drafted, or closed."""
        if not keys:
            return None
        reasons = []
        for k in keys:
            item = self.items.get(k)
            if item is None:
                return None
            if item.closed:
                reasons.append("closed")
            elif item.told_seq > since_seq and item.last_seq <= item.told_seq:
                reasons.append("told")
            else:
                return None
        if all(r == "closed" for r in reasons):
            return "its thread closed before it went"
        return "they were told about it after it was drafted"


_CODE = re.compile(r"\b(?=[A-Z0-9]*[0-9])(?=[A-Z0-9]*[A-Z])[A-Z0-9]{5,8}\b|\b[A-Z]{2}\s?[0-9]{2,4}\b")
_STOP = frozenset(
    "about after again also before being from have here into just like make more "
    "need only over remind reminder some that them then they this time today "
    "tomorrow tonight what when where will with your their there mail travel "
    "calendar".split()
)


def _codes(text: str) -> set[str]:
    return {c.replace(" ", "") for c in _CODE.findall(text or "")}


def _words(text: str) -> set[str]:
    return {w for w in normalized(text).split() if len(w) >= 4 and w not in _STOP and not w.isdigit()}


def covers(reminder: Item, item: Item) -> bool:
    """Does this reminder speak for this item? A shared flight or booking
    code, or a deadline within the absorb window after the reminder's with a
    shared word. Code, and a heuristic [A]."""
    if _codes(reminder.what) & _codes(item.what):
        return True
    due, deadline = _parse(reminder.deadline), _parse(item.deadline)
    if due is None or deadline is None:
        return False
    gap = (deadline - due).total_seconds()
    return 0 <= gap <= REMINDER_ABSORB_SECONDS and bool(_words(reminder.what) & _words(item.what))


def omega_authored(sid: str, unprompted: Iterable[int]) -> bool:
    """Was the reminder ``sid`` written by omega on its own (DL-120)? A
    reminder a turn set is ``r<for_seq>-...``; it is omega's own when that
    turn was not theirs (a look, a fire, a brief). A promise's reminder
    (``pr<n>``) is set on their yes, so it is theirs."""
    if _PROMISE_REMINDER.match(sid):
        return False
    m = _MODEL_REMINDER.match(sid)
    return bool(m) and int(m.group(1)) in set(unprompted)


class Ledger:
    """The fold. Kept at the log's head by the executor, as its other views."""

    __slots__ = ("_through", "_items", "_bulk", "_unprompted", "_inbound_keys",
                 "_fires", "_sends", "_unkeyed", "_repeats", "_counted", "_taint",
                 "_routine")

    def __init__(self) -> None:
        self._through = 0
        self._items: dict[str, Item] = {}
        self._bulk: set[str] = set()
        self._unprompted: set[int] = set()
        self._inbound_keys: dict[int, tuple[str, ...]] = {}
        self._fires: dict[int, str] = {}
        self._sends = 0
        self._unkeyed = 0
        self._repeats = 0
        self._counted: set[int] = set()
        # unprompted turn seq -> what untrusted content it read, so a reminder
        # it set carries that taint when it comes due (DL-120, DL-064).
        self._taint: dict[int, tuple[str, ...]] = {}
        self._routine = False

    def _item(self, key: str, source: str, what: str, seq: int, at: str) -> Item:
        item = self._items.get(key)
        if item is None:
            item = Item(key=key, source=source, what=what, first_seq=seq, last_seq=seq, last_at=at)
            self._items[key] = item
        return item

    def apply(self, seq: int, payload: dict[str, Any]) -> None:
        self._through = seq
        kind = payload.get("kind")
        if kind == episodes.OBSERVATION_RECORDED:
            key = payload.get("key")
            if key is None:
                self._bulk.update(str(r) for r in payload.get("refs") or ())
                return
            at = str(payload.get("at", ""))
            item = self._item(key, str(payload["source"]), str(payload["what"]), seq, at)
            item.what, item.source = str(payload["what"]), str(payload["source"])
            item.last_seq, item.last_at = seq, at
            item.observations += 1
            item.closed = False
            for name in ("when", "deadline", "url"):
                if payload.get(name) is not None:
                    setattr(item, name, str(payload[name]))
            if payload.get("ref"):
                item.refs.add(str(payload["ref"]))
            if payload.get("untrusted"):
                item.untrusted = tuple(dict.fromkeys((*item.untrusted, str(payload["untrusted"]))))
        elif kind == episodes.SCHEDULE_CREATED:
            sid = str(payload.get("id", ""))
            text = str(payload.get("instruction", ""))
            if not (payload.get("once") is True and text.startswith(_REMINDER_LEAD)):
                return
            what = text[len(_REMINDER_LEAD):].splitlines()[0].strip() or "a reminder"
            created = _parse(payload.get("at"))
            due = (
                (created + timedelta(seconds=int(payload.get("every") or 0))).isoformat()
                if created else None
            )
            key = f"reminder:{_safe(sid)}"
            at = str(payload.get("at", ""))
            item = self._item(key, "words", what, seq, at)
            # A reminder set again under its id has moved: its key reopens.
            item.what, item.deadline, item.closed = what, due, False
            item.last_seq, item.last_at = seq, at
            item.refs.add(f"{sid}@{due}")
        elif kind == episodes.SCHEDULE_CANCELLED:
            item = self._items.get(f"reminder:{_safe(str(payload.get('id', '')))}")
            if item is not None:
                item.closed = True
        elif kind == episodes.MESSAGE_INBOUND:
            if payload.get("channel") in _UNPROMPTED:
                self._unprompted.add(seq)
            elif not payload.get("untrusted"):
                text = str(payload.get("text", ""))
                if ROUTINE_OFF.search(text):
                    self._routine = False
                elif ROUTINE_ON.search(text):
                    self._routine = True
            if payload.get("keys"):
                self._inbound_keys[seq] = tuple(payload["keys"])
            if payload.get("schedule_id"):
                self._fires[seq] = str(payload["schedule_id"])
        elif kind == episodes.MESSAGE_QUEUED:
            for k in payload.get("keys") or ():
                item = self._items.get(k)
                if item is not None:
                    item.considered_seq = max(item.considered_seq, int(payload.get("for_seq") or seq))
        elif kind in episodes.TERMINAL_KINDS:
            for_seq = payload.get("for_seq")
            if not isinstance(for_seq, int):
                return
            if for_seq in self._unprompted and payload.get("tainted"):
                self._taint[for_seq] = tuple(str(t) for t in payload["tainted"])
            failed = kind == episodes.TURN_COMPLETED and payload.get("outcome") == "failed"
            sid = self._fires.get(for_seq)
            if sid is not None and not failed:
                item = self._items.get(f"reminder:{_safe(sid)}")
                if item is not None:
                    item.closed = True
            shown = self._inbound_keys.get(for_seq, ())
            keys = tuple(payload.get("keys") or ()) or shown
            for k in shown:
                item = self._items.get(k)
                if item is not None:
                    item.considered_seq = max(item.considered_seq, for_seq)
            spoke = kind == episodes.TURN_COMPLETED and payload.get("outcome") == "spoke"
            if not spoke:
                return
            if for_seq in self._unprompted and for_seq not in self._counted:
                self._counted.add(for_seq)
                self._sends += 1
                if not keys:
                    self._unkeyed += 1
            reply = payload.get("reply")
            for k in keys:
                item = self._items.get(k)
                if item is None:
                    item = self._item(k, "words", str(reply or k)[:300], for_seq, str(payload.get("at", "")))
                    item.last_seq = 0
                elif item.told_seq and item.last_seq <= item.told_seq:
                    self._repeats += 1
                item.told_seq = for_seq
                item.told_at = str(payload.get("at", ""))
                item.told_text = str(reply)[:1000] if isinstance(reply, str) else None

    def state(self) -> State:
        """A snapshot: later folding does not change a state already taken."""
        return State(
            through=self._through,
            items={k: replace(i, refs=set(i.refs)) for k, i in self._items.items()},
            bulk_refs=frozenset(self._bulk),
            sends=self._sends,
            unkeyed=self._unkeyed,
            repeats=self._repeats,
            named=frozenset(k for ks in self._inbound_keys.values() for k in ks),
            routine_warnings=self._routine,
        )

    def authored(self, sid: str) -> bool:
        """Did omega set reminder ``sid`` on its own (DL-120)? Its id names an
        unprompted turn, and that turn came before the reminder was set: a
        fire whose own seq happens to match the id is not its author."""
        m = _MODEL_REMINDER.match(sid)
        item = self._items.get(f"reminder:{_safe(sid)}")
        return (
            m is not None and item is not None
            and int(m.group(1)) < item.first_seq
            and omega_authored(sid, self._unprompted)
        )

    def authored_taint(self, sid: str) -> tuple[str, ...]:
        """What the unprompted turn that set reminder ``sid`` had read."""
        m = _MODEL_REMINDER.match(sid)
        return self._taint.get(int(m.group(1)), ()) if m else ()

    @classmethod
    def fold(cls, decoded: Iterable[tuple[int, dict[str, Any]]]) -> "Ledger":
        view = cls()
        for seq, payload in decoded:
            view.apply(seq, payload)
        return view

    def advance(self, store: "MemoryStore", *, upto: Optional[int] = None) -> "Ledger":
        for episode in store.episodes_since(self._through):
            if upto is not None and episode.seq > upto:
                break
            self.apply(episode.seq, episodes.decode(episode.payload))
        return self

    @classmethod
    def rebuild(cls, store: "MemoryStore") -> "Ledger":
        return cls().advance(store)


def lines(items: Sequence[Item], *, now: datetime) -> list[str]:
    """``[key] what (seen N times; last told ...)`` for the composer's prompt."""
    out = []
    for i in items:
        notes = []
        if i.recurrence > 1:
            notes.append(f"seen {i.recurrence} times")
        if i.told_at:
            notes.append(f"last told {i.told_at[:16]}")
        if i.deadline:
            notes.append(f"at {i.deadline[:16]}")
        tail = f" ({'; '.join(notes)})" if notes else ""
        out.append(f"[{i.key}] {i.what}{tail}")
    return out


def taint(items: Sequence[Item]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(t for i in items for t in i.untrusted))
