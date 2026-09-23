"""The clock — DL-035, DL-036.

Everything omega did before this file started with somebody typing. That is a
good tool and not yet a second brain, so this is the part that lets it act on
time. What it deliberately is *not* is an execution engine.

**It causes turns the only way anything may: by appending an episode.**
`runtime.py` already names the trap — calling ``run_turn`` from the clock
"would make M5's clock a second engine wearing omega's name" — and the way out
is that a fire is a plain :func:`omega.episodes.inbound` on the ``schedule``
channel. The queue drains it, the executor runs it, the projection reports it,
and none of them need to know a clock exists. There is no ``schedule.fired``
kind for the same reason: a fire *is* an inbound, so it is already in the log,
already recalled, and already visible to the tray.

**The table here is a cache, and throwing it away is the repair** (DL-036). The
source of truth is the log: definitions are ``schedule.created`` /
``schedule.cancelled`` records, and "when did this last fire?" is answered by
the inbound episodes carrying ``schedule_id``. So :class:`Scheduler` holds no
state that cannot be rebuilt by reading episodes it has already read, which is
what makes restart boring — the failure mode of a scheduler that persists its
own cursor is that it either goes quiet forever or re-fires everything, and
both are silent.

**Missed fires coalesce into one, and say they were late.** A laptop that slept
through eight hourly slots must not wake into eight turns; DL-036 calls that
the "re-fires everything" failure. :func:`previous_match` gives the *most
recent* matching minute rather than every one since the last fire, so
catch-up and ordinary firing are the same code path and coalescing is a
property of the algorithm rather than a special case someone has to remember.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator, Optional

from omega import episodes
from omega.queue import EventQueue

__all__ = [
    "Schedule",
    "Due",
    "Scheduler",
    "CronError",
    "cron_matches",
    "previous_match",
    "LATE_AFTER_SECONDS",
    "CATCH_UP_HORIZON_MINUTES",
    "TICK_SECONDS",
]

#: How often the heartbeat asks "is anything due?". Not how often anything
#: fires — `episodes.MIN_EVERY_SECONDS` governs that. A tick is a dict lookup
#: and a clock read; it writes nothing, which is the whole reason DL-036 can
#: leave ticks out of the log and keep the log readable.
TICK_SECONDS = 30

#: Past this much drift, a fire is reported to the turn as late. Chosen as a
#: small multiple of the tick: anything under one tick is just scheduling
#: granularity and calling it "late" would make the word meaningless.
LATE_AFTER_SECONDS = 120

#: How far back :func:`previous_match` will look for a missed slot. 25 hours
#: covers an overnight sleep plus a DST seam, and bounds the walk at ~1500
#: cheap iterations. A daily schedule missed by longer than this is not "late",
#: it is a machine that was off, and firing it on wake would be noise.
CATCH_UP_HORIZON_MINUTES = 25 * 60

#: ``minute hour day-of-week``. A subset, not a dialect: ``*``, ``N``, ``A-B``,
#: ``A,B``, ``*/N``. Day-of-week is 0=Sunday, as every crontab has it.
_FIELD_BOUNDS = ((0, 59), (0, 23), (0, 6))


class CronError(ValueError):
    """A schedule expression that cannot be matched against.

    Raised at parse time, which is the only time it can be raised usefully: a
    cron field that silently matched nothing would produce a schedule that
    exists, reports healthy, and never fires.
    """


def _parse_field(spec: str, lo: int, hi: int) -> frozenset[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise CronError(f"empty field element in {spec!r}")
        step = 1
        if "/" in part:
            part, _, raw_step = part.partition("/")
            if not raw_step.isdigit() or int(raw_step) < 1:
                raise CronError(f"bad step {raw_step!r} in {spec!r}")
            step = int(raw_step)
        if part == "*":
            start, end = lo, hi
        elif "-" in part.lstrip("-"):
            raw_start, _, raw_end = part.partition("-")
            start, end = _int(raw_start, spec), _int(raw_end, spec)
        else:
            start = end = _int(part, spec)
        if not (lo <= start <= hi and lo <= end <= hi):
            raise CronError(f"{part!r} out of range {lo}-{hi} in {spec!r}")
        if start > end:
            raise CronError(f"reversed range {part!r} in {spec!r}")
        out.update(range(start, end + 1, step))
    if not out:
        raise CronError(f"field {spec!r} matches nothing")
    return frozenset(out)


def _int(raw: str, spec: str) -> int:
    raw = raw.strip()
    if not raw.isdigit():
        raise CronError(f"expected a number, got {raw!r} in {spec!r}")
    return int(raw)


def _parse_cron(cron: str) -> tuple[frozenset[int], ...]:
    fields = cron.split()
    if len(fields) != 3:
        raise CronError(
            f"expected 3 fields (minute hour day-of-week), got {len(fields)} in {cron!r}"
        )
    return tuple(
        _parse_field(field, lo, hi)
        for field, (lo, hi) in zip(fields, _FIELD_BOUNDS)
    )


def cron_matches(cron: str, when: datetime) -> bool:
    """Does this wall-clock minute satisfy the expression?

    ``when`` is compared in whatever zone it carries, and the caller passes
    local time, because "9am" means the person's 9am and not UTC's.
    """
    minute, hour, dow = _parse_cron(cron)
    # `weekday()` is 0=Monday; crontab is 0=Sunday. Convert rather than invent
    # a private convention, because the expression is the thing a person types.
    return (
        when.minute in minute
        and when.hour in hour
        and ((when.weekday() + 1) % 7) in dow
    )


def previous_match(
    cron: str, now: datetime, *, horizon_minutes: int = CATCH_UP_HORIZON_MINUTES
) -> Optional[datetime]:
    """The most recent minute at or before ``now`` that ``cron`` matches.

    Walking backwards rather than computing a next-fire time is what collapses
    two behaviours into one. Ordinary firing is "the last matching minute is
    newer than the last fire"; catch-up after a sleep is the *same sentence*.
    And because only the most recent match is returned, eight slots missed
    overnight produce one fire, which is DL-036's coalescing rule expressed as
    an algorithm instead of a rule.
    """
    _parse_cron(cron)  # fail loudly here, not silently inside the loop
    cursor = now.replace(second=0, microsecond=0)
    for _ in range(horizon_minutes + 1):
        if cron_matches(cron, cursor):
            return cursor
        cursor -= timedelta(minutes=1)
    return None


@dataclass(frozen=True)
class Schedule:
    """A standing intention, as the log recorded it."""

    id: str
    instruction: str
    created_at: datetime
    every: Optional[int] = None
    cron: Optional[str] = None


@dataclass(frozen=True)
class Due:
    """A schedule that should fire now, and how far behind it is running."""

    schedule: Schedule
    #: The slot this fire is *for* — the matched minute, not the moment we
    #: noticed. Carried so the recorded fire is attributable to a schedule slot
    #: rather than to tick jitter.
    slot: datetime
    late_by: timedelta

    @property
    def late(self) -> bool:
        return self.late_by.total_seconds() > LATE_AFTER_SECONDS


def _parse_at(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


class Scheduler:
    """The derived table (DL-036), maintained by reading the log forward.

    Incremental rather than a full rescan per tick, but the increment is only
    an optimisation: every field is a fold over episodes, so dropping this
    object and building a new one produces the same answer. That is the
    property being protected — not the speed.
    """

    def __init__(self, queue: EventQueue) -> None:
        self._queue = queue
        self._cursor = 0
        self._defs: dict[str, Schedule] = {}
        self._last_fire: dict[str, datetime] = {}
        # seq of the fire we most recently appended per schedule. The overlap
        # guard, and in memory on purpose: a fire in flight dies with the
        # process, so persisting this would be persisting a lie.
        self._in_flight: dict[str, int] = {}

    def refresh(self) -> None:
        """Fold every episode written since the last look into the table."""
        head = self._queue.head()
        if head <= self._cursor:
            return
        for episode in self._queue.store.episodes_since(self._cursor):
            payload = episodes.decode(episode.payload)
            self._observe(payload)
            self._cursor = episode.seq

    def _observe(self, payload: dict) -> None:
        kind = payload.get("kind")
        if kind == episodes.SCHEDULE_CREATED:
            self._defs[payload["id"]] = Schedule(
                id=payload["id"],
                instruction=payload["instruction"],
                created_at=_parse_at(payload["at"]),
                every=payload.get("every"),
                cron=payload.get("cron"),
            )
        elif kind == episodes.SCHEDULE_CANCELLED:
            self._defs.pop(payload["id"], None)
        elif kind == episodes.MESSAGE_INBOUND:
            sid = payload.get("schedule_id")
            if sid:
                # The *slot*, not the append time. A fire running six hours
                # late is appended now and owed for this morning, and it is the
                # slot that says which obligations are discharged. Falling back
                # to `at` keeps fires written before the field existed readable
                # — additive schema, per `VERSION`.
                self._last_fire[sid] = _parse_at(
                    payload.get("schedule_slot") or payload["at"]
                )

    @property
    def schedules(self) -> list[Schedule]:
        return sorted(self._defs.values(), key=lambda s: s.id)

    def last_fire(self, schedule_id: str) -> Optional[datetime]:
        return self._last_fire.get(schedule_id)

    def due(self, now: datetime) -> list[Due]:
        """Which schedules should fire at ``now``, newest slot first."""
        out: list[Due] = []
        for schedule in self.schedules:
            if self._blocked_by_previous_fire(schedule.id):
                continue
            found = self._slot_for(schedule, now)
            if found is not None:
                out.append(Due(schedule=schedule, slot=found, late_by=now - found))
        return out

    def _blocked_by_previous_fire(self, schedule_id: str) -> bool:
        """DL-035's overlap guard.

        A schedule whose last fire has not finished draining does not get a
        second one. Without this, a turn slower than its own interval builds a
        queue that can never be worked off — and it builds it out of real model
        calls, so the cost of the bug grows while nobody is watching, which is
        exactly the unattended failure DL-035 exists to bound.
        """
        seq = self._in_flight.get(schedule_id)
        if seq is None:
            return False
        if self._queue.done() >= seq:
            del self._in_flight[schedule_id]
            return False
        return True

    def _slot_for(self, schedule: Schedule, now: datetime) -> Optional[datetime]:
        # A schedule cannot be late for a slot that predates it. Without this
        # baseline, creating a 9am schedule at 3pm would immediately fire for
        # this morning — technically a missed slot, and obviously not what the
        # person asked for.
        baseline = self._last_fire.get(schedule.id) or schedule.created_at

        if schedule.every is not None:
            if (now - baseline).total_seconds() >= schedule.every:
                return now.replace(second=0, microsecond=0)
            return None

        assert schedule.cron is not None  # the episode schema guarantees one
        slot = previous_match(schedule.cron, now)
        if slot is not None and slot > baseline:
            return slot
        return None

    def fire(self, due: Due, *, now: Optional[datetime] = None) -> int:
        """Append the fire as an ordinary inbound and return its seq.

        The instruction is sent as the message text, and lateness is appended
        as a plain sentence rather than a flag, because the reader is a model
        and this is the only channel it has. A schedule that is running four
        hours behind should be able to decide that a "morning brief" is no
        longer worth writing — but it can only decide that if it is told.
        """
        del now  # the slot, not the observation, is what the fire is for
        text = due.schedule.instruction
        if due.late:
            minutes = int(due.late_by.total_seconds() // 60)
            text = (
                f"{text}\n\n(Scheduled for {due.slot.isoformat()}, running "
                f"{minutes} minutes late.)"
            )
        seq = self._queue.append(
            episodes.inbound(
                text,
                channel="schedule",
                schedule_id=due.schedule.id,
                schedule_slot=due.slot.isoformat(),
            )
        )
        self._in_flight[due.schedule.id] = seq
        # Recorded locally as well as in the log so that two ticks inside one
        # drain cannot both see "never fired". `refresh` will read the same
        # fact back off the episode; agreeing with itself is the point.
        self._last_fire[due.schedule.id] = due.slot
        return seq

    def tick(self, *, now: Optional[datetime] = None) -> list[int]:
        """One heartbeat: refresh, find what is due, fire it. Returns the seqs.

        Returning the seqs rather than a count is what lets a caller *verify* —
        `CLAUDE.md` grades the world, and a scheduler that reported "fired 2"
        without saying which episodes it wrote would be asking to be believed.
        """
        self.refresh()
        moment = now or datetime.now().astimezone()
        return [self.fire(item) for item in self.due(moment)]

    def __repr__(self) -> str:
        return (
            f"<Scheduler schedules={len(self._defs)} "
            f"cursor={self._cursor} in_flight={len(self._in_flight)}>"
        )
