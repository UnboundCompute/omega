"""The task engine: a plan approved once, then run step by step (DL-095).

The person's mode here is *approve the plan, then run it*. Omega drafts a plan
— a goal, ordered steps each with its own done-criteria, and a token budget —
and files it for approval through the ordinary DL-087 #8 queue, as a ``tasks``
proposal (:mod:`omega.grants`). On their yes a ``task.started`` record is
written and the runtime drives each step as its own unattended turn, one at a
time, resumable after a restart because the whole of a task's state is a fold
over its ``task.*`` records and nothing is stored beside the log.

**Every step ends in an explicit terminal state**, recorded as a ``task.step``:
``done`` (with evidence — a record id, a url or a tool-result reference, never
bare prose), ``waiting`` (on ``tap``/``time``/``watch``, until something), or
``blocked``/``failed`` (with why). A step turn that carries the obligation and
ends in prose with no terminal state and no gated call is the DL-040 gate-leak;
the executor records it ``task.dropped``, retries once with the obligation
restated, and on a second drop blocks the step and tells the person once.

**Caps, all fold-derived:** a per-task token budget (default
:data:`omega.episodes.DEFAULT_TASK_BUDGET`, counted the way :mod:`omega.topics`
counts a turn's usage), at most :data:`omega.episodes.MAX_TASK_STEPS` steps per
task (validated at propose time), and a per-day cap on steps driven across all
tasks (:data:`DAILY_STEPS`). On any cap the current step is blocked and the
person is told once.

The money hard rule is unchanged: a step that needs payment ends
``waiting{on: "tap"}`` with a hand-off link; omega never calls a payment tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

from omega import episodes, topics

if TYPE_CHECKING:  # pragma: no cover
    from omega.memory import MemoryStore

__all__ = [
    "DAILY_STEPS",
    "Plan",
    "Step",
    "Live",
    "State",
    "Tasks",
    "obligation_of",
]

#: At most this many task steps may be driven across all tasks in one local
#: day. A smallest-reasonable ceiling (two full tasks' worth): the per-task step
#: count and token budget already bound a single task; this bounds the day.
DAILY_STEPS = 40


@dataclass(frozen=True)
class Plan:
    """A drafted plan, as approved. Its ``task`` is the ``task.proposed`` seq."""

    task: int
    goal: str
    steps: tuple[dict[str, str], ...]
    budget: int
    at: str


@dataclass(frozen=True)
class Step:
    """One step's recorded terminal state."""

    n: int
    state: str
    evidence: Optional[str]
    why: Optional[str]
    on: Optional[str]
    until: Optional[str]
    for_seq: int
    at: str


@dataclass(frozen=True)
class Live:
    """A started task as of one seq, with where its run has got to."""

    plan: Plan
    cancelled: bool
    cursor: int  # the first step not yet done
    current: Optional[Step]  # the cursor step's terminal, if any
    spent: int
    drops: dict[int, int] = field(default_factory=dict)
    #: Steps with a driven inbound out that has reached no terminal and no drop
    #: yet — so a second drive is not started while the first is still in flight.
    outstanding: frozenset[int] = frozenset()

    @property
    def task(self) -> int:
        return self.plan.task

    @property
    def total(self) -> int:
        return len(self.plan.steps)

    @property
    def completed(self) -> bool:
        return self.cursor >= self.total

    @property
    def waiting(self) -> bool:
        return self.current is not None and self.current.state == "waiting"

    @property
    def halted(self) -> bool:
        return self.current is not None and self.current.state in ("blocked", "failed")

    @property
    def drivable(self) -> bool:
        """Is the cursor step ready to be driven *fresh* as the next unattended
        turn? A step already dropped is not: restating it and, on a second
        drop, blocking it is the executor's job, not a new drive."""
        return (
            not self.cancelled
            and not self.completed
            and self.current is None
            and self.drops.get(self.cursor, 0) == 0
            and self.cursor not in self.outstanding
            and self.spent < self.plan.budget
        )

    @property
    def running(self) -> bool:
        """Started, not finished, not stopped — shown in the brief."""
        return not self.cancelled and not self.completed and not self.halted


def obligation_of(event: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The task obligation a driven step's inbound carries, or ``None``."""
    ob = event.get("obligation")
    if not isinstance(ob, dict):
        return None
    task, n = ob.get("task"), ob.get("n")
    if isinstance(task, int) and not isinstance(task, bool) and isinstance(n, int) \
            and not isinstance(n, bool):
        return {"task": task, "n": n}
    return None


def _local_day(at: Any) -> str:
    try:
        moment = datetime.fromisoformat(str(at))
    except ValueError:
        return ""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone().date().isoformat()


@dataclass(frozen=True)
class State:
    """The tasks as of one seq. Frozen, as :class:`omega.grants.State`."""

    through: int = 0
    live: dict[int, Live] = field(default_factory=dict)
    #: Local day -> steps driven that day, for the per-day cap.
    per_day: dict[str, int] = field(default_factory=dict)
    #: for_seq of every turn that wrote a ``task.step`` — the gate-leak check
    #: reads it to tell a discharged step turn from a dropped one.
    stepped: frozenset[int] = frozenset()

    def task(self, task: int) -> Optional[Live]:
        return self.live.get(task)

    def running(self) -> list[Live]:
        """Started tasks still under way, oldest first."""
        return sorted(
            (t for t in self.live.values() if t.running), key=lambda t: t.task
        )

    def drivable(self) -> Optional[tuple[int, int, dict[str, str]]]:
        """The one next step to drive — ``(task, n, step)`` — oldest task
        first, or ``None`` when nothing is ready."""
        for t in sorted(self.live.values(), key=lambda t: t.task):
            if t.drivable:
                return t.task, t.cursor, dict(t.plan.steps[t.cursor])
        return None

    def steps_today(self, day: str) -> int:
        return self.per_day.get(day, 0)

    def __bool__(self) -> bool:
        return bool(self.live)


class Tasks:
    """The fold. ``apply`` in seq order; :meth:`state` snapshots it."""

    __slots__ = (
        "_through", "_plan", "_started", "_cancelled", "_terminal", "_drops",
        "_spent", "_per_day", "_stepped", "_ob_at", "_driven",
    )

    def __init__(self) -> None:
        self._through = 0
        self._plan: dict[int, Plan] = {}
        self._started: set[int] = set()
        self._cancelled: set[int] = set()
        self._terminal: dict[int, dict[int, Step]] = {}
        self._drops: dict[int, dict[int, int]] = {}
        self._spent: dict[int, int] = {}
        self._per_day: dict[str, int] = {}
        self._stepped: set[int] = set()
        #: inbound seq -> (task, local day) for a driven step, so the turn's
        #: usage and the day's count can be attributed when the turn completes.
        self._ob_at: dict[int, tuple[int, str]] = {}
        #: task -> steps with a driven inbound out that has reached no terminal
        #: and no drop yet, so a step in flight is never driven a second time.
        self._driven: dict[int, set[int]] = {}

    @property
    def through(self) -> int:
        return self._through

    def apply(self, seq: int, payload: dict[str, Any]) -> None:
        if seq <= self._through:
            raise ValueError(
                f"episodes must be folded in order: got seq {seq} after {self._through}"
            )
        kind = payload.get("kind")
        if kind == episodes.TASK_PROPOSED:
            self._plan[seq] = Plan(
                task=seq,
                goal=str(payload["goal"]),
                steps=tuple(dict(s) for s in payload["steps"]),
                budget=int(payload["budget"]),
                at=str(payload["at"]),
            )
        elif kind == episodes.TASK_STARTED:
            self._started.add(int(payload["task"]))
        elif kind == episodes.TASK_CANCELLED:
            self._cancelled.add(int(payload["task"]))
        elif kind == episodes.TASK_STEP:
            task, n = int(payload["task"]), int(payload["n"])
            self._terminal.setdefault(task, {})[n] = Step(
                n=n, state=str(payload["state"]),
                evidence=payload.get("evidence"), why=payload.get("why"),
                on=payload.get("on"), until=payload.get("until"),
                for_seq=int(payload["for_seq"]), at=str(payload["at"]),
            )
            self._stepped.add(int(payload["for_seq"]))
            self._driven.get(task, set()).discard(n)
        elif kind == episodes.TASK_DROPPED:
            task, n = int(payload["task"]), int(payload["n"])
            self._drops.setdefault(task, {})
            self._drops[task][n] = self._drops[task].get(n, 0) + 1
            self._driven.get(task, set()).discard(n)
        elif kind == episodes.MESSAGE_INBOUND:
            ob = obligation_of(payload)
            if ob is not None:
                self._ob_at[seq] = (ob["task"], _local_day(payload.get("at")))
                day = self._ob_at[seq][1]
                self._per_day[day] = self._per_day.get(day, 0) + 1
                self._driven.setdefault(ob["task"], set()).add(ob["n"])
        elif kind == episodes.TURN_COMPLETED:
            got = self._ob_at.get(int(payload.get("for_seq", 0)))
            if got is not None:
                task, _ = got
                self._spent[task] = self._spent.get(task, 0) + topics.spent(
                    payload.get("usage")
                )
        self._through = seq

    def state(self) -> State:
        live: dict[int, Live] = {}
        for task in self._started:
            plan = self._plan.get(task)
            if plan is None:
                continue
            terms = self._terminal.get(task, {})
            cursor = 0
            while cursor < len(plan.steps) and (
                (s := terms.get(cursor)) is not None and s.state == "done"
            ):
                cursor += 1
            current = terms.get(cursor) if cursor < len(plan.steps) else None
            live[task] = Live(
                plan=plan,
                cancelled=task in self._cancelled,
                cursor=cursor,
                current=current,
                spent=self._spent.get(task, 0),
                drops=dict(self._drops.get(task, {})),
                outstanding=frozenset(self._driven.get(task, set())),
            )
        return State(
            through=self._through,
            live=live,
            per_day=dict(self._per_day),
            stepped=frozenset(self._stepped),
        )

    @classmethod
    def fold(cls, decoded: Iterable[tuple[int, dict[str, Any]]]) -> "Tasks":
        view = cls()
        for seq, payload in decoded:
            view.apply(seq, payload)
        return view

    def advance(self, store: "MemoryStore", *, upto: Optional[int] = None) -> "Tasks":
        for episode in store.episodes_since(self._through):
            if upto is not None and episode.seq > upto:
                break
            self.apply(episode.seq, episodes.decode(episode.payload))
        return self

    @classmethod
    def rebuild(cls, store: "MemoryStore") -> "Tasks":
        return cls().advance(store)
