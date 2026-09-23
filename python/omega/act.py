"""Step 4 — the `act` sub-loop. `agent/M1_SPEC.md` §1.5, §2.3, §2.4; DL-028.

**A sub-loop, never a sub-agent** (§1.5). It has no separate memory, no
separate identity and no context of its own: it runs inside the turn, over the
turn's context, writing into the turn's log. The thing that makes a sub-agent
tempting — "this work is big, give it its own head" — is exactly what produces
two agents with two beliefs about the same job, and §1.6 already refuses a
second output path for the same reason.

It asks §1.5's two questions on every pass:

*Are we done, and verified?* The signal is the **model ceasing to request
tools**, not a tool reporting success. A tool that returns ``ok`` says a
command exited zero; it does not say the job is finished, and `CLAUDE.md`'s
grade-the-world rule is precisely that those two are different facts. So what
lands in the log is what actually came back — :func:`omega.episodes.tool_returned`
carrying the real result or the real error — and the loop never upgrades
"the tool succeeded" into "the work is done".

*Is this moving?* Two answers, both of which stop rather than flail:
:data:`STALLED` when the model asks for a call it has already made verbatim,
and :data:`EXHAUSTED` when the pass cap runs out. Neither is an error. A loop
that cannot tell repetition from progress burns the cap and then reports
failure, which is the worst of the three outcomes — it costs the most and says
the least.

**What may run is decided in :mod:`omega.tools`, before dispatch, in code**
(DL-014, DL-028). This file classifies every requested call first and
dispatches nothing at all if any one of them is ``external``: the turn ends as
``turn.blocked`` with a question. Not "block that one and run the rest" —
the calls in a single pass are the model's plan, and running half a plan while
asking about the other half produces side effects for a plan nobody approved.
"""

from __future__ import annotations

from typing import Any, Optional

from omega import episodes, provider
from omega.tools import (
    EXTERNAL,
    Decision,
    ToolBox,
    ToolError,
    ToolRejected,
    schemas,
)

# `act` imports `turn`, never the other way round. `turn` names the *slot*
# (`no_act_loop_yet`) and this file fills it; the executor wires them together.
# The reverse import would be a cycle, and worse, it would make the six-step
# turn depend on the tools — which is the coupling that lets a tool change
# quietly alter what a turn is.
from omega.turn import MAX_ACT_PASSES, ActResult, TurnContext
from omega.turn import _render_event, _transcript  # the turn's own rendering

__all__ = [
    "act_loop",
    "DONE_ASKING",
    "STALLED",
    "EXHAUSTED",
    "MAX_ACT_PASSES",
]

#: The model stopped requesting tools. §1.5's "are we done, verified?" answered
#: by the only party that can answer it.
DONE_ASKING = "the model stopped asking for tools"

#: §1.5's "is this moving?" answered *no*, by repetition.
STALLED = "stalled"

#: Answered *no* by the clock.
EXHAUSTED = "pass cap reached"

_ACT_SYSTEM = (
    "You are omega, a second brain. Do the work this event needs, using the "
    "tools when they help.\n"
    "Tools are offered, not required: if you can answer without one, ask for "
    "no tools and stop. When the work is done, stop asking for tools - that "
    "is how you say you are finished.\n"
    "A tool that succeeds has not finished the job; check what actually came "
    "back before you decide you are done.\n"
    "Do not repeat a call you have already made with the same arguments - it "
    "will return the same thing and the loop will stop.\n"
    "run_code has no shell: give argv as a list, and pipes, redirection and "
    "$(...) will not work.\n"
    "Writing outside omega's own store, running anything that is not a "
    "read-only command, and fetching a URL all stop the turn to ask the "
    "person first. Ask for one only when it is genuinely what the work needs."
)


def _act_messages(ctx: TurnContext) -> list[provider.Message]:
    return [
        provider.system(_ACT_SYSTEM),
        provider.user(
            f"Recent history:\n{_transcript(ctx.recalled)}\n\n"
            f"New event:\n{_render_event(ctx.event)}"
        ),
    ]


def _signature(call: provider.ToolCall) -> tuple[str, str]:
    """A call's identity for stall detection: name plus arguments, spelled
    canonically so that key order cannot disguise a repeat."""
    return (call.name, repr(sorted(call.arguments.items(), key=lambda kv: kv[0])))


def _blocked_on(decisions: list[Decision]) -> str:
    """The question a blocked turn asks, in words the person can answer.

    Names every external call in the pass, not just the first: the person is
    being asked to approve a plan, and showing them one step of it invites a
    yes to something they have not seen.
    """
    wanted = [d.why for d in decisions if d.tier == EXTERNAL]
    if len(wanted) == 1:
        return f"I want to {wanted[0]}. May I?"
    joined = "\n".join(f"- {w}" for w in wanted)
    return f"I want to do these, and each needs your go:\n{joined}"


def act_loop(
    ctx: TurnContext,
    *,
    box: Optional[ToolBox] = None,
    max_passes: int = MAX_ACT_PASSES,
) -> ActResult:
    """Run tool passes until done, blocked, stalled or out of passes.

    ``box`` is injectable for tests; in production it is built from the store
    the queue is already writing to, so "inside the store" means the same
    directory the log lives in and nothing has to be configured twice.
    """
    if box is None:
        box = ToolBox(store_root=ctx.queue.store.root)

    offered = schemas()
    messages = _act_messages(ctx)
    used: list[str] = []
    seen: set[tuple[str, str]] = set()

    for _ in range(max_passes):
        response = ctx.complete(provider.ACT, messages, offered)
        calls = response.tool_calls

        if not calls:
            # Done. The model asking for nothing is the signal (§1.5) — and it
            # is the *only* signal, because no tool result can say "the job is
            # finished", only "this command came back".
            return ActResult(tools=tuple(used), stop_reason=DONE_ASKING)

        repeats = [call for call in calls if _signature(call) in seen]
        if repeats:
            # Repetition is not progress. The honest cost: a second identical
            # read really can be the right move (read, write, read back), and
            # this stops it. That is the deliberate trade — a loop that cannot
            # recognise repetition spends the whole cap re-running one call and
            # then reports nothing useful, and re-reading is recoverable by
            # asking again next turn while flailing is not.
            names = ", ".join(sorted({call.name for call in repeats}))
            return ActResult(
                tools=tuple(used),
                stop_reason=f"{STALLED}: asked for {names} again with the same "
                f"arguments, which would return the same thing",
            )
        seen.update(_signature(call) for call in calls)

        # Classify **everything first**. Nothing below this line may run until
        # every call in the pass has a tier, because the external check has to
        # see the whole plan before any of it has happened.
        decisions: list[tuple[provider.ToolCall, Optional[Decision], Optional[str]]] = []
        for call in calls:
            try:
                decisions.append((call, box.classify(call.name, call.arguments), None))
            except ToolRejected as exc:
                # Nothing ran: an unreadable call is a failed call, and it goes
                # back to the model as one rather than ending the turn.
                decisions.append((call, None, str(exc)))

        cleared = [d for _, d, _ in decisions if d is not None]
        if any(d.tier == EXTERNAL for d in cleared):
            return ActResult(
                tools=tuple(used),
                blocked_on=_blocked_on(cleared),
                stop_reason="waiting on a person",
            )

        messages.append(provider.assistant_tool_calls(calls, response.text))
        for call, decision, rejection in decisions:
            if decision is None:
                assert rejection is not None
                # The pair is written even though nothing ran, so that every
                # `tool.returned` has a `tool.called` and the log can be read
                # by pairing rather than by guessing. The error says plainly
                # that it was refused before dispatch.
                _called(ctx, call.name, dict(call.arguments))
                _returned(ctx, call.name, ok=False, error=f"refused: {rejection}")
                messages.append(provider.tool_result(call.id, rejection))
                continue

            # `called` lands **before** the work, not after. That ordering is
            # the mechanism, not bookkeeping: a crash mid-tool then leaves
            # evidence that it started, which is the only way a restart can
            # tell "never ran" from "ran and we never heard back".
            _called(ctx, decision.tool, decision.args)
            text, ok = _dispatch(box, decision)
            _returned(
                ctx,
                decision.tool,
                ok=ok,
                result=text if ok else None,
                error=None if ok else text,
            )
            if ok:
                used.append(decision.tool)
            messages.append(provider.tool_result(call.id, text))

    # The cap, and it is **not** an error (§1.5). The turn goes on to compose a
    # reply out of what did get done; the stop_reason is what says the work was
    # cut off rather than finished, so the record does not read like a clean
    # completion.
    return ActResult(
        tools=tuple(used),
        stop_reason=f"{EXHAUSTED}: stopped after {max_passes} tool passes",
    )


def _dispatch(box: ToolBox, decision: Decision) -> tuple[str, bool]:
    """Run one cleared call. Returns its text and whether it worked.

    A failure comes back as text, never as an exception through the loop, and
    it is **never retried here** (§2.4). Retry policy is explicitly still open,
    and the one thing worse than no retry policy is an implicit one invented by
    a loop — a silent retry of a write or a run is how one action becomes two.
    """
    try:
        return box.dispatch(decision), True
    except ToolError as exc:
        return str(exc), False
    except Exception as exc:  # noqa: BLE001
        # A bug inside a tool is still a tool failure, not a failed turn: it is
        # recorded with its type so the log says what actually happened, and
        # the next pass gets to react to it.
        return f"{decision.tool} raised {type(exc).__name__}: {exc}", False


def _called(ctx: TurnContext, tool: str, args: dict[str, Any]) -> None:
    """``tool.called``, carrying ``for_seq = ctx.seq`` so every tool episode is
    attributable to the turn that caused it."""
    ctx.queue.append(episodes.tool_called(for_seq=ctx.seq, tool=tool, args=args))


def _returned(
    ctx: TurnContext,
    tool: str,
    *,
    ok: bool,
    result: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """``tool.returned``, carrying what actually came back.

    Surfaced, never swallowed (§2.4). The result is already capped by the tool
    itself, so the log and the model saw the same text — a log that kept more
    than the model was shown would make the transcript a plausible lie about
    what omega was reasoning over.
    """
    ctx.queue.append(
        episodes.tool_returned(
            for_seq=ctx.seq, tool=tool, ok=ok, result=result, error=error
        )
    )
