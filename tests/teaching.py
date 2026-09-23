"""Shared fixtures for the teaching path — DL-043, DL-044.

Lifted out of ``test_learn.py`` when a second module needed them, rather than
imported across test files: a test module importing another test module's
underscored names is the same coupling smell it would be in production, and it
resolves differently depending on which file pytest collects first.

``TRAY_INSTRUCTION`` is the load-bearing one. It reproduces the literal string
``TrayViewModel.swift:523-529`` builds, deliberately copied rather than shared,
because the property under test is that two languages agree while having no way
to hold one constant between them.
"""

from __future__ import annotations

import json

from omega import derive, episodes, provider
from omega.executor import Executor
from omega.queue import EventQueue

TRAY_INSTRUCTION = (
    "Teaching note from me. Treat this as something to remember and apply in "
    "future conversations, not as a task to execute. Briefly confirm what you "
    "learned."
)


def _teach_text(note: str) -> str:
    return f"{TRAY_INSTRUCTION}\n\n{note}"


def _claim_obj(seq: int, text: str, **over) -> derive.Claim:
    fields = dict(
        seq=seq,
        text=text,
        trigger=None,
        situation="taught in the tray",
        source_seq=seq - 1,
        explicit=True,
    )
    fields.update(over)
    return derive.Claim(**fields)


def _answer(*claims: dict, retract=()) -> str:
    out: dict = {"claims": list(claims)}
    if retract:
        out["retract"] = list(retract)
    return json.dumps(out)


def _a_claim(text="keep status updates short", **over) -> dict:
    out = {"text": text, "situation": "asked in the tray", "trigger": None}
    out.update(over)
    return out


class _Teaching:
    """A provider scripted for all three roles, keeping what it was asked."""

    def __init__(self, *, learn_answer, verdict: str = "SPEAK", reply: str = "ok"):
        self.prompts: list[str] = []
        self._fake = provider.FakeProvider(
            {
                provider.JUDGE: self._record(verdict),
                provider.ACT: self._record(reply),
                provider.LEARN: self._record(learn_answer),
            }
        )

    def _record(self, answer):
        def respond(role, messages):
            self.prompts.append(
                "\n".join(_text_of(m) for m in messages)
            )
            if callable(answer):
                return answer(role, messages)
            return answer

        return respond

    @property
    def complete(self):
        return self._fake.complete

    @property
    def calls(self):
        return self._fake.calls

    def last(self) -> str:
        return self.prompts[-1]


def _text_of(message: dict) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "") for part in content if isinstance(part, dict)
    )


LEARNED_HEADING = "What you have learned about working with this person"


def _learned_section_of(prompt: str) -> str:
    """Just the learned section, which is what "applies to this turn" means.

    The rest of the prompt legitimately contains superseded claims — the
    receipt that announced one is an ordinary reply and shows up in recall
    forever. Reading the whole prompt would make a passing case out of the
    wrong fact.
    """
    if LEARNED_HEADING not in prompt:
        return ""
    return prompt.split(LEARNED_HEADING, 1)[1].split("Recent history:", 1)[0]


def _drain(q: EventQueue, teaching: _Teaching) -> None:
    ex = Executor(q, complete=teaching.complete)
    ex.recover()
    ex.drain()


def _log(q: EventQueue) -> list[tuple[int, dict]]:
    """Every episode, decoded. Read back from the store rather than from any
    object the turn built, because what the turn believed it wrote is the one
    thing these cases are not allowed to trust."""
    return [(e.seq, episodes.decode(e.payload)) for e in q.store.episodes_since(0)]


def _claims_in(q: EventQueue) -> list[tuple[int, dict]]:
    return [
        (seq, p) for seq, p in _log(q) if p.get("kind") == episodes.CLAIM_EXTRACTED
    ]


def _retractions_in(q: EventQueue) -> list[tuple[int, dict]]:
    return [
        (seq, p) for seq, p in _log(q) if p.get("kind") == episodes.CLAIM_RETRACTED
    ]


def _replies(q: EventQueue) -> list[str]:
    return [
        p["reply"]
        for _, p in _log(q)
        if p.get("kind") == episodes.TURN_COMPLETED and p.get("reply")
    ]
