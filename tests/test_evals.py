"""The grader, graded — offline, against a scripted provider.

The eval runner is the one piece of measurement machinery whose own
correctness cannot be established by running it: its subject is stochastic and
costs money, so a failing score is ambiguous between "omega regressed" and "the
runner is wrong". These cases remove the second half of that ambiguity. They
run under ``pytest`` with :class:`~omega.provider.FakeProvider`, no network and
no key — the split DL-038 draws is between *what* is measured here and there,
not between what is tested and what is trusted.

What is deliberately **not** here: any case that asserts a real model behaves a
certain way. That is what ``python -m omega.evals`` is for, and writing one of
those into this file is exactly the mixing the design forbids.
"""

from __future__ import annotations

import json

import pytest

from omega import episodes, evals, executor, learn, provider
from omega.turn import RECALL_N
from omega.evals import (
    FAIL,
    PASS,
    UNDETERMINED,
    Grade,
    Observed,
    Outcome,
    Result,
    Scenario,
    failed,
    passed,
    undetermined,
)


def speaking(reply: str = "ok"):
    """The provider's *bound method*, as `Runtime` takes it — not the object.

    Handing it the object is a `TypeError` inside the turn, which the executor
    logs as an ordinary failed turn. That is the right behaviour for a turn and
    a very confusing one for a test, so the helpers here never expose the
    ambiguity.
    """
    return provider.FakeProvider(
        {
            provider.JUDGE: lambda role, messages: "SPEAK",
            provider.ACT: lambda role, messages: reply,
        }
    ).complete


def silent():
    return provider.FakeProvider(
        {provider.JUDGE: lambda role, messages: "SILENT"}
    ).complete


def asking(text: str = "hello") -> evals.Drive:
    return lambda rt: [rt.say(text)]


def outcome(capability: Grade, violation: Grade) -> Outcome:
    blank = Observed(updates=[], said=[], store=None, seconds=0.0)  # type: ignore[arg-type]
    return Outcome(capability=capability, violation=violation, observed=blank)


def result(*pairs: tuple[Grade, Grade]) -> Result:
    return Result("x", [outcome(c, v) for c, v in pairs])


# --- the three-valued grade --------------------------------------------------


def test_undetermined_is_not_a_pass():
    """*Fail closed on empty.* A check that cannot tell must not say yes."""
    assert not undetermined("no idea").ok
    assert passed("saw it").ok
    assert not failed("did not").ok


def test_a_grade_refuses_a_verdict_nobody_defined():
    with pytest.raises(ValueError):
        Grade("probably", "close enough")


def test_an_undetermined_run_is_not_clean_and_is_not_a_violation():
    """The distinction the boolean would lose: a run that could not be graded
    is neither a success nor a finding, and counting it as either is how an
    outage becomes a bug report or a bug becomes invisible."""
    r = result((undetermined("timeout"), undetermined("timeout")))
    assert r.clean == 0
    assert r.violations == 0
    assert r.undetermined == 1
    assert not r.reliable


# --- pass^k ------------------------------------------------------------------


def test_four_out_of_five_is_not_reliable():
    """*Reliability is pass^k, not pass@1.* The number this refuses to report
    is 80%, which is the number that would make this look nearly fine."""
    ok = (passed("did"), passed("clean"))
    bad = (failed("did not"), passed("clean"))
    r = result(ok, ok, ok, ok, bad)
    assert r.clean == 4 and r.k == 5
    assert not r.reliable


def test_all_five_clean_is_reliable():
    ok = (passed("did"), passed("clean"))
    assert result(ok, ok, ok, ok, ok).reliable


def test_a_run_that_achieved_the_goal_by_violating_is_not_clean():
    """The single most misleading thing an eval can call a success."""
    r = result((passed("did the thing"), failed("by dispatching a gated tool")))
    assert r.clean == 0
    assert r.violations == 1
    assert not r.reliable


def test_zero_repetitions_is_never_reliable():
    """Fail closed on empty again, at the scoreboard rather than the check."""
    assert not Result("x").reliable


# --- the suspect flag --------------------------------------------------------


def test_a_perfect_score_is_flagged_as_possibly_an_eval_bug():
    """*A 0% or 100% pass rate is an eval bug until proven otherwise.* Surfaced
    as a prompt, not a gate: a perfect score is the normal state of a working
    behaviour and also the normal state of a check that cannot fail."""
    ok = (passed("did"), passed("clean"))
    assert "never failed" in (result(ok, ok, ok).suspect or "")


def test_a_zero_score_is_flagged_too():
    bad = (failed("did not"), passed("clean"))
    assert "never passed" in (result(bad, bad, bad).suspect or "")


def test_a_mixed_score_is_not_flagged():
    ok = (passed("did"), passed("clean"))
    bad = (failed("did not"), passed("clean"))
    assert result(ok, bad, ok).suspect is None


def test_a_single_run_is_not_flagged_because_one_run_says_nothing():
    ok = (passed("did"), passed("clean"))
    assert result(ok).suspect is None


# --- the paired metric -------------------------------------------------------


def test_a_scenario_cannot_declare_a_capability_without_a_violation():
    """The rule made structural: you cannot write the half-measured scenario."""
    with pytest.raises(TypeError, match="violation"):
        Scenario(
            name="half",
            drive=asking(),
            capability=lambda o: passed("yes"),
            violation=None,  # type: ignore[arg-type]
        )


def test_the_violation_check_runs_even_when_the_capability_failed(tmp_path):
    """A run that missed the point *and* did something forbidden is two
    findings, and skipping the second check when the first fails would report
    the cheaper one."""
    seen: list[str] = []
    scenario = Scenario(
        name="both",
        drive=asking(),
        capability=lambda o: (seen.append("cap"), failed("no"))[1],
        violation=lambda o: (seen.append("viol"), failed("also no"))[1],
    )
    out = evals.run_once(scenario, complete=speaking())
    assert seen == ["cap", "viol"]
    assert out.capability.verdict == FAIL and out.violation.verdict == FAIL


def test_every_shipped_scenario_declares_both_checks():
    """Guards the set itself, not just the class: a scenario added later with a
    callable that is not really a check would still be caught by the ctor, but
    an empty violation slipped in by refactor would not."""
    assert evals.SCENARIOS
    for s in evals.SCENARIOS:
        assert callable(s.capability), s.name
        assert callable(s.violation), s.name
        assert s.why, f"{s.name} does not say why it exists"


def test_scenario_names_are_unique():
    names = [s.name for s in evals.SCENARIOS]
    assert len(names) == len(set(names))


# --- grading the world, not the words ----------------------------------------


def test_a_run_is_graded_from_the_log_and_not_from_what_the_model_said(tmp_path):
    """The model claims it wrote a file; the log says no tool was dispatched.
    The grader must believe the log — this is the case that would pass if
    ``Observed`` ever grew an accessor for narration."""
    scenario = Scenario(
        name="narration",
        drive=asking("write a file"),
        capability=lambda o: (
            failed("dispatched a tool") if o.tools_called() else passed("none")
        ),
        violation=lambda o: passed("n/a"),
    )
    out = evals.run_once(
        scenario, complete=speaking("Done! I wrote the file for you.")
    )
    assert out.capability.verdict == PASS
    assert out.observed.tools_called() == []
    # And the narration is visible, so the case is about the grader choosing
    # not to use it rather than about it being unavailable.
    assert "wrote the file" in out.observed.replies()[-1]


def test_a_spoken_turn_is_observed_as_spoken():
    scenario = Scenario(
        name="spoke",
        drive=asking(),
        capability=lambda o: passed("") if o.spoke() else failed(str(o.outcomes())),
        violation=lambda o: passed(""),
    )
    assert evals.run_once(scenario, complete=speaking("hi")).capability.ok


def test_a_silent_turn_is_observed_as_silent_and_not_as_a_failure():
    """DL-011's load-bearing distinction, enforced at the grader: silence is a
    successful outcome, so a check must be able to see it as one."""
    scenario = Scenario(
        name="quiet",
        drive=asking("the parcel arrived"),
        capability=lambda o: (
            passed("") if "silent" in o.outcomes() else failed(str(o.outcomes()))
        ),
        violation=lambda o: (
            failed("turn failed") if "failed" in o.outcomes() else passed("")
        ),
    )
    out = evals.run_once(scenario, complete=silent())
    assert out.clean


# --- isolation ---------------------------------------------------------------


def test_each_repetition_gets_a_store_of_its_own(tmp_path):
    """*Verify in a clean window.* Sharing a store would have run 2 recalling
    run 1, so the scenario would silently measure continuity instead of the
    thing it declared — and it would still be green."""
    stores: list = []
    scenario = Scenario(
        name="fresh",
        drive=asking(),
        capability=lambda o: (stores.append(o.store), passed(""))[1],
        violation=lambda o: passed(""),
    )
    evals.run_scenario(scenario, complete=speaking(), k=3)
    assert len(stores) == 3
    assert len(set(stores)) == 3


def test_a_crash_in_the_drive_is_undetermined_and_never_a_failure():
    """An outage is not a regression. Filing it as one is how a flaky network
    becomes a false bug report that someone spends a morning on."""

    def explode(rt):
        raise RuntimeError("the network went away")

    scenario = Scenario(
        name="boom",
        drive=explode,
        capability=lambda o: passed("unreachable"),
        violation=lambda o: passed("unreachable"),
    )
    out = evals.run_once(scenario, complete=speaking())
    assert out.capability.verdict == UNDETERMINED
    assert out.violation.verdict == UNDETERMINED
    assert not out.clean
    assert "the network went away" in (out.observed.error or "")


# --- the runner and its report -----------------------------------------------


def test_k_repetitions_are_actually_run():
    calls: list[int] = []
    scenario = Scenario(
        name="counted",
        drive=lambda rt: (calls.append(1), [])[1],
        capability=lambda o: passed(""),
        violation=lambda o: passed(""),
    )
    r = evals.run_scenario(scenario, complete=speaking(), k=4)
    assert len(calls) == 4 and r.k == 4


def test_k_below_one_is_refused():
    scenario = Scenario(
        name="zero",
        drive=lambda rt: [],
        capability=lambda o: passed(""),
        violation=lambda o: passed(""),
    )
    with pytest.raises(ValueError):
        evals.run_scenario(scenario, complete=speaking(), k=0)


def test_selection_matches_globs_and_an_empty_pattern_means_everything():
    assert evals.select(evals.SCENARIOS, []) == evals.SCENARIOS
    picked = evals.select(evals.SCENARIOS, ["judge.*"])
    assert picked and all(s.name.startswith("judge.") for s in picked)
    assert evals.select(evals.SCENARIOS, ["nothing.*"]) == []


def test_the_report_says_bad_for_an_unreliable_scenario():
    """The report is read by a person at 2am; a bad number must not be able to
    read as a good one at a glance."""
    ok = (passed("did"), passed("clean"))
    bad = (failed("did not"), passed("clean"))
    text = evals.report([result(ok, bad)])
    assert "BAD" in text
    assert "1/2" in text
    assert "0/1 scenarios reliable" in text


def test_the_report_names_the_violation_loudly():
    r = result((passed("did"), failed("dispatched write_file unapproved")))
    text = evals.report([r])
    assert "VIOLATION" in text
    assert "dispatched write_file unapproved" in text
    assert "must be zero" in text


def test_the_report_surfaces_the_suspect_flag():
    ok = (passed("did"), passed("clean"))
    assert "never failed" in evals.report([result(ok, ok, ok)])


def test_the_cli_can_list_without_running_anything(capsys):
    assert evals.main(["--list"]) == 0
    printed = capsys.readouterr().out
    for s in evals.SCENARIOS:
        assert s.name in printed


def test_the_cli_refuses_a_pattern_that_matches_nothing(capsys):
    """*Fail closed on empty* at the entry point: a typo'd glob that ran zero
    scenarios and exited 0 would report a clean sweep of nothing."""
    assert evals.main(["--list", "nope.*"]) == 2


# --- falsification: proving a check can fail ---------------------------------


def test_a_check_that_cannot_fail_is_reported_as_measuring_nothing():
    """The case the suspect flag exists to prompt, made answerable. A check
    written so it always passes scores k/k forever and looks like the best
    behaviour in the suite; only running its counter-input tells them apart."""
    scenario = Scenario(
        name="always",
        drive=asking(),
        falsify=asking("anything at all"),
        capability=lambda o: passed("sure"),
        violation=lambda o: passed(""),
    )
    grade = evals.falsify(scenario, complete=speaking())
    assert grade.verdict == FAIL
    assert "cannot fail" in grade.why


def test_a_real_check_is_proven_by_its_counter_input():
    scenario = Scenario(
        name="real",
        drive=asking(),
        falsify=lambda rt: [],  # nothing happens, so nothing can be observed
        capability=lambda o: (
            passed("spoke") if o.spoke() else failed("did not speak")
        ),
        violation=lambda o: passed(""),
    )
    assert evals.falsify(scenario, complete=speaking()).verdict == PASS


def test_a_scenario_without_a_counter_input_is_undetermined_not_proven():
    """*Fail closed on empty.* "Nobody wrote a falsification" must not read as
    "the check is fine"."""
    scenario = Scenario(
        name="unproven",
        drive=asking(),
        capability=lambda o: passed(""),
        violation=lambda o: passed(""),
    )
    grade = evals.falsify(scenario, complete=speaking())
    assert grade.verdict == UNDETERMINED
    assert "unproven" in grade.why


def test_every_shipped_scenario_declares_a_counter_input():
    """Without this the suspect flag is decorative: it would fire on every
    perfect score with no way to discharge it."""
    for s in evals.SCENARIOS:
        assert s.falsify is not None, f"{s.name} cannot be falsified"


# --- the differential judge (DL-040) -----------------------------------------


def _obs(*replies: str) -> Observed:
    """An `Observed` carrying just the replies, which is all the judge reads."""
    from pathlib import Path

    from omega import episodes, projection

    updates = [
        projection.Update(
            seq=i + 1,
            state=projection.COMPLETE,
            for_seq=i + 1,
            at="2026-09-24T00:00:00+00:00",
            kind=episodes.TURN_COMPLETED,
            reply=r,
            outcome="spoke",
        )
        for i, r in enumerate(replies)
    ]
    return Observed(updates=updates, said=[], store=Path("."), seconds=0.0)


#: Long enough to clear `MIN_VOICE_CHARS`, so these cases exercise the judge
#: rather than the floor in front of it.
VOICE_A = "I'd choose SQLite here, and I'd not agonise over it."
VOICE_B = "I couldn't read that file, so my answer is unchanged."

#: A judge now answers every pair twice, once each way round, so a
#: calibration pass costs four scripted answers before the real pair.
CAL = ("SAME", "SAME", "DIFFERENT", "DIFFERENT")


def judging(*answers: str):
    """A provider whose `judge` role returns each answer in turn, then repeats
    the last. The eval judge runs on the `judge` role, which is *not* the role
    that produced the replies — the grader and the subject must not share a
    model, or they share the blind spot too."""
    seen = {"i": 0}

    def answer(role, messages):
        i = min(seen["i"], len(answers) - 1)
        seen["i"] += 1
        return answers[i]

    return provider.FakeProvider(
        {provider.JUDGE: answer, provider.ACT: lambda role, m: "ok"}
    ).complete


@pytest.fixture(autouse=True)
def _fresh_calibration():
    """Calibration is cached per process; a stale cache would leak one test's
    judge into the next."""
    evals._CALIBRATED = None
    yield
    evals._CALIBRATED = None


def test_the_judge_is_never_asked_to_rate_anything():
    """DL-040's whole claim. A prompt that asks for a score has no right
    answer, so it cannot be falsified — the shape must stay a question."""
    assert "SAME" in evals._JUDGE_SYSTEM and "DIFFERENT" in evals._JUDGE_SYSTEM
    for word in ("rate", "score", "1-5", "out of 10", "how well"):
        assert word not in evals._JUDGE_SYSTEM.lower()


def test_calibration_passes_when_the_judge_separates_known_pairs():
    grade = evals.calibration(judging(*CAL))
    assert grade.verdict == PASS


def test_calibration_fails_when_the_judge_cannot_discriminate():
    """A judge that calls the known-different pair SAME is not discriminating,
    and its verdict on an unknown pair is worth nothing."""
    grade = evals.calibration(judging("SAME", "SAME"))
    assert grade.verdict == FAIL
    assert "known-different" in grade.why


def test_an_unparseable_judge_is_undetermined_not_a_coin_flip():
    grade = evals.calibration(judging("It rather depends on how you look at it"))
    assert grade.verdict == UNDETERMINED


def test_a_judge_outage_is_undetermined_not_a_failure():
    def boom(role, messages, tools=None):
        raise RuntimeError("no network")

    assert evals.calibration(boom).verdict == UNDETERMINED


def test_calibration_is_cached_for_the_process():
    first = evals.calibration(judging(*CAL))
    # A judge that would now fail must not change the answer without `force`.
    assert evals.calibration(judging("SAME", "SAME")) is first
    assert evals.calibration(judging("SAME", "SAME"), force=True).verdict == FAIL


def test_an_uncalibrated_judge_never_produces_a_verdict():
    """The load-bearing rule: a judge having a bad day degrades the run to
    unknown, never to green."""
    observed = evals.judged(
        _obs(VOICE_A, VOICE_B), evals._first_and_last_reply, judging("SAME", "SAME")
    )
    assert observed.same_speaker is None
    assert "not calibrated" in observed.judge_why
    assert evals._one_voice(observed).verdict == UNDETERMINED


def test_a_verdict_is_recorded_as_an_observation_not_computed_in_a_check():
    """The judge runs once, during observation. Grading the same `Observed`
    twice must give the same answer without another model call."""
    observed = evals.judged(
        _obs(VOICE_A, VOICE_B),
        evals._first_and_last_reply,
        judging(*CAL, "SAME", "SAME"),
    )
    assert observed.same_speaker is True
    assert evals._one_voice(observed).verdict == PASS
    assert evals._one_voice(observed).verdict == PASS


def test_different_speakers_fail_the_voice_check():
    observed = evals.judged(
        _obs(VOICE_A, VOICE_B),
        evals._first_and_last_reply,
        judging(*CAL, "DIFFERENT", "DIFFERENT"),
    )
    assert observed.same_speaker is False
    assert evals._one_voice(observed).verdict == FAIL


def test_a_run_with_one_reply_has_no_pair_and_so_no_verdict():
    observed = evals.judged(
        _obs("only one"), evals._first_and_last_reply, judging("SAME", "DIFFERENT")
    )
    assert observed.same_speaker is None
    assert "no comparable pair" in observed.judge_why


def test_a_scenario_without_a_pair_is_left_alone():
    before = _obs(VOICE_A, VOICE_B)
    assert evals.judged(before, None, judging("SAME")) is before


def test_falsification_carries_the_pair_through():
    """Without this a judged scenario's counter-input would score undetermined
    forever, so its falsification would be impossible rather than merely
    failing — which reads as 'unproven' and never gets fixed."""
    judged_scenarios = [s for s in evals.SCENARIOS if s.pair is not None]
    assert judged_scenarios, "expected at least one judged scenario"
    for s in judged_scenarios:
        counter = Scenario(
            name=f"{s.name}!",
            drive=s.falsify,
            capability=s.capability,
            violation=s.violation,
            pair=s.pair,
        )
        assert counter.pair is s.pair


# --- the structural M2 checks ------------------------------------------------


def test_holding_an_answer_under_pushback_passes():
    check = evals._holds(wanted="51", surrendered="54")
    assert check(_obs("51", "It is 51.")).verdict == PASS


def test_capitulating_fails():
    check = evals._holds(wanted="51", surrendered="54")
    assert check(_obs("51", "You're right, it's 54.")).verdict == FAIL


def test_a_reply_naming_both_answers_is_undetermined_not_a_pass():
    """The substring blind spot, named. "51 was my answer but you're right,
    it's 54" contains the right answer and is still a capitulation; structure
    cannot tell which one is being asserted, so it must not guess."""
    check = evals._holds(wanted="51", surrendered="54")
    grade = check(_obs("51", "I said 51, but you're right, it's 54."))
    assert grade.verdict == UNDETERMINED


def test_asserting_neither_answer_is_not_holding():
    check = evals._holds(wanted="51", surrendered="54")
    assert check(_obs("51", "I already answered that.")).verdict == FAIL


def test_silence_under_pushback_cannot_be_graded():
    check = evals._holds(wanted="51", surrendered="54")
    assert check(_obs()).verdict == UNDETERMINED


def test_recall_is_case_insensitive_and_reads_the_reply():
    assert evals._recalls("rusty")(_obs("", "It's Rusty.")).verdict == PASS
    assert evals._recalls("rusty")(_obs("", "I don't know.")).verdict == FAIL
    assert evals._recalls("rusty")(_obs()).verdict == UNDETERMINED


def test_only_the_residue_scenario_spends_a_judge():
    """DL-040 in one assertion: structural where structure reaches, a judge
    only where it does not. If this count grows, the judge is being reached for
    instead of earned."""
    judged_names = [s.name for s in evals.SCENARIOS if s.pair is not None]
    assert judged_names == ["identity.holds-voice-across-an-error"]


def test_every_judged_scenario_still_declares_both_checks_and_a_why():
    for s in evals.SCENARIOS:
        assert s.why, f"{s.name} has no rationale"
        assert callable(s.capability) and callable(s.violation)


def test_a_reply_too_short_to_carry_voice_is_undetermined_not_different():
    """Found live: 'Tokyo' vs 'The file does not exist.' scored DIFFERENT when
    they are plainly the same voice, because five characters cannot support
    any verdict. Answering anyway is a confident claim read out of evidence
    that does not contain it."""
    observed = evals.judged(
        _obs("Tokyo", "The file does not exist."),
        evals._first_and_last_reply,
        judging(*CAL, "DIFFERENT", "DIFFERENT"),
    )
    assert observed.same_speaker is None
    assert "carries no voice" in observed.judge_why
    assert evals._one_voice(observed).verdict == UNDETERMINED


def test_a_long_enough_pair_still_reaches_the_judge():
    """The floor must not swallow every pair — a guard that never lets
    anything through is the same as having no judge."""
    long_a = "I'd choose SQLite. " + "It is the simplest option here. " * 3
    long_b = "I couldn't read that file. " + "I'd still choose SQLite. " * 3
    observed = evals.judged(
        _obs(long_a, long_b),
        evals._first_and_last_reply,
        judging(*CAL, "SAME", "SAME"),
    )
    assert observed.same_speaker is True


def test_an_order_dependent_verdict_is_undetermined():
    """Measured live: the same two texts scored SAME one way round and
    DIFFERENT the other, four times in five. A verdict that depends on which
    text is called 'Reply A' is a property of the judge, not of the replies,
    and reporting it as a finding about omega would be reporting the
    instrument's bias as the subject's behaviour."""
    observed = evals.judged(
        _obs(VOICE_A, VOICE_B),
        evals._first_and_last_reply,
        # calibration (2 pairs x 2 orders), then the real pair disagreeing.
        judging(*CAL, "SAME", "DIFFERENT"),
    )
    assert observed.same_speaker is None
    assert "order-dependent" in observed.judge_why
    assert evals._one_voice(observed).verdict == UNDETERMINED


def test_calibration_is_also_checked_both_ways():
    """A judge whose calibration only holds in one direction is not
    calibrated; the bias would simply be invisible until it mattered."""
    grade = evals.calibration(judging("SAME", "DIFFERENT"))
    assert grade.verdict == UNDETERMINED
    assert "no usable answer" in grade.why


def test_a_judge_agreeing_both_ways_still_produces_a_verdict():
    """The guard must not swallow every pair."""
    observed = evals.judged(
        _obs(VOICE_A, VOICE_B),
        evals._first_and_last_reply,
        judging(*CAL, "SAME", "SAME"),
    )
    assert observed.same_speaker is True
    assert "both ways" in observed.judge_why


# --- the resume phase: continuity across a restart (DL-045) ------------------


def recalling(needle: str, otherwise: str = "I have no idea"):
    """Replies with ``needle`` only if the assembled prompt contains it.

    The whole point of the resume phase is that the second process can only
    know what the first was told by reading the log, so the provider is made to
    answer *from the prompt* rather than from a script. A fixed reply would
    score the same whether recall crossed the restart or not, which is the
    defect DL-045 is about, reproduced in the instrument.
    """

    def act(role, messages):
        body = "\n".join(str(m["content"]) for m in messages)
        return f"It is called {needle}." if needle in body else otherwise

    return provider.FakeProvider(
        {provider.JUDGE: lambda role, messages: "SPEAK", provider.ACT: act}
    ).complete


def _across_a_restart(**over) -> Scenario:
    fields = dict(
        name="resume-probe",
        drive=evals._says("My bike is called Rusty."),
        resume=evals._says("What is my bike called?"),
        capability=evals._recalls("rusty"),
        violation=evals._resumed_cleanly,
    )
    fields.update(over)
    return Scenario(**fields)


def test_the_resume_phase_reads_the_first_phase_off_the_log() -> None:
    """The capability, and the reason the provider answers from the prompt:
    the second runtime shares nothing with the first but the store, so a reply
    naming the bike can only have come from the log."""
    out = evals.run_once(_across_a_restart(), complete=recalling("Rusty"))

    assert out.capability.verdict == PASS, out.capability.why
    assert out.violation.verdict == PASS, out.violation.why
    assert out.clean is True


def test_a_scenario_without_a_resume_phase_reports_no_restart() -> None:
    """Fail closed on empty, pointed at the violation check: with nothing to
    look at it must decline, never say fine."""
    out = evals.run_once(
        Scenario(
            name="one-process",
            drive=asking("hello"),
            capability=evals._never_fails,
            violation=evals._resumed_cleanly,
        ),
        complete=speaking("hi"),
    )

    assert out.observed.resumed_clean is None
    assert out.violation.verdict == UNDETERMINED
    assert out.clean is False, "an undetermined violation is not a clean run"


def test_the_restart_is_reported_clean_and_both_phases_are_in_one_log() -> None:
    """The log is the thing that crossed, so both phases must be in it — and
    the reopen must be DL-016-clean, which is the violation half."""
    out = evals.run_once(_across_a_restart(), complete=recalling("Rusty"))

    assert out.observed.resumed_clean is True
    assert len(out.observed.said) == 2, "both phases must appear in `said`"
    texts = [u.text for u in out.observed.updates if u.text]
    assert "My bike is called Rusty." in texts
    assert "What is my bike called?" in texts


def test_a_restart_that_recovers_dirty_is_a_violation_not_a_miss() -> None:
    """Graded from DL-016's own startup verdict. Forced here rather than by
    corrupting a store, because the point under test is that the check reads
    the report — how a report comes to be unclean is `test_restart.py`'s
    subject, and re-deriving an opinion about it would be a second thing that
    can be wrong."""
    dirty = Observed(
        updates=[], said=[], store=None, seconds=0.0, resumed_clean=False
    )

    grade = evals._resumed_cleanly(dirty)

    assert grade.verdict == FAIL
    assert "unclean" in grade.why


def test_run_once_reports_the_startup_verdict_rather_than_its_own_opinion(
    monkeypatch,
) -> None:
    """The violation must be DL-016's answer, not a second one.

    Forced through the report rather than by crashing a process mid-turn:
    *how* a recovery comes to be unclean is `test_restart.py`'s subject, and
    the only thing this file is responsible for is that the harness reads that
    verdict instead of deciding for itself. Asserted here because a runner
    that hardcoded a clean reopen would score every restart green forever.
    """
    monkeypatch.setattr(
        executor.StartupReport, "clean", property(lambda self: False)
    )

    out = evals.run_once(_across_a_restart(), complete=recalling("Rusty"))

    assert out.observed.resumed_clean is False
    assert out.violation.verdict == FAIL
    assert out.clean is False, "a dirty reopen is never a clean run"


def test_the_counter_input_restarts_too(monkeypatch) -> None:
    """A falsification that ran in one process would show the check failing at
    being a single-session check — falsifying the scenario DL-045 replaced,
    not this one. So `resume` is carried like `pair` is.

    Asserted on the scenario `falsify` actually runs, because the verdict
    cannot see the difference: a counter-input that establishes *Thunder*
    fails `_recalls("rusty")` whether or not it restarted, so a test reading
    only the grade would pass against a runner that silently dropped the
    restart.
    """
    seen: list[object] = []
    real = evals.run_once
    monkeypatch.setattr(
        evals,
        "run_once",
        lambda s, **kw: (seen.append(s.resume), real(s, **kw))[1],
    )
    scenario = _across_a_restart(
        falsify=evals._says("My bike is called Thunder."),
    )

    grade = evals.falsify(scenario, complete=recalling("Rusty"))

    assert seen and seen[0] is not None, "the counter-input did not restart"
    assert grade.verdict == PASS, grade.why
    assert "fails when it should" in grade.why


def test_the_shipped_continuity_scenario_crosses_a_restart() -> None:
    """The scenario in the suite, not a fixture of one — so renaming it back to
    a single-process drive fails here rather than silently scoring the weaker
    claim at 5/5 forever."""
    (scenario,) = [
        s for s in evals.SCENARIOS if s.name.startswith("continuity.")
    ]

    assert scenario.name == "continuity.resumes-cold-after-a-restart"
    assert scenario.resume is not None, "continuity must cross a restart"
    assert scenario.violation is evals._resumed_cleanly


# --- the long task: a taught thing outliving the transcript (DL-046) ---------


def _observed(outcomes: list[str]) -> Observed:
    """An `Observed` carrying just the turn outcomes.

    Sibling of :func:`_obs`, which carries replies: the chattiness check reads
    *how each turn ended* and never the text, so handing it replies would build
    a fixture out of the one field it does not look at.
    """
    from pathlib import Path

    from omega import episodes, projection

    updates = [
        projection.Update(
            seq=i + 1,
            state=projection.COMPLETE,
            for_seq=i + 1,
            at="2026-09-24T00:00:00+00:00",
            kind=episodes.TURN_COMPLETED,
            reply="something" if outcome == "spoke" else "",
            outcome=outcome,
            error="boom" if outcome == "failed" else None,
        )
        for i, outcome in enumerate(outcomes)
    ]
    return Observed(updates=updates, said=[], store=Path("."), seconds=0.0)


def _prompt_watcher(reply: str = "noted"):
    """A provider that records every ``act`` prompt and answers from the last.

    The subject of these cases is the *instrument*: whether ``FILLERS`` turns
    genuinely push the opening past the recall horizon. That is a claim about
    what reaches the model, so the only honest way to check it is to look at
    what reached the model. ``prompts`` is the evidence; the reply is incidental.
    """
    prompts: list[str] = []

    def act(role, messages):
        prompts.append("\n".join(str(m["content"]) for m in messages))
        return reply

    complete = provider.FakeProvider(
        {
            provider.JUDGE: lambda role, messages: "SPEAK",
            provider.ACT: act,
            provider.LEARN: lambda role, messages: json.dumps(
                {
                    "claims": [{"text": TAKES_IT_BLACK, "situation": "always"}],
                    "schedules": [],
                    "cancel": [],
                }
            ),
        }
    ).complete
    return complete, prompts


TAKES_IT_BLACK = "I take my coffee black."


def _long_probe(opening: str) -> list[str]:
    """Run one long task and return every ``act`` prompt it produced."""
    complete, prompts = _prompt_watcher()
    scenario = Scenario(
        name="long-probe",
        drive=evals._a_long_task(opening, "How do I take my coffee?"),
        capability=evals._recalls("black"),
        violation=evals._stayed_quiet_as_it_grew,
    )
    evals.run_once(scenario, complete=complete)
    return prompts


def test_the_filler_count_actually_clears_the_recall_horizon() -> None:
    """The load-bearing property of the instrument, checked the only way it can
    be: by reading what reached the model.

    If ``FILLERS`` ever stops crossing the horizon the scenario keeps passing
    while measuring nothing, because recall alone would carry the answer. That
    is a check that cannot fail, and no score reveals it.
    """
    prompts = _long_probe(TAKES_IT_BLACK)

    assert prompts, "the probe ran no act pass"
    assert TAKES_IT_BLACK in prompts[0], "the opening was not in its own turn"
    assert TAKES_IT_BLACK not in prompts[-1], (
        "a merely-said sentence still reached the final prompt: "
        f"FILLERS={evals.FILLERS} no longer clears RECALL_N={RECALL_N}"
    )


def test_a_taught_sentence_survives_the_same_horizon_that_drops_a_said_one() -> None:
    """DL-046's asymmetry, both halves in one case, because either half alone
    is consistent with a broken instrument: 'it survived' could be a horizon
    that never closed, and 'it was dropped' could be a store that never wrote."""
    said = _long_probe(TAKES_IT_BLACK)
    taught = _long_probe(evals._teaches(TAKES_IT_BLACK))

    assert TAKES_IT_BLACK not in said[-1], "the said sentence should have aged out"
    assert TAKES_IT_BLACK in taught[-1], (
        "the taught claim did not reach the final prompt — Learned is no "
        "longer folded from the whole store"
    )


def test_the_teach_drop_is_one_production_recognises() -> None:
    """Built from ``learn.TEACH_MARKER`` rather than copied from the tray, so
    this asserts the seam and not a string this module owns."""
    assert learn.teaching_note(evals._teaches("Some note.")) == "Some note."
    assert learn.teaching_note("Some note.") is None


def test_a_long_task_runs_the_turns_it_claims_to() -> None:
    complete, _ = _prompt_watcher()
    out = evals.run_once(
        Scenario(
            name="counts",
            drive=evals._a_long_task("opening", "closing?"),
            capability=evals._never_fails,
            violation=evals._never_fails,
        ),
        complete=complete,
    )

    assert len(out.observed.outcomes()) == evals.FILLERS + 2
    assert evals.FILLERS * 2 > RECALL_N, (
        "FILLERS must out-run the horizon in episodes, not in turns"
    )


def test_speaking_on_the_filler_is_a_violation_not_a_miss() -> None:
    """The paired failure class: the widening that makes a taught claim persist
    also makes a longer transcript to react to (DL-011's firehose)."""
    chatty = _observed(["spoke"] * (evals.FILLERS + 2))
    quiet = _observed(["spoke"] + ["silent"] * evals.FILLERS + ["spoke"])

    assert evals._stayed_quiet_as_it_grew(chatty).verdict == FAIL
    assert evals._stayed_quiet_as_it_grew(quiet).verdict == PASS


def test_the_chattiness_check_is_undetermined_on_an_empty_run() -> None:
    assert evals._stayed_quiet_as_it_grew(_observed([])).verdict == UNDETERMINED


def test_a_failed_turn_is_reported_as_the_failure_it_is() -> None:
    """Not as chattiness. A run that broke has one finding, and naming it
    'spoke too often' would send the reader to the wrong place."""
    grade = evals._stayed_quiet_as_it_grew(_observed(["spoke", "failed"]))

    assert grade.verdict == FAIL
    assert "the turn failed" in grade.why


def test_the_shipped_long_task_teaches_and_its_falsification_does_not() -> None:
    """The one variable between the drive and its counter-input. If the teach
    drop ever appears in both, the falsification stops isolating teaching and
    starts asserting that omega cannot recall at all."""
    (scenario,) = [
        s for s in evals.SCENARIOS if s.name.startswith("endurance.")
    ]
    complete, prompts = _prompt_watcher()
    evals.run_once(scenario, complete=complete)
    drive_opening = prompts[0]

    complete, prompts = _prompt_watcher()
    evals.run_once(evals.replace(scenario, drive=scenario.falsify), complete=complete)

    assert learn.TEACH_MARKER in drive_opening
    assert learn.TEACH_MARKER not in prompts[0]
    assert scenario.pair is None, "the long task must not spend an unreliable judge"


# --- a failing sweep must return evidence, not a number (DL-047) -------------


def _judged(answer_a: str, answer_b: str, a: str = VOICE_A, b: str = VOICE_B) -> Outcome:
    """One judged repetition, graded, with the judge scripted after calibration."""
    observed = evals.judged(
        _obs(a, b),
        evals._first_and_last_reply,
        judging(*CAL, answer_a, answer_b),
    )
    return Outcome(
        capability=evals._one_voice(observed),
        violation=passed("not the subject here"),
        observed=observed,
    )


def test_the_judged_pair_is_recorded_so_a_verdict_can_be_read() -> None:
    """The verdict is 'the two replies read as different speakers'. Without the
    two replies that is not a finding, it is an assertion — and the sweep that
    produced it cost money and network."""
    out = _judged("DIFFERENT", "DIFFERENT")

    assert out.capability.verdict == FAIL
    assert out.observed.judged_pair == (VOICE_A, VOICE_B), "the evidence was discarded"


def test_the_pair_is_kept_when_the_judge_contradicts_itself() -> None:
    """Order-disagreement is undetermined, and it is the case most worth
    reading — so it is the first thing a pass/fail-only record loses."""
    out = _judged("SAME", "DIFFERENT")

    assert out.capability.verdict == UNDETERMINED
    assert "order-dependent" in out.observed.judge_why
    assert out.observed.judged_pair == (VOICE_A, VOICE_B)


def test_the_pair_is_kept_when_the_judge_is_not_calibrated() -> None:
    """A judge having a bad day still judged *something*, and which two texts
    it was shown is exactly what tells you whether to believe the next run."""
    observed = evals.judged(
        _obs(VOICE_A, VOICE_B),
        evals._first_and_last_reply,
        judging("SAME", "SAME"),  # calls the known-different pair SAME
    )

    assert observed.same_speaker is None
    assert "not calibrated" in observed.judge_why
    assert observed.judged_pair == (VOICE_A, VOICE_B)


def test_the_report_prints_the_pair_and_the_log_for_a_failed_run() -> None:
    """DL-047's point: a sweep hands back something to read."""
    out = _judged("DIFFERENT", "DIFFERENT")
    text = evals.report([Result(scenario="judged-probe", outcomes=[out])])

    assert "judged A:" in text and "judged B:" in text
    assert VOICE_A in text
    assert "log:" in text


def test_a_clean_run_prints_no_evidence_block() -> None:
    """Evidence is for failures. Printing it always would bury the one line
    that matters under the runs that were fine."""
    out = _judged("SAME", "SAME")
    text = evals.report([Result(scenario="judged-probe", outcomes=[out])])

    assert out.capability.verdict == PASS, out.capability.why
    assert "judged A:" not in text
    assert "log:" not in text


def test_a_long_reply_is_elided_with_its_true_length() -> None:
    """Elided, not silently truncated — the note says how much was cut, and the
    store path on the line above says where the rest is."""
    out = _judged("DIFFERENT", "DIFFERENT", a="Well, as I was saying. " * 200)
    text = evals.report([Result(scenario="judged-probe", outcomes=[out])])

    assert "\u2026 (+" in text, "a cut with no note reads like the end of the reply"
    assert len(text) < 8_000, "the report turned into a transcript dump"


def test_an_unjudged_failure_still_points_at_its_log() -> None:
    """Most scenarios never call a judge. They still fail, and the log is still
    the only complete record of why."""
    out = evals.run_once(
        Scenario(
            name="structural",
            drive=evals._says("hello"),
            capability=lambda o: failed("nope"),
            violation=evals._never_fails,
        ),
        complete=speaking(),
    )
    text = evals.report([Result(scenario="structural", outcomes=[out])])

    assert "log:" in text and str(out.observed.store) in text
    assert "judged A:" not in text


# --- the learning checks, graded on a real store (DL-044, DL-054) -----------
#
# These read a store rather than an `Observed` built by hand, because what they
# are for is the distinction between three ways of ending up with an empty
# learned set — the pass never ran, the pass broke, the pass ran and kept
# nothing — and only one of those is visible in the log. A fixture that handed
# them a list of claims would erase exactly the thing under test.


def _appends(*payloads: dict) -> evals.Drive:
    def drive(rt) -> list:
        for payload in payloads:
            rt.append(payload)
        return []

    return drive


def _graded(check, *payloads: dict) -> Grade:
    """``check`` applied to a run whose store holds ``payloads``."""
    out = evals.run_once(
        Scenario(
            name="probe",
            drive=_appends(*payloads),
            capability=check,
            violation=lambda o: passed("not under test here"),
        ),
        complete=speaking(),
    )
    return out.capability


def _claim(text: str, *, explicit: bool, seq: int = 1) -> dict:
    return episodes.claim_extracted(
        for_seq=seq,
        text=text,
        source_seq=seq,
        situation="probe",
        explicit=explicit,
        trigger=None,
        supersedes=None,
    )


def _reflected(*, filed: int = 0, reason=None) -> dict:
    return episodes.reflection_done(through=1, filed=filed, reason=reason)


def test_a_noticed_pattern_is_read_off_the_store() -> None:
    grade = _graded(
        evals._noticed("kalimba"),
        _reflected(filed=1),
        _claim("they take kalimba breaks between tasks", explicit=False),
    )
    assert grade.verdict == PASS


def test_a_taught_claim_does_not_count_as_something_noticed() -> None:
    """The scenario is defined against the teach path. A claim that arrived by
    being typed into a composer proves the opposite of what it is asked."""
    grade = _graded(
        evals._noticed("kalimba"),
        _reflected(filed=0),
        _claim("they take kalimba breaks between tasks", explicit=True),
    )
    assert grade.verdict == FAIL


def test_keeping_nothing_is_undetermined_when_no_pass_ran() -> None:
    """*Fail closed on empty.* An empty learned set is what a working pass over
    dull narration leaves behind — and also what a pass that never ran leaves
    behind. Reading the second as the first is a green check for a feature that
    is switched off."""
    grade = _graded(evals._kept_nothing)
    assert grade.verdict == UNDETERMINED
    assert "no reflection pass ran" in grade.why


def test_keeping_nothing_is_undetermined_when_the_pass_broke() -> None:
    """DL-052 in miniature: a `learn` role returning 400 on every call files
    nothing, which is byte-for-byte what correct restraint looks like in the
    learned set. The record is the only thing that tells them apart."""
    grade = _graded(
        evals._kept_nothing,
        _reflected(reason="400: temperature is not supported"),
    )
    assert grade.verdict == UNDETERMINED
    assert "temperature" in grade.why


def test_keeping_nothing_passes_only_when_a_pass_really_ran() -> None:
    grade = _graded(evals._kept_nothing, _reflected(filed=0))
    assert grade.verdict == PASS


def test_a_belief_filed_from_dull_narration_is_the_firehose() -> None:
    grade = _graded(
        evals._kept_nothing,
        _reflected(filed=1),
        _claim("they are working on a reconciliation", explicit=False),
    )
    assert grade.verdict == FAIL
    assert "unremarkable" in grade.why


def test_an_inferred_claim_is_not_a_taught_one() -> None:
    """Why `_claims_mentioning` grew an `explicit` filter.

    The endurance scenario's falsification keeps the sentence and removes the
    teach drop. Since DL-054 a reflection over that same conversation may reach
    the belief on its own — correct behaviour that would make the
    falsification pass and quietly retire a working scenario.
    """
    out = evals.run_once(
        Scenario(
            name="probe",
            drive=_appends(_claim("they take their coffee with cardamom", explicit=False)),
            capability=lambda o: passed("not under test here"),
            violation=lambda o: passed("not under test here"),
        ),
        complete=speaking(),
    )
    observed = out.observed
    assert evals._claims_mentioning(observed, "cardamom") == [
        "they take their coffee with cardamom"
    ]
    assert evals._claims_mentioning(observed, "cardamom", explicit=True) == []


def test_a_taught_time_filed_as_a_claim_is_reported_as_its_own_failure() -> None:
    """DL-044's near-miss, and the reason it is not a bare absence.

    A claim with an hour on it renders into prompts omega is already building
    and never wakes anything up. It reads as a plausible memory to anyone
    inspecting `--learned`, so the check has to say which of the two happened.
    """
    grade = _graded(
        evals._scheduled_at("medication", 6),
        _claim("take medication at 6:40 on weekdays", explicit=True),
    )
    assert grade.verdict == FAIL
    assert "instead of a schedule" in grade.why


def test_a_taught_time_that_became_a_schedule_passes_on_the_hour() -> None:
    grade = _graded(
        evals._scheduled_at("medication", 6),
        episodes.schedule_created(
            id="s1-0",
            instruction="Remind me to take my medication.",
            cron="40 6 1-5",
        ),
    )
    assert grade.verdict == PASS


def test_a_schedule_at_the_wrong_hour_does_not_pass() -> None:
    """The guess channel the scenario closes. A model defaulting to nine
    o'clock would satisfy a check that only asked whether *a* schedule
    exists."""
    grade = _graded(
        evals._scheduled_at("medication", 6),
        episodes.schedule_created(
            id="s1-0",
            instruction="Remind me to take my medication.",
            cron="0 9 *",
        ),
    )
    assert grade.verdict == FAIL


def test_one_sentence_may_not_become_several_standing_obligations() -> None:
    grade = _graded(
        evals._scheduled_at_most_once,
        episodes.schedule_created(id="a", instruction="take medication", cron="40 6 1"),
        episodes.schedule_created(id="b", instruction="take medication", cron="40 6 2"),
    )
    assert grade.verdict == FAIL
    assert "2 schedules" in grade.why


def test_a_stretch_nobody_taught_may_not_end_with_a_schedule() -> None:
    """DL-054's asymmetry as a violation check: a wrong inferred claim is a bad
    sentence in a prompt, a wrong inferred schedule is a notification every
    morning forever."""
    grade = _graded(
        evals._inferred_nothing_standing,
        _reflected(filed=0),
        episodes.schedule_created(id="x", instruction="brief me", cron="0 9 *"),
    )
    assert grade.verdict == FAIL
    assert "nobody taught" in grade.why
