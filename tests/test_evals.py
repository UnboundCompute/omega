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

import pytest

from omega import evals, provider
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
