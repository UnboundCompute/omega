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
