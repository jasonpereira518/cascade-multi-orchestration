"""Audit log, review loop, hints, and status. Synthetic tasks only."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from cascade.taskgen import audit as au
from cascade.taskgen.audit import AuditError
from tests.phase3_helpers import GOLD_PATCH, make_task, make_tasks

NOW = dt.datetime(2026, 2, 3, 4, 5, 6, tzinfo=dt.UTC)


class Script:
    """Scripted auditor: answers are consumed in order; prompts and output are recorded."""

    def __init__(self, *answers: str) -> None:
        self.answers: Iterator[str] = iter(answers)
        self.prompts: list[str] = []
        self.out: list[str] = []

    def ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self.answers)

    def echo(self, text: str) -> None:
        self.out.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self.out)


def review(tasks, path: Path, *answers: str, **kw):  # type: ignore[no-untyped-def]
    s = Script(*answers)
    stats = au.run_review(tasks, path, ask=s.ask, echo=s.echo, now=lambda: NOW, **kw)
    return stats, s


# --- the log ------------------------------------------------------------------------------


def test_latest_verdict_wins_and_nothing_is_rewritten(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    au.append_verdict(path, au.make_verdict("t1", "ok", "first", now=NOW))
    au.append_verdict(path, au.make_verdict("t2", "leaks_fix", now=NOW))
    first = path.read_text()
    au.append_verdict(path, au.make_verdict("t1", "underspecified", "changed my mind", now=NOW))
    assert path.read_text().startswith(first)  # append-only
    assert len(au.read_log(path)) == 3
    latest = au.load_verdicts(path)
    assert latest["t1"].verdict == "underspecified" and latest["t1"].notes == "changed my mind"
    assert latest["t2"].verdict == "leaks_fix"


def test_make_verdict_strips_notes_and_stamps_time() -> None:
    v = au.make_verdict("t", "ok", "  spaced  ", now=NOW)
    assert v.notes == "spaced" and v.ts == NOW.isoformat()


def test_missing_log_is_empty_and_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    assert au.read_log(path) == [] and au.load_verdicts(path) == {}
    au.append_verdict(path, au.make_verdict("t", "ok", now=NOW))
    path.write_text(path.read_text() + "\n\n")
    assert len(au.read_log(path)) == 1


def test_unreadable_or_invalid_lines_name_the_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    good = au.make_verdict("t", "ok", now=NOW).model_dump_json()
    path.write_text(good + "\n{broken\n")
    with pytest.raises(AuditError, match=r"audit.jsonl:2"):
        au.read_log(path)
    bad_verdict = json.dumps({"task_id": "t", "verdict": "maybe", "ts": "x"})
    path.write_text(bad_verdict + "\n")
    with pytest.raises(AuditError, match=r":1"):
        au.read_log(path)


# --- ordering ----------------------------------------------------------------------------


def test_order_is_deterministic_independent_of_input_order_and_mixes_repos() -> None:
    tasks = make_tasks({"o/a": 6, "o/b": 6})
    a = [t.task_id for t in au.order_tasks(tasks)]
    b = [t.task_id for t in au.order_tasks(list(reversed(tasks)))]
    assert a == b and sorted(a) == sorted(t.task_id for t in tasks)
    assert a != sorted(a)  # not simply repo by repo
    assert [t.task_id for t in au.order_tasks(tasks, seed="other")] != a


# --- the review loop ---------------------------------------------------------------------


def test_each_verdict_key_is_recorded_with_its_note(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    tasks = au.order_tasks(make_tasks({"o/a": 4}))
    stats, _ = review(tasks, path, "o", "good", "u", "", "l", "has the patch", "a", "odd assert")
    got = au.load_verdicts(path)
    assert [got[t.task_id].verdict for t in tasks] == [
        "ok", "underspecified", "leaks_fix", "arbitrary_test",
    ]  # fmt: skip
    assert [got[t.task_id].notes for t in tasks] == ["good", "", "has the patch", "odd assert"]
    assert stats.recorded == 4 and stats.skipped == 0 and stats.remaining == 0 and not stats.quit
    assert dict(stats.by_verdict) == {
        "ok": 1, "underspecified": 1, "leaks_fix": 1, "arbitrary_test": 1,
    }  # fmt: skip
    assert all(v.ts == NOW.isoformat() for v in got.values())


def test_answers_are_case_and_whitespace_insensitive(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    [t] = make_tasks({"o/a": 1})
    review([t], path, "  O  ", "")
    assert au.load_verdicts(path)[t.task_id].verdict == "ok"


def test_invalid_answer_reasks_without_recording(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    [t] = make_tasks({"o/a": 1})
    stats, s = review([t], path, "x", "okay", "", "o", "")
    assert s.text.count("Please answer o, u, l, a, s, or q.") == 3
    assert stats.recorded == 1 and au.load_verdicts(path)[t.task_id].verdict == "ok"


def test_skip_records_nothing_and_the_task_comes_back_next_session(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    tasks = au.order_tasks(make_tasks({"o/a": 2}))
    stats, _ = review(tasks, path, "s", "o", "")
    assert stats.skipped == 1 and stats.recorded == 1 and stats.remaining == 0
    assert set(au.load_verdicts(path)) == {tasks[1].task_id}
    stats2, _ = review(tasks, path, "o", "")  # next session: only the skipped one is shown
    assert stats2.recorded == 1 and len(au.load_verdicts(path)) == 2


def test_quit_stops_immediately_and_keeps_everything_so_far(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    tasks = au.order_tasks(make_tasks({"o/a": 4}))
    stats, _ = review(tasks, path, "o", "", "u", "", "q")
    assert stats.quit and stats.recorded == 2 and stats.remaining == 2
    assert set(au.load_verdicts(path)) == {tasks[0].task_id, tasks[1].task_id}


def test_resume_shows_only_unreviewed_tasks_and_redo_shows_all(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    tasks = au.order_tasks(make_tasks({"o/a": 3}))
    review(tasks, path, "o", "", "q")
    stats, s = review(tasks, path, "u", "", "u", "")
    assert stats.recorded == 2 and "2 task(s) to review." in s.text
    done = au.load_verdicts(path)
    assert [done[t.task_id].verdict for t in tasks] == ["ok", "underspecified", "underspecified"]

    stats, s = review(tasks, path, "l", "", "q", redo=True)
    assert "3 task(s) to review." in s.text and "previous verdict: ok" in s.text
    assert au.load_verdicts(path)[tasks[0].task_id].verdict == "leaks_fix"  # newest wins


def test_nothing_to_review_asks_nothing(tmp_path: Path) -> None:
    stats, s = review([], tmp_path / "audit.jsonl")
    assert stats.recorded == 0 and s.prompts == [] and "0 task(s) to review." in s.text


def test_rubric_is_shown_once_at_the_start(tmp_path: Path) -> None:
    [t] = make_tasks({"o/a": 1})
    _, s = review([t], tmp_path / "a.jsonl", "q")
    assert s.text.count("VERDICTS") == 1 and "arbitrary_test" in s.text


# --- rendering ---------------------------------------------------------------------------


def test_render_shows_issue_gold_and_test_patch_and_the_tests_to_pass() -> None:
    t = make_task("o/a", 7, split="test")
    out = au.render_task(t, position=2, total=9)
    assert "Task 2/9: o__a-7" in out and "split test" in out
    assert t.issue_title in out and "returns ' x '" in out
    assert "return flag.strip()" in out  # gold patch
    assert "def test_strip" in out  # test patch
    assert "GOLD PATCH (hidden from agents)" in out and "TEST PATCH (hidden from agents)" in out
    assert "fail_to_pass: tests/test_core.py::test_strip" in out
    assert "split unassigned" in au.render_task(make_task(), position=1, total=1)


def test_render_truncates_long_patches_unless_full() -> None:
    long = "\n".join(f"+line {i}" for i in range(500))
    t = make_task(gold_patch=GOLD_PATCH + long)
    clipped = au.render_task(t, position=1, total=1)
    assert "+line 10" in clipped and "+line 499" not in clipped
    assert "more lines; rerun with --full" in clipped
    assert "+line 499" in au.render_task(t, position=1, total=1, full=True)


def test_render_lists_at_most_eight_failing_tests() -> None:
    t = make_task(fail_to_pass=[f"tests/t.py::t{i}" for i in range(12)])
    out = au.render_task(t, position=1, total=1)
    assert "t7" in out and "t8" not in out and "(+4)" in out


def test_render_shows_the_previous_verdict_with_a_placeholder_for_no_note() -> None:
    t = make_task()
    prev = au.make_verdict(t.task_id, "ok", now=NOW)
    assert "previous verdict: ok (no note)" in au.render_task(t, position=1, total=1, previous=prev)


# --- hints -------------------------------------------------------------------------------


def test_clean_task_has_no_hints() -> None:
    assert au.audit_hints(make_task()) == []


def test_fix_line_pasted_into_the_issue_is_flagged_as_a_possible_leak() -> None:
    body = "Please change it:\n    return flag.strip() + 'trailing text'\n"
    gold = GOLD_PATCH.replace("return flag.strip()", "return flag.strip() + 'trailing text'")
    hints = au.audit_hints(make_task(issue_body=body, gold_patch=gold))
    assert any(h.startswith("LEAK?") and "verbatim" in h for h in hints)


def test_short_added_lines_are_not_treated_as_leaks() -> None:
    # the gold's added line "return flag.strip()" (19 chars) is quoted in the issue, but is too
    # short and common to count as copying the fix
    hints = au.audit_hints(make_task(issue_body="maybe: return flag.strip() somewhere?"))
    assert not any("verbatim" in h for h in hints)


def test_diff_fragments_in_the_issue_are_flagged() -> None:
    assert any(
        "diff or patch" in h for h in au.audit_hints(make_task(issue_body="```diff\n-a\n```"))
    )
    body = "see:\n@@ -1,3 +1,3 @@\n-x\n+y"
    assert any("diff or patch" in h for h in au.audit_hints(make_task(issue_body=body)))


def test_new_public_name_used_by_tests_but_absent_from_the_issue_is_flagged() -> None:
    gold = GOLD_PATCH + "+def brand_new_helper(x):\n+    return x\n"
    test = (
        "diff --git a/tests/test_core.py b/tests/test_core.py\n"
        "--- a/tests/test_core.py\n+++ b/tests/test_core.py\n@@ -1,1 +1,2 @@\n"
        " x\n+assert brand_new_helper(1) == 1\n"
    )
    flagged = au.audit_hints(make_task(gold_patch=gold, test_patch=test))
    assert any(h.startswith("UNDERSPECIFIED?") and "brand_new_helper" in h for h in flagged)
    named = au.audit_hints(
        make_task(gold_patch=gold, test_patch=test, issue_body="add brand_new_helper(x)")
    )
    assert not any(h.startswith("UNDERSPECIFIED?") for h in named)


def test_literal_asserted_by_tests_but_not_in_the_issue_is_flagged_arbitrary() -> None:
    gold = GOLD_PATCH + '+    raise ValueError("flag must not be empty here")\n'
    test = (
        "diff --git a/tests/test_core.py b/tests/test_core.py\n"
        "--- a/tests/test_core.py\n+++ b/tests/test_core.py\n@@ -1,1 +1,2 @@\n x\n"
        '+    assert str(e) == "flag must not be empty here"\n'
    )
    hints = au.audit_hints(make_task(gold_patch=gold, test_patch=test))
    assert any(h.startswith("ARBITRARY?") and "flag must not be empty here" in h for h in hints)
    quoted = au.audit_hints(
        make_task(gold_patch=gold, test_patch=test, issue_body="error: flag must not be empty here")
    )
    assert not any(h.startswith("ARBITRARY?") for h in quoted)


def test_huge_fail_to_pass_list_gets_an_informational_note() -> None:
    hints = au.audit_hints(make_task(fail_to_pass=[f"t.py::t{i}" for i in range(51)]))
    assert any(h.startswith("NOTE: 51 fail_to_pass") for h in hints)
    assert not au.audit_hints(make_task(fail_to_pass=[f"t.py::t{i}" for i in range(50)]))


# --- status ------------------------------------------------------------------------------


def _verdicts(pairs: dict[str, au.Verdict]) -> dict[str, au.AuditVerdict]:
    return {k: au.make_verdict(k, v, now=NOW) for k, v in pairs.items()}


def test_experiment_tasks_are_only_the_latest_ok_ones_optionally_by_split() -> None:
    tasks = [
        make_task("o/a", 1, split="train"),
        make_task("o/a", 2, split="test"),
        make_task("o/b", 1, split="test"),
        make_task("o/b", 2, split="train"),
    ]
    v = _verdicts({
        "o__a-1": "ok", "o__a-2": "ok", "o__b-1": "leaks_fix",
    })  # fmt: skip
    assert [t.task_id for t in au.experiment_tasks(tasks, v)] == ["o__a-1", "o__a-2"]
    assert [t.task_id for t in au.experiment_tasks(tasks, v, split="test")] == ["o__a-2"]
    assert au.experiment_tasks(tasks, {}) == []  # unaudited tasks never qualify


def test_changing_a_verdict_from_ok_removes_the_task_from_experiments(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    [t] = make_tasks({"o/a": 1})
    au.append_verdict(path, au.make_verdict(t.task_id, "ok", now=NOW))
    assert au.experiment_tasks([t], au.load_verdicts(path))
    au.append_verdict(path, au.make_verdict(t.task_id, "leaks_fix", now=NOW))
    assert au.experiment_tasks([t], au.load_verdicts(path)) == []


def test_status_counts_ignore_verdicts_for_tasks_not_in_the_manifest() -> None:
    tasks = [
        make_task("o/a", 1, split="train"),
        make_task("o/b", 1, split="test"),
        make_task("o/b", 2),
    ]
    v = _verdicts({"o__a-1": "ok", "o__b-1": "underspecified", "ghost-1": "ok"})
    s = au.audit_status(tasks, v)
    assert (s.total, s.audited, s.pending, s.ok_total) == (3, 2, 1, 1)
    assert s.by_verdict == {"ok": 1, "underspecified": 1}
    assert s.ok_by_repo == {"o/a": 1} and s.ok_by_split == {"train": 1}


def test_unsplit_ok_tasks_are_counted_as_unassigned() -> None:
    s = au.audit_status([make_task("o/a", 1)], _verdicts({"o__a-1": "ok"}))
    assert s.ok_by_split == {"unassigned": 1}


@pytest.mark.parametrize(
    ("ok", "test_ok", "target_line", "test_line"),
    [
        (60, 25, "meets target", "in range"),
        (59, 30, "meets minimum", "in range"),
        (40, 24, "meets minimum", "below range"),
        (39, 31, "1 short of the minimum", "above range"),
        (0, 0, "40 short of the minimum", "below range"),
    ],
)
def test_format_status_reports_progress_against_the_targets(
    ok: int, test_ok: int, target_line: str, test_line: str
) -> None:
    s = au.AuditStatus(
        total=100, audited=ok, pending=100 - ok, by_verdict={"ok": ok},
        ok_by_repo={"o/a": ok} if ok else {}, ok_by_split={"test": test_ok}, ok_total=ok,
    )  # fmt: skip
    text = au.format_status(s)
    assert f"audited ok: {ok} (target 60, minimum 40): {target_line}" in text
    assert f"test-split ok: {test_ok} (target 25-30): {test_line}" in text
    assert ("ok by repo: o/a" in text) == bool(ok)
