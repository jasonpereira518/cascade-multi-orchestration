from __future__ import annotations

import datetime as dt
from pathlib import Path

from cascade.taskgen import attrition
from cascade.taskgen.store import DROPPED, ENV, ERROR, FILTER, OK, PATCH, VALIDATE, StageStore

WINDOW = (dt.date(2025, 1, 1), dt.date(2025, 3, 31))  # 90 days
NOW = dt.datetime(2026, 10, 4, 12, 0, tzinfo=dt.UTC)


def seed(complete: bool) -> StageStore:
    s = StageStore(":memory:")
    for n in range(1, 11):
        s.upsert_candidate(n, {"n": n})
    for n in (1, 2, 3, 4, 5, 6):
        s.put_stage(n, FILTER, OK)
    for n in (7, 8):
        s.put_stage(n, FILTER, DROPPED, "bot_author")
    s.put_stage(9, FILTER, DROPPED, "outside_date_window")
    if complete:
        s.put_stage(10, FILTER, DROPPED, "no_closing_issue")
    for n in (1, 2, 3, 4):
        s.put_stage(n, PATCH, OK)
    s.put_stage(5, PATCH, DROPPED, "diff_mismatch")
    s.put_stage(6, PATCH, OK if complete else ERROR, None if complete else "git_error")
    for n in (1, 2, 3):
        s.put_stage(n, ENV, OK)
    s.put_stage(4, ENV, DROPPED, "env_collect_failed")
    if complete:
        s.put_stage(6, ENV, DROPPED, "env_no_tests")
    for n in (1, 2):
        s.put_stage(n, VALIDATE, OK)
    s.put_stage(3, VALIDATE, DROPPED, "no_fail_to_pass")
    return s


def test_funnel_counts_come_straight_from_the_logged_stage_results() -> None:
    t = attrition.compute(seed(complete=False))
    assert t.candidates == 10
    assert (t.after_filters, t.env_builds, t.validated) == (4, 3, 2)
    by = {r.stage: r for r in t.rows}
    assert (by[FILTER].passed, by[FILTER].dropped, by[FILTER].pending) == (6, 3, 1)
    assert (by[PATCH].passed, by[PATCH].dropped, by[PATCH].errors) == (4, 1, 1)
    assert by[PATCH].pending == 0  # 6 entered = 4 ok + 1 dropped + 1 error; errors are not pending
    assert by[ENV].entered == 4 and by[ENV].pending == 0
    assert by[VALIDATE].reasons == {"no_fail_to_pass": 1}
    assert not t.complete  # a pending PR and an error row


def test_incomplete_run_is_labelled_and_gets_no_projection() -> None:
    md = attrition.render_markdown(
        attrition.compute(seed(complete=False)), repo="o/r", window=WINDOW, generated_at=NOW,
        state_path="runs/x/state.sqlite",
    )  # fmt: skip
    assert "10 candidates -> 4 after filters -> 3 env builds -> 2 validated" in md
    assert "**Projection:** none" in md
    assert "| filter (metadata rules) | 10 | 6 | 3 | 0 | 1 |" in md
    assert "Drops at filter: bot_author (2), outside_date_window (1)" in md
    assert "runs/x/state.sqlite" in md and "2026-10-04 12:00 UTC" in md


def test_complete_run_projects_linearly_and_says_so() -> None:
    s = seed(complete=True)
    t = attrition.compute(s)
    assert t.complete and (t.after_filters, t.env_builds, t.validated) == (5, 3, 2)
    md = attrition.render_markdown(t, repo="o/r", window=WINDOW, generated_at=NOW, state_path="p")
    # 2 validated / 90 days * 30.44 days = 0.68 per month; x12 = 8
    assert "about 0.7 per month (8 per 12 months)" in md
    assert "Linear extrapolation" in md


def test_status_md_block_is_created_replaced_and_isolated_per_repo(tmp_path: Path) -> None:
    path = tmp_path / "STATUS.md"
    path.write_text("# Status\n\nSome earlier content.\n")
    attrition.update_status_md(path, "o/a", "block A v1\n")
    attrition.update_status_md(path, "o/b", "block B v1\n")
    attrition.update_status_md(path, "o/a", "block A v2\n")
    text = path.read_text()
    assert text.startswith("# Status\n\nSome earlier content.\n")
    assert text.count("## Task pipeline attrition") == 1
    assert "block A v2" in text and "block A v1" not in text
    assert "block B v1" in text
    assert (
        text.count("<!-- attrition:o/a:begin -->") == 1
        and text.count("<!-- attrition:o/b:end -->") == 1
    )


def test_status_md_content_after_the_block_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "STATUS.md"
    attrition.update_status_md(path, "o/a", "first\n")
    path.write_text(path.read_text() + "\n## Later section\n\nkeep me\n")
    attrition.update_status_md(path, "o/a", "second\n")
    text = path.read_text()
    assert "second" in text and "first" not in text and "## Later section\n\nkeep me" in text


def test_status_md_is_created_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "new" / "STATUS.md"
    path.parent.mkdir()
    attrition.update_status_md(path, "o/a", "hello\n")
    assert path.read_text().startswith("## Task pipeline attrition")
