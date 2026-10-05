"""Pilot reviews and metrics. All SYNTHETIC: invented drafts, invented costs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cascade.cli import app
from cascade.errors import CascadeError
from cascade.pilot import reviews as rev

runner = CliRunner()


def make_draft(root: Path, repo: str, issue: int, *, cost: float = 0.5, arms: int = 1) -> None:
    d = root / repo.replace("/", "__") / str(issue)
    d.mkdir(parents=True)
    (d / "draft.json").write_text(
        json.dumps(
            {
                "repo": repo, "issue": issue, "cost_usd": cost,
                "arms_tried": ["m@low", "m@high", "n@max"][:arms],
            }
        )
    )  # fmt: skip


def review(root: Path, log: Path, repo: str, issue: int, outcome: str, **kw: object) -> rev.Review:
    args: dict[str, object] = {"lines_edited": None, "merged_upstream": None, "notes": ""}
    args.update(kw)
    return rev.record_review(
        reviews_path=log, drafts_root=root, repo=repo, issue=issue, outcome=outcome,
        **args,  # type: ignore[arg-type]
    )  # fmt: skip


@pytest.fixture
def drafts(tmp_path: Path) -> Path:
    return tmp_path / "drafts"


@pytest.fixture
def log(tmp_path: Path) -> Path:
    return tmp_path / "tasks" / "pilot_reviews.jsonl"


# --- recording ----------------------------------------------------------------------------------


def test_a_review_copies_arms_and_cost_from_the_draft_and_appends_a_row(
    drafts: Path, log: Path
) -> None:
    make_draft(drafts, "a/b", 3, cost=1.25, arms=2)
    row = review(drafts, log, "a/b", 3, "minor_edits", lines_edited=4, notes="renamed a var")
    assert (row.arms_tried, row.cost_usd, row.lines_edited) == (["m@low", "m@high"], 1.25, 4)
    [line] = log.read_text().splitlines()
    assert json.loads(line)["notes"] == "renamed a var" and json.loads(line)["issue"] == 3


def test_reviewing_a_draft_that_does_not_exist_is_an_error(drafts: Path, log: Path) -> None:
    with pytest.raises(rev.ReviewError, match="run `cascade draft`"):
        review(drafts, log, "a/b", 3, "rejected")
    assert not log.exists()


@pytest.mark.parametrize(
    ("outcome", "kw", "match"),
    [
        ("accepted_as_is", {"lines_edited": 2}, "no lines were edited"),
        ("minor_edits", {}, "lines_edited >= 1"),
        ("minor_edits", {"lines_edited": 0}, "lines_edited >= 1"),
        ("rejected", {"merged_upstream": True}, "cannot be merged"),
        ("mostly_fine", {}, "outcome must be one of"),
        ("minor_edits", {"lines_edited": -1}, "greater than or equal to 0"),
    ],
)
def test_inconsistent_reviews_are_refused_and_nothing_is_written(
    drafts: Path, log: Path, outcome: str, kw: dict[str, object], match: str
) -> None:
    make_draft(drafts, "a/b", 3)
    with pytest.raises(rev.ReviewError, match=match):
        review(drafts, log, "a/b", 3, outcome, **kw)
    assert not log.exists()


def test_the_latest_row_for_a_draft_wins_and_the_log_is_never_rewritten(
    drafts: Path, log: Path
) -> None:
    make_draft(drafts, "a/b", 3)
    review(drafts, log, "a/b", 3, "minor_edits", lines_edited=5)
    first = log.read_text()
    review(drafts, log, "a/b", 3, "minor_edits", lines_edited=5, merged_upstream=True)
    assert log.read_text().startswith(first) and len(log.read_text().splitlines()) == 2
    [only] = rev.latest(rev.read_reviews(log))
    assert only.merged_upstream is True


def test_a_malformed_log_row_is_an_error_with_its_line_number(log: Path) -> None:
    log.parent.mkdir(parents=True)
    log.write_text('{"repo": "a/b"}\n')
    with pytest.raises(rev.ReviewError, match=r"pilot_reviews.jsonl:1"):
        rev.read_reviews(log)
    assert rev.read_reviews(log.parent / "absent.jsonl") == []


# --- metrics, from the log only -----------------------------------------------------------------


def seeded(drafts: Path, log: Path) -> None:
    for i, cost in enumerate([1.0, 2.0, 3.0, 4.0, 5.0], start=1):
        make_draft(drafts, "o/a" if i <= 3 else "o/b", i, cost=cost)
    review(drafts, log, "o/a", 1, "accepted_as_is", merged_upstream=True)
    review(drafts, log, "o/a", 2, "minor_edits", lines_edited=3, merged_upstream=False)
    review(drafts, log, "o/a", 3, "minor_edits", lines_edited=9)  # merged not recorded
    review(drafts, log, "o/b", 4, "rejected", notes="wrong approach")
    review(drafts, log, "o/b", 5, "rejected")


def test_the_metrics_are_computed_from_the_review_rows(drafts: Path, log: Path) -> None:
    seeded(drafts, log)
    m = rev.pilot_metrics(rev.latest(rev.read_reviews(log)))
    assert (m.n_reviewed, m.n_repos, m.n_accepted) == (5, 2, 3)
    assert (m.n_accepted_as_is, m.n_minor_edits, m.n_rejected) == (1, 2, 2)
    assert m.accept_rate == pytest.approx(0.6)
    lo, hi = m.accept_rate_ci or (0, 0)
    assert 0 < lo < 0.6 < hi < 1
    assert (m.n_merge_known, m.n_merged) == (2, 1) and m.merge_rate == pytest.approx(0.5)
    assert m.median_lines_edited == 3.0  # accepted drafts edited 0, 3, 9 lines
    assert m.total_cost_usd == pytest.approx(15.0)  # rejected drafts' cost counts
    assert m.cost_per_accepted_usd == pytest.approx(5.0)


def test_the_metrics_need_only_the_log_not_the_drafts(drafts: Path, log: Path) -> None:
    seeded(drafts, log)
    for p in sorted(drafts.rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    m = rev.pilot_metrics(rev.latest(rev.read_reviews(log)))
    assert m.n_reviewed == 5 and m.cost_per_accepted_usd == pytest.approx(5.0)


def test_with_no_accepted_draft_the_per_accepted_figures_are_undefined(
    drafts: Path, log: Path
) -> None:
    make_draft(drafts, "o/a", 1)
    review(drafts, log, "o/a", 1, "rejected")
    m = rev.pilot_metrics(rev.latest(rev.read_reviews(log)))
    assert m.accept_rate == 0.0 and m.cost_per_accepted_usd is None
    assert m.median_lines_edited is None and m.merge_rate is None
    text = rev.format_metrics(m, [])
    assert "per accepted draft: n/a" in text and "merge rate: n/a" in text


def test_an_empty_log_gives_empty_metrics() -> None:
    m = rev.pilot_metrics([])
    assert m.n_reviewed == 0 and m.accept_rate is None and m.accept_rate_ci is None


def test_unreviewed_drafts_are_listed_and_excluded(drafts: Path, log: Path) -> None:
    seeded(drafts, log)
    make_draft(drafts, "o/c", 9)
    rows = rev.latest(rev.read_reviews(log))
    assert rev.unreviewed_drafts(drafts, rows) == ["o/c#9"]
    assert rev.pilot_metrics(rows).n_reviewed == 5
    assert "NOT YET REVIEWED (excluded above): o/c#9" in rev.format_metrics(
        rev.pilot_metrics(rows), ["o/c#9"]
    )


def test_review_errors_are_cascade_errors() -> None:
    assert issubclass(rev.ReviewError, CascadeError)


# --- the CLI ------------------------------------------------------------------------------------


def test_review_and_pilot_metrics_through_the_cli(drafts: Path, log: Path) -> None:
    make_draft(drafts, "o/a", 1, cost=2.0)
    make_draft(drafts, "o/a", 2, cost=2.0)
    common = ["--reviews-file", str(log), "--drafts-dir", str(drafts)]
    ok = runner.invoke(
        app,
        ["review", "--repo", "o/a", "--issue", "1", "--outcome", "accepted_as_is", "--merged",
         *common],
    )  # fmt: skip
    assert ok.exit_code == 0, ok.output
    bad = runner.invoke(
        app, ["review", "--repo", "o/a", "--issue", "2", "--outcome", "minor_edits", *common]
    )
    assert bad.exit_code == 2 and "lines_edited" in bad.output
    out = runner.invoke(app, ["pilot-metrics", *common])
    assert out.exit_code == 0, out.output
    assert "1 reviewed draft(s)" in out.output and "NOT YET REVIEWED" in out.output
    assert "o/a#2" in out.output and "merge rate: 100.0% (1/1" in out.output


def test_pilot_metrics_with_no_reviews_says_what_to_do(tmp_path: Path) -> None:
    r = runner.invoke(app, ["pilot-metrics", "--reviews-file", str(tmp_path / "none.jsonl")])
    assert r.exit_code == 2 and "cascade review" in r.output
