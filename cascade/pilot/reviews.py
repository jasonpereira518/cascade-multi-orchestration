"""Human review of pilot drafts, and the pilot metrics computed from the review log ONLY.

``tasks/pilot_reviews.jsonl`` is append-only: every ``cascade review`` adds a row, and when a draft
is reviewed again (say, to record that it was later merged) the LATEST row for that (repo, issue)
wins. Arms tried and cost are copied from the draft's own ``draft.json`` (written by ``cascade
draft`` from the ledger) at review time, so the reviewer never types a number the ledger knows and
the metrics need nothing but this file.

Definitions (stated in the output so nobody has to guess):
- accepted: ``accepted_as_is`` or ``minor_edits``. Everything else is ``rejected``.
- accept rate: accepted / reviewed, with a 95% Wilson interval (the pilot is 5-10 drafts: wide).
- merge rate: of ACCEPTED drafts whose ``merged_upstream`` is recorded (true or false), the share
  merged. Drafts with ``merged_upstream`` unset are excluded, not counted as unmerged.
- median lines edited: over accepted drafts (``accepted_as_is`` counts 0 edited lines).
- cost per accepted draft: the cost of ALL reviewed drafts (rejected ones included) divided by the
  number accepted; undefined when none was accepted.
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from cascade.errors import CascadeError
from cascade.pilot.draft import draft_dir
from cascade.routing.calibrate import wilson

Outcome = Literal["accepted_as_is", "minor_edits", "rejected"]
OUTCOMES: tuple[str, ...] = ("accepted_as_is", "minor_edits", "rejected")
ACCEPTED: frozenset[str] = frozenset({"accepted_as_is", "minor_edits"})


class ReviewError(CascadeError):
    """A review cannot be recorded, or the review log is malformed."""


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str
    issue: int = Field(ge=1)
    outcome: Outcome
    lines_edited: int | None = Field(default=None, ge=0)
    merged_upstream: bool | None = None
    arms_tried: list[str]
    cost_usd: float = Field(ge=0)
    notes: str = ""
    reviewed_at: str

    @model_validator(mode="after")
    def _consistent(self) -> Review:
        if self.outcome == "accepted_as_is" and self.lines_edited not in (None, 0):
            raise ValueError("accepted_as_is means no lines were edited")
        if self.outcome == "minor_edits" and not self.lines_edited:
            raise ValueError("minor_edits needs lines_edited >= 1 (count the lines you changed)")
        if self.outcome == "rejected" and self.merged_upstream:
            raise ValueError("a rejected draft cannot be merged upstream")
        return self


def read_reviews(path: Path) -> list[Review]:
    """Every row in the log, in order. A malformed row is an error, never silently skipped."""
    if not path.exists():
        return []
    rows: list[Review] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(Review.model_validate_json(line))
        except ValidationError as e:
            raise ReviewError(f"{path}:{n}: not a valid review row: {e}") from e
    return rows


def latest(rows: list[Review]) -> list[Review]:
    """The latest row per (repo, issue), in first-reviewed order."""
    by_key: dict[tuple[str, int], Review] = {}
    for r in rows:
        by_key[(r.repo, r.issue)] = r
    return list(by_key.values())


def record_review(
    *,
    reviews_path: Path,
    drafts_root: Path,
    repo: str,
    issue: int,
    outcome: str,
    lines_edited: int | None,
    merged_upstream: bool | None,
    notes: str,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
) -> Review:
    meta_path = draft_dir(drafts_root, repo, issue) / "draft.json"
    if not meta_path.exists():
        raise ReviewError(f"no draft at {meta_path}: run `cascade draft` for {repo}#{issue} first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if outcome not in OUTCOMES:
        raise ReviewError(f"outcome must be one of {', '.join(OUTCOMES)}")
    try:
        review = Review.model_validate(
            {
                "repo": repo, "issue": issue, "outcome": outcome, "lines_edited": lines_edited,
                "merged_upstream": merged_upstream,
                "arms_tried": [str(a) for a in meta["arms_tried"]],
                "cost_usd": float(meta["cost_usd"]), "notes": notes,
                "reviewed_at": now().isoformat(),
            }
        )  # fmt: skip
    except ValidationError as e:
        raise ReviewError("; ".join(str(err["msg"]) for err in e.errors())) from e
    reviews_path.parent.mkdir(parents=True, exist_ok=True)
    with reviews_path.open("a", encoding="utf-8") as f:
        f.write(review.model_dump_json() + "\n")
    return review


@dataclass(frozen=True)
class PilotMetrics:
    n_reviewed: int
    n_repos: int
    n_accepted: int
    n_accepted_as_is: int
    n_minor_edits: int
    n_rejected: int
    accept_rate: float | None
    accept_rate_ci: tuple[float, float] | None
    n_merge_known: int
    n_merged: int
    merge_rate: float | None  # None when no accepted draft has merged_upstream recorded
    median_lines_edited: float | None  # over accepted drafts; None when none was accepted
    total_cost_usd: float
    cost_per_accepted_usd: float | None  # None when none was accepted


def pilot_metrics(rows: list[Review]) -> PilotMetrics:
    """Computed from the review rows given (the caller passes ``latest(read_reviews(...))``)."""
    n = len(rows)
    accepted = [r for r in rows if r.outcome in ACCEPTED]
    known = [r for r in accepted if r.merged_upstream is not None]
    merged = sum(1 for r in known if r.merged_upstream)
    edits = [float(r.lines_edited or 0) for r in accepted]
    cost = sum(r.cost_usd for r in rows)
    return PilotMetrics(
        n_reviewed=n,
        n_repos=len({r.repo for r in rows}),
        n_accepted=len(accepted),
        n_accepted_as_is=sum(1 for r in rows if r.outcome == "accepted_as_is"),
        n_minor_edits=sum(1 for r in rows if r.outcome == "minor_edits"),
        n_rejected=sum(1 for r in rows if r.outcome == "rejected"),
        accept_rate=len(accepted) / n if n else None,
        accept_rate_ci=wilson(len(accepted), n) if n else None,
        n_merge_known=len(known),
        n_merged=merged,
        merge_rate=merged / len(known) if known else None,
        median_lines_edited=statistics.median(edits) if edits else None,
        total_cost_usd=cost,
        cost_per_accepted_usd=cost / len(accepted) if accepted else None,
    )


def unreviewed_drafts(drafts_root: Path, rows: list[Review]) -> list[str]:
    """Drafts on disk with no review row yet (reported so a pending draft is never forgotten)."""
    reviewed = {(r.repo, r.issue) for r in rows}
    pending: list[str] = []
    for meta_path in sorted(drafts_root.glob("*/*/draft.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (str(meta["repo"]), int(meta["issue"])) not in reviewed:
            pending.append(f"{meta['repo']}#{meta['issue']}")
    return pending


def format_metrics(m: PilotMetrics, pending: list[str]) -> str:
    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{100 * x:.1f}%"

    ci = ""
    if m.accept_rate_ci is not None:
        ci = f" (95% Wilson {100 * m.accept_rate_ci[0]:.0f}-{100 * m.accept_rate_ci[1]:.0f}%)"
    median = "n/a" if m.median_lines_edited is None else f"{m.median_lines_edited:g}"
    per = "n/a" if m.cost_per_accepted_usd is None else f"${m.cost_per_accepted_usd:.4f}"
    lines = [
        f"pilot: {m.n_reviewed} reviewed draft(s) across {m.n_repos} repo(s)",
        f"  accepted {m.n_accepted} ({m.n_accepted_as_is} as-is, {m.n_minor_edits} minor edits), "
        f"rejected {m.n_rejected}",
        f"  accept rate: {pct(m.accept_rate)}{ci}",
        f"  merge rate: {pct(m.merge_rate)} ({m.n_merged}/{m.n_merge_known} accepted drafts with "
        "merged_upstream recorded)",
        f"  median lines edited (accepted drafts): {median}",
        f"  cost: ${m.total_cost_usd:.4f} total; per accepted draft: {per}",
    ]
    if pending:
        lines.append(f"  NOT YET REVIEWED (excluded above): {', '.join(pending)}")
    return "\n".join(lines)
