"""Attrition table, computed only from logged stage results (hard rule 1).

Funnel: candidates -> after filters (metadata + patch stages) -> env builds -> validated.
A row's "pending" count is PRs that reached the stage but have no result yet (stage not run, or run
interrupted), and "error" is transient failures that a rerun will retry. Both are reported so a
partial run cannot read as a complete one, and no projection is made until nothing is pending.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from cascade.taskgen.store import DROPPED, ENV, ERROR, FILTER, OK, PATCH, VALIDATE, StageStore

_STAGE_LABELS = {
    FILTER: "filter (metadata rules)",
    PATCH: "patch (git diff, size rules)",
    ENV: "env build + smoke run",
    VALIDATE: "validation (3x pre/post)",
}


@dataclass(frozen=True)
class StageRow:
    stage: str
    entered: int
    passed: int
    dropped: int
    errors: int
    pending: int
    reasons: dict[str, int]


@dataclass(frozen=True)
class AttritionTable:
    candidates: int
    rows: list[StageRow]

    @property
    def after_filters(self) -> int:
        return self._passed(PATCH)

    @property
    def env_builds(self) -> int:
        return self._passed(ENV)

    @property
    def validated(self) -> int:
        return self._passed(VALIDATE)

    @property
    def complete(self) -> bool:
        return all(r.pending == 0 and r.errors == 0 for r in self.rows)

    def _passed(self, stage: str) -> int:
        return next(r.passed for r in self.rows if r.stage == stage)


def compute(store: StageStore) -> AttritionTable:
    rows: list[StageRow] = []
    entered = store.candidate_count()
    for stage in (FILTER, PATCH, ENV, VALIDATE):
        passed = store.count(stage, OK)
        dropped = store.count(stage, DROPPED)
        errors = store.count(stage, ERROR)
        rows.append(
            StageRow(
                stage=stage,
                entered=entered,
                passed=passed,
                dropped=dropped,
                errors=errors,
                pending=entered - passed - dropped - errors,
                reasons=store.drop_reasons(stage),
            )
        )
        entered = passed
    return AttritionTable(candidates=store.candidate_count(), rows=rows)


def render_markdown(
    table: AttritionTable,
    *,
    repo: str,
    window: tuple[dt.date, dt.date],
    generated_at: dt.datetime,
    state_path: str,
) -> str:
    start, end = window
    lines = [
        f"### {repo}",
        "",
        f"Window {start} to {end}. Computed from `{state_path}` at "
        f"{generated_at.strftime('%Y-%m-%d %H:%M UTC')}; every number below is a count of logged "
        "stage results.",
        "",
        "| Stage | Entered | Passed | Dropped | Error (retry) | Pending |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in table.rows:
        lines.append(
            f"| {_STAGE_LABELS[r.stage]} | {r.entered} | {r.passed} | {r.dropped} | "
            f"{r.errors} | {r.pending} |"
        )
    lines += [
        "",
        f"**Funnel:** {table.candidates} candidates -> {table.after_filters} after filters -> "
        f"{table.env_builds} env builds -> {table.validated} validated.",
        "",
    ]
    for r in table.rows:
        if r.reasons:
            joined = ", ".join(f"{reason} ({n})" for reason, n in r.reasons.items())
            lines.append(f"- Drops at {r.stage}: {joined}")
    lines.append("")
    if table.complete:
        days = (end - start).days + 1
        per_month = table.validated / days * 30.44
        lines.append(
            f"**Projection:** {table.validated} validated over {days} days is about "
            f"{per_month:.1f} per month ({per_month * 12:.0f} per 12 months) for this repo. "
            "Linear extrapolation from one window; treat as a rough order of magnitude."
        )
    else:
        lines.append(
            "**Projection:** none. The run is incomplete (pending or error rows above), so any "
            "extrapolation would be a guess."
        )
    return "\n".join(lines) + "\n"


def update_status_md(path: Path, repo: str, block: str) -> None:
    """Replace this repo's block between markers in STATUS.md, appending the section if absent."""
    begin = f"<!-- attrition:{repo}:begin -->"
    end = f"<!-- attrition:{repo}:end -->"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    wrapped = f"{begin}\n{block}{end}\n"
    if begin in text and end in text:
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        new = head + wrapped + tail.lstrip("\n")
    else:
        heading = "## Task pipeline attrition"
        sep = "" if text.endswith("\n\n") or not text else "\n"
        section = f"{heading}\n\n" if heading not in text else ""
        new = text + sep + section + wrapped
    path.write_text(new, encoding="utf-8")
