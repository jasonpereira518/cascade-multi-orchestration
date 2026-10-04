"""Human audit of tasks: an append-only verdict log, a one-task-at-a-time review loop, and status.

Only tasks with the latest verdict "ok" enter experiments (docs/SPEC.md). The log
(`tasks/audit.jsonl`) is the source of truth: one JSON line per verdict, newest wins, nothing is
ever rewritten, so an interrupted session loses nothing and a changed mind is just another line.

The four verdicts, shown to the auditor and meant literally:
  ok              the issue specifies the behaviour well enough that a competent engineer could
                  write a fix that passes the hidden tests;
  underspecified  the hidden tests need behaviour, names, or messages the issue does not give;
  leaks_fix       the issue text contains the fix itself (code, a patch), so the task is trivial;
  arbitrary_test  the hidden tests check incidental details unrelated to the issue's request.

``audit_hints`` are model-free pointers (verbatim overlap, undeclared names) that save the human
time. They are hints only; the verdict is always the human's.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from cascade.errors import CascadeError
from cascade.taskgen.features import gold_symbols, split_patch
from cascade.taskgen.schema import Task

Verdict = Literal["ok", "underspecified", "leaks_fix", "arbitrary_test"]
VERDICTS: tuple[Verdict, ...] = ("ok", "underspecified", "leaks_fix", "arbitrary_test")
_KEYS: dict[str, Verdict] = {
    "o": "ok", "u": "underspecified", "l": "leaks_fix", "a": "arbitrary_test",
}  # fmt: skip

TARGET_OK = 60
MINIMUM_OK = 40
TARGET_TEST_OK = (25, 30)

RUBRIC = """\
VERDICTS
  o  ok              the issue specifies the behaviour well enough for a competent engineer to write
                     a fix that passes the hidden tests
  u  underspecified  the hidden tests need behaviour, names or messages the issue does not give
  l  leaks_fix       the issue text contains the fix itself (code or a patch), so it is trivial
  a  arbitrary_test  the hidden tests check incidental details unrelated to what the issue asks
  s  skip (decide later)      q  quit (everything so far is saved)
"""

_MAX_SHOWN_LINES = 300
_MIN_LEAK_LINE = 25
_LITERAL = re.compile(r"""["']([^"'\n]{12,})["']""")


class AuditError(CascadeError):
    """The audit log is unreadable."""


class AuditVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    verdict: Verdict
    notes: str = ""
    ts: str


# --- the log -------------------------------------------------------------------------------


def read_log(path: Path) -> list[AuditVerdict]:
    if not path.exists():
        return []
    out: list[AuditVerdict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(AuditVerdict.model_validate_json(line))
        except ValueError as e:
            raise AuditError(f"{path}:{lineno}: unreadable audit line: {e}") from e
    return out


def load_verdicts(path: Path) -> dict[str, AuditVerdict]:
    """The latest verdict per task."""
    return {v.task_id: v for v in read_log(path)}


def append_verdict(path: Path, verdict: AuditVerdict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(verdict.model_dump_json() + "\n")


def make_verdict(
    task_id: str, verdict: Verdict, notes: str = "", *, now: dt.datetime | None = None
) -> AuditVerdict:
    stamp = (now or dt.datetime.now(dt.UTC)).isoformat()
    return AuditVerdict(task_id=task_id, verdict=verdict, notes=notes.strip(), ts=stamp)


# --- ordering and hints --------------------------------------------------------------------


def order_tasks(tasks: Sequence[Task], *, seed: str = "audit-v1") -> list[Task]:
    """A fixed, repo-mixing order: sorted by a hash of the id, so auditing is not repo by repo."""
    return sorted(tasks, key=lambda t: hashlib.sha256(f"{seed}:{t.task_id}".encode()).hexdigest())


def _added_lines(patch: str, *, source_only: bool) -> list[str]:
    from cascade.taskgen.filter import classify_path

    lines: list[str] = []
    for path, body in split_patch(patch):
        if source_only and (not path.endswith(".py") or classify_path(path) != "source"):
            continue
        lines += [ln[1:] for ln in body if ln.startswith("+") and not ln.startswith("+++")]
    return lines


def audit_hints(task: Task) -> list[str]:
    """Model-free pointers for the auditor. Each is a reason to look closer, never a verdict."""
    issue = f"{task.issue_title}\n{task.issue_body}"
    hints: list[str] = []

    gold_added = {ln.strip() for ln in _added_lines(task.gold_patch, source_only=True)}
    verbatim = sorted(ln for ln in gold_added if len(ln) >= _MIN_LEAK_LINE and ln in issue)
    if verbatim:
        hints.append(
            f"LEAK? {len(verbatim)} added line(s) of the fix appear verbatim in the issue, "
            f"e.g. {verbatim[0]!r}"
        )
    if "```diff" in task.issue_body or re.search(r"^(\+\+\+ |@@ )", task.issue_body, re.MULTILINE):
        hints.append("LEAK? the issue contains a diff or patch fragment")

    _touched, new_public = gold_symbols(task.gold_patch)
    test_text = "\n".join(_added_lines(task.test_patch, source_only=False))
    unnamed = sorted(
        s
        for s in new_public
        if re.search(rf"\b{re.escape(s)}\b", test_text)
        and not re.search(rf"\b{re.escape(s)}\b", issue)
    )
    if unnamed:
        hints.append(
            "UNDERSPECIFIED? tests use new name(s) the issue never states: "
            + ", ".join(unnamed[:5])
        )

    gold_literals = {m for ln in gold_added for m in _LITERAL.findall(ln)}
    asserted = sorted(s for s in gold_literals if s in test_text and s not in issue)
    if asserted:
        hints.append(f"ARBITRARY? tests assert text the issue never states, e.g. {asserted[0]!r}")

    if len(task.fail_to_pass) > 50:
        hints.append(
            f"NOTE: {len(task.fail_to_pass)} fail_to_pass tests; usually a test module that could "
            "not import before the fix, so the count says nothing about difficulty"
        )
    return hints


# --- rendering and the review loop ---------------------------------------------------------


def _clip(text: str, *, full: bool) -> str:
    lines = text.rstrip("\n").splitlines()
    if full or len(lines) <= _MAX_SHOWN_LINES:
        return "\n".join(lines)
    more = len(lines) - _MAX_SHOWN_LINES
    return "\n".join(lines[:_MAX_SHOWN_LINES]) + f"\n... ({more} more lines; rerun with --full)"


def render_task(
    task: Task,
    *,
    position: int,
    total: int,
    previous: AuditVerdict | None = None,
    full: bool = False,
) -> str:
    f = task.features
    parts = [
        "=" * 78,
        f"Task {position}/{total}: {task.task_id}   repo {task.repo}   "
        f"split {task.split or 'unassigned'}",
        f"features: {f.src_files} source file(s), {f.src_lines} changed lines, "
        f"issue {f.issue_words} words, "
        f"{f.f2p_count} fail_to_pass, {f.p2p_count} pass_to_pass, test run {f.test_runtime_s:.1f}s",
    ]
    if previous is not None:
        parts.append(f"previous verdict: {previous.verdict} ({previous.notes or 'no note'})")
    hints = audit_hints(task)
    if hints:
        parts += ["hints (model-free, not verdicts):", *[f"  * {h}" for h in hints]]
    parts += [
        "-" * 30 + " ISSUE (all the agent sees) " + "-" * 19,
        f"{task.issue_title}\n\n{_clip(task.issue_body, full=full)}",
        "-" * 30 + " GOLD PATCH (hidden from agents) " + "-" * 14,
        _clip(task.gold_patch, full=full),
        "-" * 30 + " TEST PATCH (hidden from agents) " + "-" * 14,
        _clip(task.test_patch, full=full),
        "-" * 30 + " TESTS THAT MUST PASS " + "-" * 25,
        "fail_to_pass: "
        + ", ".join(task.fail_to_pass[:8])
        + (f" ... (+{len(task.fail_to_pass) - 8})" if len(task.fail_to_pass) > 8 else ""),
        "",
    ]
    return "\n".join(parts)


@dataclass
class ReviewStats:
    recorded: int = 0
    skipped: int = 0
    remaining: int = 0
    quit: bool = False
    by_verdict: Counter[str] = field(default_factory=lambda: Counter())


def run_review(
    tasks: Sequence[Task],
    audit_path: Path,
    *,
    ask: Callable[[str], str],
    echo: Callable[[str], None],
    redo: bool = False,
    full: bool = False,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
) -> ReviewStats:
    """Show tasks one at a time and append each verdict as it is given."""
    verdicts = load_verdicts(audit_path)
    queue = [t for t in order_tasks(tasks) if redo or t.task_id not in verdicts]
    stats = ReviewStats(remaining=len(queue))
    echo(RUBRIC)
    echo(f"{len(queue)} task(s) to review.")
    for position, task in enumerate(queue, start=1):
        echo(
            render_task(
                task,
                position=position,
                total=len(queue),
                previous=verdicts.get(task.task_id),
                full=full,
            )
        )
        while True:
            answer = ask("verdict [o/u/l/a/s/q]").strip().lower()
            if answer in _KEYS or answer in ("s", "q"):
                break
            echo("Please answer o, u, l, a, s, or q.")
        if answer == "q":
            stats.quit = True
            break
        stats.remaining -= 1
        if answer == "s":
            stats.skipped += 1
            continue
        notes = ask("note (optional, press enter to skip)")
        append_verdict(audit_path, make_verdict(task.task_id, _KEYS[answer], notes, now=now()))
        stats.recorded += 1
        stats.by_verdict[_KEYS[answer]] += 1
    return stats


# --- status --------------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditStatus:
    total: int
    audited: int
    pending: int
    by_verdict: dict[str, int]
    ok_by_repo: dict[str, int]
    ok_by_split: dict[str, int]
    ok_total: int


def experiment_tasks(
    tasks: Sequence[Task], verdicts: Mapping[str, AuditVerdict], *, split: str | None = None
) -> list[Task]:
    """The only tasks experiments may use: latest verdict ok (and optionally one split)."""
    return [
        t for t in tasks
        if (v := verdicts.get(t.task_id)) is not None and v.verdict == "ok"
        and (split is None or t.split == split)
    ]  # fmt: skip


def audit_status(tasks: Sequence[Task], verdicts: Mapping[str, AuditVerdict]) -> AuditStatus:
    known = {t.task_id for t in tasks}
    latest = {k: v for k, v in verdicts.items() if k in known}
    oks = experiment_tasks(tasks, verdicts)
    return AuditStatus(
        total=len(tasks),
        audited=len(latest),
        pending=len(tasks) - len(latest),
        by_verdict=dict(Counter(v.verdict for v in latest.values())),
        ok_by_repo=dict(Counter(t.repo for t in oks)),
        ok_by_split=dict(Counter(t.split or "unassigned" for t in oks)),
        ok_total=len(oks),
    )


def format_status(s: AuditStatus) -> str:
    lo, hi = TARGET_TEST_OK
    test_ok = s.ok_by_split.get("test", 0)
    lines = [
        f"tasks: {s.total} | audited: {s.audited} | pending: {s.pending}",
        "verdicts: " + (", ".join(f"{k} {s.by_verdict.get(k, 0)}" for k in VERDICTS)),
        f"audited ok: {s.ok_total} (target {TARGET_OK}, minimum {MINIMUM_OK}): "
        + (
            "meets target"
            if s.ok_total >= TARGET_OK
            else "meets minimum"
            if s.ok_total >= MINIMUM_OK
            else f"{MINIMUM_OK - s.ok_total} short of the minimum"
        ),
        f"test-split ok: {test_ok} (target {lo}-{hi}): "
        + ("in range" if lo <= test_ok <= hi else "below range" if test_ok < lo else "above range"),
    ]
    if s.ok_by_repo:
        lines.append(
            "ok by repo: " + ", ".join(f"{r} {n}" for r, n in sorted(s.ok_by_repo.items()))
        )
    return "\n".join(lines)
