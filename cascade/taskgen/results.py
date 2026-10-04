"""Parse the plugin's JSONL report into one outcome per test, and describe a finished run."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from cascade.errors import CascadeError

PASSED = "passed"
FAILED = "failed"
ERROR = "error"
SKIPPED = "skipped"
XFAILED = "xfailed"
XPASSED = "xpassed"
# Not an outcome the plugin emits: used when a test appears in one run but not another.
MISSING = "missing"

NOT_PASSING = frozenset({FAILED, ERROR, MISSING})


class ResultsError(CascadeError):
    """The pytest report was unreadable."""


@dataclass(frozen=True)
class RunResult:
    exit_code: int | None
    duration_s: float
    timed_out: bool
    outcomes: dict[str, str] = field(default_factory=lambda: {})
    stdout_tail: str = ""


def aggregate(phases: list[tuple[str, str]]) -> str:
    """Collapse the (when, outcome) phases of one test into a single outcome.

    - a failed call is "failed"
    - a failed setup, teardown or collection is "error"
    - otherwise the call outcome stands (passed, xpassed, xfailed, skipped)
    - a test that only has a non-failing setup phase (skipped before running) is "skipped"
    """
    failed_whens = {when for when, outcome in phases if outcome == FAILED}
    if "call" in failed_whens:
        return FAILED
    if failed_whens:
        return ERROR
    for when, outcome in phases:
        if when == "call":
            return outcome
    return SKIPPED


def parse_report(text: str) -> dict[str, str]:
    """Return {nodeid: outcome}. Raises ResultsError on a malformed line rather than skipping it."""
    phases: dict[str, list[tuple[str, str]]] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            nodeid, when, outcome = row["nodeid"], row["when"], row["outcome"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ResultsError(f"malformed report line {lineno}: {line[:120]!r}") from e
        if not (isinstance(nodeid, str) and isinstance(when, str) and isinstance(outcome, str)):
            raise ResultsError(f"malformed report line {lineno}: {line[:120]!r}")
        phases.setdefault(nodeid, []).append((when, outcome))
    return {nodeid: aggregate(p) for nodeid, p in phases.items()}
