"""Test-failure parser: turn a finished pytest run into a short, fixed-shape summary for prompts.

Deterministic and bounded: the model sees counts, the first few failing test ids, and the tail of
each failure's text, never raw multi-thousand-line pytest output.
"""

from __future__ import annotations

from collections import Counter

from cascade.taskgen.results import ERROR, FAILED, PASSED, SKIPPED, XFAILED, XPASSED, RunResult

_DETAIL_LINES = 12
_ORDER = (PASSED, FAILED, ERROR, SKIPPED, XFAILED, XPASSED)


def outcome_counts(run: RunResult) -> dict[str, int]:
    counts = Counter(run.outcomes.values())
    return {k: counts[k] for k in _ORDER if counts[k]}


def failing_ids(run: RunResult) -> list[str]:
    return sorted(n for n, o in run.outcomes.items() if o in (FAILED, ERROR))


def passed_ids(run: RunResult) -> set[str]:
    return {n for n, o in run.outcomes.items() if o == PASSED}


def tail_lines(text: str, count: int = _DETAIL_LINES) -> str:
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    return "\n".join(lines[-count:])


def summarize_run(run: RunResult, *, max_failures: int = 5, max_chars: int = 2500) -> str:
    if run.timed_out:
        return f"The test run TIMED OUT after {run.duration_s:.0f}s and produced no results."
    counts = outcome_counts(run)
    head = ", ".join(f"{n} {k}" for k, n in counts.items()) or "no tests reported"
    parts = [f"Test run: {head} (exit code {run.exit_code}, {run.duration_s:.1f}s)."]
    failing = failing_ids(run)
    if not run.outcomes and run.stdout_tail:
        parts.append("Raw output tail:\n" + tail_lines(run.stdout_tail))
    for nodeid in failing[:max_failures]:
        detail = run.details.get(nodeid, "")
        label = run.outcomes[nodeid].upper()
        parts.append(f"{label} {nodeid}" + (f"\n{tail_lines(detail)}" if detail else ""))
    if len(failing) > max_failures:
        parts.append(f"... and {len(failing) - max_failures} more failing tests.")
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[: max_chars - 40].rstrip() + "\n... (summary truncated)"
    return text
