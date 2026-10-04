from __future__ import annotations

from cascade.agents.failures import (
    failing_ids,
    outcome_counts,
    passed_ids,
    summarize_run,
    tail_lines,
)
from cascade.taskgen.results import RunResult


def run(outcomes: dict[str, str], *, details: dict[str, str] | None = None, exit_code: int | None = 1,
        timed_out: bool = False, tail: str = "") -> RunResult:  # fmt: skip
    return RunResult(exit_code, 2.5, timed_out, outcomes, tail, details or {})


def test_counts_are_ordered_and_omit_zeros() -> None:
    r = run({"a": "failed", "b": "passed", "c": "passed", "d": "skipped"})
    assert outcome_counts(r) == {"passed": 2, "failed": 1, "skipped": 1}
    assert list(outcome_counts(r)) == ["passed", "failed", "skipped"]


def test_failing_and_passed_ids() -> None:
    r = run({"z": "failed", "a": "error", "m": "passed", "s": "skipped"})
    assert failing_ids(r) == ["a", "z"]  # sorted, failed and error only
    assert passed_ids(r) == {"m"}


def test_summary_has_counts_failing_ids_and_detail_tail() -> None:
    detail = "\n".join(f"line {i}" for i in range(40))
    r = run({"t.py::a": "failed", "t.py::b": "passed"}, details={"t.py::a": detail})
    text = summarize_run(r)
    assert "1 passed, 1 failed" in text and "exit code 1" in text and "2.5s" in text
    assert "FAILED t.py::a" in text
    assert "line 39" in text and "line 28" in text and "line 27" not in text  # last 12 lines
    assert text.count("\n") < 30


def test_summary_limits_how_many_failures_are_listed() -> None:
    r = run({f"t.py::f{i}": "failed" for i in range(9)})
    text = summarize_run(r, max_failures=3)
    assert text.count("FAILED t.py::") == 3 and "and 6 more failing tests" in text


def test_collection_errors_are_labelled_error() -> None:
    r = run({"tests/test_x.py": "error"}, details={"tests/test_x.py": "ImportError: no module"})
    text = summarize_run(r)
    assert "ERROR tests/test_x.py" in text and "ImportError" in text


def test_timeout_has_its_own_message() -> None:
    r = RunResult(None, 120.0, True, {}, "")
    assert "TIMED OUT after 120s" in summarize_run(r)


def test_no_report_falls_back_to_the_raw_output_tail() -> None:
    r = run({}, exit_code=2, tail="a\nb\nusage: pytest [options]\n")
    text = summarize_run(r)
    assert "no tests reported" in text and "usage: pytest [options]" in text


def test_summary_is_hard_capped() -> None:
    r = run({"t.py::a": "failed"}, details={"t.py::a": "x" * 5000})
    text = summarize_run(r, max_chars=300)
    assert len(text) <= 300 and text.endswith("(summary truncated)")


def test_tail_lines_strips_and_limits() -> None:
    assert tail_lines("a\n\nb  \nc\n", 2) == "b\nc"
    assert tail_lines("", 3) == ""
