from __future__ import annotations

import json

import pytest

from cascade.taskgen.results import (
    ERROR,
    FAILED,
    PASSED,
    SKIPPED,
    XFAILED,
    XPASSED,
    ResultsError,
    aggregate,
    parse_report,
)


def line(nodeid: str, when: str, outcome: str) -> str:
    return json.dumps({"nodeid": nodeid, "when": when, "outcome": outcome})


@pytest.mark.parametrize(
    ("phases", "expected"),
    [
        ([("call", "passed")], PASSED),
        ([("call", "failed")], FAILED),
        ([("setup", "failed")], ERROR),
        ([("call", "passed"), ("teardown", "failed")], ERROR),
        ([("call", "failed"), ("teardown", "failed")], FAILED),
        ([("setup", "skipped")], SKIPPED),
        ([("call", "skipped")], SKIPPED),
        ([("call", "xfailed")], XFAILED),
        ([("call", "xpassed")], XPASSED),
        ([("collect", "failed")], ERROR),
    ],
)
def test_aggregate_phases(phases: list[tuple[str, str]], expected: str) -> None:
    assert aggregate(phases) == expected


def test_parse_report_keeps_exact_ids_with_spaces_and_dashes() -> None:
    nodeid = "tests/test_a.py::test_x[2-a b - expect 'it']"
    text = "\n".join([line(nodeid, "call", "passed"), line("tests/t.py::test_y", "call", "failed")])
    assert parse_report(text) == {nodeid: PASSED, "tests/t.py::test_y": FAILED}


def test_parse_report_merges_phases_per_test() -> None:
    text = "\n".join(
        [
            line("t.py::a", "call", "passed"),
            line("t.py::a", "teardown", "failed"),
            line("t.py::b", "setup", "skipped"),
        ]
    )
    assert parse_report(text) == {"t.py::a": ERROR, "t.py::b": SKIPPED}


def test_blank_lines_are_ignored_and_empty_report_is_empty() -> None:
    assert parse_report("") == {}
    assert parse_report("\n\n") == {}


@pytest.mark.parametrize(
    "bad",
    ["not json", '{"nodeid": "x"}', '{"nodeid": 1, "when": "call", "outcome": "passed"}', "[]"],
)
def test_malformed_lines_raise_instead_of_being_skipped(bad: str) -> None:
    with pytest.raises(ResultsError):
        parse_report(bad)
