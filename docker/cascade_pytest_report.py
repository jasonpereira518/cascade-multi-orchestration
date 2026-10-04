"""pytest plugin baked into the task image: one JSON line per test phase result.

Why a plugin and not `-rA` output: short-summary lines drop skipped node ids, and parametrized ids
can contain spaces and " - ", so splitting them back out of text is ambiguous. This writes exact
node ids and outcomes to the file named by $CASCADE_REPORT_PATH.

Line format: {"nodeid": str, "when": "setup|call|teardown|collect", "outcome": str}
Outcomes: passed, failed, skipped, xfailed, xpassed. Setup/teardown phases are written only when
they did not pass, so a passing test produces exactly one line (its call phase).
"""

from __future__ import annotations

import json
import os
from typing import Any


def _path() -> str | None:
    return os.environ.get("CASCADE_REPORT_PATH")


def _emit(nodeid: str, when: str, outcome: str) -> None:
    path = _path()
    if path is None:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"nodeid": nodeid, "when": when, "outcome": outcome}) + "\n")


def pytest_configure(config: Any) -> None:
    path = _path()
    if path is not None:
        open(path, "w", encoding="utf-8").close()  # truncate: one run, one report


def pytest_runtest_logreport(report: Any) -> None:
    wasxfail = hasattr(report, "wasxfail")
    if report.failed:
        outcome = "failed"
    elif report.skipped:
        outcome = "xfailed" if wasxfail else "skipped"
    else:
        outcome = "xpassed" if wasxfail else "passed"
    if report.when == "call" or outcome != "passed":
        _emit(report.nodeid, report.when, outcome)


def pytest_collectreport(report: Any) -> None:
    if report.failed:
        _emit(report.nodeid, "collect", "failed")
