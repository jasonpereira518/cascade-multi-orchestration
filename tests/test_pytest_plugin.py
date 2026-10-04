"""Runs the real plugin under the real pytest in a subprocess, on a throwaway project."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from cascade.taskgen.results import ERROR, FAILED, PASSED, SKIPPED, XFAILED, XPASSED, parse_report
from tests.conftest import ROOT

SAMPLE = """
import pytest


def test_ok():
    assert True


def test_fails():
    assert False


@pytest.mark.parametrize("value", ["a b", "x - y"])
def test_param_ids_with_awkward_characters(value):
    assert value


def test_skipped():
    pytest.skip("not today")


@pytest.mark.xfail(reason="known bug")
def test_expected_failure():
    assert False


@pytest.mark.xfail(reason="fixed already")
def test_unexpected_pass():
    assert True


@pytest.fixture
def broken():
    raise RuntimeError("fixture exploded")


def test_setup_error(broken):
    pass
"""


def run_pytest(project: Path, *extra: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    report = project / "report.jsonl"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "docker"),
        "CASCADE_REPORT_PATH": str(report),
    }
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-p", "cascade_pytest_report",
            "--continue-on-collection-errors", "-p", "no:cacheprovider", "-q", "--tb=no", *extra,
        ],
        cwd=project, env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    return proc, report


def test_plugin_reports_every_outcome_kind(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text(textwrap.dedent(SAMPLE))
    proc, report = run_pytest(tmp_path)
    assert proc.returncode == 1, proc.stdout + proc.stderr  # some tests fail, run not interrupted
    out = parse_report(report.read_text())
    p = "test_sample.py::"
    assert out[p + "test_ok"] == PASSED
    assert out[p + "test_fails"] == FAILED
    assert out[p + "test_param_ids_with_awkward_characters[a b]"] == PASSED
    assert out[p + "test_param_ids_with_awkward_characters[x - y]"] == PASSED
    assert out[p + "test_skipped"] == SKIPPED
    assert out[p + "test_expected_failure"] == XFAILED
    assert out[p + "test_unexpected_pass"] == XPASSED
    assert out[p + "test_setup_error"] == ERROR


def test_plugin_reports_collection_errors_and_still_runs_other_modules(tmp_path: Path) -> None:
    (tmp_path / "test_good.py").write_text("def test_fine():\n    assert True\n")
    (tmp_path / "test_broken.py").write_text("import module_that_does_not_exist\n")
    proc, report = run_pytest(tmp_path)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    out = parse_report(report.read_text())
    assert out["test_good.py::test_fine"] == PASSED
    assert out["test_broken.py"] == ERROR  # collection failure keyed by the module path


def test_plugin_truncates_a_stale_report(tmp_path: Path) -> None:
    (tmp_path / "test_one.py").write_text("def test_one():\n    pass\n")
    (tmp_path / "report.jsonl").write_text(
        '{"nodeid": "stale::t", "when": "call", "outcome": "passed"}\n'
    )
    _, report = run_pytest(tmp_path)
    assert list(parse_report(report.read_text())) == ["test_one.py::test_one"]


def test_plugin_is_inert_without_report_path(tmp_path: Path) -> None:
    (tmp_path / "test_one.py").write_text("def test_one():\n    pass\n")
    env = {k: v for k, v in os.environ.items() if k != "CASCADE_REPORT_PATH"}
    env["PYTHONPATH"] = str(ROOT / "docker")
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-p", "cascade_pytest_report",
            "-p", "no:cacheprovider", "-q",
        ],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
