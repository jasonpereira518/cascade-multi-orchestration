"""The environment checks and the live progress line. Fake client, fake docker, no network."""

from __future__ import annotations

import io
import subprocess
from collections.abc import Sequence
from typing import Any

import pytest
from typer.testing import CliRunner

from cascade import cli
from cascade.cli import app
from cascade.taskgen.github import GitHubClient
from cascade.taskgen.runtime import CheckResult, ProgressPrinter, check_docker, check_github

runner = CliRunner()


def client_returning(remaining: int, limit: int = 5000) -> GitHubClient:
    def transport(_q: str, _v: dict[str, Any]) -> dict[str, Any]:
        rate = {"limit": limit, "remaining": remaining, "resetAt": "2026-10-05T12:00:00Z"}
        return {"data": {"rateLimit": rate}}

    return GitHubClient(transport=transport)


# --- GitHub -------------------------------------------------------------------------------------


def test_a_missing_token_is_reported_with_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    r = check_github(lambda: client_returning(5000))
    assert not r.ok and "GITHUB_TOKEN is not set" in r.detail and "export GITHUB_TOKEN" in r.detail


def test_a_working_token_reports_the_quota_left(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "synthetic-token")
    r = check_github(lambda: client_returning(4321))
    assert r.ok and "4321/5000" in r.detail and "2026-10-05T12:00:00Z" in r.detail
    assert "synthetic-token" not in r.detail  # the token is never echoed


def test_a_rejected_token_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    monkeypatch.setenv("GITHUB_TOKEN", "synthetic-token")

    def transport(_q: str, _v: dict[str, Any]) -> dict[str, Any]:
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)  # type: ignore[arg-type]

    r = check_github(lambda: GitHubClient(transport=transport))
    assert not r.ok and "rejected" in r.detail and "401" in r.detail


def test_an_almost_exhausted_quota_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "synthetic-token")
    r = check_github(lambda: client_returning(3))
    assert not r.ok and "rate limit nearly exhausted" in r.detail


# --- Docker -------------------------------------------------------------------------------------


def docker_run(rc: int, out: str = "", err: str = "") -> Any:
    def run(argv: Sequence[str], timeout: float | None) -> subprocess.CompletedProcess[str]:
        assert list(argv)[:2] == ["docker", "info"]
        return subprocess.CompletedProcess(list(argv), rc, stdout=out, stderr=err)

    return run


def test_a_running_daemon_reports_its_version() -> None:
    r = check_docker(docker_run(0, "28.5.1\n"))
    assert r.ok and "28.5.1" in r.detail


def test_a_stopped_daemon_says_to_start_docker_and_why() -> None:
    err = "Cannot connect to the Docker daemon at unix:///var/run/docker.sock"
    r = check_docker(docker_run(1, "", "warning\n" + err))
    assert not r.ok and "start Docker" in r.detail and "Cannot connect" in r.detail


def test_a_missing_docker_command_is_reported() -> None:
    def run(argv: Sequence[str], timeout: float | None) -> Any:
        raise FileNotFoundError("docker")

    r = check_docker(run)
    assert not r.ok and "not installed" in r.detail


def test_a_hung_docker_cli_is_reported_not_raised() -> None:
    def run(argv: Sequence[str], timeout: float | None) -> Any:
        raise subprocess.TimeoutExpired(list(argv), 30)

    r = check_docker(run)
    assert not r.ok and "could not run" in r.detail


# --- the CLI command ----------------------------------------------------------------------------


def test_the_check_command_lists_every_problem_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "check_github", lambda: CheckResult("github", False, "no token"))
    monkeypatch.setattr(cli, "check_docker", lambda: CheckResult("docker", False, "not running"))
    r = runner.invoke(app, ["taskgen", "check"])
    assert r.exit_code == 1
    assert "FAIL github: no token" in r.output and "FAIL docker: not running" in r.output


def test_the_check_command_passes_when_both_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "check_github", lambda: CheckResult("github", True, "fine"))
    monkeypatch.setattr(cli, "check_docker", lambda: CheckResult("docker", True, "fine"))
    r = runner.invoke(app, ["taskgen", "check"])
    assert r.exit_code == 0 and "OK   github: fine" in r.output and "OK   docker: fine" in r.output


# --- the progress line --------------------------------------------------------------------------


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_on_a_terminal_the_line_is_rewritten_in_place_and_ends_with_a_newline() -> None:
    out = Tty()
    p = ProgressPrinter(out)
    p("o/r", "validate", 3, 10, 2)
    p("o/r", "validate", 10, 10, 7)
    text = out.getvalue()
    assert text.count("\r") == 2
    assert "[o/r] validate 3/10 PRs | validated so far: 2" in text
    assert text.endswith("validate 10/10 PRs | validated so far: 7\n")
    assert text.count("\n") == 1  # only the finished stage ends its line


def test_off_a_terminal_updates_are_throttled_but_a_finished_stage_always_prints() -> None:
    now = [0.0]
    out = io.StringIO()
    p = ProgressPrinter(out, every_s=5.0, clock=lambda: now[0])
    p("o/r", "patch", 1, 4, 0)  # first update prints
    now[0] = 1.0
    p("o/r", "patch", 2, 4, 0)  # within 5 s: suppressed
    now[0] = 6.5
    p("o/r", "patch", 3, 4, 0)  # past 5 s: printed
    p("o/r", "patch", 4, 4, 0)  # the stage is done: printed regardless
    lines = out.getvalue().splitlines()
    assert [ln.split()[2] for ln in lines] == ["1/4", "3/4", "4/4"]
    assert "\r" not in out.getvalue()


def test_the_mine_stage_counts_month_slices() -> None:
    out = io.StringIO()
    ProgressPrinter(out)("o/r", "mine", 1, 4, 0)
    assert "1/4 month slices" in out.getvalue()


def test_a_note_does_not_leave_a_half_drawn_line_behind() -> None:
    out = Tty()
    p = ProgressPrinter(out)
    p("o/r", "env", 1, 5, 0)
    p.note("warning: something")
    assert out.getvalue().endswith("env      1/5 PRs | validated so far: 0\nwarning: something\n")
