"""CLI for Phase 3: run-all, report, split, tasks finalize, audit, triage. Synthetic data only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from cascade import cli
from cascade.cli import app
from cascade.taskgen import audit as au
from cascade.taskgen import split as sp
from cascade.taskgen.manifest import read_manifest, write_tasks
from cascade.taskgen.orchestrate import RepoOutcome
from tests.phase3_helpers import make_tasks

runner = CliRunner()

REPOS_YAML = """\
date_window: {start: 2025-01-01, end: 2025-03-31}
repos:
  - {name: o/a, license: MIT, python_version: "3.12", install_cmd: x, test_cmd: y}
  - {name: o/b, license: MIT, python_version: "3.12", install_cmd: x, test_cmd: y}
  - {name: o/off, license: MIT, python_version: "3.12", install_cmd: x, test_cmd: y, enabled: false}
"""


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, str]:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "repos.yaml").write_text(REPOS_YAML)
    write_tasks(tmp_path / "m.jsonl", make_tasks({"o/a": 3, "o/b": 3, "o/c": 3}))
    return {
        "manifest": str(tmp_path / "m.jsonl"),
        "split": str(tmp_path / "split.json"),
        "audit": str(tmp_path / "audit.jsonl"),
        "configs": str(tmp_path / "configs"),
        "runs": str(tmp_path / "runs"),
        "status": str(tmp_path / "STATUS.md"),
    }


def invoke(args: list[str], input: str | None = None):  # type: ignore[no-untyped-def]
    return runner.invoke(app, args, input=input)


# --- split -------------------------------------------------------------------------------


def test_split_freeze_prints_assignments_and_never_overwrites(paths: dict[str, str]) -> None:
    r = invoke(["split", "freeze", "--manifest", paths["manifest"], "--split-file", paths["split"]])
    assert r.exit_code == 0, r.output
    assert "train" in r.output and "test" in r.output and "target 40%" in r.output
    before = Path(paths["split"]).read_text()

    again = invoke(
        ["split", "freeze", "--manifest", paths["manifest"], "--split-file", paths["split"]]
    )
    assert again.exit_code == 2 and "frozen" in again.output
    assert Path(paths["split"]).read_text() == before

    forced = invoke(
        ["split", "freeze", "--manifest", paths["manifest"], "--split-file", paths["split"],
         "--salt", "other", "--force"]
    )  # fmt: skip
    assert forced.exit_code == 0 and sp.read_split(Path(paths["split"])).salt == "other"


def test_split_freeze_needs_two_repos(tmp_path: Path) -> None:
    write_tasks(tmp_path / "m.jsonl", make_tasks({"o/a": 3}))
    r = invoke(["split", "freeze", "--manifest", str(tmp_path / "m.jsonl"),
                "--split-file", str(tmp_path / "s.json")])  # fmt: skip
    assert r.exit_code == 2 and "at least two repos" in r.output
    assert not (tmp_path / "s.json").exists()


def test_split_show_reports_counts_and_late_repos(paths: dict[str, str]) -> None:
    invoke(["split", "freeze", "--manifest", paths["manifest"], "--split-file", paths["split"]])
    r = invoke(["split", "show", "--manifest", paths["manifest"], "--split-file", paths["split"]])
    assert r.exit_code == 0 and "frozen " in r.output and "o/a (3 tasks)" in r.output
    assert "assigned after the freeze" not in r.output

    missing = invoke(["split", "show", "--split-file", paths["split"] + ".nope"])
    assert missing.exit_code == 2 and "no split" in missing.output


# --- tasks finalize ----------------------------------------------------------------------


def test_finalize_command_stamps_split_and_audit(paths: dict[str, str]) -> None:
    invoke(["split", "freeze", "--manifest", paths["manifest"], "--split-file", paths["split"]])
    au.append_verdict(Path(paths["audit"]), au.make_verdict("o__a-1", "ok", "fine"))
    r = invoke(["tasks", "finalize", "--manifest", paths["manifest"],
                "--split-file", paths["split"], "--audit-file", paths["audit"]])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "9 task(s) finalized" in r.output and "ok 1" in r.output and "pending 8" in r.output
    tasks = {t.task_id: t for t in read_manifest(Path(paths["manifest"]))}
    assert tasks["o__a-1"].audit.status == "ok" and tasks["o__a-1"].split in ("train", "test")


def test_finalize_without_a_split_is_a_clear_error(paths: dict[str, str]) -> None:
    r = invoke(["tasks", "finalize", "--manifest", paths["manifest"],
                "--split-file", paths["split"], "--audit-file", paths["audit"]])  # fmt: skip
    assert r.exit_code == 2 and "no split" in r.output


# --- audit -------------------------------------------------------------------------------


def _review_args(paths: dict[str, str], *extra: str) -> list[str]:
    return [
        "audit",
        "review",
        "--manifest",
        paths["manifest"],
        "--audit-file",
        paths["audit"],
        *extra,
    ]


def test_audit_review_records_verdicts_from_typed_input(paths: dict[str, str]) -> None:
    r = invoke(_review_args(paths, "--repo", "o/a"), input="o\nlooks fine\nu\n\nq\n")
    assert r.exit_code == 0, r.output
    assert "GOLD PATCH (hidden from agents)" in r.output and "TEST PATCH" in r.output
    assert "recorded 2, skipped 0, left 1 (quit early)" in r.output
    log = au.read_log(Path(paths["audit"]))
    assert [v.verdict for v in log] == ["ok", "underspecified"]
    assert log[0].notes == "looks fine" and log[1].notes == ""
    assert {v.task_id.split("-")[0] for v in log} == {"o__a"}  # --repo respected


def test_audit_review_resumes_where_it_left_off(paths: dict[str, str]) -> None:
    invoke(_review_args(paths, "--repo", "o/a"), input="o\n\nq\n")
    r = invoke(_review_args(paths, "--repo", "o/a"), input="o\n\no\n\n")
    assert "2 task(s) to review." in r.output and "recorded 2, skipped 0, left 0" in r.output
    assert len(au.load_verdicts(Path(paths["audit"]))) == 3


def test_audit_review_survives_input_ending_and_keeps_what_was_saved(paths: dict[str, str]) -> None:
    r = invoke(_review_args(paths, "--repo", "o/a"), input="o\nnote\n")  # then EOF
    assert r.exit_code == 0 and "input ended; every verdict so far is saved" in r.output
    assert len(au.read_log(Path(paths["audit"]))) == 1


def test_audit_review_filters_by_task_and_split_and_errors_when_nothing_matches(
    paths: dict[str, str],
) -> None:
    r = invoke(_review_args(paths, "--task", "o__b-2"), input="a\nodd\n")
    assert [v.task_id for v in au.read_log(Path(paths["audit"]))] == ["o__b-2"] and r.exit_code == 0

    none = invoke(_review_args(paths, "--repo", "o/zzz"))
    assert none.exit_code == 2 and "no tasks match" in none.output
    unsplit = invoke(_review_args(paths, "--split", "test"))  # manifest has no splits yet
    assert unsplit.exit_code == 2 and "no tasks match" in unsplit.output


def test_audit_review_redo_reshows_reviewed_tasks(paths: dict[str, str]) -> None:
    invoke(_review_args(paths, "--task", "o__a-1"), input="o\n\n")
    r = invoke(_review_args(paths, "--task", "o__a-1", "--redo"), input="l\nnow I see it\n")
    assert "previous verdict: ok" in r.output
    assert au.load_verdicts(Path(paths["audit"]))["o__a-1"].verdict == "leaks_fix"


def test_audit_review_reports_an_unreadable_manifest(tmp_path: Path) -> None:
    bad = tmp_path / "m.jsonl"
    bad.write_text("{not json}\n")
    r = invoke(["audit", "review", "--manifest", str(bad), "--audit-file", str(tmp_path / "a")])
    assert r.exit_code == 2 and "unreadable manifest" in r.output


def test_audit_status_reports_progress_and_a_corrupt_log(paths: dict[str, str]) -> None:
    au.append_verdict(Path(paths["audit"]), au.make_verdict("o__a-1", "ok"))
    r = invoke(["audit", "status", "--manifest", paths["manifest"], "--audit-file", paths["audit"]])
    assert r.exit_code == 0, r.output
    assert "tasks: 9 | audited: 1 | pending: 8" in r.output
    assert "40 short of the minimum" not in r.output and "39 short of the minimum" in r.output

    Path(paths["audit"]).write_text("garbage\n")
    bad = invoke(
        ["audit", "status", "--manifest", paths["manifest"], "--audit-file", paths["audit"]]
    )
    assert bad.exit_code == 2 and "unreadable audit line" in bad.output


# --- taskgen report / run-all ------------------------------------------------------------


def test_report_with_no_state_lists_every_repo_as_missing_and_still_shows_manifest_counts(
    paths: dict[str, str],
) -> None:
    r = invoke(["taskgen", "report", "--configs-dir", paths["configs"], "--runs-dir", paths["runs"],
                "--manifest", paths["manifest"]])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "PER-STAGE ATTRITION" in r.output and "TOTAL" in r.output
    for repo in ("o/a", "o/b", "o/off"):
        assert f"{repo}" in r.output and "(no state yet)" in r.output


def test_run_all_requires_a_token_before_doing_anything(
    paths: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    called: list[object] = []
    monkeypatch.setattr(cli, "run_all", lambda *a, **k: called.append(1) or [])
    r = invoke(["taskgen", "run-all", "--configs-dir", paths["configs"]])
    assert r.exit_code == 2 and "GITHUB_TOKEN is not set" in r.output and called == []


class _Spy:
    def __init__(self, outcomes: list[RepoOutcome] | None = None) -> None:
        self.kwargs: dict[str, Any] = {}
        self.repos: list[str] = []
        self.outcomes = outcomes

    def __call__(self, repos: list[Any], **kwargs: Any) -> list[RepoOutcome]:
        self.repos = [r.name for r in repos]
        self.kwargs = kwargs
        return (
            self.outcomes
            if self.outcomes is not None
            else [RepoOutcome(r.name, None, None) for r in repos]
        )


@pytest.fixture
def token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-not-real")


def _run_all_args(paths: dict[str, str], *extra: str) -> list[str]:
    return ["taskgen", "run-all", "--configs-dir", paths["configs"], "--runs-dir", paths["runs"],
            "--manifest", paths["manifest"], "--status-md", paths["status"], *extra]  # fmt: skip


def test_run_all_defaults_to_enabled_repos_only(
    paths: dict[str, str], token: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _Spy()
    monkeypatch.setattr(cli, "run_all", spy)
    r = invoke(_run_all_args(paths))
    assert r.exit_code == 0, r.output
    assert spy.repos == ["o/a", "o/b"]
    assert spy.kwargs["workers"] == 4 and spy.kwargs["jobs"] == 2
    assert spy.kwargs["stop_after"] == "validate"


def test_run_all_include_disabled_and_explicit_repo_selection(
    paths: dict[str, str], token: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _Spy()
    monkeypatch.setattr(cli, "run_all", spy)
    invoke(_run_all_args(paths, "--include-disabled"))
    assert spy.repos == ["o/a", "o/b", "o/off"]
    invoke(_run_all_args(paths, "--repo", "o/off", "--workers", "1", "--jobs", "3"))
    assert spy.repos == ["o/off"] and spy.kwargs["workers"] == 1 and spy.kwargs["jobs"] == 3


def test_run_all_exits_nonzero_and_reports_when_a_repo_stopped_early(
    paths: dict[str, str], token: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _Spy([RepoOutcome("o/a", None, "CascadeError: boom"), RepoOutcome("o/b", None, None)])
    monkeypatch.setattr(cli, "run_all", spy)
    r = invoke(_run_all_args(paths))
    assert r.exit_code == 1
    assert "STOPPED EARLY" in r.output and "o/a: CascadeError: boom" in r.output


def test_run_all_rejects_an_unknown_repo_and_an_unfilled_config(
    paths: dict[str, str], token: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "run_all", _Spy())
    unknown = invoke(_run_all_args(paths, "--repo", "o/ghost"))
    assert unknown.exit_code == 2 and "not listed in repos.yaml" in unknown.output
    (Path(paths["configs"]) / "repos.yaml").write_text(
        REPOS_YAML.replace("{start: 2025-01-01, end: 2025-03-31}", "{start: TODO, end: TODO}")
    )
    todo = invoke(_run_all_args(paths))
    assert todo.exit_code == 2 and "date_window" in todo.output


# --- triage ------------------------------------------------------------------------------


def test_triage_command_with_and_without_runs(tmp_path: Path) -> None:
    assert invoke(["triage", "--runs-dir", str(tmp_path / "none")]).output.strip() == (
        "no attempts found"
    )
    from cascade.routing.matrix import MatrixRow, MatrixStore

    store = MatrixStore(tmp_path / "matrix.sqlite")
    store.put(
        MatrixRow(
            task_id="o__a-1", arm_id="cheap", seed=0, true_success=False, verifier_claimed=False,
            cost_usd=0.0, input_tokens=0, output_tokens=0, reasoning_tokens=0, wall_s=0.0,
            turns=0, patch_sha=None, pipeline_config="{}", error="BudgetExceededError: cap",
        )
    )  # fmt: skip
    store.close()
    r = invoke(["triage", "--runs-dir", str(tmp_path)])
    assert r.exit_code == 0 and "stopped_by_cap_or_switch 1" in r.output
    assert json.loads(json.dumps(r.output))  # plain text, no tracebacks
