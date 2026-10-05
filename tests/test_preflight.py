"""Pre-flight leakage check over audited tasks. Synthetic fixture task, real git, no Docker.

A clean task must pass, and each planted leak must be reported against the task that has it. The
planted leaks are the same kinds tests/test_leakage.py plants into one task, now exercised through
the manifest-wide entry point the CLI uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cascade.agents import prompts
from cascade.routing import preflight as pf
from cascade.taskgen.schema import Task
from tests.conftest import FixtureRepo
from tests.taskgen_helpers import commit_files

HIDDEN_TEST = "test_above_range"


def run_one(fx: FixtureRepo, task: Task) -> pf.TaskPreflight:
    return pf.run_preflight([task], lambda _t: fx.path)[0]


def test_the_clean_fixture_task_passes_with_real_checks_behind_it(
    fixture_repo: FixtureRepo,
) -> None:
    r = run_one(fixture_repo, fixture_repo.task)
    assert r.ok and r.violations == []
    assert r.checks >= 7  # view, two workspace checks, and every role prompt


def test_a_task_with_no_checks_is_never_reported_ok() -> None:
    assert not pf.TaskPreflight("t", [], checks=0).ok
    assert pf.TaskPreflight("t", ["x"], checks=5).ok is False


def test_the_full_gold_patch_in_the_issue_is_reported_against_the_prompts(
    fixture_repo: FixtureRepo,
) -> None:
    task = fixture_repo.task
    leaky = task.model_copy(update={"issue_body": task.issue_body + "\n\n" + task.gold_patch})
    r = run_one(fixture_repo, leaky)
    assert not r.ok
    assert any(v.startswith("prompt input:triager") and "gold patch" in v for v in r.violations)
    assert any(v.startswith("prompt input:patcher") for v in r.violations)


def test_a_hidden_test_name_in_the_issue_is_reported(fixture_repo: FixtureRepo) -> None:
    task = fixture_repo.task
    leaky = task.model_copy(update={"issue_body": f"please make {HIDDEN_TEST} pass"})
    r = run_one(fixture_repo, leaky)
    assert any(HIDDEN_TEST in v and v.startswith("prompt ") for v in r.violations)


def test_a_post_base_commit_hash_in_the_issue_is_reported(fixture_repo: FixtureRepo) -> None:
    task = fixture_repo.task
    leaky = task.model_copy(update={"issue_body": f"fixed in {fixture_repo.head_sha}"})
    r = run_one(fixture_repo, leaky)
    assert any("post-base commit" in v for v in r.violations)


def test_a_gold_patch_sitting_in_the_base_tree_is_reported_against_the_workspace(
    fixture_repo: FixtureRepo,
) -> None:
    task = fixture_repo.task
    sha = commit_files(fixture_repo.path, {"notes.patch": task.gold_patch}, "oops: patch in tree")
    r = run_one(fixture_repo, task.model_copy(update={"base_commit": sha}))
    assert not r.ok
    assert any(v.startswith("workspace:") and "gold patch" in v for v in r.violations)


def test_a_hidden_test_name_present_in_the_base_tree_is_reported(fixture_repo: FixtureRepo) -> None:
    task = fixture_repo.task
    files: dict[str, str | None] = {"docs/todo.txt": f"remember to write {HIDDEN_TEST}\n"}
    sha = commit_files(fixture_repo.path, files, "notes")
    r = run_one(fixture_repo, task.model_copy(update={"base_commit": sha}))
    assert any(v.startswith("workspace:") and HIDDEN_TEST in v for v in r.violations)


def test_a_missing_clone_is_a_failure_not_a_skip(fixture_repo: FixtureRepo, tmp_path: Path) -> None:
    [r] = pf.run_preflight([fixture_repo.task], lambda _t: tmp_path / "nowhere")
    assert not r.ok and "no clone available" in r.violations[0]
    [none] = pf.run_preflight([fixture_repo.task], lambda _t: None)
    assert not none.ok


def test_an_unexportable_base_commit_is_reported(fixture_repo: FixtureRepo) -> None:
    broken = fixture_repo.task.model_copy(update={"base_commit": "f" * 40})
    r = run_one(fixture_repo, broken)
    assert not r.ok and any("cannot export the base commit" in v for v in r.violations)


def test_the_view_guard_notices_an_agent_view_that_grew_a_field(
    fixture_repo: FixtureRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pf, "ALLOWED_VIEW_FIELDS", frozenset({"task_id"}))
    r = run_one(fixture_repo, fixture_repo.task)
    assert any(v.startswith("view: AgentView fields") for v in r.violations)


def test_each_task_is_reported_separately_and_in_order(fixture_repo: FixtureRepo) -> None:
    good = fixture_repo.task
    bad = good.model_copy(update={"task_id": "bad-1", "issue_body": good.gold_patch})
    results = pf.run_preflight([bad, good], lambda _t: fixture_repo.path)
    assert [r.task_id for r in results] == ["bad-1", good.task_id]
    assert [r.ok for r in results] == [False, True]


def test_no_prompt_a_task_can_produce_contains_hidden_material(fixture_repo: FixtureRepo) -> None:
    task = fixture_repo.task
    rendered = pf.rendered_prompts(task.agent_view())
    assert {"input:triager", "input:reproducer", "input:localizer", "input:patcher"} <= set(
        rendered
    )
    assert {f"instructions:{k}" for k in prompts.INSTRUCTIONS} <= set(rendered)
    blob = "\n".join(rendered.values())
    assert task.gold_patch.strip() not in blob and task.test_patch.strip() not in blob
    assert HIDDEN_TEST not in blob


def test_the_report_names_each_failing_task_and_its_violations(fixture_repo: FixtureRepo) -> None:
    good = fixture_repo.task
    bad = good.model_copy(update={"task_id": "bad-1", "issue_body": good.gold_patch})
    text = pf.format_preflight(pf.run_preflight([bad, good], lambda _t: fixture_repo.path))
    assert "2 audited task(s) checked, 1 failed" in text and "- bad-1" in text
    assert "gold patch" in text and good.task_id not in text.split("- bad-1")[0]
    clean = pf.format_preflight(pf.run_preflight([good], lambda _t: fixture_repo.path))
    assert "1 audited task(s) checked, 0 failed" in clean and "all checks passed" in clean
