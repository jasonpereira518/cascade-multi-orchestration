"""Leakage tests (CLAUDE.md hard rule 3), against the synthetic fixture task.

What is asserted, and where it comes from:
  1. the agent sandbox contains no gold patch, no test patch, and no git history at all;
  2. no model call of a full attempt ever carried the gold patch, the test patch, a hidden test's
     name, the fix commit's hash or its message;
  3. the tools cannot be used to reach anything outside the base-commit export;
  4. the agent-side leak guard actually refuses a contaminated workspace during a real run;
  5. the audit that backs 2 detects each kind of leak when one is planted (positive controls).
Network-off is asserted at the Docker level by tests/test_sandbox.py and tests/test_grading.py.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from cascade.agents import prompts
from cascade.agents.audit import added_test_names, audit_text, audit_trace, post_base_commits
from cascade.agents.tools import Toolbox, ToolError
from cascade.errors import LeakError
from cascade.harness.sandbox import (
    LeakGuard,
    SandboxRunner,
    SandboxSpec,
    prepare_workspace,
)
from cascade.harness.trace import TraceWriter
from cascade.taskgen.schema import AgentView
from tests.agent_helpers import (
    AttemptEnv,
    NativeExecutor,
    good_plans,
    request_text,
)
from tests.conftest import FixtureRepo

ROLES = ("triager", "reproducer", "localizer", "patcher", "reviewer")
FIX_COMMIT_MESSAGE = "fix clamp (gold + test)"


def all_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


# --- 1. the sandbox contents ---------------------------------------------------------------


def test_the_agent_workspace_has_no_gold_no_test_patch_and_no_git_history(
    attempt_env: AttemptEnv,
) -> None:
    report = attempt_env.run()
    d = report.attempt_dir
    assert d is not None
    gold, test = attempt_env.task.gold_patch, attempt_env.task.test_patch
    for name in ("base", "work"):
        root = d / name
        assert not (root / ".git").exists()  # no history, so none after the base commit
        for path in all_files(root):
            assert ".git" not in path.relative_to(root).parts
            assert path.suffix not in {".patch", ".diff"}
            text = path.read_text(errors="ignore")
            assert gold.strip() not in text and test.strip() not in text
            assert attempt_env.fx.head_sha not in text and FIX_COMMIT_MESSAGE not in text
    # The workspace is the base commit: the fix and the hidden test are absent from the files.
    clamp = (d / "base" / "tinylib/clamp.py").read_text()
    assert "return high" not in clamp
    assert "test_above_range" not in (d / "base" / "tests/test_clamp.py").read_text()


def test_the_post_base_commit_exists_in_the_clone_but_not_in_any_workspace(
    fixture_repo: FixtureRepo, tmp_path: Path
) -> None:
    assert fixture_repo.head_sha in post_base_commits(fixture_repo.path, fixture_repo.base_sha)
    ws = prepare_workspace(fixture_repo.path, fixture_repo.base_sha, tmp_path / "ws")
    blob = "\n".join(p.read_text(errors="ignore") for p in all_files(ws))
    assert fixture_repo.head_sha not in blob and "return high" not in blob


def test_the_mounted_directory_set_is_exactly_one_clean_workspace(
    attempt_env: AttemptEnv, fixture_repo: FixtureRepo
) -> None:
    report = attempt_env.run()
    assert report.attempt_dir is not None
    guard = LeakGuard.from_task(attempt_env.task)
    runner = SandboxRunner(guard, sandbox_root=report.attempt_dir)
    cmd = runner.build_command(
        SandboxSpec(image="img", workspace=report.attempt_dir / "work"), ["pytest"], name="n"
    )
    mounts = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--mount"]
    assert (
        len(mounts) == 1 and "source=" + str((report.attempt_dir / "work").resolve()) in mounts[0]
    )
    assert cmd[cmd.index("--network") + 1] == "none"
    assert guard.scan(report.attempt_dir / "work") == []


# --- 2. what the models were actually sent -------------------------------------------------


def test_no_model_call_in_a_full_attempt_carries_hidden_material(attempt_env: AttemptEnv) -> None:
    source = attempt_env.source()
    report = attempt_env.run(source=source)
    assert report.row.true_success  # a full, successful run: every role was exercised
    task, fx = attempt_env.task, attempt_env.fx

    forbidden: dict[str, str] = {
        "the gold patch": task.gold_patch.strip(),
        "the test patch": task.test_patch.strip(),
        "the fix commit hash": fx.head_sha,
        "the fix commit message": FIX_COMMIT_MESSAGE,
    }
    for name in added_test_names(task.test_patch):
        forbidden[f"hidden test name {name}"] = name
    for test_id in task.fail_to_pass:
        forbidden[f"hidden test id {test_id}"] = test_id

    calls = [(role, c) for role in ROLES for c in source.calls(role)]
    assert {r for r, _ in calls} == set(ROLES) and len(calls) == 10
    for role, call in calls:
        text = request_text(call)
        escaped = json.dumps(text)[1:-1]
        for label, needle in forbidden.items():
            if role == "reviewer" and label == "the gold patch":
                continue  # the reviewer is shown the agent's own patch (see agents.audit)
            assert needle not in text and needle[:80] not in escaped, (
                f"{role} prompt contains {label}"
            )


def test_the_trace_audit_agrees_that_a_full_attempt_is_clean(attempt_env: AttemptEnv) -> None:
    report = attempt_env.run()
    assert report.leak_violations == []
    trace = (report.attempt_dir / "trace.jsonl").read_text()  # type: ignore[union-attr]
    assert (
        audit_trace(
            trace,
            attempt_env.task,
            post_base_shas=post_base_commits(attempt_env.fx.path, attempt_env.task.base_commit),
        )
        == []
    )


def test_prompt_builders_cannot_receive_a_task() -> None:
    assert "Task" not in vars(prompts)  # the module never even imports the full task type
    for name, fn in inspect.getmembers(prompts, inspect.isfunction):
        if fn.__module__ != prompts.__name__ or name.startswith("_"):
            continue
        annotations = " ".join(str(p.annotation) for p in inspect.signature(fn).parameters.values())
        assert not re.search(r"\bTask\b", annotations), name


def test_agent_view_has_no_field_that_could_carry_hidden_data(attempt_env: AttemptEnv) -> None:
    assert set(AgentView.model_fields) == {
        "task_id",
        "repo",
        "base_commit",
        "issue_title",
        "issue_body",
    }
    dumped = attempt_env.task.agent_view().model_dump_json()
    assert attempt_env.task.gold_patch not in dumped and attempt_env.task.test_patch not in dumped


# --- 3. the tools cannot escape the export -------------------------------------------------


@pytest.fixture
def box(attempt_env: AttemptEnv, tmp_path: Path) -> Toolbox:
    ws = prepare_workspace(attempt_env.fx.path, attempt_env.task.base_commit, tmp_path / "tool-ws")
    return Toolbox(ws, role="patcher", trace=TraceWriter(None), executor=NativeExecutor())


def test_tools_cannot_read_git_internals_or_the_upstream_repository(
    box: Toolbox, attempt_env: AttemptEnv
) -> None:
    upstream = attempt_env.fx.path
    for target in (".git/HEAD", ".git/refs/heads/main", str(upstream / ".git/HEAD"),
                   str(upstream / "tinylib/clamp.py"), "../upstream/tinylib/clamp.py"):  # fmt: skip
        with pytest.raises(ToolError):
            box.read_file(target, 1, 20)
    with pytest.raises(ToolError):
        box.list_files(str(upstream))


def test_searching_cannot_surface_content_that_is_not_in_the_export(
    box: Toolbox, attempt_env: AttemptEnv
) -> None:
    # The gold fix and the hidden test exist only in the post-base commit, never in the export.
    assert box.grep("test_above_range", "*") == "no matches"  # the exact hidden test name
    assert box.search_code("above", 5) == "no matches"
    assert "return high" not in box.grep("return", "*")
    assert box.grep(attempt_env.fx.head_sha, "*") == "no matches"  # exact hash, not BM25 noise
    assert box.grep("gold \\+ test", "*") == "no matches"  # the fix commit message


# --- 4. the leak guard is live during a real run --------------------------------------------


def test_a_contaminated_workspace_stops_the_run_instead_of_running_tests(
    attempt_env: AttemptEnv,
) -> None:
    source = attempt_env.source()
    pipeline, base, work = attempt_env.pipeline(source)
    from cascade.harness.sandbox import SandboxRunner as Runner

    guard = LeakGuard.from_task(attempt_env.task)
    executor = NativeExecutor(Runner(guard, sandbox_root=work.parent))
    pipeline._executor = executor
    (work / "notes.txt").write_text(attempt_env.task.gold_patch)  # contamination appears mid-run
    with pytest.raises(LeakError):
        pipeline.run(attempt_env.task.agent_view(), base, work)
    assert executor.calls == []  # no test process was ever started


# --- 5. positive controls: the audit really detects each kind of leak -----------------------


def test_the_audit_detects_each_kind_of_planted_leak(attempt_env: AttemptEnv) -> None:
    task, fx = attempt_env.task, attempt_env.fx
    shas = post_base_commits(fx.path, task.base_commit)

    assert audit_text("harmless text", task, post_base_shas=shas) == []
    assert any("gold patch" in v for v in audit_text(f"prefix {task.gold_patch} suffix", task))
    assert any("test patch" in v for v in audit_text(f"x {task.test_patch} y", task))
    escaped_gold = json.dumps(task.gold_patch)[1:-1]  # how it looks inside a JSONL trace line
    assert any("gold patch" in v for v in audit_text(f'{{"input": "{escaped_gold}"}}', task))
    assert any("test_above_range" in v for v in audit_text("run test_above_range please", task))
    assert not audit_text("test_above_range_extended is a different name", task)  # word boundary
    assert any(
        fx.head_sha[:12] in v
        for v in audit_text(f"commit {fx.head_sha}", task, post_base_shas=shas)
    )
    assert not audit_text(f"commit {fx.base_sha}", task, post_base_shas=shas)  # the base is allowed


def test_the_trace_audit_exempts_only_whole_patch_matches_in_reviewer_calls(
    attempt_env: AttemptEnv,
) -> None:
    task = attempt_env.task
    line = lambda role, text: json.dumps({"kind": "llm", "role": role, "input": text})  # noqa: E731
    gold = task.gold_patch
    assert audit_trace(line("reviewer", gold), task) == []  # the agent's own patch, shown back
    assert any("gold patch" in v for v in audit_trace(line("triager", gold), task))
    assert any("gold patch" in v for v in audit_trace(line("patcher", gold), task))
    # Other checks still apply to reviewer lines.
    assert any(
        "test_above_range" in v for v in audit_trace(line("reviewer", "test_above_range"), task)
    )
    assert audit_trace("", task) == [] and audit_trace("not json\n\n", task) == []


def test_every_planted_leak_into_a_real_prompt_would_be_caught(attempt_env: AttemptEnv) -> None:
    """End to end: a Patcher output that echoes hidden material ends up flagged on the attempt."""
    plans = good_plans()
    from agents.testing import ModelStep

    from cascade.agents.schemas import PatchOutput
    from cascade.harness.providers.mock import final_step
    from tests.agent_helpers import patch_steps

    echoed = attempt_env.task.test_patch.strip()
    plans["patcher"] = [
        *patch_steps()[:-1],
        final_step(PatchOutput(summary=echoed, files_changed=[], confidence="high")),
    ]
    assert isinstance(plans["patcher"][0], ModelStep)
    report = attempt_env.run(plans)
    assert any("test patch" in v for v in report.leak_violations)


def test_the_trace_audit_finds_a_patch_hidden_inside_several_layers_of_json(
    attempt_env: AttemptEnv,
) -> None:
    """A model can wrap content in JSON inside JSON inside the trace record's own JSON."""
    task = attempt_env.task
    wrapped = json.dumps({"payload": json.dumps({"p": task.test_patch.strip()})})
    line = json.dumps({"kind": "llm", "role": "patcher", "input": wrapped})
    assert any("test patch" in v for v in audit_trace(line, task))
    # And the unwrapped control is clean, so the hit above is caused by the nesting handling.
    clean = json.dumps({"kind": "llm", "role": "patcher", "input": json.dumps({"p": "nothing"})})
    assert audit_trace(clean, task) == []
