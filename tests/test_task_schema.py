from __future__ import annotations

import pytest
from pydantic import ValidationError

from cascade.taskgen.schema import AgentView, Task
from tests.conftest import SYNTHETIC


def test_synthetic_task_loads_and_patch_files_match_embedded_patches(task: Task) -> None:
    assert task.gold_patch == (SYNTHETIC / "gold.patch").read_text()
    assert task.test_patch == (SYNTHETIC / "test.patch").read_text()
    assert task.audit.status == "ok" and task.split is None


@pytest.mark.parametrize("sha", ["abc123", "A" * 40, "g" * 40, "0" * 39, "0" * 41])
def test_base_commit_must_be_full_lowercase_sha(task: Task, sha: str) -> None:
    data = task.model_dump()
    data["base_commit"] = sha
    with pytest.raises(ValidationError):
        Task.model_validate(data)


def test_fail_to_pass_cannot_be_empty(task: Task) -> None:
    data = task.model_dump()
    data["fail_to_pass"] = []
    with pytest.raises(ValidationError):
        Task.model_validate(data)


def test_agent_view_exposes_only_issue_text_and_pin(task: Task) -> None:
    view = task.agent_view()
    assert set(view.model_dump()) == {"task_id", "repo", "base_commit", "issue_title", "issue_body"}
    blob = view.model_dump_json()
    assert task.gold_patch not in blob and task.test_patch not in blob
    for test_id in task.fail_to_pass + task.pass_to_pass:
        assert test_id not in blob


def test_agent_view_cannot_be_built_with_gold_fields(task: Task) -> None:
    with pytest.raises(ValidationError):
        AgentView.model_validate(task.model_dump())
