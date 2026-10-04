"""End-to-end smoke test (`make smoke`): fixture repo -> workspace -> stub pipeline -> ledger row.

Uses the mock provider and synthetic data only. No Docker, no network, $0.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cascade.agents.pipeline import run_stub_pipeline
from cascade.config import ModelsConfig, ResolvedBudget
from cascade.harness.ledger import Ledger
from cascade.harness.metered import MeteredClient
from cascade.harness.providers.mock import MockProvider, MockReply
from cascade.harness.sandbox import LeakGuard, SandboxRunner, SandboxSpec, prepare_workspace
from cascade.taskgen.schema import Task
from tests.conftest import SYNTHETIC, FixtureRepo

pytestmark = pytest.mark.smoke


def test_pipeline_stub_on_fixture_writes_one_ledger_row(
    tmp_path: Path,
    fixture_repo: FixtureRepo,
    task: Task,
    models: ModelsConfig,
    budget: ResolvedBudget,
) -> None:
    # 1. Build the agent workspace from the base commit only, and run the leak guard on it.
    sandbox_root = tmp_path / "sandboxes"
    workspace = prepare_workspace(fixture_repo.path, fixture_repo.base_sha, sandbox_root / "ws")
    guard = LeakGuard.from_task(task, host_paths=[SYNTHETIC])
    SandboxRunner(guard, sandbox_root=sandbox_root).validate(
        SandboxSpec(image="unused-in-smoke", workspace=workspace)
    )

    # 2. Run the stub pipeline with the mock provider through the metered client.
    db = tmp_path / "ledger.sqlite"
    ledger = Ledger(db, budget=budget, models=models)
    provider = MockProvider(
        [
            MockReply(
                text="Likely tinylib/clamp.py",
                input_tokens=400,
                output_tokens=20,
                reasoning_tokens=100,
                latency_s=0.5,
            )
        ]
    )
    arm = models.arms[0]
    result = run_stub_pipeline(
        task.agent_view(), workspace, MeteredClient(provider, ledger), arm, run_id="smoke"
    )
    assert result.triage_text == "Likely tinylib/clamp.py" and result.calls == 1

    # 3. Exactly one ledger row, costed from the (synthetic) models config.
    [row] = ledger.rows()
    prices = models.prices_for(arm.model)
    expected = (
        400 * prices.input_usd_per_mtok
        + 20 * prices.output_usd_per_mtok
        + 100 * prices.reasoning_usd_per_mtok
    ) / 1e6
    assert row.provider == "mock" and row.model == arm.model and row.arm_id == arm.arm_id
    assert (row.input_tokens, row.output_tokens, row.reasoning_tokens) == (400, 20, 100)
    assert row.cost_usd == pytest.approx(expected) and row.cost_usd > 0
    assert row.latency_s == 0.5 and row.status == "ok"
    ledger.close()

    # 4. The row is durable: a fresh ledger on the same file sees the same spend.
    reopened = Ledger(db, budget=budget, models=models)
    assert reopened.spent_usd(task_id=task.task_id) == pytest.approx(expected)
    reopened.close()

    # 5. What the agent saw: the issue and the base-commit file list, never the fix or hidden test.
    prompt = "\n".join(m.content for m in provider.calls[0].messages)
    assert task.issue_title in prompt and "tinylib/clamp.py" in prompt
    assert "return high" not in prompt and "test_above_range" not in prompt
    assert task.gold_patch not in prompt and task.test_patch not in prompt
