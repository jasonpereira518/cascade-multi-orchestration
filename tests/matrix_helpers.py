"""Shared builders for the Phase 4 tests. All SYNTHETIC: mock models, invented prices and caps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cascade.agents.pipeline import PipelineConfig
from cascade.config import ModelsConfig, ResolvedBudget, load_budget, load_models
from cascade.harness.ledger import CallContext, Ledger
from cascade.harness.providers.base import Usage
from cascade.routing.matrix import MatrixRow
from cascade.taskgen.schema import Task
from tests.conftest import SYNTHETIC
from tests.phase3_helpers import make_task

MATRIX_MODELS = SYNTHETIC / "models_matrix.yaml"
MATRIX_BUDGET = SYNTHETIC / "budget_matrix.yaml"

CHEAP = "mock-luna@low"
MID = "mock-luna@high"
MAX = "mock-sol@medium"
ARMS = (CHEAP, MID, MAX)


def matrix_models() -> ModelsConfig:
    return load_models(MATRIX_MODELS)


def matrix_budget(tmp_path: Path, **over: Any) -> ResolvedBudget:
    b = load_budget(MATRIX_BUDGET).resolve()
    return b.model_copy(update={"kill_switch_path": tmp_path / "KILL_SWITCH", **over})


def new_ledger(tmp_path: Path, **budget_over: Any) -> Ledger:
    return Ledger(
        tmp_path / "ledger.sqlite",
        budget=matrix_budget(tmp_path, **budget_over),
        models=matrix_models(),
    )


def ctx(
    arm_id: str = CHEAP,
    *,
    task_id: str = "t1",
    attempt_id: str = "a1",
    call_id: str = "c1",
    bucket: str = "default",
    tier: str = "standard",
) -> CallContext:
    return CallContext(call_id, "run", task_id, attempt_id, arm_id, "patcher", bucket, tier)


def spend(
    ledger: Ledger,
    arm_id: str,
    input_tokens: int,
    *,
    task_id: str = "t1",
    attempt_id: str = "a1",
    call_id: str = "c1",
    bucket: str = "default",
    tier: str = "standard",
) -> float:
    """Record one call of ``input_tokens`` input tokens on an arm's model; return its cost."""
    model = arm_id.split("@")[0]
    rec = ledger.record(
        ctx(
            arm_id, task_id=task_id, attempt_id=attempt_id, call_id=call_id, bucket=bucket,
            tier=tier,
        ),
        provider="mock",
        model=model,
        usage=Usage(input_tokens=input_tokens, output_tokens=0),
        latency_s=0.0,
    )  # fmt: skip
    return rec.cost_usd


def row(task_id: str, arm_id: str, **over: Any) -> MatrixRow:
    base: dict[str, Any] = {
        "task_id": task_id, "arm_id": arm_id, "seed": 0, "true_success": False,
        "verifier_claimed": False, "cost_usd": 0.01, "input_tokens": 100, "output_tokens": 10,
        "reasoning_tokens": 0, "wall_s": 1.0, "turns": 3, "patch_sha": None,
        "pipeline_config": PipelineConfig().canonical(), "error": None,
    }  # fmt: skip
    return MatrixRow(**{**base, **over})


def tasks_with_splits(spec: dict[str, tuple[str, int]]) -> list[Task]:
    """{task_id: (split, src_lines)} -> synthetic tasks across two repos (a and b alternating)."""
    out: list[Task] = []
    for i, (task_id, (split, lines)) in enumerate(spec.items()):
        repo = "o/a" if i % 2 == 0 else "o/b"
        t = make_task(repo, i + 1, task_id=task_id, split=split)
        features = t.features.model_copy(update={"src_lines": lines})
        out.append(t.model_copy(update={"features": features}))
    return out
