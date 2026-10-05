from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from cascade.agents.pipeline import PipelineConfig
from cascade.routing.matrix import MatrixRow, MatrixStore, config_id_of


def row(**over: Any) -> MatrixRow:
    base: dict[str, Any] = {
        "task_id": "t1", "arm_id": "m@low", "seed": 0, "true_success": True, "verifier_claimed": True,
        "cost_usd": 0.0123, "input_tokens": 1000, "output_tokens": 200, "reasoning_tokens": 50,
        "wall_s": 12.5, "turns": 9, "patch_sha": "abc", "pipeline_config": PipelineConfig().canonical(),
        "error": None,
    }  # fmt: skip
    return MatrixRow(**{**base, **over})


def test_rows_carry_the_spec_fields_plus_the_phase_4_fields() -> None:
    assert list(MatrixRow.model_fields) == [
        "task_id", "arm_id", "seed", "true_success", "verifier_claimed", "cost_usd", "input_tokens",
        "output_tokens", "reasoning_tokens", "wall_s", "turns", "patch_sha", "pipeline_config", "error",
        "model", "reasoning_effort", "processing_tier", "cached_tokens", "cache_write_tokens",
        "started_at",
    ]  # fmt: skip


def test_rows_written_before_phase_4_still_load_with_defaults() -> None:
    legacy = row().model_dump()
    for key in ("model", "reasoning_effort", "processing_tier", "cached_tokens",
                "cache_write_tokens", "started_at"):  # fmt: skip
        legacy.pop(key)
    loaded = MatrixRow.model_validate(legacy)
    assert loaded.processing_tier == "standard" and loaded.cached_tokens == 0
    assert loaded.model == "" and loaded.started_at == ""


def test_delete_removes_only_the_named_cell() -> None:
    store = MatrixStore(":memory:")
    a, b = row(), row(arm_id="m@high")
    store.put(a)
    store.put(b)
    store.delete(a.task_id, a.arm_id, a.seed, a.config_id)
    assert store.rows() == [b]
    store.delete("nope", "nope", 0, "x")  # deleting a missing cell is a no-op
    assert store.rows() == [b]


def test_put_get_roundtrip_and_upsert() -> None:
    store = MatrixStore(":memory:")
    r = row()
    store.put(r)
    assert store.get("t1", "m@low", 0, r.config_id) == r
    store.put(row(true_success=False, cost_usd=0.5))  # same key: replaced, not duplicated
    assert len(store.rows()) == 1 and store.rows()[0].true_success is False


def test_the_key_includes_task_arm_seed_and_pipeline_config() -> None:
    store = MatrixStore(":memory:")
    ablated = PipelineConfig(reviewer=False).canonical()
    for r in (
        row(),
        row(task_id="t2"),
        row(arm_id="m@high"),
        row(seed=1),
        row(pipeline_config=ablated),
    ):
        store.put(r)
    assert len(store.rows()) == 5  # ablations of the same attempt are separate rows
    assert store.get("t1", "m@low", 0, config_id_of(ablated)) is not None
    assert store.get("t1", "m@low", 7, row().config_id) is None


def test_rows_come_back_in_a_stable_order() -> None:
    store = MatrixStore(":memory:")
    for r in (row(task_id="b"), row(task_id="a", seed=2), row(task_id="a", seed=1)):
        store.put(r)
    assert [(r.task_id, r.seed) for r in store.rows()] == [("a", 1), ("a", 2), ("b", 0)]


def test_the_store_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "m.sqlite"
    store = MatrixStore(path)
    store.put(row())
    store.close()
    assert MatrixStore(path).rows() == [row()]


def test_config_ids_are_stable_and_distinguish_configs() -> None:
    a, b = PipelineConfig().canonical(), PipelineConfig(max_repairs=5).canonical()
    assert config_id_of(a) == config_id_of(a) != config_id_of(b) and len(config_id_of(a)) == 10


@pytest.mark.parametrize(
    "bad", [{"cost_usd": -0.01}, {"input_tokens": -1}, {"turns": -1}, {"wall_s": -1.0}]
)
def test_negative_quantities_are_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        row(**bad)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        row(model_name="not-in-the-spec")
