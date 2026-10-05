"""The loader's guarantees: provenance, one tier, default rows only, cap normalization, splits."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from cascade.agents.pipeline import PipelineConfig
from cascade.evals import data as dt
from cascade.evals import parquet_io
from cascade.evals.data import DataError, MixedTierError, ProvenanceError
from cascade.routing.matrix import MatrixRow
from cascade.taskgen.audit import append_verdict, make_verdict
from tests.eval_helpers import make_real_like, rewrite_matrix, write_synthetic
from tests.synthetic import gen

ARMS = gen.ARM_IDS
DEFAULT = PipelineConfig().canonical()


def load(d: Path, splits: tuple[str, ...] = ("train",)) -> dt.Dataset:
    return dt.load_dataset(
        data_dir=d, manifest=d / "manifest.jsonl", audit_file=d / "audit.jsonl", arms=ARMS,
        splits=splits, repo_root=Path("."),
    )  # fmt: skip


@pytest.fixture
def d(tmp_path: Path) -> Path:
    return write_synthetic(tmp_path / "data", 0)


def _ids(split: str) -> set[str]:
    return {t.task_id for t in gen.generate(0).tasks if t.split == split}


# --- provenance ---------------------------------------------------------------------------------


def test_the_marker_makes_data_synthetic_and_it_is_stamped(d: Path) -> None:
    p = load(d).provenance
    assert p.data_source == "synthetic"
    assert p.matrix_hash == hashlib.sha256((d / "matrix.parquet").read_bytes()).hexdigest()
    stamp = p.stamp()
    assert "data_source=synthetic" in stamp and p.matrix_hash[:16] in stamp
    assert f"commit={p.git_commit[:12]}" in stamp and "tier=standard" in stamp
    assert p.as_dict()["data_source"] == "synthetic"


def test_unmarked_data_without_a_ledger_is_refused(d: Path) -> None:
    (d / "SYNTHETIC.json").unlink()
    (d / "ledger.parquet").unlink()  # the generator writes an empty one; remove it
    with pytest.raises(ProvenanceError, match="cannot establish"):
        load(d)


def test_unmarked_data_with_an_openai_only_ledger_is_real(d: Path) -> None:
    make_real_like(d)
    assert load(d).provenance.data_source == "real"


@pytest.mark.parametrize("providers", [["mock", "openai"], ["mock"]])
def test_a_ledger_with_any_mock_call_is_not_real_data(d: Path, providers: list[str]) -> None:
    make_real_like(d, providers)
    with pytest.raises(ProvenanceError, match="only openai calls count as real"):
        load(d)


def test_an_empty_ledger_is_refused(d: Path) -> None:
    make_real_like(d, providers=[])
    with pytest.raises(ProvenanceError, match="no logged calls"):
        load(d)


def test_matrix_cost_above_the_ledger_total_is_refused(d: Path) -> None:
    make_real_like(d, extra_cost=-1e9)
    with pytest.raises(ProvenanceError, match="not backed by logged calls"):
        load(d)


def test_a_missing_matrix_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="no matrix"):
        load(tmp_path)


# --- one tier -----------------------------------------------------------------------------------


def test_mixed_processing_tiers_are_refused_before_anything_is_evaluated(d: Path) -> None:
    def flip(rows: list[dict[str, Any]]) -> None:
        rows[0]["processing_tier"] = "flex"

    rewrite_matrix(d, flip)
    with pytest.raises(MixedTierError, match="mixes processing tiers"):
        load(d)


def test_a_single_non_standard_tier_is_fine_and_is_reported(d: Path) -> None:
    def all_flex(rows: list[dict[str, Any]]) -> None:
        for r in rows:
            r["processing_tier"] = "flex"

    rewrite_matrix(d, all_flex)
    assert load(d).provenance.tier == "flex"


def test_a_mixed_tier_in_a_test_row_is_caught_even_by_a_train_only_load(d: Path) -> None:
    test_ids = _ids("test")

    def flip(rows: list[dict[str, Any]]) -> None:
        for r in rows:
            if r["task_id"] in test_ids:
                r["processing_tier"] = "flex"
                break

    rewrite_matrix(d, flip)
    with pytest.raises(
        MixedTierError
    ):  # the tier column of ALL rows is checked; no outcome is read
        load(d, ("train",))


# --- default rows only, and what is kept apart --------------------------------------------------


def test_policies_see_only_default_seed_zero_rows(d: Path) -> None:
    ds = load(d)
    assert ds.provenance.default_config == DEFAULT
    assert all(r.pipeline_config == DEFAULT and r.seed == 0 for r in ds.default_rows.values())
    assert len(ds.records) == 36 and ds.incomplete == []
    for (task, arm), row in ds.default_rows.items():
        assert (task, arm) == (row.task_id, row.arm_id)


def test_ablation_and_repeat_rows_are_kept_apart_not_mixed_into_policies(d: Path) -> None:
    ds = load(d)
    assert set(ds.ablation_rows) == {"no_reproducer", "no_reviewer"}
    assert all(len(v) == 16 for v in ds.ablation_rows.values())
    assert all(r.arm_id == ARMS["mid"] for v in ds.ablation_rows.values() for r in v.values())
    assert all(r.pipeline_config != DEFAULT for v in ds.ablation_rows.values() for r in v.values())
    assert len(ds.repeat_rows) == 24 and all(r.seed != 0 for r in ds.repeat_rows.values())


def test_two_distinct_default_configs_make_the_default_ambiguous(d: Path) -> None:
    other = PipelineConfig(max_repairs=7).canonical()

    def change(rows: list[dict[str, Any]]) -> None:
        for r in rows:
            if r["seed"] == 0 and r["pipeline_config"] == DEFAULT:
                r["pipeline_config"] = other
                break

    rewrite_matrix(d, change)
    with pytest.raises(DataError, match="ambiguous"):
        load(d)


def test_no_default_rows_is_an_error(d: Path) -> None:
    def drop(rows: list[dict[str, Any]]) -> None:
        rows[:] = [r for r in rows if r["pipeline_config"] != DEFAULT]

    rewrite_matrix(d, drop)
    with pytest.raises(DataError, match="no default-pipeline rows"):
        load(d)


# --- cap-exceeded cells -------------------------------------------------------------------------


def test_cap_exceeded_cells_are_failures_that_keep_their_cost(d: Path) -> None:
    ids = {t.task_id for t in gen.generate(0).tasks}
    raw = [MatrixRow.model_validate(r) for r in parquet_io.read_rows(d / "matrix.parquet", ids)]
    capped = [
        r for r in raw
        if r.seed == 0 and r.pipeline_config == DEFAULT
        and dt.is_cap_exceeded(r.error) and (r.true_success or r.verifier_claimed)
    ]  # fmt: skip
    assert capped, "the generator should include capped cells that look like successes"
    ds = load(d, ("train", "test"))
    assert ds.provenance.n_cap_normalized >= len(capped)
    for r in capped:
        got = ds.default_rows[(r.task_id, r.arm_id)]
        assert got.true_success is False and got.verifier_claimed is False
        assert got.cost_usd == r.cost_usd  # the incurred cost is kept


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("cap_exceeded", True),
        ("BudgetExceededError: attempt cap: spent", True),
        ("BudgetExceededError: attempt token cap: used", True),
        ("BudgetExceededError: task cap: spent", True),
        ("patch_error: x", False),
        ("", False),
        (None, False),
    ],
)
def test_is_cap_exceeded(error: str | None, expected: bool) -> None:
    assert dt.is_cap_exceeded(error) is expected


def test_normalize_leaves_ordinary_rows_untouched() -> None:
    row = gen.generate(0).rows[0]
    ok = row.model_copy(update={"true_success": True, "error": None})
    assert dt.normalize(ok) is ok
    capped = row.model_copy(update={"true_success": True, "error": "attempt cap"})
    assert dt.normalize(capped).true_success is False


# --- splits -------------------------------------------------------------------------------------


def test_a_train_load_never_materializes_a_test_row(
    d: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[set[str]] = []
    real = parquet_io.read_rows

    def spy(path: Path, task_ids: Any) -> list[dict[str, Any]]:
        asked.append(set(task_ids))
        return real(path, task_ids)

    monkeypatch.setattr(parquet_io, "read_rows", spy)
    ds = load(d, ("train",))
    assert asked and all(not (ids & _ids("test")) for ids in asked)
    assert {r.task.split for r in ds.records} == {"train"}
    assert ds.provenance.splits == ("train",)
    assert ds.audited_by_split == {"train": 36, "test": 17}  # counts come from the manifest only


def test_a_test_load_returns_only_test_tasks(d: Path) -> None:
    ds = load(d, ("test",))
    assert {r.task.split for r in ds.records} == {"test"} and len(ds.records) == 17


def test_only_audited_ok_tasks_are_used(d: Path) -> None:
    victim = sorted(_ids("train"))[0]
    append_verdict(d / "audit.jsonl", make_verdict(victim, "leaks_fix"))
    ds = load(d)
    assert victim not in {r.task.task_id for r in ds.records} and len(ds.records) == 35


def test_tasks_missing_an_arm_are_excluded_and_counted(d: Path) -> None:
    victim = sorted(_ids("train"))[0]

    def drop(rows: list[dict[str, Any]]) -> None:
        rows[:] = [
            r for r in rows
            if not (
                r["task_id"] == victim and r["arm_id"] == ARMS["max"]
                and r["seed"] == 0 and r["pipeline_config"] == DEFAULT
            )
        ]  # fmt: skip

    rewrite_matrix(d, drop)
    ds = load(d)
    assert ds.incomplete == [victim] and ds.provenance.n_incomplete_tasks == 1
    assert len(ds.records) == 35


def test_arm_labels_must_be_cheap_mid_max(d: Path) -> None:
    with pytest.raises(DataError, match="arms must be labelled"):
        dt.load_dataset(
            data_dir=d, manifest=d / "manifest.jsonl", audit_file=d / "audit.jsonl",
            arms={"a": "x", "b": "y"}, splits=("train",), repo_root=Path("."),
        )  # fmt: skip


def test_default_rows_using_an_unknown_arm_are_refused(d: Path) -> None:
    def rename(rows: list[dict[str, Any]]) -> None:
        rows[0]["arm_id"] = "surprise@arm"
        rows[0]["pipeline_config"] = DEFAULT
        rows[0]["seed"] = 0

    rewrite_matrix(d, rename)
    with pytest.raises(DataError, match="not in the arm map"):
        load(d, ("train", "test"))


def test_git_info_outside_a_repo_is_unknown_and_dirty(tmp_path: Path) -> None:
    assert dt.git_info(tmp_path) == ("unknown", True)
