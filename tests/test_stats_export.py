"""`cascade stats` numbers and the parquet export. Synthetic data; every figure checked from source."""

from __future__ import annotations

import datetime as dt
import textwrap
from pathlib import Path

import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs]
import pytest

from cascade.harness.providers.base import Usage
from cascade.routing import stats as st
from cascade.routing.export import export_parquet
from cascade.routing.matrix import MatrixStore
from cascade.taskgen.audit import AuditVerdict, Verdict, make_verdict
from tests.matrix_helpers import (
    CHEAP,
    MAX,
    MID,
    ctx,
    matrix_models,
    new_ledger,
    row,
    tasks_with_splits,
)

NOW = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


def fake_collect(_cwd: Path) -> str:
    return "tests/test_a.py::test_x\n\n609 tests collected in 0.42s\n"


def verdicts(ids: list[str], v: Verdict = "ok") -> dict[str, AuditVerdict]:
    return {i: make_verdict(i, v, now=NOW) for i in ids}


# --- counting tests -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("609 tests collected in 0.42s", 609),
        ("1 test collected in 0.01s", 1),
        ("collected 5 items", 5),
        ("noise\n12 tests collected\n", 12),
        ("no summary here", None),
        ("", None),
    ],
)
def test_count_tests_parses_pytests_collection_summary(output: str, expected: int | None) -> None:
    assert st.count_tests(output) == expected


def test_pytest_collect_really_counts_a_directory_of_tests(tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text(
        textwrap.dedent(
            """
            import pytest

            def test_a(): pass
            def test_b(): pass

            @pytest.mark.parametrize("x", [1, 2])
            def test_c(x): pass
            """
        )
    )
    assert st.count_tests(st.pytest_collect(tmp_path)) == 4


# --- collecting stats ---------------------------------------------------------------------------


def test_stats_come_from_the_ledger_matrix_manifest_and_audit(tmp_path: Path) -> None:
    ledger = new_ledger(tmp_path)
    ledger.record(
        ctx(CHEAP, bucket="pilot", call_id="c1"), provider="mock", model="mock-luna",
        usage=Usage(input_tokens=1000, output_tokens=100, reasoning_tokens=10, cached_tokens=200),
        latency_s=0.1,
    )  # fmt: skip
    ledger.record(
        ctx(MAX, bucket="matrix", call_id="c2", attempt_id="a2"), provider="mock",
        model="mock-sol",
        usage=Usage(input_tokens=500, output_tokens=50, cache_write_tokens=30), latency_s=0.1,
    )  # fmt: skip
    ledger.close()

    spec = {"a1": ("train", 5), "a2": ("test", 6), "a3": ("train", 7), "a4": ("train", 8)}
    tasks = tasks_with_splits(spec)
    v = verdicts(["a1", "a2", "a3"])  # a4 is not audited
    v["a3"] = make_verdict("a3", "leaks_fix", now=NOW)  # and a3 is audited but not ok
    rows = [row("a1", CHEAP), row("a1", MID), row("a2", MAX)]

    s = st.collect_stats(
        tasks=tasks, verdicts=v, models=matrix_models(), matrix_rows=rows,
        ledger_db=tmp_path / "ledger.sqlite", repo_root=tmp_path, collect=fake_collect,
    )  # fmt: skip
    assert (s.audited_ok, s.audited_train, s.audited_test) == (2, 1, 1)
    assert s.repos == ["o/a", "o/b"] and s.ok_by_repo == {"o/a": 1, "o/b": 1}
    assert s.arms == [CHEAP, MID, MAX]  # optional arms are not counted
    assert s.attempts == 3 and s.ledger_calls == 2
    assert (s.input_tokens, s.output_tokens, s.reasoning_tokens) == (1500, 150, 10)
    assert (s.cached_tokens, s.cache_write_tokens) == (200, 30)
    cheap = (800 * 1.0 + 200 * 0.25 + 100 * 4.0 + 10 * 4.0) / 1e6
    top = (470 * 10.0 + 30 * 12.5 + 50 * 40.0) / 1e6
    assert s.spend_by_arm == {CHEAP: pytest.approx(cheap), MAX: pytest.approx(top)}
    assert s.spend_by_bucket == {"matrix": pytest.approx(top), "pilot": pytest.approx(cheap)}
    assert s.spend_total == pytest.approx(cheap + top)
    assert s.tests_in_repo == 609


def test_stats_with_no_artifacts_are_zeros_not_errors(tmp_path: Path) -> None:
    s = st.collect_stats(
        tasks=[], verdicts={}, models=matrix_models(), matrix_rows=[], ledger_db=None,
        repo_root=tmp_path, collect=fake_collect,
    )  # fmt: skip
    assert (s.audited_ok, s.attempts, s.ledger_calls, s.spend_total) == (0, 0, 0, 0.0)
    assert s.spend_by_arm == {} and s.spend_by_bucket == {}
    text = st.format_stats(s)
    assert "audited-ok tasks: 0" in text and "(none)" in text


def test_a_failed_test_collection_is_reported_as_unknown_never_guessed(tmp_path: Path) -> None:
    s = st.collect_stats(
        tasks=[], verdicts={}, models=matrix_models(), matrix_rows=[], ledger_db=None,
        repo_root=tmp_path, collect=lambda _p: "ERROR: no tests ran",
    )  # fmt: skip
    assert s.tests_in_repo is None
    assert "unknown (pytest collection failed)" in st.format_stats(s)


def test_the_printed_stats_contain_every_requested_number(tmp_path: Path) -> None:
    ledger = new_ledger(tmp_path)
    ledger.record(
        ctx(CHEAP, bucket="pilot"), provider="mock", model="mock-luna",
        usage=Usage(input_tokens=1_234_567, output_tokens=2, reasoning_tokens=3, cached_tokens=4),
        latency_s=0.0,
    )  # fmt: skip
    ledger.close()
    tasks = tasks_with_splits({"a1": ("train", 5)})
    s = st.collect_stats(
        tasks=tasks, verdicts=verdicts(["a1"]), models=matrix_models(),
        matrix_rows=[row("a1", CHEAP)], ledger_db=tmp_path / "ledger.sqlite",
        repo_root=tmp_path, collect=fake_collect,
    )  # fmt: skip
    text = st.format_stats(s)
    for needle in (
        "repos with audited-ok tasks: 1", "audited-ok tasks: 1 (train 1, test 0)", "arms: 3",
        "total attempts (matrix rows): 1", "input 1,234,567", "reasoning 3", "cached 4",
        "by arm:", "by budget bucket:", "pilot:",
        "tests in this repository (pytest --collect-only): 609",
    ):  # fmt: skip
        assert needle in text, needle


# --- parquet export -----------------------------------------------------------------------------


def test_export_is_a_straight_copy_of_both_stores(tmp_path: Path) -> None:
    ledger = new_ledger(tmp_path)
    ledger.record(
        ctx(CHEAP, bucket="pilot", tier="flex"), provider="mock", model="mock-luna",
        usage=Usage(input_tokens=10, output_tokens=1, cached_tokens=2), latency_s=0.5,
    )  # fmt: skip
    ledger.close()
    store = MatrixStore(tmp_path / "matrix.sqlite")
    store.put(row("t2", MID, true_success=True))
    store.put(row("t1", CHEAP, started_at="2026-01-01T00:00:00+00:00"))
    stored = store.rows()
    store.close()

    out = tmp_path / "out"
    counts = export_parquet(tmp_path / "matrix.sqlite", tmp_path / "ledger.sqlite", out)
    assert counts == {"matrix": 2, "ledger": 1}
    m = pq.read_table(out / "matrix.parquet").to_pylist()
    assert m == [r.model_dump() for r in stored]  # same fields, same values, same order
    assert [r["task_id"] for r in m] == ["t1", "t2"]  # deterministic order
    ldg = pq.read_table(out / "ledger.parquet").to_pylist()
    assert ldg[0]["processing_tier"] == "flex" and ldg[0]["bucket"] == "pilot"
    assert ldg[0]["cached_tokens"] == 2 and ldg[0]["cost_usd"] > 0


def test_export_of_missing_stores_writes_empty_files_with_the_full_columns(tmp_path: Path) -> None:
    out = tmp_path / "out"
    counts = export_parquet(tmp_path / "no.sqlite", tmp_path / "no-ledger.sqlite", out)
    assert counts == {"matrix": 0, "ledger": 0}
    cols = pq.read_table(out / "matrix.parquet").column_names
    assert {"task_id", "processing_tier", "started_at", "pipeline_config"} <= set(cols)
    assert "bucket" in pq.read_table(out / "ledger.parquet").column_names


def test_exporting_twice_gives_identical_files(tmp_path: Path) -> None:
    store = MatrixStore(tmp_path / "matrix.sqlite")
    store.put(row("t1", CHEAP))
    store.close()
    new_ledger(tmp_path).close()
    export_parquet(tmp_path / "matrix.sqlite", tmp_path / "ledger.sqlite", tmp_path / "a")
    export_parquet(tmp_path / "matrix.sqlite", tmp_path / "ledger.sqlite", tmp_path / "b")
    first = pq.read_table(tmp_path / "a" / "matrix.parquet")
    assert first.equals(pq.read_table(tmp_path / "b" / "matrix.parquet"))
