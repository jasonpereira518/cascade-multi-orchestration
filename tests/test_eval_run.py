"""The dev (train/CV) analysis end to end on the labeled-synthetic matrix, and `cascade eval`.

Everything here is SYNTHETIC; the point is the plumbing: reproducibility given the seed, the stamp on
every artifact, the train-only guarantee, and the refusals. No number here is a result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from cascade.cli import app
from cascade.eval_cli import dev_result
from cascade.evals import bootstrap as bs
from cascade.evals import parquet_io, report
from cascade.evals.data import Dataset, load_dataset
from cascade.evals.run import EvalResult, evaluate_dataset
from tests.eval_helpers import init_repo, rewrite_matrix, write_synthetic
from tests.synthetic import gen

runner = CliRunner()
B = 400


@pytest.fixture
def d(tmp_path: Path) -> Path:
    return write_synthetic(tmp_path / "data", 0)


def train_set(d: Path) -> Dataset:
    return load_dataset(
        data_dir=d, manifest=d / "manifest.jsonl", audit_file=d / "audit.jsonl", arms=gen.ARM_IDS,
        splits=("train",), repo_root=Path("."),
    )  # fmt: skip


def run_dev(d: Path, b: int = B) -> EvalResult:
    return dev_result(train_set(d), b)


def invoke(d: Path, out: Path, *extra: str) -> Result:
    args = [
        "eval", "--data-dir", str(d), "--manifest", str(d / "manifest.jsonl"),
        "--audit-file", str(d / "audit.jsonl"), "--out-dir", str(out), "--resamples", str(B),
    ]  # fmt: skip
    return runner.invoke(app, [*args, *extra])


# --- reproducibility and stamping ---------------------------------------------------------------


def test_the_dev_run_is_reproducible_given_the_seed(d: Path, tmp_path: Path) -> None:
    a, b = run_dev(d), run_dev(d)
    assert report.to_json(a) == report.to_json(b)
    assert report.to_markdown(a) == report.to_markdown(b)
    assert report.to_svg(a) == report.to_svg(b)
    again = run_dev(write_synthetic(tmp_path / "again", 0))  # regenerated from the same seed
    assert report.to_json(again) == report.to_json(a)
    assert a.stamp.split("matrix=")[1] == again.stamp.split("matrix=")[1]  # identical matrix hash


def test_a_different_generator_seed_gives_a_different_analysis(d: Path, tmp_path: Path) -> None:
    other = run_dev(write_synthetic(tmp_path / "other", 1))
    assert report.to_json(other) != report.to_json(run_dev(d))


def test_a_different_bootstrap_seed_changes_intervals_but_not_points(d: Path) -> None:
    train = train_set(d)
    kw: dict[str, Any] = {"mode": "dev", "train": train, "target": train, "margin": 0.15}
    a = evaluate_dataset(**kw, margin_basis="x", b=B, seed=1)
    b = evaluate_dataset(**kw, margin_basis="x", b=B, seed=2)
    pa, pb = a.metrics[0], b.metrics[0]
    assert pa.resolve_rate.point == pb.resolve_rate.point
    assert (pa.resolve_rate.lo, pa.resolve_rate.hi) != (pb.resolve_rate.lo, pb.resolve_rate.hi)


def test_every_artifact_carries_the_stamp_and_says_it_is_synthetic_dev(d: Path) -> None:
    r = run_dev(d)
    md, js, svg = report.to_markdown(r), json.loads(report.to_json(r)), report.to_svg(r)
    assert r.stamp in md and r.stamp in svg and js["stamp"] == r.stamp
    assert "data_source=synthetic" in r.stamp and js["provenance"]["data_source"] == "synthetic"
    assert js["provenance"]["matrix_hash"][:16] in svg
    assert "SYNTHETIC DATA" in md and "TRAIN / CV (dev)" in md and "not evidence" in md
    assert "TRAIN / CV (dev)" in svg and svg.startswith("<svg")


def test_the_json_is_strict_json_with_no_nan_tokens(d: Path) -> None:
    text = report.to_json(run_dev(d))

    def refuse(token: str) -> Any:
        raise AssertionError(f"non-standard JSON constant {token}")

    json.loads(text, parse_constant=refuse)


# --- what the dev run contains ------------------------------------------------------------------


def test_the_dev_run_covers_train_only_with_every_policy_and_secondary_analysis(d: Path) -> None:
    r = run_dev(d)
    assert r.mode == "dev" and r.n_tasks == 36 and r.n_repos == 3
    assert r.provenance["splits"] == ["train"]
    names = [m.name for m in r.metrics]
    assert names == [
        "always_cheap", "always_mid", "always_max", "cascade", "oracle",
        "predictive_issue", "predictive_all",
    ]  # fmt: skip
    assert [c.policy for c in r.comparisons] == [n for n in names if n != "always_max"]
    assert {v.arm for v in r.verifier} == {"cheap", "mid", "max"}
    assert {a.variant for a in r.ablation} == {"default", "no_reproducer", "no_reviewer"}
    assert all(a.n_shared == 16 for a in r.ablation)
    assert len(r.noise) == 2 and all(n.n_pairs == 12 for n in r.noise)
    assert set(r.selections) == {"issue", "all"} and r.notes == []  # 3 repos: nested CV is possible
    assert {x.repo for x in r.per_repo} == {"syn/a", "syn/b", "syn/c"}


def test_the_oracle_resolves_at_least_as_many_as_any_other_policy(d: Path) -> None:
    r = run_dev(d)
    rates = {m.name: m.resolve_rate.point for m in r.metrics}
    assert all(rates["oracle"] >= v - 1e-12 for v in rates.values())


def test_the_margin_comes_from_the_manifest_count_of_test_tasks_not_from_outcomes(d: Path) -> None:
    r = run_dev(d)
    assert r.margin == bs.margin_for(17) and "17 audited test tasks" in r.margin_basis
    assert (r.resamples, r.seed) == (B, bs.SEED_DEFAULT)


def test_cap_normalization_is_counted_and_small_samples_are_warned_about(d: Path) -> None:
    r = run_dev(d)
    assert r.provenance["n_cap_normalized"] >= 1
    md = report.to_markdown(r)
    assert "Confidence intervals are wide: only 36 tasks" in md
    assert "Repo-clustered intervals rest on only 3 repo(s)" in md
    assert f"{B:,} resamples" in md and "SYNTHETIC" in md


def test_the_pareto_figure_has_a_point_per_policy_and_labels_the_axes(d: Path) -> None:
    svg = report.to_svg(run_dev(d))
    for name in ("always_cheap", "always_max", "cascade", "oracle", "predictive_issue"):
        assert name in svg
    assert "mean cost per task (USD)" in svg and "resolve rate" in svg
    assert svg.count('opacity="0.6"') == 10  # the faint CV sweep points, one per multiplier


# --- the real CLI -------------------------------------------------------------------------------


def test_cascade_eval_writes_the_three_artifacts_and_prints_the_report(
    d: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    r = invoke(d, out)
    assert r.exit_code == 0, r.output
    for name in ("report.md", "results.json", "pareto.svg"):
        assert (out / name).exists(), name
    assert "SYNTHETIC DATA" in r.output and "data_source=synthetic" in r.output
    assert (out / "report.md").read_text() == report.to_markdown(run_dev(d))


def test_cascade_eval_without_final_never_reads_a_test_row(
    d: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[set[str]] = []
    real = parquet_io.read_rows

    def spy(path: Path, task_ids: Any) -> list[dict[str, Any]]:
        asked.append(set(task_ids))
        return real(path, task_ids)

    monkeypatch.setattr(parquet_io, "read_rows", spy)
    assert invoke(d, tmp_path / "o").exit_code == 0
    test_ids = {t.task_id for t in gen.generate(0).tasks if t.split == "test"}
    assert asked and all(not (ids & test_ids) for ids in asked)


def test_cascade_eval_refuses_a_mixed_tier_matrix(d: Path, tmp_path: Path) -> None:
    def flip(rows: list[dict[str, Any]]) -> None:
        rows[0]["processing_tier"] = "flex"

    rewrite_matrix(d, flip)
    r = invoke(d, tmp_path / "o")
    assert r.exit_code == 2 and "mixes processing tiers" in r.output
    assert not (tmp_path / "o").exists()  # nothing was evaluated or written


def test_cascade_eval_refuses_data_with_no_provenance_cleanly(d: Path, tmp_path: Path) -> None:
    (d / "SYNTHETIC.json").unlink()
    (d / "ledger.parquet").unlink()
    r = runner.invoke(
        app,
        ["eval", "--data-dir", str(d), "--manifest", str(d / "manifest.jsonl"),
         "--audit-file", str(d / "audit.jsonl"), "--configs-dir", str(tmp_path)],
    )  # fmt: skip
    assert r.exit_code == 2 and "error:" in r.output and "Traceback" not in r.output
    assert "models.yaml" in r.output  # no marker, so arms come from models.yaml, which is absent


def test_cascade_eval_final_refuses_synthetic_data_and_writes_no_lock(
    d: Path, tmp_path: Path
) -> None:
    repo = init_repo(tmp_path / "repo")
    r = runner.invoke(
        app,
        ["eval", "--final", "--data-dir", str(d), "--manifest", str(d / "manifest.jsonl"),
         "--audit-file", str(d / "audit.jsonl"), "--repo-root", str(repo),
         "--out-dir", str(tmp_path / "final")],
    )  # fmt: skip
    assert r.exit_code == 2 and "needs real data" in r.output
    assert not (repo / "FINAL.lock.json").exists() and not (tmp_path / "final").exists()


def test_default_dev_output_dir_is_named_by_source_and_matrix_hash(
    d: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(
        app,
        ["eval", "--data-dir", str(d), "--manifest", str(d / "manifest.jsonl"),
         "--audit-file", str(d / "audit.jsonl"), "--resamples", str(B)],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    [made] = list((tmp_path / "results" / "dev").iterdir())
    assert made.name.startswith("synthetic-") and (made / "report.md").exists()
