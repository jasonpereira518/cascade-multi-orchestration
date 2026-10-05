"""The labeled-synthetic generator: reproducible by seed, structured, and unmistakably synthetic."""

from __future__ import annotations

import json
from pathlib import Path

from cascade.agents.pipeline import PipelineConfig
from cascade.routing.matrix import MatrixRow
from tests.synthetic import gen


def test_the_generator_lives_only_under_tests_synthetic() -> None:
    path = Path(gen.__file__).resolve()
    assert path.parent.name == "synthetic" and path.parent.parent.name == "tests"


def test_the_same_seed_gives_identical_data_and_another_seed_gives_different_data() -> None:
    a, b, c = gen.generate(0), gen.generate(0), gen.generate(1)
    assert [r.model_dump() for r in a.rows] == [r.model_dump() for r in b.rows]
    assert [t.model_dump() for t in a.tasks] == [t.model_dump() for t in b.tasks]
    assert [r.model_dump() for r in a.rows] != [r.model_dump() for r in c.rows]


def test_the_structure_the_eval_code_relies_on_is_present() -> None:
    d = gen.generate(0)
    assert len(d.tasks) == sum(gen.REPOS.values())
    assert {t.split for t in d.tasks} == {"train", "test"}
    assert {t.repo for t in d.tasks if t.split == "test"} == gen.TEST_REPOS  # whole repos held out
    default = PipelineConfig().canonical()
    full = [r for r in d.rows if r.pipeline_config == default and r.seed == 0]
    assert len(full) == 3 * len(d.tasks)  # every task has all three arms
    assert {r.arm_id for r in d.rows} == set(gen.ARM_IDS.values())
    assert {r.processing_tier for r in d.rows} == {"standard"}
    abl = [r for r in d.rows if r.pipeline_config != default]
    assert len(abl) == 32 and {r.arm_id for r in abl} == {gen.ARM_IDS["mid"]}  # 16 tasks x 2
    rep = [r for r in d.rows if r.seed == 1]
    assert len(rep) == 24 and {r.arm_id for r in rep} == {gen.ARM_IDS["cheap"], gen.ARM_IDS["mid"]}
    train_ids = {t.task_id for t in d.tasks if t.split == "train"}
    assert {r.task_id for r in abl} <= train_ids and {r.task_id for r in rep} <= train_ids


def test_some_cap_exceeded_cells_look_like_successes_so_normalization_has_work_to_do() -> None:
    d = gen.generate(0)
    capped = [r for r in d.rows if r.error and "attempt cap" in r.error]
    assert capped and any(r.true_success or r.verifier_claimed for r in capped)


def test_costs_rise_with_the_arm_and_the_verifier_is_imperfect() -> None:
    d = gen.generate(0)
    default = PipelineConfig().canonical()
    by_arm: dict[str, list[MatrixRow]] = {}
    for r in d.rows:
        if r.pipeline_config == default and r.seed == 0:
            by_arm.setdefault(r.arm_id, []).append(r)
    mean = {a: sum(x.cost_usd for x in rs) / len(rs) for a, rs in by_arm.items()}
    assert mean[gen.ARM_IDS["cheap"]] < mean[gen.ARM_IDS["mid"]] < mean[gen.ARM_IDS["max"]]
    rows = [x for rs in by_arm.values() for x in rs]
    assert any(x.verifier_claimed and not x.true_success for x in rows)  # false positives exist
    assert any(x.true_success and not x.verifier_claimed for x in rows)  # false negatives exist


def test_write_lays_out_the_files_and_marks_the_data_synthetic(tmp_path: Path) -> None:
    out = gen.write(gen.generate(3), tmp_path / "d")
    for name in ("manifest.jsonl", "audit.jsonl", "matrix.parquet", "arms.json", "SYNTHETIC.json"):
        assert (out / name).exists(), name
    marker = json.loads((out / "SYNTHETIC.json").read_text())
    assert marker["data_source"] == "synthetic" and marker["seed"] == 3
    assert json.loads((out / "arms.json").read_text()) == gen.ARM_IDS
    assert not (out / "matrix.sqlite").exists()
