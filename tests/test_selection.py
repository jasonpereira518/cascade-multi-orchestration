"""Task selection: audited-ok only, pilot/ablation/repeat draws are TRAIN only, deterministic."""

from __future__ import annotations

import datetime as dt

import pytest

from cascade.routing import selection as sel
from cascade.routing.selection import SelectionError
from cascade.taskgen.audit import AuditVerdict, make_verdict
from cascade.taskgen.schema import Task
from tests.matrix_helpers import tasks_with_splits

NOW = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


def ok(tasks: list[Task]) -> dict[str, AuditVerdict]:
    return {t.task_id: make_verdict(t.task_id, "ok", now=NOW) for t in tasks}


def pool(n_train: int = 18, n_test: int = 8) -> list[Task]:
    spec = {f"tr{i:02d}": ("train", 5 + i * 7) for i in range(n_train)}
    spec |= {f"te{i:02d}": ("test", 10 + i) for i in range(n_test)}
    return tasks_with_splits(spec)


def test_the_pilot_draws_train_tasks_only_even_when_asked_for_more_than_exist() -> None:
    tasks = pool()
    for n in (1, 5, 10, 18, 1000):
        drawn = sel.draw_train(tasks, ok(tasks), n, seed=0, purpose="pilot")
        assert drawn and all(t.split == "train" for t in drawn)
        assert not {t.task_id for t in drawn} & {t.task_id for t in tasks if t.split == "test"}
        assert len(drawn) == min(n, 18)


@pytest.mark.parametrize("purpose", ["pilot", "ablate", "repeat"])
@pytest.mark.parametrize("seed", range(6))
def test_no_purpose_and_no_seed_ever_reaches_the_test_split(purpose: str, seed: int) -> None:
    tasks = pool()
    drawn = sel.draw_train(tasks, ok(tasks), 12, seed=seed, purpose=purpose)
    assert {t.split for t in drawn} == {"train"}


def test_only_audited_ok_tasks_are_eligible() -> None:
    tasks = pool(6, 2)
    verdicts = ok(tasks)
    verdicts["tr00"] = make_verdict("tr00", "leaks_fix", now=NOW)
    verdicts["tr01"] = make_verdict("tr01", "underspecified", now=NOW)
    del verdicts["tr02"]  # never audited
    drawn = {t.task_id for t in sel.draw_train(tasks, verdicts, 100, seed=0, purpose="pilot")}
    assert drawn == {"tr03", "tr04", "tr05"}
    assert {t.task_id for t in sel.audited_ok(tasks, verdicts)} == drawn | {"te00", "te01"}


def test_an_audited_ok_task_without_a_split_is_an_error_not_a_silent_skip() -> None:
    tasks = pool(4, 1)
    tasks[0] = tasks[0].model_copy(update={"split": None})
    with pytest.raises(SelectionError, match="no split"):
        sel.draw_train(tasks, ok(tasks), 3, seed=0, purpose="pilot")


def test_nothing_to_draw_from_is_an_error() -> None:
    tasks = pool(0, 3)
    with pytest.raises(SelectionError, match="no audited-ok train tasks"):
        sel.draw_train(tasks, ok(tasks), 3, seed=0, purpose="pilot")
    tasks = pool(3, 3)
    with pytest.raises(SelectionError, match="no audited-ok train tasks"):
        sel.draw_train(tasks, {}, 3, seed=0, purpose="pilot")


def test_draws_are_deterministic_and_vary_with_seed_and_purpose() -> None:
    tasks = pool()
    v = ok(tasks)

    def ids(seed: int, purpose: str, source: list[Task] = tasks) -> list[str]:
        return [t.task_id for t in sel.draw_train(source, v, 6, seed=seed, purpose=purpose)]

    a = ids(0, "pilot")
    assert a == ids(0, "pilot")
    assert a == ids(0, "pilot", list(reversed(tasks)))
    assert a != ids(1, "pilot")
    assert a != ids(0, "ablate")


def test_a_small_sample_covers_every_repo_and_every_size_band() -> None:
    tasks = pool(18, 0)
    drawn = sel.draw_train(tasks, ok(tasks), 6, seed=0, purpose="pilot")
    assert {t.repo for t in drawn} == {"o/a", "o/b"}
    bands = sel.size_bands(tasks)
    assert {bands[t.task_id] for t in drawn} == {"small", "medium", "large"}


def test_size_bands_are_terciles_of_changed_source_lines() -> None:
    lines = [1, 2, 3, 4, 5, 6]
    tasks = tasks_with_splits({f"t{i}": ("train", n) for i, n in enumerate(lines)})
    bands = sel.size_bands(tasks)
    expected = ["small", "small", "medium", "medium", "large", "large"]
    assert [bands[f"t{i}"] for i in range(6)] == expected
    assert sel.size_bands([]) == {}


def test_sample_size_edges() -> None:
    tasks = pool(5, 0)
    assert sel.stratified_sample(tasks, 0, seed=0, purpose="x") == []
    assert len(sel.stratified_sample(tasks, 99, seed=0, purpose="x")) == 5
    with pytest.raises(SelectionError, match="negative"):
        sel.stratified_sample(tasks, -1, seed=0, purpose="x")


def test_task_major_order_is_a_fixed_shuffle_independent_of_input_order() -> None:
    tasks = pool()
    a = [t.task_id for t in sel.task_major_order(tasks, 0)]
    assert a == [t.task_id for t in sel.task_major_order(list(reversed(tasks)), 0)]
    assert sorted(a) == sorted(t.task_id for t in tasks)
    assert a != sorted(a)  # actually shuffled
    assert a != [t.task_id for t in sel.task_major_order(tasks, 1)]
