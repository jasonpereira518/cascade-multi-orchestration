"""The frozen repo-level split: determinism, immutability, late repos, and the fitting guard.

All inputs are synthetic task counts; nothing here says anything about real repos.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cascade.taskgen import split as sp
from cascade.taskgen.split import SplitError
from tests.phase3_helpers import make_task, make_tasks

COUNTS = {"o/alpha": 10, "o/beta": 10, "o/gamma": 10, "o/delta": 10, "o/eps": 10}
NOW = dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.UTC)


def test_bucket_is_a_pure_function_of_salt_and_repo() -> None:
    expected = int(hashlib.sha256(b"salt:o/alpha").hexdigest()[:12], 16) / 16**12
    assert sp.bucket("o/alpha", "salt") == expected
    assert 0 <= sp.bucket("o/alpha", "salt") < 1
    assert sp.bucket("o/alpha", "salt") != sp.bucket("o/alpha", "other-salt")


def test_hash_order_ignores_input_order_and_uses_sha256() -> None:
    repos = list(COUNTS)
    assert sp.hash_order(repos, "s") == sp.hash_order(list(reversed(repos)), "s")
    digests = [hashlib.sha256(f"s:{r}".encode()).hexdigest() for r in sp.hash_order(repos, "s")]
    assert digests == sorted(digests)


def test_split_is_identical_across_processes_with_different_hash_seeds() -> None:
    """Python's hash() is randomized per process; the split must not depend on it."""
    code = (
        "import json;from cascade.taskgen import split as s;"
        f"print(json.dumps(s.choose_test_repos({COUNTS!r}, salt='s', target_test_fraction=0.4)))"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout
        for seed in ("0", "1", "12345")
    }
    assert len(outs) == 1


def test_test_repos_are_a_hash_order_prefix_closest_to_the_target() -> None:
    test = sp.choose_test_repos(COUNTS, salt="s", target_test_fraction=0.4)
    assert len(test) == 2  # 2 of 5 equal repos = exactly 40%
    assert test == sp.hash_order(list(COUNTS), "s")[:2]


def test_choice_does_not_depend_on_dict_order_or_on_outcomes() -> None:
    a = sp.choose_test_repos(COUNTS, salt="s", target_test_fraction=0.4)
    b = sp.choose_test_repos(dict(reversed(COUNTS.items())), salt="s", target_test_fraction=0.4)
    assert a == b


def test_uneven_counts_pick_the_prefix_whose_share_is_closest() -> None:
    order = sp.hash_order(["o/a", "o/b", "o/c"], "s")
    counts = {order[0]: 50, order[1]: 30, order[2]: 20}
    # prefixes: 50% (gap .10), 80% (gap .40) -> the first repo only
    assert sp.choose_test_repos(counts, salt="s", target_test_fraction=0.4) == [order[0]]
    counts = {order[0]: 5, order[1]: 30, order[2]: 65}
    # prefixes: 5% (gap .35), 35% (gap .05) -> two repos
    assert sp.choose_test_repos(counts, salt="s", target_test_fraction=0.4) == order[:2]


def test_at_least_one_repo_always_stays_in_train() -> None:
    order = sp.hash_order(["o/a", "o/b"], "s")
    test = sp.choose_test_repos({order[0]: 1, order[1]: 1}, salt="s", target_test_fraction=0.99)
    assert test == [order[0]]


def test_repos_with_no_tasks_are_ignored_and_one_repo_cannot_be_split() -> None:
    with pytest.raises(SplitError, match="at least two repos"):
        sp.choose_test_repos({"o/a": 5, "o/b": 0}, salt="s", target_test_fraction=0.4)
    with pytest.raises(SplitError, match="at least two repos"):
        sp.choose_test_repos({}, salt="s", target_test_fraction=0.4)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_target_fraction_must_be_strictly_between_zero_and_one(bad: float) -> None:
    with pytest.raises(SplitError, match="strictly between"):
        sp.choose_test_repos(COUNTS, salt="s", target_test_fraction=bad)


def test_freeze_writes_the_assignments_and_everything_needed_to_audit_them(tmp_path: Path) -> None:
    path = tmp_path / "tasks" / "split.json"
    s = sp.freeze(COUNTS, path=path, salt="s", target_test_fraction=0.4, now=NOW)
    on_disk = json.loads(path.read_text())
    assert on_disk["assignments"] == s.assignments
    assert on_disk["salt"] == "s" and on_disk["counts_at_freeze"] == COUNTS
    assert on_disk["frozen_at"] == NOW.isoformat() and on_disk["late_assignments"] == {}
    assert sorted(s.assignments) == sorted(COUNTS)
    assert sum(1 for v in s.assignments.values() if v == "test") == 2
    assert not list(path.parent.glob("*.tmp"))  # atomic write leaves no temp file


def test_freeze_refuses_to_overwrite_and_leaves_the_file_untouched(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    sp.freeze(COUNTS, path=path, salt="s")
    before = path.read_text()
    with pytest.raises(SplitError, match="frozen"):
        sp.freeze(COUNTS, path=path, salt="different")
    assert path.read_text() == before


def test_force_overwrites_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    sp.freeze(COUNTS, path=path, salt="s")
    again = sp.freeze(COUNTS, path=path, salt="other", force=True)
    assert sp.read_split(path).salt == "other" == again.salt


def test_read_split_roundtrips_and_reports_problems(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    with pytest.raises(SplitError, match="no split"):
        sp.read_split(path)
    frozen = sp.freeze(COUNTS, path=path)
    assert sp.read_split(path) == frozen
    path.write_text("{not json")
    with pytest.raises(SplitError, match="unreadable"):
        sp.read_split(path)
    path.write_text(json.dumps({**frozen.model_dump(), "version": 99}))
    with pytest.raises(SplitError, match="version 99"):
        sp.read_split(path)


def test_freeze_is_deterministic_for_the_same_inputs(tmp_path: Path) -> None:
    a = sp.freeze(COUNTS, path=tmp_path / "a.json", now=NOW)
    b = sp.freeze(COUNTS, path=tmp_path / "b.json", now=NOW)
    assert (tmp_path / "a.json").read_text() == (tmp_path / "b.json").read_text()
    assert a == b


def _frozen(tmp_path: Path) -> sp.SplitFile:
    return sp.freeze(COUNTS, path=tmp_path / "split.json", salt="s", now=NOW)


def test_assign_new_repo_appends_only_and_is_idempotent(tmp_path: Path) -> None:
    s = _frozen(tmp_path)
    s2 = sp.assign_new_repo(s, "o/late", now=NOW)
    assert {k: v for k, v in s2.assignments.items() if k != "o/late"} == s.assignments
    assert s2.late_assignments == {"o/late": NOW.isoformat()}
    expected = "test" if sp.bucket("o/late", "s") < s.target_test_fraction else "train"
    assert s2.assignments["o/late"] == expected
    assert sp.assign_new_repo(s2, "o/late", now=dt.datetime(2030, 1, 1, tzinfo=dt.UTC)) == s2
    assert sp.assign_new_repo(s, "o/alpha") == s  # already assigned: never reassigned


def test_late_assignment_is_independent_of_task_counts_and_other_repos(tmp_path: Path) -> None:
    s = _frozen(tmp_path)
    sides = {sp.assign_new_repo(s, f"o/late{i}").assignments[f"o/late{i}"] for i in range(60)}
    assert sides == {"train", "test"}  # both occur; it is a hash bucket, not a constant
    frac = (
        sum(sp.assign_new_repo(s, f"o/x{i}").assignments[f"o/x{i}"] == "test" for i in range(400))
        / 400
    )
    assert 0.3 < frac < 0.5  # bucket < 0.4, so about 40% land in test


def test_apply_split_stamps_tasks_and_refuses_unassigned_repos(tmp_path: Path) -> None:
    s = _frozen(tmp_path)
    tasks = make_tasks({"o/alpha": 2, "o/beta": 1})
    out = sp.apply_split(tasks, s)
    assert [t.split for t in out] == [s.assignments["o/alpha"]] * 2 + [s.assignments["o/beta"]]
    assert all(t.split is None for t in tasks)  # inputs are not mutated
    with pytest.raises(SplitError, match="o/unknown"):
        sp.apply_split([make_task("o/unknown")], s)


def test_whole_repos_are_never_split_across_train_and_test(tmp_path: Path) -> None:
    s = _frozen(tmp_path)
    out = sp.apply_split(make_tasks(COUNTS), s)
    by_repo: dict[str, set[str | None]] = {}
    for t in out:
        by_repo.setdefault(t.repo, set()).add(t.split)
    assert all(len(sides) == 1 for sides in by_repo.values())


def test_fitting_tasks_returns_train_only(tmp_path: Path) -> None:
    s = _frozen(tmp_path)
    out = sp.apply_split(make_tasks(COUNTS), s)
    fit = sp.fitting_tasks(out)
    assert fit and all(t.split == "train" for t in fit)
    assert len(fit) == sum(1 for t in out if t.split == "train")
    assert not {t.task_id for t in fit} & {t.task_id for t in out if t.split == "test"}


def test_fitting_tasks_refuses_a_manifest_with_any_unsplit_task(tmp_path: Path) -> None:
    s = _frozen(tmp_path)
    tasks = sp.apply_split(make_tasks({"o/alpha": 2}), s)
    tasks.append(make_task("o/beta", 1))  # split is None
    with pytest.raises(SplitError, match="no split assigned"):
        sp.fitting_tasks(tasks)
    with pytest.raises(SplitError, match="1 task"):
        sp.fitting_tasks([make_task("o/a", 1)])
    assert sp.fitting_tasks([]) == []
