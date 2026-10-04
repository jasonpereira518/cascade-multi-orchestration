"""Manifest merge/finalize, store thread safety, and the repo ``enabled`` flag. Synthetic data."""

from __future__ import annotations

import datetime as dt
import json
import threading
from pathlib import Path

import pytest
import yaml

from cascade.config import load_repos
from cascade.taskgen import audit as au
from cascade.taskgen import split as sp
from cascade.taskgen.manifest import (
    finalize_manifest,
    merge_repo_tasks,
    read_manifest,
    write_tasks,
)
from cascade.taskgen.store import OK, StageStore
from tests.phase3_helpers import GOLD_PATCH, make_task, make_tasks

NOW = dt.datetime(2026, 3, 4, tzinfo=dt.UTC)


def ids(path: Path) -> list[str]:
    return [t.task_id for t in read_manifest(path)]


def test_missing_manifest_reads_as_empty(tmp_path: Path) -> None:
    assert read_manifest(tmp_path / "nope.jsonl") == []


def test_write_is_sorted_atomic_and_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "tasks" / "manifest.jsonl"
    tasks = [make_task("o/b", 2), make_task("o/a", 1), make_task("o/b", 1)]
    write_tasks(path, tasks)
    assert ids(path) == ["o__a-1", "o__b-1", "o__b-2"]
    assert {t.task_id: t for t in read_manifest(path)} == {t.task_id: t for t in tasks}
    assert not list(path.parent.glob("*.tmp"))


def test_merge_replaces_only_that_repos_lines(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    write_tasks(path, make_tasks({"o/a": 2, "o/b": 2}))
    merge_repo_tasks(path, [make_task("o/a", 9)], "o/a")
    assert ids(path) == ["o__a-9", "o__b-1", "o__b-2"]  # a's old lines gone, b untouched


def test_merge_into_a_missing_manifest_creates_it(tmp_path: Path) -> None:
    path = tmp_path / "new" / "m.jsonl"
    merge_repo_tasks(path, make_tasks({"o/a": 2}), "o/a")
    assert ids(path) == ["o__a-1", "o__a-2"]


def test_merge_keeps_the_split_and_audit_a_task_already_had(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    stamped = make_task("o/a", 1, split="test", audit={"status": "ok", "notes": "fine"})
    write_tasks(path, [stamped])
    merge_repo_tasks(path, [make_task("o/a", 1), make_task("o/a", 2)], "o/a")  # mine again
    by_id = {t.task_id: t for t in read_manifest(path)}
    assert by_id["o__a-1"].split == "test" and by_id["o__a-1"].audit.status == "ok"
    assert by_id["o__a-1"].audit.notes == "fine"
    assert by_id["o__a-2"].split is None and by_id["o__a-2"].audit.status == "pending"


def test_merge_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    merge_repo_tasks(path, make_tasks({"o/a": 2}), "o/a")
    first = path.read_text()
    merge_repo_tasks(path, make_tasks({"o/a": 2}), "o/a")
    assert path.read_text() == first


# --- finalize ----------------------------------------------------------------------------


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest, split, audit = tmp_path / "m.jsonl", tmp_path / "split.json", tmp_path / "a.jsonl"
    write_tasks(manifest, make_tasks({"o/a": 3, "o/b": 3, "o/c": 3}))
    sp.freeze({"o/a": 3, "o/b": 3, "o/c": 3}, path=split, salt="s", target_test_fraction=0.34)
    return manifest, split, audit


def test_finalize_stamps_split_audit_and_recomputed_features(tmp_path: Path) -> None:
    manifest, split, audit = _setup(tmp_path)
    au.append_verdict(audit, au.make_verdict("o__a-1", "ok", "clear", now=NOW))
    au.append_verdict(audit, au.make_verdict("o__a-2", "leaks_fix", now=NOW))

    result = finalize_manifest(manifest, split, audit)
    assert result.tasks == 9 and result.new_repos == []
    assert sum(result.by_split.values()) == 9 and set(result.by_split) <= {"train", "test"}
    assert result.by_audit == {"ok": 1, "leaks_fix": 1, "pending": 7}

    frozen = sp.read_split(split).assignments
    by_id = {t.task_id: t for t in read_manifest(manifest)}
    assert all(t.split == frozen[t.repo] for t in by_id.values())
    assert by_id["o__a-1"].audit.status == "ok" and by_id["o__a-1"].audit.notes == "clear"
    assert by_id["o__a-3"].audit.status == "pending"
    f = by_id["o__a-1"].features
    assert f.src_files == 1 and f.f2p_count == 1 and f.p2p_count == 1 and f.src_added == 1
    assert f.test_runtime_s == 1.5  # the one measured feature is preserved, not recomputed


def test_finalize_is_idempotent(tmp_path: Path) -> None:
    manifest, split, audit = _setup(tmp_path)
    finalize_manifest(manifest, split, audit)
    first = manifest.read_text()
    finalize_manifest(manifest, split, audit)
    assert manifest.read_text() == first


def test_finalize_follows_the_latest_verdict_and_can_demote_a_task(tmp_path: Path) -> None:
    manifest, split, audit = _setup(tmp_path)
    au.append_verdict(audit, au.make_verdict("o__a-1", "ok", now=NOW))
    finalize_manifest(manifest, split, audit)
    au.append_verdict(audit, au.make_verdict("o__a-1", "underspecified", "on reflection", now=NOW))
    finalize_manifest(manifest, split, audit)
    t = {t.task_id: t for t in read_manifest(manifest)}["o__a-1"]
    assert t.audit.status == "underspecified" and t.audit.notes == "on reflection"


def test_finalize_never_changes_an_existing_assignment(tmp_path: Path) -> None:
    manifest, split, audit = _setup(tmp_path)
    before = sp.read_split(split)
    finalize_manifest(manifest, split, audit)
    assert sp.read_split(split) == before


def test_new_repo_after_the_freeze_gets_a_recorded_hash_bucket_assignment(tmp_path: Path) -> None:
    manifest, split, audit = _setup(tmp_path)
    merge_repo_tasks(manifest, make_tasks({"o/late": 2}), "o/late")
    before = sp.read_split(split).assignments
    result = finalize_manifest(manifest, split, audit)
    assert result.new_repos == ["o/late"]
    after = sp.read_split(split)
    assert "o/late" in after.late_assignments
    assert {k: v for k, v in after.assignments.items() if k != "o/late"} == before
    side = after.assignments["o/late"]
    assert {t.split for t in read_manifest(manifest) if t.repo == "o/late"} == {side}


def test_without_assign_new_an_unassigned_repo_is_an_error(tmp_path: Path) -> None:
    manifest, split, audit = _setup(tmp_path)
    merge_repo_tasks(manifest, [make_task("o/late", 1)], "o/late")
    with pytest.raises(sp.SplitError, match="o/late"):
        finalize_manifest(manifest, split, audit, assign_new=False)
    assert "o/late" not in sp.read_split(split).assignments


def test_finalize_without_a_frozen_split_fails_and_writes_nothing(tmp_path: Path) -> None:
    manifest = tmp_path / "m.jsonl"
    write_tasks(manifest, make_tasks({"o/a": 1}))
    before = manifest.read_text()
    with pytest.raises(sp.SplitError, match="no split"):
        finalize_manifest(manifest, tmp_path / "split.json", tmp_path / "a.jsonl")
    assert manifest.read_text() == before


def test_features_are_recomputed_from_the_patches_not_trusted_from_the_file(tmp_path: Path) -> None:
    manifest, split, audit = _setup(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["features"]["f2p_count"] = 999  # a stale or hand-edited value
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    finalize_manifest(manifest, split, audit)
    assert all(t.features.f2p_count == 1 for t in read_manifest(manifest))


def test_old_manifest_rows_without_new_feature_fields_still_load(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    row = json.loads(make_task().model_dump_json())
    row["features"] = {"src_files": 1, "src_lines": 2, "issue_chars": 3, "test_runtime_s": 1.0}
    row.pop("split")
    row.pop("audit")
    path.write_text(json.dumps(row) + "\n")
    [t] = read_manifest(path)
    assert t.features.f2p_count == 0 and t.audit.status == "pending" and t.split is None
    assert t.gold_patch == GOLD_PATCH


# --- thread-safe store -------------------------------------------------------------------


def test_store_survives_many_threads_writing_and_reading(tmp_path: Path) -> None:
    store = StageStore(tmp_path / "state.sqlite")
    errors: list[BaseException] = []

    def work(offset: int) -> None:
        try:
            for i in range(60):
                n = offset * 1000 + i
                store.upsert_candidate(n, {"n": n})
                store.put_stage(n, "filter", OK, data={"n": n})
                assert store.get_stage(n, "filter") == (OK, None, {"n": n})
                store.passed("filter")
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert store.candidate_count() == 480 and store.count("filter", OK) == 480
    store.close()


# --- enabled flag ------------------------------------------------------------------------


def test_repos_are_enabled_by_default_and_the_flag_disables_them(tmp_path: Path) -> None:
    flags: dict[str, bool | None] = {"o/a": None, "o/b": False, "o/c": True}
    repos: list[dict[str, object]] = []
    for name, enabled in flags.items():
        spec: dict[str, object] = {
            "name": name, "license": "MIT", "python_version": "3.12.13",
            "install_cmd": "pip install -e .", "test_cmd": "pytest",
        }  # fmt: skip
        if enabled is not None:
            spec["enabled"] = enabled
        repos.append(spec)
    path = tmp_path / "repos.yaml"
    path.write_text(
        yaml.safe_dump(
            {"date_window": {"start": "2025-01-01", "end": "2025-02-01"}, "repos": repos}
        )
    )
    cfg = load_repos(path)
    assert cfg.enabled_repos() == ["o/a", "o/c"]
    assert cfg.resolve("o/b").name == "o/b"  # a disabled repo can still be named explicitly
