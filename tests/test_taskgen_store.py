from __future__ import annotations

import datetime as dt
from pathlib import Path

from cascade.taskgen.store import DROPPED, ENV, ERROR, FILTER, OK, PATCH, StageStore


def test_candidates_roundtrip_and_upsert_replaces() -> None:
    s = StageStore(":memory:")
    s.upsert_candidate(7, {"title": "old"})
    s.upsert_candidate(3, {"title": "three"})
    s.upsert_candidate(7, {"title": "new"})
    assert s.candidates() == [(3, {"title": "three"}), (7, {"title": "new"})]
    assert s.candidate_count() == 2


def test_ok_and_dropped_are_final_but_error_is_retried() -> None:
    s = StageStore(":memory:")
    assert not s.is_final(1, FILTER)
    s.put_stage(1, FILTER, OK)
    s.put_stage(2, FILTER, DROPPED, "bot_author")
    s.put_stage(3, FILTER, ERROR, "docker_hiccup", {"message": "boom"})
    assert s.is_final(1, FILTER) and s.is_final(2, FILTER)
    assert not s.is_final(3, FILTER)  # a rerun must retry transient errors
    s.put_stage(3, FILTER, OK)  # the retry overwrites the error row
    assert s.get_stage(3, FILTER) == (OK, None, {})


def test_stage_data_roundtrips_and_stages_are_independent() -> None:
    s = StageStore(":memory:")
    s.put_stage(5, PATCH, OK, data={"base_commit": "abc", "n": [1, 2]})
    s.put_stage(5, ENV, DROPPED, "env_collect_failed")
    assert s.get_stage(5, PATCH) == (OK, None, {"base_commit": "abc", "n": [1, 2]})
    assert s.get_stage(5, ENV) == (DROPPED, "env_collect_failed", {})
    assert s.get_stage(5, FILTER) is None


def test_counts_passed_and_drop_reasons() -> None:
    s = StageStore(":memory:")
    for n in (1, 2, 3):
        s.put_stage(n, FILTER, OK)
    for n in (4, 5, 6):
        s.put_stage(n, FILTER, DROPPED, "bot_author")
    s.put_stage(7, FILTER, DROPPED, "no_closing_issue")
    s.put_stage(8, FILTER, ERROR, "x")
    assert s.passed(FILTER) == [1, 2, 3]
    assert (
        s.count(FILTER, OK) == 3 and s.count(FILTER, DROPPED) == 4 and s.count(FILTER, ERROR) == 1
    )
    assert s.drop_reasons(FILTER) == {"bot_author": 3, "no_closing_issue": 1}  # most common first


def test_slices_and_kv() -> None:
    s = StageStore(":memory:")
    a, b = dt.date(2025, 1, 1), dt.date(2025, 1, 31)
    assert not s.slice_done(a, b)
    s.mark_slice_done(a, b)
    s.mark_slice_done(a, b)  # idempotent
    assert s.slice_done(a, b) and not s.slice_done(a, dt.date(2025, 1, 30))
    assert s.get_kv("k") is None
    s.set_kv("k", "v1")
    s.set_kv("k", "v2")
    assert s.get_kv("k") == "v2"


def test_state_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "state.sqlite"  # parent directory is created on demand
    s = StageStore(path)
    s.upsert_candidate(1, {"a": 1})
    s.put_stage(1, FILTER, OK)
    s.set_kv("x", "y")
    s.close()
    again = StageStore(path)
    assert again.candidates() == [(1, {"a": 1})]
    assert again.is_final(1, FILTER) and again.get_kv("x") == "y"
    again.close()
