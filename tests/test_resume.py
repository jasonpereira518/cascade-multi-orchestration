"""Resume stats, bullets, and the generated README/write-up blocks. All SYNTHETIC.

The stats tests run the real final-evaluation guard on the labeled-synthetic matrix (the test-only
``allow_synthetic`` path) inside a throwaway git repo. The bullet tests use hand-built stats so each
rule (gating, rounding direction, wording flip, limits) can be pinned with exact numbers.
"""

# pyright: basic
from __future__ import annotations

import copy
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from cascade.cli import app
from cascade.evals.final import run_final
from cascade.resume import bullets as bl
from cascade.resume import render as rd
from cascade.resume.stats import StatsError, StatsInputs, build_stats, load_stats, write_stats
from tests.conftest import SYNTHETIC
from tests.eval_helpers import init_repo, write_synthetic
from tests.synthetic import gen

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = REPO_ROOT / "docs/resume_templates.yaml"

LEDGER_ROWS = [  # purpose, input, output, reasoning, cost
    ("experiment", 1000, 200, 50, 1.25),
    ("experiment", 2000, 400, 0, 2.50),
    ("live_pilot", 500, 100, 25, 0.75),
]
REVIEWS = [
    {"repo": "o/a", "issue": 1, "outcome": "accepted_as_is", "lines_edited": 0,
     "merged_upstream": True, "arms_tried": ["m@low"], "cost_usd": 0.5, "notes": "",
     "reviewed_at": "2026-01-01T00:00:00+00:00"},
    {"repo": "o/b", "issue": 2, "outcome": "minor_edits", "lines_edited": 6,
     "merged_upstream": None, "arms_tried": ["m@low", "m@high"], "cost_usd": 1.5, "notes": "",
     "reviewed_at": "2026-01-01T00:00:00+00:00"},
    {"repo": "o/b", "issue": 3, "outcome": "rejected", "lines_edited": None,
     "merged_upstream": None, "arms_tried": ["m@low"], "cost_usd": 1.0, "notes": "",
     "reviewed_at": "2026-01-01T00:00:00+00:00"},
]  # fmt: skip

REPOS_YAML = """\
date_window: {start: 2026-03-01, end: 2026-09-30}
repos:
  - {name: syn/a, license: MIT, python_version: "3.12", install_cmd: x, test_cmd: y}
"""


def write_ledger(data: Path, *, purpose_column: bool = True) -> None:
    cols: dict[str, Any] = {
        "input_tokens": pa.array([r[1] for r in LEDGER_ROWS], pa.int64()),
        "output_tokens": pa.array([r[2] for r in LEDGER_ROWS], pa.int64()),
        "reasoning_tokens": pa.array([r[3] for r in LEDGER_ROWS], pa.int64()),
        "cost_usd": pa.array([r[4] for r in LEDGER_ROWS], pa.float64()),
    }
    if purpose_column:
        cols["purpose"] = pa.array([r[0] for r in LEDGER_ROWS], pa.string())
    pq.write_table(pa.table(cols), data / "ledger.parquet")


class World:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.data = write_synthetic(tmp / "data", 0)
        write_ledger(self.data)
        self.repo = init_repo(tmp / "repo")
        self.final = tmp / "final"
        run_final(
            repo_root=self.repo, data_dir=self.data, manifest=self.data / "manifest.jsonl",
            audit_file=self.data / "audit.jsonl", arms=gen.ARM_IDS, out_dir=self.final,
            allow_synthetic=True, resamples=300,
        )  # fmt: skip
        self.configs = tmp / "configs"
        self.configs.mkdir()
        shutil.copy(SYNTHETIC / "models_matrix.yaml", self.configs / "models.yaml")
        (self.configs / "repos.yaml").write_text(REPOS_YAML)
        self.reviews = tmp / "reviews.jsonl"
        self.reviews.write_text("".join(json.dumps(r) + "\n" for r in REVIEWS))
        self.stats_path = tmp / "artifacts" / "resume_stats.json"

    @property
    def inputs(self) -> StatsInputs:
        return StatsInputs(
            final_dir=self.final, lock_path=self.repo / "FINAL.lock.json", data_dir=self.data,
            manifest=self.data / "manifest.jsonl", audit_file=self.data / "audit.jsonl",
            reviews_file=self.reviews, dev_dir=self.tmp / "dev", configs_dir=self.configs,
        )  # fmt: skip

    def stats(self) -> dict[str, Any]:
        return build_stats(self.inputs, allow_synthetic=True)

    def cli(self, *args: str) -> Any:
        i = self.inputs
        return runner.invoke(
            app,
            ["resume-stats", "--out", str(self.stats_path), "--final-dir", str(i.final_dir),
             "--lock-path", str(i.lock_path), "--data-dir", str(i.data_dir),
             "--manifest", str(i.manifest), "--audit-file", str(i.audit_file),
             "--reviews-file", str(i.reviews_file), "--dev-dir", str(i.dev_dir),
             "--configs-dir", str(i.configs_dir), *args],
        )  # fmt: skip


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def results_json(w: World) -> dict[str, Any]:
    return json.loads((w.final / "results.json").read_text())


# --- resume-stats -------------------------------------------------------------------------------


def test_every_stat_is_read_from_an_artifact(world: World) -> None:
    s = world.stats()
    raw = results_json(world)
    n_rows = pq.ParquetFile(world.data / "matrix.parquet").metadata.num_rows
    assert s["data_source"] == "synthetic" and s["synthetic_warning"]
    assert s["tasks"]["audited"] == {"train": 36, "test": 17, "total": 53}
    assert s["tasks"]["evaluated"] == {"train": 36, "test": 17, "total": 53}
    assert s["tasks"]["all_audited_tasks_evaluated"] and s["tasks"]["repos"]["n"] == 5
    assert s["arms"] == {"n": 3, "ids": gen.ARM_IDS}
    assert s["attempts"]["n_matrix_rows"] == n_rows
    assert s["tokens"] == {"input": 3500, "output": 700, "reasoning": 75, "total": 4275}
    assert s["spend_usd"] == {"experiment": 3.75, "live_pilot": 0.75, "total": 4.5}
    assert s["final"]["margin"] == raw["margin"] and s["final"]["n_test_tasks"] == raw["n_tasks"]
    metrics = {m["name"]: m for m in raw["metrics"]}
    for name, p in s["policies"].items():
        assert p["resolve_rate"]["point"] == metrics[name]["resolve_rate"]["point"]
        assert p["cost_per_task_usd"]["lo"] == metrics[name]["cost_per_task"]["lo"]
    comp = {c["policy"]: c for c in raw["comparisons"]}
    for name, c in s["comparisons"].items():
        assert c["cost_per_task_reduction"]["point"] == comp[name]["cost_reduction"]["point"]
        assert c["verdict"] == comp[name]["verdict"]
        assert c["oracle_savings_share"]["point"] == comp[name]["oracle_savings_share"]["point"]


def test_cost_per_resolved_reduction_is_computed_from_the_point_estimates(world: World) -> None:
    s = world.stats()
    metrics = {m["name"]: m for m in results_json(world)["metrics"]}
    base = metrics["always_max"]["cost_per_resolved"]["point"]
    for name, c in s["comparisons"].items():
        own = metrics[name]["cost_per_resolved"]["point"]
        want = None if own is None else 1.0 - own / base
        assert c["cost_per_resolved_reduction_point"] == pytest.approx(want)


def test_the_verifier_rate_is_pooled_from_the_per_arm_counts(world: World) -> None:
    rows = results_json(world)["verifier"]
    failed = sum(r["n"] - r["passed"] for r in rows)
    fps = sum(r["false_pos"] for r in rows)
    pooled = world.stats()["verifier"]["pooled"]
    assert (pooled["n_failed_attempts"], pooled["false_positives"]) == (failed, fps)
    assert pooled["fp_rate"]["point"] == pytest.approx(fps / failed)
    assert pooled["fp_rate"]["lo"] <= pooled["fp_rate"]["point"] <= pooled["fp_rate"]["hi"]


def test_pilot_counts_come_from_the_review_log_latest_row_per_draft(world: World) -> None:
    p = world.stats()["pilot"]
    assert (p["n_reviewed"], p["n_repos"], p["n_accepted"]) == (3, 2, 2)
    assert p["merge_rate"] == 1.0 and p["n_merge_known"] == 1
    assert p["median_lines_edited"] == 3.0 and p["total_cost_usd"] == 3.0
    again = dict(REVIEWS[1], merged_upstream=False)  # a later row for o/b#2 supersedes
    with world.reviews.open("a") as f:
        f.write(json.dumps(again) + "\n")
    p2 = world.stats()["pilot"]
    assert p2["n_reviewed"] == 3 and p2["n_merge_known"] == 2 and p2["merge_rate"] == 0.5


def test_an_empty_review_log_means_a_pilot_of_zero(world: World) -> None:
    world.reviews.write_text("")
    p = world.stats()["pilot"]
    assert p["n_reviewed"] == 0 and p["accept_rate"] is None


def test_stats_are_byte_identical_when_regenerated_and_strict_json(world: World) -> None:
    write_stats(world.stats(), world.stats_path)
    first = world.stats_path.read_bytes()
    write_stats(world.stats(), world.stats_path)
    assert world.stats_path.read_bytes() == first

    def refuse(token: str) -> Any:
        raise AssertionError(f"non-standard JSON constant {token}")

    json.loads(first, parse_constant=refuse)
    assert load_stats(world.stats_path)["schema"] == 1


def test_the_inputs_hashes_are_recorded(world: World) -> None:
    hashes = world.stats()["inputs_sha256"]
    assert set(hashes) == {
        "final_results", "final_lock", "matrix", "ledger", "manifest", "audit", "pilot_reviews"
    }  # fmt: skip
    assert all(len(h) == 64 for h in hashes.values())


def test_synthetic_data_is_refused_without_the_test_only_flag(world: World) -> None:
    with pytest.raises(StatsError, match="not evidence"):
        build_stats(world.inputs)
    r = world.cli()
    assert r.exit_code == 2 and "not evidence" in r.output and not world.stats_path.exists()


@pytest.mark.parametrize(
    ("victim", "match"),
    [
        ("final", "has not been run"),
        ("lock", "has not been run"),
        ("ledger", "the ledger export"),
        ("matrix", "the matrix export"),
        ("reviews", "the pilot review log"),
        ("manifest", "the task manifest"),
        ("audit", "the audit log"),
    ],
)
def test_a_missing_input_fails_loudly_and_names_it(world: World, victim: str, match: str) -> None:
    paths = {
        "final": world.final / "results.json", "lock": world.repo / "FINAL.lock.json",
        "ledger": world.data / "ledger.parquet", "matrix": world.data / "matrix.parquet",
        "reviews": world.reviews, "manifest": world.data / "manifest.jsonl",
        "audit": world.data / "audit.jsonl",
    }  # fmt: skip
    paths[victim].unlink()
    with pytest.raises(StatsError, match=match):
        world.stats()
    r = world.cli("--allow-synthetic")
    assert r.exit_code == 2 and match in r.output and not world.stats_path.exists()


def test_a_final_result_edited_after_the_lock_is_refused(world: World) -> None:
    p = world.final / "results.json"
    p.write_text(p.read_text().replace('"margin"', '"margin" ', 1))
    with pytest.raises(StatsError, match="does not match the hash"):
        world.stats()


def test_a_matrix_changed_after_the_final_evaluation_is_refused(world: World) -> None:
    table = pq.read_table(world.data / "matrix.parquet")
    pq.write_table(table.slice(0, table.num_rows - 1), world.data / "matrix.parquet")
    with pytest.raises(StatsError, match="changed since the final evaluation"):
        world.stats()


def test_a_manifest_edited_after_the_final_evaluation_is_refused(world: World) -> None:
    lines = (world.data / "audit.jsonl").read_text().splitlines()
    manifest = [
        json.loads(x) for x in (world.data / "manifest.jsonl").read_text().splitlines() if x
    ]
    victim = sorted(t["task_id"] for t in manifest if t["split"] == "test")[0]
    keep = [ln for ln in lines if json.loads(ln)["task_id"] != victim]
    (world.data / "audit.jsonl").write_text("\n".join(keep) + "\n")
    with pytest.raises(StatsError, match="changed after the final evaluation"):
        world.stats()


def test_a_ledger_export_without_a_purpose_column_counts_everything_as_experiment(
    world: World,
) -> None:
    write_ledger(world.data, purpose_column=False)
    s = world.stats()
    assert s["spend_usd"] == {"experiment": 4.5, "live_pilot": 0.0, "total": 4.5}
    assert s["ledger"]["purpose_column_present"] is False


def test_the_noise_estimate_comes_from_the_matching_dev_analysis_or_is_absent(
    world: World,
) -> None:
    s = world.stats()
    assert s["noise"] is None
    dev = world.tmp / "dev" / f"{s['data_source']}-{s['final']['matrix_hash'][:8]}"
    dev.mkdir(parents=True)
    noise = [
        {"arm": "mock-luna@low", "n_pairs": 10, "flips": 2, "claimed_flip_rate": 0.1,
         "flip_rate": {"point": 0.2, "lo": 0.05, "hi": 0.5, "n_undefined": 0}}
    ]  # fmt: skip
    (dev / "results.json").write_text(json.dumps({"noise": noise}))
    assert world.stats()["noise"] == noise


def test_the_cli_writes_the_file_with_the_test_only_flag(world: World) -> None:
    r = world.cli("--allow-synthetic")
    assert r.exit_code == 0, r.output
    assert load_stats(world.stats_path)["data_source"] == "synthetic"


# --- bullets: rounding ---------------------------------------------------------------------------


def test_floor_and_ceil_never_round_in_the_wrong_direction() -> None:
    assert bl.floor_to(0.4129, 1, 100) == "41.2" and bl.floor_to(0.29, 1, 100) == "29.0"
    assert bl.ceil_to(0.4121, 1, 100) == "41.3" and bl.ceil_to(0.29, 1, 100) == "29.0"
    assert bl.ceil_to(37.501, 2) == "37.51" and bl.ceil_to(37.5, 2) == "37.50"
    assert bl.floor_to(-0.0333, 1, 100) == "-3.4" and bl.ceil_to(-0.0333, 1, 100) == "-3.3"
    rng = random.Random(1)  # noqa: S311
    for _ in range(500):
        x = rng.random()
        assert float(bl.floor_to(x, 1, 100)) <= 100 * x + 1e-12
        assert float(bl.ceil_to(x, 1, 100)) >= 100 * x - 1e-12
        assert float(bl.floor_to(x, 1, 100)) > 100 * x - 0.1 - 1e-9


# --- bullets: rendering --------------------------------------------------------------------------


def iv(point: float | None) -> dict[str, Any] | None:
    return None if point is None else {"point": point, "lo": point - 0.1, "hi": point + 0.1}


def base_stats(**over: Any) -> dict[str, Any]:
    s: dict[str, Any] = {
        "schema": 1,
        "data_source": "real", "designated_policy": "cascade", "claim_supported": True,
        "final": {"margin": 0.15},
        "tasks": {"evaluated": {"total": 58}, "repos": {"n": 5}},
        "arms": {"n": 3}, "attempts": {"n_matrix_rows": 412},
        "spend_usd": {"experiment": 37.501},
        "verifier": {"pooled": {"fp_rate": {"point": 0.2131}}},
        "pilot": {"n_reviewed": 7, "n_repos": 3, "n_accepted": 4},
        "comparisons": {
            "cascade": {
                "cost_per_resolved_reduction_point": 0.4129, "oracle_savings_share": iv(0.5551),
                "resolve_rate_diff": iv(-0.0333), "verdict": "MET",
            },
            "predictive_issue": {
                "cost_per_resolved_reduction_point": 0.1, "oracle_savings_share": iv(0.2),
                "resolve_rate_diff": iv(0.0), "verdict": "INCONCLUSIVE",
            },
        },
    }  # fmt: skip
    s.update(over)
    return s


def render(stats: dict[str, Any], **kw: Any) -> dict[str, bl.Bullet]:
    out = bl.render_bullets(stats, bl.load_templates(TEMPLATES), **kw)
    return {b.name: b for b in out}


def test_the_template_file_holds_the_briefs_templates_verbatim() -> None:
    loaded = bl.load_templates(TEMPLATES)
    t = {x.name: x.text for x in loaded}
    assert t["pipeline"] == (
        "Built a 6-agent pipeline (triage, reproduce, localize, patch, review, verify) with OpenAI "
        "Agents SDK that turns GitHub issues into test-verified PRs, benchmarked on {n_tasks} "
        "post-cutoff hidden-test tasks from {n_repos} repos in sandboxed Docker"
    )
    assert t["router_supported"] == (
        "Designed a cost-aware router (cascade and learned policies) choosing model and reasoning "
        "effort per issue, cutting cost per resolved issue {cost_cut_pct}% vs an always-max-effort "
        "baseline at equal resolve rate, capturing {oracle_share_pct}% of oracle savings"
    )
    assert t["router_neutral"] == (
        "Evaluated cost-aware routing across {n_arms} model and effort arms on {n_tasks} "
        "post-cutoff issues, mapping the cost-accuracy frontier with paired bootstrap CIs; cascade "
        "routing cut cost {cost_cut_pct}% vs always-max at a {rr_gap_pts}-point lower resolve rate"
    )
    assert t["matrix"] == (
        "Cached a {n_attempts}-attempt outcome matrix to replay routing policies offline, scored "
        "with bootstrap CIs on ${spend_usd} of API spend"
    )
    assert t["verifier"] == (
        "Isolated an LLM-free test Verifier from the patcher, measuring a {verifier_fp_pct}% "
        "false-positive rate that bounds cascade savings"
    )
    assert t["leakage"] == (
        "Blocked eval leakage via network-off Docker sandboxes, truncated git history, and a "
        "frozen repo-level train/test split"
    )
    assert t["pilot"] == (
        "Piloted on {n_pilot} real issues across {n_pilot_repos} repos with human review: "
        "{n_accepted} AI-drafted PRs accepted as-is or with minor edits"
    )
    two = [x.name for x in loaded if x.kind == "two_line"]
    assert two == ["pipeline", "router_supported", "router_neutral"]
    assert len(loaded) == 7


def test_a_supported_claim_uses_router_supported_with_truncated_favorable_numbers() -> None:
    b = render(base_stats())
    text = b["router_supported"].text
    assert text is not None and "cutting cost per resolved issue 41.2%" in text
    assert "capturing 55.5% of oracle savings" in text  # 41.29 and 55.51 are truncated, not rounded
    assert b["router_neutral"].text is None
    assert "router_supported applies" in b["router_neutral"].reason
    assert any("headline MET" in n for n in b["router_supported"].notes)


def test_an_unsupported_claim_uses_router_neutral_with_the_gap_rounded_up() -> None:
    b = render(base_stats(claim_supported=False, designated_policy="always_max"))
    assert b["router_supported"].text is None and "does not support" in b["router_supported"].reason
    text = b["router_neutral"].text
    assert text is not None
    assert "cut cost 41.2% vs always-max at a 3.4-point lower resolve rate" in text  # 3.33 -> 3.4
    assert "across 3 model and effort arms on 58 post-cutoff issues" in text


def test_the_neutral_wording_flips_when_cascade_resolved_more() -> None:
    s = base_stats(claim_supported=False, designated_policy="always_max")
    s["comparisons"]["cascade"]["resolve_rate_diff"] = iv(0.0349)
    text = render(s)["router_neutral"].text
    assert text is not None and "at a 3.4-point higher resolve rate" in text  # favorable: truncated
    s["comparisons"]["cascade"]["resolve_rate_diff"] = iv(0.0)
    text = render(s)["router_neutral"].text
    assert text is not None and "at the same resolve rate" in text and "lower" not in text


def test_a_cascade_that_did_not_cut_cost_does_not_get_a_bullet() -> None:
    s = base_stats(claim_supported=False, designated_policy="always_max")
    s["comparisons"]["cascade"]["cost_per_resolved_reduction_point"] = -0.05
    b = render(s)["router_neutral"]
    assert b.text is None and "did not cut cost" in b.reason
    s2 = base_stats()
    s2["comparisons"]["cascade"]["cost_per_resolved_reduction_point"] = 0.0
    assert render(s2)["router_supported"].text is None


def test_a_template_renders_only_if_every_metric_exists() -> None:
    s = base_stats()
    s["tasks"]["repos"]["n"] = None
    s["verifier"]["pooled"] = None
    s["comparisons"]["cascade"]["oracle_savings_share"] = None
    b = render(s)
    assert b["pipeline"].text is None and "n_repos" in b["pipeline"].reason
    assert b["verifier"].text is None and "verifier_fp_pct" in b["verifier"].reason
    assert b["router_supported"].text is None and "oracle_share_pct" in b["router_supported"].reason
    assert b["matrix"].text is not None and b["leakage"].text is not None  # unaffected


def test_pilot_needs_an_accepted_draft_and_the_other_numbers_round_the_unflattering_way() -> None:
    s = base_stats()
    s["pilot"]["n_accepted"] = 0
    assert render(s)["pilot"].text is None
    b = render(base_stats())
    assert b["matrix"].text is not None and "412-attempt" in b["matrix"].text
    assert "on $37.51 of API spend" in b["matrix"].text  # spend rounded UP to the cent
    assert b["verifier"].text is not None and "21.4%" in b["verifier"].text  # 21.31 rounded UP
    assert b["pilot"].text is not None and "7 real issues across 3 repos" in b["pilot"].text
    assert "4 AI-drafted PRs" in b["pilot"].text


def test_character_counts_and_flags_follow_the_limits() -> None:
    templates = [
        bl.Template("short", "one_line", "x" * 100),
        bl.Template("exact", "one_line", "y" * 119),
        bl.Template("long", "one_line", "z" * 120),
        bl.Template("slack", "one_line", "w" * 104),
        bl.Template("under", "one_line", "v" * 103),
        bl.Template("t_ok", "two_line", "a" * 239),
        bl.Template("t_long", "two_line", "a" * 240),
        bl.Template("t_under", "two_line", "a" * 223),
    ]
    out = {b.name: b for b in bl.render_bullets(base_stats(), templates)}
    assert out["exact"].chars == 119 and out["exact"].flags == []
    assert out["long"].flags == ["OVER the 119-character limit by 1"]
    assert out["slack"].flags == []  # exactly 15 under is allowed
    assert out["under"].flags and out["under"].flags[0].startswith("UNDER the 119")
    assert out["short"].flags
    assert out["t_ok"].flags == [] and out["t_long"].flags[0].startswith("OVER the 239")
    assert out["t_under"].flags[0].startswith("UNDER the 239")


def test_wording_is_never_edited_to_fit() -> None:
    long_text = "Built " + "very " * 60 + "much"
    [b] = bl.render_bullets(base_stats(), [bl.Template("x", "two_line", long_text)])
    assert b.text == long_text and b.flags


def test_synthetic_stats_are_refused_unless_allowed_and_then_marked() -> None:
    s = base_stats(data_source="synthetic")
    with pytest.raises(bl.BulletsError, match="real data only"):
        render(s)
    b = render(s, allow_synthetic=True)
    assert all(x.text.startswith("[SYNTHETIC] ") for x in b.values() if x.text)
    assert "SYNTHETIC DATA" in bl.format_markdown(list(b.values()), s)


def test_the_markdown_lists_each_count_flag_and_reason() -> None:
    s = base_stats(claim_supported=False, designated_policy="always_max")
    bullets = list(render(s).values())
    md = bl.format_markdown(bullets, s)
    for b in bullets:
        assert f"## {b.name}" in md
        if b.text is not None:
            assert f"**{len(b.text)} characters**" in md
    assert "**FLAG: " in md and "not used" in md and "Check the wording" in md


def test_the_real_templates_rendered_report_their_true_lengths() -> None:
    for b in render(base_stats()).values():
        if b.text is not None:
            assert b.chars == len(b.text) and b.chars is not None
            assert bool(b.flags) == (b.chars > b.limit or b.chars < b.limit - 15)


def test_the_bullets_cli_writes_the_file_and_prints_flags(tmp_path: Path) -> None:
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps(base_stats()))
    out = tmp_path / "bullets.md"
    r = runner.invoke(
        app,
        ["resume-bullets", "--stats-file", str(stats), "--out", str(out),
         "--templates", str(TEMPLATES)],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "characters" in out.read_text() and "router_supported (" in r.output
    bad = runner.invoke(app, ["resume-bullets", "--stats-file", str(tmp_path / "none.json")])
    assert bad.exit_code == 2 and "resume-stats" in bad.output


def test_the_bullets_cli_refuses_synthetic_stats(tmp_path: Path) -> None:
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps(base_stats(data_source="synthetic")))
    r = runner.invoke(
        app, ["resume-bullets", "--stats-file", str(stats), "--out", str(tmp_path / "o.md")]
    )
    assert r.exit_code == 2 and "real data only" in r.output and not (tmp_path / "o.md").exists()


# --- README and write-up -------------------------------------------------------------------------

README_TEXT = f"""# T

intro text

{rd.BEGIN}
old block
{rd.END}

## After

hand-written tail
"""


def test_the_results_block_is_regenerated_between_the_markers_only(world: World) -> None:
    write_stats(world.stats(), world.stats_path)
    readme = world.tmp / "README.md"
    readme.write_text(README_TEXT)
    args = [
        "--stats", str(world.stats_path), "--configs-dir", str(world.configs),
        "--readme", str(readme), "--allow-synthetic",
    ]  # fmt: skip
    assert rd.readme_main(args) == 0
    text = readme.read_text()
    assert text.startswith("# T\n\nintro text\n\n" + rd.BEGIN)
    assert text.endswith(rd.END + "\n\n## After\n\nhand-written tail\n") and "old block" not in text
    assert "SYNTHETIC DATA" in text and "### Headline" in text
    assert rd.readme_main([*args, "--check"]) == 0  # up to date
    readme.write_text(text.replace("### Headline", "### Edited by hand"))
    assert rd.readme_main([*args, "--check"]) == 1  # a hand edit is detected
    assert rd.readme_main(args) == 0 and rd.readme_main([*args, "--check"]) == 0  # and repaired


def test_every_number_in_the_block_matches_the_artifacts(world: World) -> None:
    s = world.stats()
    raw = results_json(world)
    block = rd.results_block(s, world.configs, script="x")
    m = {x["name"]: x for x in raw["metrics"]}["always_max"]
    assert f"{m['resolve_rate']['point']:.1%}" in block
    assert f"{m['cost_per_task']['point']:,.4f}" in block
    for c in raw["comparisons"]:
        assert f"{c['cost_reduction']['point']:.1%}" in block
    assert "knowledge cutoff 2024-01-01" in block and "2024-06-01" in block  # models.yaml
    assert "2026-03-01 to 2026-09-30" in block  # repos.yaml
    assert "not available" in block  # no dev analysis ran, and it says so
    assert f"{raw['margin']}" in block and "$4.5000" in block  # total ledger spend
    assert "3 draft(s) on 2" in block


def test_the_block_says_plainly_when_no_policy_met_the_headline(world: World) -> None:
    s = world.stats()
    s["claim_supported"], s["designated_policy"] = False, "always_max"
    assert "No primary policy met the headline" in rd.headline_block(s)
    s["claim_supported"], s["designated_policy"] = True, "cascade"
    assert "`cascade` met the headline" in rd.headline_block(s)


def test_a_block_needs_filled_config_values(world: World) -> None:
    s = world.stats()
    (world.configs / "repos.yaml").write_text(REPOS_YAML.replace("2026-03-01", "TODO"))
    with pytest.raises(rd.RenderError, match="date_window is not filled"):
        rd.facts_block(s, world.configs)
    (world.configs / "repos.yaml").write_text(REPOS_YAML)
    s2 = copy.deepcopy(s)
    s2["arms"]["ids"]["cheap"] = "ghost@low"
    with pytest.raises(rd.RenderError, match="ghost@low"):
        rd.facts_block(s2, world.configs)


def test_the_readme_script_fails_loudly_without_stats_markers_or_real_data(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    readme = world.tmp / "README.md"
    readme.write_text(README_TEXT)
    base = ["--readme", str(readme), "--configs-dir", str(world.configs)]
    assert rd.readme_main([*base, "--stats", str(world.tmp / "none.json")]) == 2
    assert "resume-stats" in capsys.readouterr().err
    write_stats(world.stats(), world.stats_path)
    assert rd.readme_main([*base, "--stats", str(world.stats_path)]) == 2  # synthetic, no flag
    assert "real data only" in capsys.readouterr().err
    readme.write_text("# no markers\n")
    assert rd.readme_main([*base, "--stats", str(world.stats_path), "--allow-synthetic"]) == 2
    assert "exactly one" in capsys.readouterr().err
    readme.write_text(README_TEXT + rd.BEGIN + "\n")
    assert rd.readme_main([*base, "--stats", str(world.stats_path), "--allow-synthetic"]) == 2


def test_the_placeholder_block_needs_no_artifacts(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(README_TEXT)
    assert rd.readme_main(["--readme", str(readme), "--placeholder"]) == 0
    assert "No results yet" in readme.read_text()


def test_the_committed_readme_has_one_marker_pair_and_a_current_placeholder() -> None:
    text = (REPO_ROOT / "README.md").read_text()
    assert text.count(rd.BEGIN) == 1 and text.count(rd.END) == 1
    for heading in ("## Architecture", "## Reproduce from a clean clone", "## Limitations"):
        assert heading in text
    for topic in (
        "Contamination", "One run per matrix cell", "independence assumption",
        "Verifier false positives", "Wide intervals", "owned or opt-in",
    ):  # fmt: skip
        assert topic in text
    block = text.split(rd.BEGIN)[1].split(rd.END)[0]
    assert block.strip() == rd.placeholder_block("scripts/make_readme_results.py").strip()


def test_the_writeup_is_filled_from_the_template_and_checked(world: World) -> None:
    write_stats(world.stats(), world.stats_path)
    out = world.tmp / "WRITEUP.md"
    args = [
        "--stats", str(world.stats_path), "--configs-dir", str(world.configs),
        "--template", str(REPO_ROOT / "docs/WRITEUP.template.md"), "--out", str(out),
        "--allow-synthetic",
    ]  # fmt: skip
    assert rd.writeup_main(args) == 0
    text = out.read_text()
    assert "{{" not in text and "SYNTHETIC DATA" in text and "### Headline" in text
    assert "## Limitations" in text and "knowledge cutoff" in text
    assert rd.writeup_main([*args, "--check"]) == 0
    out.write_text(text + "\nedited by hand\n")
    assert rd.writeup_main([*args, "--check"]) == 1


def test_the_writeup_refuses_unknown_tokens_and_synthetic_data(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    write_stats(world.stats(), world.stats_path)
    template = world.tmp / "t.md"
    template.write_text("{{PROVENANCE}} {{MADE_UP}}")
    base = [
        "--stats", str(world.stats_path), "--configs-dir", str(world.configs),
        "--template", str(template), "--out", str(world.tmp / "o.md"),
    ]  # fmt: skip
    assert rd.writeup_main([*base, "--allow-synthetic"]) == 2
    assert "MADE_UP" in capsys.readouterr().err
    assert rd.writeup_main(base) == 2 and "real data only" in capsys.readouterr().err
    assert not (world.tmp / "o.md").exists()


def test_the_writeup_template_states_no_number_of_its_own() -> None:
    text = (REPO_ROOT / "docs/WRITEUP.template.md").read_text()
    prose = re.sub(r"\{\{[A-Z_]+\}\}", "", text)
    numbers = re.findall(r"(?<![\w`-])\d+(?:\.\d+)?%?(?![\w`])", prose)
    assert set(numbers) <= {"1", "2", "3", "4"}, numbers  # only the numbered method steps


# --- the designated policy and the final-artifact checks -----------------------------------------


def _cmp(policy: str, verdict: str, cut: float) -> dict[str, Any]:
    return {"policy": policy, "verdict": verdict, "cost_reduction": {"point": cut}}


def test_the_designated_policy_is_the_met_primary_policy_with_the_larger_saving() -> None:
    from cascade.evals.designate import designated_policy

    both = {"comparisons": [_cmp("cascade", "MET", 0.3), _cmp("predictive_issue", "MET", 0.5)]}
    assert designated_policy(both) == "predictive_issue"
    swapped = {"comparisons": [_cmp("cascade", "MET", 0.6), _cmp("predictive_issue", "MET", 0.5)]}
    assert designated_policy(swapped) == "cascade"
    only = {
        "comparisons": [_cmp("cascade", "INCONCLUSIVE", 0.9), _cmp("predictive_issue", "MET", 0.1)]
    }
    assert designated_policy(only) == "predictive_issue"


def test_without_a_met_primary_policy_the_baseline_is_designated() -> None:
    from cascade.evals.designate import designated_policy

    for verdict in ("INCONCLUSIVE", "INFERIOR"):
        r = {"comparisons": [_cmp("cascade", verdict, 0.9), _cmp("predictive_issue", verdict, 0.9)]}
        assert designated_policy(r) == "always_max"
    # a MET verdict on a non-primary policy (the oracle, a fixed arm) never designates it
    other = {"comparisons": [_cmp("oracle", "MET", 0.9), _cmp("always_cheap", "MET", 0.95)]}
    assert designated_policy(other) == "always_max"
    assert designated_policy({"comparisons": []}) == "always_max"


def test_a_lock_that_did_not_complete_is_refused(world: World) -> None:
    lock = world.repo / "FINAL.lock.json"
    data = json.loads(lock.read_text())
    data["status"] = "failed"
    lock.write_text(json.dumps(data))
    with pytest.raises(StatsError, match="did not complete"):
        world.stats()


def test_the_repo_count_is_withheld_when_any_audited_task_was_not_evaluated(world: World) -> None:
    manifest = [
        json.loads(x) for x in (world.data / "manifest.jsonl").read_text().splitlines() if x
    ]
    victim = sorted(t["task_id"] for t in manifest if t["split"] == "train")[0]
    audit = (world.data / "audit.jsonl").read_text().splitlines()
    (world.data / "audit.jsonl").write_text(
        "\n".join(ln for ln in audit if json.loads(ln)["task_id"] != victim) + "\n"
    )
    t = world.stats()["tasks"]  # the test split is intact, so the final lock still agrees
    assert t["audited"]["train"] == 35 and t["evaluated"]["train"] == 36
    assert t["all_audited_tasks_evaluated"] is False and t["repos"]["n"] is None
    assert t["repos"]["names"]  # names are still listed
