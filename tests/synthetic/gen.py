"""A labeled-SYNTHETIC outcome matrix, for developing and testing the routing and eval code.

Everything here is invented: tasks, features, per-arm success, Verifier claims, costs, and
latencies are drawn from made-up distributions with a seeded generator. It exists so the policy and
evaluation code can be tested end to end before any real matrix exists, and it must never be
mistaken for one: ``write`` drops a ``SYNTHETIC.json`` marker that the loader reads, and every
output derived from this data is stamped ``data_source: synthetic``.

The data has deliberate structure so the code under test has something to find: a latent difficulty
per task drives success (cheap < mid < max) and cost, and correlates with the issue-side and
gold-derived features; repos differ by an offset; the Verifier is imperfect (false positives and
false negatives); a few attempts are cap-exceeded yet "succeeded" (to test normalization); and there
are ablation rows (mid arm, no_reproducer / no_reviewer) and repeat-seed rows (cheap and mid).
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cascade.agents.pipeline import PipelineConfig
from cascade.routing.analysis import ablation_configs
from cascade.routing.export import export_parquet
from cascade.routing.matrix import MatrixRow, MatrixStore
from cascade.taskgen.audit import append_verdict, make_verdict
from cascade.taskgen.manifest import write_tasks
from cascade.taskgen.schema import Features, Task

ARM_IDS = {"cheap": "mock-luna@low", "mid": "mock-luna@high", "max": "mock-sol@medium"}
REPOS = {"syn/a": 14, "syn/b": 12, "syn/c": 10, "syn/d": 9, "syn/e": 8}
TEST_REPOS = {"syn/d", "syn/e"}
_PATCH = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
_BASE_COST = {"cheap": 0.05, "mid": 0.12, "max": 0.60}
_BASE_WALL = {"cheap": 60.0, "mid": 110.0, "max": 200.0}
# success logit = intercept - slope * difficulty
_LOGIT = {"cheap": (-0.3, 1.4), "mid": (0.5, 1.1), "max": (1.2, 0.9)}
_CAP_ERROR = "BudgetExceededError: attempt cap: spent $1 + worst-case $1 > cap $1"


@dataclass(frozen=True)
class SyntheticData:
    tasks: list[Task]
    rows: list[MatrixRow]
    seed: int


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def _task(i: int, repo: str, d: float, rng: np.random.Generator) -> Task:
    noise = rng.normal(0, 0.25, 6)
    issue_chars = int(np.exp(6.0 + 0.35 * d + noise[0]))
    features = Features(
        src_files=int(1 + max(0.0, d + noise[1])),
        src_lines=int(np.exp(3.0 + 0.6 * d + noise[2])),
        issue_chars=issue_chars,
        test_runtime_s=float(np.exp(0.5 + 0.3 * d + noise[3])),
        issue_words=issue_chars // 6,
        issue_code_blocks=int(rng.poisson(1.0)),
        issue_has_traceback=bool(rng.random() < 0.3),
        issue_identifiers=int(rng.poisson(3.0 + max(0.0, d))),
        src_hunks=int(1 + max(0.0, d + noise[4])),
        f2p_count=int(1 + rng.poisson(1.0)),
        changed_symbols=int(1 + max(0.0, d + noise[5])),
    )
    return Task(
        task_id=f"{repo.replace('/', '__')}-{i}", repo=repo, base_commit="a" * 40,
        issue_title=f"synthetic issue {i} in {repo}", issue_body="x" * issue_chars,
        gold_patch=_PATCH, test_patch=_PATCH, fail_to_pass=["t.py::t"], pass_to_pass=[],
        image="synthetic", features=features, split="test" if repo in TEST_REPOS else "train",
    )  # fmt: skip


def _row(
    task: Task, label: str, success: bool, rng: np.random.Generator, d: float, config: str,
    seed: int, *, capped: bool = False,
) -> MatrixRow:  # fmt: skip
    claimed = bool(rng.random() < (0.9 if success else 0.25))
    cost = _BASE_COST[label] * float(np.exp(0.3 * d + rng.normal(0, 0.25)))
    cost *= 1.0 if success else 1.3
    wall = _BASE_WALL[label] * float(np.exp(0.2 * d + rng.normal(0, 0.2)))
    arm = ARM_IDS[label]
    return MatrixRow(
        task_id=task.task_id, arm_id=arm, seed=seed, true_success=success,
        verifier_claimed=claimed, cost_usd=cost, input_tokens=1000, output_tokens=100,
        reasoning_tokens=0, wall_s=wall, turns=8, patch_sha=None, pipeline_config=config,
        error=_CAP_ERROR if capped else None, model=arm.split("@")[0],
        reasoning_effort=arm.split("@")[1], processing_tier="standard",
        started_at="2026-01-01T00:00:00+00:00",
    )  # fmt: skip


def generate(seed: int = 0) -> SyntheticData:
    rng = np.random.default_rng(seed)
    default = PipelineConfig().canonical()
    ablations = {k: v.canonical() for k, v in ablation_configs(PipelineConfig()).items()}
    repo_offset = {r: float(rng.normal(0, 0.4)) for r in REPOS}
    tasks: list[Task] = []
    rows: list[MatrixRow] = []
    difficulty: dict[str, float] = {}
    for repo, n in REPOS.items():
        for i in range(n):
            d = float(rng.normal(0, 1)) + repo_offset[repo]
            t = _task(i, repo, d, rng)
            tasks.append(t)
            difficulty[t.task_id] = d
            for label, (a, b) in _LOGIT.items():
                ok = bool(rng.random() < _sigmoid(a - b * d))
                capped = label != "cheap" and d > 1.0 and bool(rng.random() < 0.15)
                if capped:
                    ok = bool(rng.random() < 0.5)  # a capped attempt may still have passed
                rows.append(_row(t, label, ok, rng, d, default, 0, capped=capped))

    train = [t for t in tasks if t.split == "train"]
    first_pass = {
        (r.task_id, r.arm_id): r.true_success for r in rows if r.pipeline_config == default
    }
    for t in train[:16]:  # ablations: mid arm, 16 shared train tasks
        d = difficulty[t.task_id]
        for name, cfg in ablations.items():
            shift = -0.3 if name == "no_reproducer" else -0.1
            ok = bool(rng.random() < _sigmoid(_LOGIT["mid"][0] + shift - _LOGIT["mid"][1] * d))
            rows.append(_row(t, "mid", ok, rng, d, cfg, 0))
    for t in train[:12]:  # repeat-subset: cheap and mid, seed 1
        d = difficulty[t.task_id]
        for label in ("cheap", "mid"):
            first = first_pass[(t.task_id, ARM_IDS[label])]
            ok = first if rng.random() < 0.8 else not first  # about 20% of cells flip
            rows.append(_row(t, label, ok, rng, d, default, 1))
    return SyntheticData(tasks, rows, seed)


def write(data: SyntheticData, out: Path) -> Path:
    """Materialize the dataset the way the real exports are laid out, plus the SYNTHETIC marker."""
    out.mkdir(parents=True, exist_ok=True)
    write_tasks(out / "manifest.jsonl", data.tasks)
    audit = out / "audit.jsonl"
    audit.write_text("")
    for t in data.tasks:
        append_verdict(audit, make_verdict(t.task_id, "ok"))
    db = out / "matrix.sqlite"
    db.unlink(missing_ok=True)
    store = MatrixStore(db)
    for r in data.rows:
        store.put(r)
    store.close()
    export_parquet(db, out / "no-ledger.sqlite", out)
    db.unlink()
    (out / "arms.json").write_text(json.dumps(ARM_IDS, indent=2) + "\n")
    marker = {"data_source": "synthetic", "generator": "tests/synthetic/gen.py", "seed": data.seed}
    (out / "SYNTHETIC.json").write_text(json.dumps(marker, indent=2) + "\n")
    return out


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="synthetic-"))
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(write(generate(seed), out))


if __name__ == "__main__":
    main()
