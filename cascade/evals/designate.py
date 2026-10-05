"""Reading the final evaluation back, and the rule for "the policy the final eval designated".

``results/final/results.json`` is what ``cascade eval --final`` wrote once. Everything downstream
(the live pilot's default policy, ``resume-stats``, the README and write-up generators) reads it
through ``load_final``, which refuses to hand over a result that cannot be trusted: no completed
lock, outputs whose hashes no longer match the lock, or a matrix that changed after the evaluation.

DESIGNATION RULE (documented in DECISIONS.md; it does not change what the evaluation computes):
among the plan's two primary policies, ``cascade`` and ``predictive_issue``, take the one whose
final-eval verdict is MET (non-inferior and a saving), preferring the larger point cost reduction.
If neither is MET the designated policy is ``always_max``: nothing was shown to beat the baseline,
so the baseline is what would be used. An inconclusive result designates the baseline, deliberately.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cascade.errors import CascadeError

PRIMARY_POLICIES = ("cascade", "predictive_issue")
BASELINE_POLICY = "always_max"
FINAL_FILES = ("report.md", "results.json", "pareto.svg")


class FinalArtifactError(CascadeError):
    """The final evaluation's artifacts are missing or cannot be trusted."""


@dataclass(frozen=True)
class FinalEval:
    results: dict[str, Any]  # results.json of the final evaluation
    lock: dict[str, Any]  # FINAL.lock.json

    @property
    def data_source(self) -> str:
        return str(self.results["provenance"]["data_source"])


def designated_policy(results: Mapping[str, Any]) -> str:
    comparisons: list[dict[str, Any]] = list(results["comparisons"])
    met = [c for c in comparisons if c["policy"] in PRIMARY_POLICIES and c["verdict"] == "MET"]
    if not met:
        return BASELINE_POLICY
    best = max(met, key=lambda c: float(c["cost_reduction"]["point"]))
    return str(best["policy"])


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()


def load_final(final_dir: Path, lock_path: Path, *, matrix_path: Path | None = None) -> FinalEval:
    """Read the final results, verifying the lock, output hashes, and (optionally) the matrix."""
    needed = [lock_path, *[final_dir / f for f in FINAL_FILES]]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FinalArtifactError(
            "the final evaluation has not been run (missing: " + ", ".join(missing) + "). "
            "Run `cascade eval --final` once, after confirming docs/EVAL_PLAN.md."
        )
    lock: dict[str, Any] = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "completed":
        raise FinalArtifactError(
            f"{lock_path} has status {lock.get('status')!r}: the final evaluation did not complete"
        )
    for name in FINAL_FILES:
        want = lock.get("outputs_sha256", {}).get(name)
        if want != _sha256_text(final_dir / name):
            raise FinalArtifactError(
                f"{final_dir / name} does not match the hash in {lock_path}: it was edited or "
                "regenerated after the final evaluation"
            )
    results: dict[str, Any] = json.loads((final_dir / "results.json").read_text(encoding="utf-8"))
    if results.get("mode") != "final":
        raise FinalArtifactError(f"{final_dir / 'results.json'} is not a final-mode result")
    if matrix_path is not None:
        now = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
        if now != lock.get("matrix_hash"):
            raise FinalArtifactError(
                f"{matrix_path} changed since the final evaluation (hash differs from the lock)"
            )
    return FinalEval(results, lock)
