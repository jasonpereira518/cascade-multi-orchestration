"""Classify finished attempts so failures are not all blamed on the model.

Input is only what is already logged: the matrix row and the attempt's ``result.json``. Output is a
category per attempt, the reasons, and what to do next. The categories follow the three buckets that
matter for deciding what to fix:

  pipeline_bug_suspected  our code or environment failed (inapplicable patch, grading could not run,
                          the visible suite does not even run on the base commit): FIX, and do not
                          count it against the model;
  verifier_issue          the Verifier's verdict disagreed with hidden-test grading (false positive
                          or false negative): check whether the Verifier or the reproducer is wrong;
  genuine_model_failure   the pipeline worked and the model's patch or output was simply wrong;

plus cases that are none of those: success, hitting a budget cap or the kill switch (a cost data
point, not a defect), a provider/config error (fix the config), and a leak audit hit (investigate
first, always).

These are heuristics over logged facts, not proof; each diagnosis lists the evidence it used.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from cascade.agents.runner import _safe  # pyright: ignore[reportPrivateUsage]
from cascade.routing.matrix import MatrixRow, MatrixStore

SUCCESS = "success"
PIPELINE_BUG = "pipeline_bug_suspected"
VERIFIER_FALSE_POSITIVE = "verifier_false_positive"
VERIFIER_FALSE_NEGATIVE = "verifier_false_negative"
VERIFIER_ISSUE = "verifier_issue"
MODEL_FAILURE = "genuine_model_failure"
MODEL_OUTPUT_FAILURE = "model_output_failure"
CAP = "stopped_by_cap_or_switch"
PROVIDER = "provider_or_config_error"
LEAK = "leak_audit_hit"

# What to do, per category.
ACTIONS = {
    SUCCESS: "nothing",
    PIPELINE_BUG: "fix the pipeline; exclude this attempt from model comparisons until fixed",
    VERIFIER_FALSE_POSITIVE: (
        "inspect the reproducer and visible tests; the Verifier accepted a wrong patch"
    ),
    VERIFIER_FALSE_NEGATIVE: (
        "inspect why the evidence failed (reproducer invalid, flaky regression?)"
    ),
    VERIFIER_ISSUE: "inspect the Verifier report in result.json",
    MODEL_FAILURE: (
        "expected: the model's patch was wrong; look at prompts only if many tasks fail alike"
    ),
    MODEL_OUTPUT_FAILURE: (
        "the model did not produce valid structured output or finish; "
        "tune prompts or raise --max-output-tokens"
    ),
    CAP: "a cost data point; raise the cap or accept it",
    PROVIDER: "fix models.yaml (model or effort string) or the provider settings",
    LEAK: "investigate before anything else: open the trace and find where the text came from",
}

_PIPELINE_GRADE_REASONS = {
    "patch_apply_failed": "our patch builder produced a patch that does not apply to a fresh base",
    "test_patch_conflict": "the agent's source patch conflicts with the hidden test patch",
    "grading_run_error": "the grading run did not produce usable test results",
    "grading_timeout": "the grading run timed out",
}
_MODEL_FORMAT_MARKERS = (
    "ModelBehaviorError",
    "MaxTurnsExceeded",
    "Invalid JSON",
    "ModelRefusalError",
)
_PROVIDER_MARKERS = (
    "AuthenticationError",
    "NotFoundError",
    "BadRequestError",
    "PermissionDeniedError",
    "HTTP 4",
)


@dataclass(frozen=True)
class Diagnosis:
    task_id: str
    arm_id: str
    seed: int
    category: str
    evidence: list[str]
    flags: list[str]  # extra categories that also apply (e.g. a leak hit on a successful attempt)

    @property
    def action(self) -> str:
        return ACTIONS[self.category]


def _as_obj(value: object) -> dict[str, Any]:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _obj(parent: dict[str, Any], key: str) -> dict[str, Any]:
    return _as_obj(parent.get(key))


def _items(parent: dict[str, Any], key: str) -> list[Any]:
    value = parent.get(key)
    return cast("list[Any]", value) if isinstance(value, list) else []


def diagnose(row: MatrixRow, result: dict[str, Any] | None) -> Diagnosis:
    """Classify one attempt. ``result`` is its parsed result.json (None if missing)."""
    evidence: list[str] = []
    flags: list[str] = []
    result = result or {}
    grade = _obj(result, "grade")
    grade_reason = str(grade.get("reason", ""))
    verifier = _obj(result, "verifier")
    stages = [_as_obj(s) for s in _items(result, "stages")]
    leaks = _items(result, "leak_violations")
    error = row.error or ""

    if leaks:
        flags.append(LEAK)
        evidence.append(f"leak audit: {leaks[0]}")

    def done(category: str) -> Diagnosis:
        return Diagnosis(row.task_id, row.arm_id, row.seed, category, evidence, flags)

    if result == {}:
        evidence.append("no result.json found, so only the matrix row was used")

    if grade_reason in _PIPELINE_GRADE_REASONS:
        evidence.append(f"grading: {grade_reason} ({_PIPELINE_GRADE_REASONS[grade_reason]})")
        return done(PIPELINE_BUG)
    if error.startswith("patch_error"):
        evidence.append(f"could not turn the edits into a patch: {error}")
        return done(PIPELINE_BUG)
    if verifier and not verifier.get("baseline_ok", True):
        evidence.append("the visible test suite does not run on the unpatched base commit")
        return done(PIPELINE_BUG)

    if "BudgetExceededError" in error or "KillSwitchEngagedError" in error:
        evidence.append(f"attempt stopped: {error[:160]}")
        return done(CAP)

    failed_stage_text = " | ".join(
        str(s.get("detail", "")) for s in stages if not s.get("ok", True)
    )
    if any(m in failed_stage_text for m in _PROVIDER_MARKERS):
        evidence.append(f"a model call failed at the provider: {failed_stage_text[:160]}")
        return done(PROVIDER)
    if any(m in failed_stage_text for m in _MODEL_FORMAT_MARKERS):
        evidence.append(f"a role failed to produce valid output: {failed_stage_text[:160]}")
        return done(MODEL_OUTPUT_FAILURE)

    if row.true_success and row.verifier_claimed:
        return done(SUCCESS)
    if row.verifier_claimed and not row.true_success:
        evidence.append(
            f"Verifier claimed success, hidden tests disagree ({grade_reason or 'tests failed'})"
        )
        failed = _items(grade, "failed")
        if failed:
            evidence.append(f"failing hidden tests: {', '.join(str(f) for f in failed[:3])}")
        return done(VERIFIER_FALSE_POSITIVE)
    if row.true_success and not row.verifier_claimed:
        evidence += [f"Verifier said: {r}" for r in _items(verifier, "reasons")]
        return done(VERIFIER_FALSE_NEGATIVE)

    # Both outcomes negative: the model's patch was wrong, unless the evidence says otherwise.
    if grade_reason == "empty_patch":
        evidence.append("the model produced no source change")
    elif grade_reason:
        evidence.append(f"grading: {grade_reason}")
    evidence += [f"Verifier: {r}" for r in _items(verifier, "reasons")[:3]]
    return done(MODEL_FAILURE)


def triage_runs(runs_dir: Path) -> list[Diagnosis]:
    """Diagnose every attempt in a runs directory's matrix."""
    db = runs_dir / "matrix.sqlite"
    if not db.exists():
        return []
    store = MatrixStore(db)
    try:
        rows = store.rows()
    finally:
        store.close()
    out: list[Diagnosis] = []
    for row in rows:
        path = (
            runs_dir / "attempts" / _safe(row.task_id) / _safe(row.arm_id)
            / f"seed{row.seed}-{row.config_id}" / "result.json"
        )  # fmt: skip
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        out.append(diagnose(row, result))
    return out


def format_triage(diagnoses: list[Diagnosis]) -> str:
    if not diagnoses:
        return "no attempts found"
    counts = Counter(d.category for d in diagnoses)
    lines = [
        "TRIAGE",
        f"{len(diagnoses)} attempt(s): " + ", ".join(f"{k} {v}" for k, v in counts.most_common()),
    ]
    flagged = [d for d in diagnoses if d.flags]
    if flagged:
        lines.append(f"LEAK AUDIT HITS: {len(flagged)} attempt(s); investigate these first")
    for d in diagnoses:
        if d.category == SUCCESS and not d.flags:
            continue
        lines.append(
            f"- {d.task_id} | {d.arm_id} | seed {d.seed}: {d.category}"
            + (f" (+{', '.join(d.flags)})" if d.flags else "")
        )
        lines += [f"    evidence: {e}" for e in d.evidence]
        lines.append(f"    next: {d.action}")
    return "\n".join(lines)
