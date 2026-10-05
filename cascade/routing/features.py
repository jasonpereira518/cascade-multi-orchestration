"""Model-free triage features for the predictive router. No LLM-derived feature appears here.

Every feature is a deterministic function of a task record (the manifest's ``Features`` plus the
issue text). They fall in two groups, and the split matters for what a result may claim:

- **issue-side** features come from the issue text alone, so a router could compute them before
  it has attempted anything. This is the group a deployable router may use.
- **gold-derived** features (``src_files``, ``src_lines``, ``test_runtime_s``, patch hunk counts,
  the fail_to_pass count, symbols the fix touches) are measured on the gold patch or the test run,
  which a router in production would not have until after the fix exists. They are in the manifest
  and the SPEC lists some of them as triage features, so they are available here, but a policy that
  uses them is an upper bound on what triage could achieve, not a deployable router.
  ``docs/EVAL_PLAN.md`` therefore reports the two groups separately and treats the issue-side group
  as primary.

Count-like features are log1p-transformed so that a few huge tasks do not dominate a linear model.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cascade.taskgen.schema import Task

_URL = re.compile(r"https?://\S+")
_VERSION = re.compile(r"\b\d+\.\d+(?:\.\d+)?\b|\bv\d+\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    fn: Callable[[Task], float]
    gold_derived: bool
    log: bool = False  # apply log1p after computing


def _text(t: Task) -> str:
    return f"{t.issue_title}\n{t.issue_body}"


FEATURES: tuple[FeatureSpec, ...] = (
    # --- issue-side: computable before any attempt ---
    FeatureSpec("issue_chars", lambda t: float(t.features.issue_chars), False, log=True),
    FeatureSpec("title_chars", lambda t: float(len(t.issue_title)), False, log=True),
    FeatureSpec("issue_words", lambda t: float(t.features.issue_words), False, log=True),
    FeatureSpec("issue_lines", lambda t: float(t.issue_body.count("\n") + 1), False, log=True),
    FeatureSpec("code_blocks", lambda t: float(t.features.issue_code_blocks), False),
    FeatureSpec("has_traceback", lambda t: float(t.features.issue_has_traceback), False),
    FeatureSpec("identifiers", lambda t: float(t.features.issue_identifiers), False, log=True),
    FeatureSpec("n_urls", lambda t: float(len(_URL.findall(_text(t)))), False),
    FeatureSpec("mentions_version", lambda t: float(bool(_VERSION.search(_text(t)))), False),
    FeatureSpec(
        "n_sentences", lambda t: float(len(_SENTENCE_END.findall(t.issue_body))), False, log=True
    ),
    # --- gold-derived: measured on the fix or the test run, not available to a deployed router ---
    FeatureSpec("src_files", lambda t: float(t.features.src_files), True),
    FeatureSpec("src_lines", lambda t: float(t.features.src_lines), True, log=True),
    FeatureSpec("test_runtime_s", lambda t: float(t.features.test_runtime_s), True, log=True),
    FeatureSpec("src_hunks", lambda t: float(t.features.src_hunks), True, log=True),
    FeatureSpec("f2p_count", lambda t: float(t.features.f2p_count), True, log=True),
    FeatureSpec("changed_symbols", lambda t: float(t.features.changed_symbols), True, log=True),
    FeatureSpec(
        "changed_symbols_in_issue",
        lambda t: float(t.features.changed_symbols_in_issue),
        True,
    ),
)

BY_NAME = {f.name: f for f in FEATURES}
ISSUE_FEATURES: tuple[str, ...] = tuple(f.name for f in FEATURES if not f.gold_derived)
GOLD_FEATURES: tuple[str, ...] = tuple(f.name for f in FEATURES if f.gold_derived)
ALL_FEATURES: tuple[str, ...] = tuple(f.name for f in FEATURES)
FEATURE_SETS: dict[str, tuple[str, ...]] = {"issue": ISSUE_FEATURES, "all": ALL_FEATURES}


def feature_value(spec: FeatureSpec, task: Task) -> float:
    v = spec.fn(task)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"feature {spec.name} is not a finite non-negative number: {task.task_id}")
    return math.log1p(v) if spec.log else v


def feature_matrix(tasks: Sequence[Task], names: Sequence[str]) -> NDArray[np.float64]:
    """Rows are tasks (in the given order), columns are the named features."""
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        raise KeyError(f"unknown feature(s): {', '.join(unknown)}")
    specs = [BY_NAME[n] for n in names]
    values = [[feature_value(s, t) for s in specs] for t in tasks]
    return np.array(values, dtype=np.float64).reshape(len(tasks), len(specs))
