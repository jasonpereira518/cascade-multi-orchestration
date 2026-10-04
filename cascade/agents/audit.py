"""Post-hoc leak audit: scan an attempt's local trace for anything an agent must never see.

The trace stores every prompt and tool result verbatim, so this checks what actually went over the
wire, independently of the code that built the prompts. It looks for:
  - the whole gold patch or the whole test patch (raw or JSON-escaped, as traces are JSONL);
  - the names of test functions the test patch adds (hidden tests: they do not exist at the base
    commit, so they can only reach a prompt through leakage or a coincidental name);
  - hashes of commits made after the base commit.
Ids of tests that already exist at the base commit are deliberately not checked: agents see those in
their own test output.

A hit means "investigate", not "proven leak": a reproducer test can coincidentally reuse a hidden
test's name. The runner records hits on the attempt; it does not hide them.

One deliberate exemption: the Reviewer's prompt is built from the agent's OWN patch, so a
whole-patch match inside a reviewer call is the agent's work, not a leak. This matters because a
one-line fix can coincide with the gold patch byte for byte. ``audit_trace`` therefore skips only
the whole-patch check on reviewer lines; every other check runs on every line.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Iterator
from pathlib import Path
from typing import Any, cast

from cascade.errors import CascadeError
from cascade.taskgen.gitrepo import run_git
from cascade.taskgen.schema import Task

_ADDED_TEST_NAME = re.compile(r"^\+\s*(?:async\s+)?def (test_\w+)\s*\(", re.MULTILINE)
_SHA = re.compile(r"\b[0-9a-f]{40}\b")
MIN_PATCH_CHARS = 20


def added_test_names(test_patch: str) -> set[str]:
    return set(_ADDED_TEST_NAME.findall(test_patch))


def _escaped(text: str) -> str:
    return json.dumps(text)[1:-1]


def audit_text(
    text: str,
    task: Task,
    *,
    post_base_shas: Collection[str] = (),
    check_whole_patches: bool = True,
) -> list[str]:
    violations: list[str] = []
    for label, patch in (("gold patch", task.gold_patch), ("test patch", task.test_patch)):
        stripped = patch.strip()
        if not check_whole_patches or len(stripped) < MIN_PATCH_CHARS:
            continue
        if stripped in text or _escaped(stripped) in text:
            violations.append(f"the full {label} appears in the trace")
    for name in sorted(added_test_names(task.test_patch)):
        if re.search(rf"\b{re.escape(name)}\b", text):
            violations.append(f"hidden test name appears in the trace: {name}")
    if post_base_shas:
        leaked = sorted({s for s in _SHA.findall(text) if s in post_base_shas})
        violations += [f"post-base commit appears in the trace: {s[:12]}" for s in leaked]
    return violations


def post_base_commits(clone: Path, base_commit: str, *, limit: int = 100_000) -> set[str]:
    """Hashes of commits reachable from the clone's HEAD but not from the base commit."""
    try:
        out = run_git(clone, "rev-list", f"--max-count={limit}", f"{base_commit}..HEAD")
    except CascadeError:
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def _strings(value: Any, depth: int = 0) -> Iterator[str]:
    """Every string inside a decoded JSON value, decoding JSON-in-JSON strings recursively.

    Trace lines nest JSON (a model's structured output is a JSON string inside an output item
    inside the trace record), so a hidden patch can be escaped two or three times. Decoding the
    layers is more reliable than guessing escape levels.
    """
    if isinstance(value, str):
        yield value
        if depth < 4 and value.lstrip()[:1] in ("{", "["):
            try:
                inner = json.loads(value)
            except json.JSONDecodeError:
                return
            yield from _strings(inner, depth + 1)
    elif isinstance(value, dict):
        for item in value.values():  # pyright: ignore[reportUnknownVariableType]
            yield from _strings(item, depth)
    elif isinstance(value, list):
        for item in value:  # pyright: ignore[reportUnknownVariableType]
            yield from _strings(item, depth)


def audit_trace(trace_text: str, task: Task, *, post_base_shas: Collection[str] = ()) -> list[str]:
    """Audit a JSONL trace line by line; see the module docstring for the reviewer exemption."""
    found: list[str] = []
    for line in trace_text.splitlines():
        if not line.strip():
            continue
        reviewer_call = False
        text = line
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            pass  # audit unparseable lines as plain text
        else:
            if isinstance(row, dict):
                record = cast("dict[str, Any]", row)
                reviewer_call = record.get("kind") == "llm" and record.get("role") == "reviewer"
            text = line + "\n" + "\n".join(_strings(row))
        for v in audit_text(
            text, task, post_base_shas=post_base_shas, check_whole_patches=not reviewer_call
        ):
            if v not in found:
                found.append(v)
    return found
