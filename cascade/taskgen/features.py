"""Model-free difficulty features, computed from the task's own data. No LLM, no repo access.

They exist for two reasons: the predictive router (Tier 2) needs cheap pre-attempt features, and the
auditor benefits from seeing how big and how well-specified a task looks. Everything here is a
deterministic function of (issue text, gold patch, test patch, test id lists, measured runtime), so
it can be recomputed at any time and never depends on a model's opinion.

Patch parsing is deliberately simple (unified-diff headers, `@@` hunks, `+`/`-` lines, and `def` /
`class` names), because the inputs are patches this pipeline derived itself.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from cascade.taskgen.filter import classify_path
from cascade.taskgen.schema import Features

_FILE_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE)
_HUNK_CONTEXT = re.compile(r"^@@ [^@]+ @@\s*(?:async\s+)?(?:def|class)\s+(\w+)")
_DEF = re.compile(r"^[+-]\s*(?:async\s+)?(?:def|class)\s+(\w+)")
_BACKTICKED = re.compile(r"`([A-Za-z_][\w.]*)(?:\(\))?`")
_WORD = re.compile(r"\w+")
_MIN_SYMBOL_LEN = 3


def split_patch(patch: str) -> list[tuple[str, list[str]]]:
    """[(path, lines)] for each file block of a unified diff."""
    matches = list(_FILE_HEADER.finditer(patch))
    blocks: list[tuple[str, list[str]]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(patch)
        blocks.append((m.group(2), patch[m.end() : end].splitlines()))
    return blocks


def _is_source_py(path: str) -> bool:
    return path.endswith(".py") and classify_path(path) == "source"


def _line_counts(lines: Sequence[str]) -> tuple[int, int, int]:
    """(hunks, added, removed) over the hunk bodies of one file block."""
    hunks = added = removed = 0
    in_hunk = False
    for line in lines:
        if line.startswith("@@"):
            hunks += 1
            in_hunk = True
        elif in_hunk and line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif in_hunk and line.startswith("-") and not line.startswith("---"):
            removed += 1
    return hunks, added, removed


def _symbols(lines: Sequence[str]) -> tuple[set[str], set[str], set[str]]:
    """(touched, added_defs, removed_defs): names of functions and classes the patch touches."""
    touched: set[str] = set()
    added: set[str] = set()
    removed: set[str] = set()
    for line in lines:
        context = _HUNK_CONTEXT.match(line)
        if context:
            touched.add(context.group(1))
        found = _DEF.match(line)
        if found:
            touched.add(found.group(1))
            (added if line.startswith("+") else removed).add(found.group(1))
    return touched, added, removed


def gold_symbols(gold_patch: str) -> tuple[set[str], set[str]]:
    """(touched, new_public): functions/classes the patch touches, and the public ones it adds."""
    touched: set[str] = set()
    new_defs: set[str] = set()
    removed_defs: set[str] = set()
    for path, lines in split_patch(gold_patch):
        if _is_source_py(path):
            t, added, removed = _symbols(lines)
            touched |= t
            new_defs |= added
            removed_defs |= removed
    return touched, {s for s in new_defs - removed_defs if not s.startswith("_")}


def compute_features(
    *,
    issue_title: str,
    issue_body: str,
    gold_patch: str,
    test_patch: str,
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
    test_runtime_s: float,
) -> Features:
    issue = f"{issue_title}\n{issue_body}"

    gold_blocks = split_patch(gold_patch)
    src_blocks = [(p, ls) for p, ls in gold_blocks if _is_source_py(p)]
    hunks = added = removed = 0
    touched: set[str] = set()
    new_defs: set[str] = set()
    removed_defs: set[str] = set()
    for _path, lines in src_blocks:
        h, a, r = _line_counts(lines)
        hunks, added, removed = hunks + h, added + a, removed + r
        t, ad, rd = _symbols(lines)
        touched |= t
        new_defs |= ad
        removed_defs |= rd

    test_blocks = split_patch(test_patch)
    test_added = sum(_line_counts(lines)[1] for _p, lines in test_blocks)

    symbols = {s for s in touched if len(s) >= _MIN_SYMBOL_LEN}
    in_issue = {s for s in symbols if re.search(rf"\b{re.escape(s)}\b", issue)}
    public_new = {s for s in new_defs - removed_defs if not s.startswith("_")}

    return Features(
        src_files=len(src_blocks),
        src_lines=added + removed,
        issue_chars=len(issue_title) + len(issue_body),
        test_runtime_s=test_runtime_s,
        src_hunks=hunks,
        src_added=added,
        src_removed=removed,
        gold_files=len(gold_blocks),
        test_files=len(test_blocks),
        test_added=test_added,
        f2p_count=len(fail_to_pass),
        p2p_count=len(pass_to_pass),
        issue_words=len(_WORD.findall(issue)),
        issue_code_blocks=issue_body.count("```") // 2,
        issue_has_traceback="Traceback (most recent call last)" in issue_body,
        issue_identifiers=len(set(_BACKTICKED.findall(issue))),
        changed_symbols=len(symbols),
        changed_symbols_in_issue=len(in_issue),
        new_public_symbols=len(public_new),
    )
