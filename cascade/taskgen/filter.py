"""Rule-based PR filters (no LLM). Each rule has a stable reason code used in the attrition table.

Rules are checked in a fixed order and the first failure is the recorded reason, so per-reason
counts sum to the number dropped. The spec's rules: merged inside the date window, closes an issue,
touches source and tests, 1-5 source files, about 150 source lines or fewer, and not a bot,
docs-only, or dependency PR.

Choices beyond the spec (see DECISIONS.md): exactly one closing issue, in the same repo and with a
non-empty body; PRs into a non-default branch are dropped (backports duplicate mainline fixes).
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from cascade.config import SelectionLimits
from cascade.taskgen.github import ChangedFile, PRCandidate

FileKind = Literal["test", "source", "docs", "meta", "other"]

# Reason codes, in the order they are checked. Shared with attrition reporting.
OUTSIDE_DATE_WINDOW = "outside_date_window"
NOT_DEFAULT_BRANCH = "not_default_branch"
NO_MERGE_COMMIT = "no_merge_commit"
BOT_AUTHOR = "bot_author"
DEPENDENCY_PR = "dependency_pr"
FILES_TRUNCATED = "files_truncated"
NO_CLOSING_ISSUE = "no_closing_issue"
MULTIPLE_CLOSING_ISSUES = "multiple_closing_issues"
CROSS_REPO_ISSUE = "cross_repo_issue"
EMPTY_ISSUE_BODY = "empty_issue_body"
NO_TEST_FILES = "no_test_files"
DOCS_ONLY = "docs_only"
NO_SOURCE_FILES = "no_source_files"
TOO_FEW_SOURCE_FILES = "too_few_source_files"
TOO_MANY_SOURCE_FILES = "too_many_source_files"
TOO_MANY_SOURCE_LINES = "too_many_source_lines"

_BOT_LOGINS = frozenset({"dependabot", "pre-commit-ci", "github-actions", "renovate"})
_DEP_TITLE = re.compile(
    r"^(\[pre-commit\.ci\]|pre-commit autoupdate|bump\b|update .*dependenc|upgrade .*dependenc)",
    re.IGNORECASE,
)
_DEP_LABELS = frozenset({"dependencies", "dependency"})
_DOC_SUFFIXES = frozenset({".md", ".rst", ".txt", ".adoc"})
_DOC_DIRS = frozenset({"docs", "doc", "examples"})
_TEST_DIRS = frozenset({"tests", "test"})


def classify_path(path: str) -> FileKind:
    p = PurePosixPath(path)
    parts = p.parts
    name = p.name
    if (
        any(part in _TEST_DIRS for part in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    ):
        return "test"
    # Dependency/CI files are checked before docs: requirements/*.txt would otherwise match .txt.
    if (
        parts[0] in {".github", "requirements"}
        or name.endswith(".lock")
        or (name.startswith("requirements") and p.suffix == ".txt")
        or name == ".pre-commit-config.yaml"
    ):
        return "meta"
    if p.suffix in _DOC_SUFFIXES or any(part in _DOC_DIRS for part in parts[:-1]):
        return "docs"
    if p.suffix == ".py":
        return "source"
    return "other"


@dataclass(frozen=True)
class FileSummary:
    src_files: int
    src_lines: int  # additions + deletions over source .py files
    test_files: int
    docs_files: int
    meta_files: int
    other_files: int

    @property
    def total(self) -> int:
        return (
            self.src_files + self.test_files + self.docs_files + self.meta_files + self.other_files
        )


def summarize_files(files: Sequence[ChangedFile]) -> FileSummary:
    counts = {"source": 0, "test": 0, "docs": 0, "meta": 0, "other": 0}
    src_lines = 0
    for f in files:
        kind = classify_path(f.path)
        counts[kind] += 1
        if kind == "source":
            src_lines += f.additions + f.deletions
    return FileSummary(
        src_files=counts["source"],
        src_lines=src_lines,
        test_files=counts["test"],
        docs_files=counts["docs"],
        meta_files=counts["meta"],
        other_files=counts["other"],
    )


def check_size_limits(summary: FileSummary, limits: SelectionLimits) -> str | None:
    """The source-size rules, shared by the metadata stage and the patch stage."""
    if summary.src_files == 0:
        only_docs = summary.docs_files + summary.meta_files == summary.total
        return DOCS_ONLY if only_docs else NO_SOURCE_FILES
    if summary.src_files < limits.min_source_files:
        return TOO_FEW_SOURCE_FILES
    if summary.src_files > limits.max_source_files:
        return TOO_MANY_SOURCE_FILES
    if summary.src_lines > limits.max_source_lines:
        return TOO_MANY_SOURCE_LINES
    return None


@dataclass(frozen=True)
class FilterContext:
    window_start: dt.date
    window_end: dt.date
    default_branch: str
    repo: str  # owner/name; closing issues must belong to it
    limits: SelectionLimits


def is_bot(candidate: PRCandidate) -> bool:
    login = (candidate.author_login or "").lower()
    return (
        candidate.author_type == "Bot"
        or login.endswith("[bot]")
        or login.removesuffix("[bot]") in _BOT_LOGINS
    )


def is_dependency_pr(candidate: PRCandidate) -> bool:
    if _DEP_TITLE.search(candidate.title.strip()):
        return True
    if any(label.lower() in _DEP_LABELS for label in candidate.labels):
        return True
    kinds = {classify_path(f.path) for f in candidate.files}
    return bool(kinds) and kinds <= {"meta"}


def filter_metadata(candidate: PRCandidate, ctx: FilterContext) -> str | None:
    """Return the first failing reason code, or None if the PR passes every metadata rule."""
    merged_on = candidate.merged_at.astimezone(dt.UTC).date()
    if not ctx.window_start <= merged_on <= ctx.window_end:
        return OUTSIDE_DATE_WINDOW
    if candidate.base_ref != ctx.default_branch:
        return NOT_DEFAULT_BRANCH
    if not candidate.merge_commit:
        return NO_MERGE_COMMIT
    if is_bot(candidate):
        return BOT_AUTHOR
    if is_dependency_pr(candidate):
        return DEPENDENCY_PR
    if candidate.files_total > len(candidate.files):
        return FILES_TRUNCATED
    if candidate.closing_issues_total == 0:
        return NO_CLOSING_ISSUE
    if candidate.closing_issues_total > 1:
        return MULTIPLE_CLOSING_ISSUES
    issue = candidate.closing_issues[0]
    if issue.repo.lower() != ctx.repo.lower():
        return CROSS_REPO_ISSUE
    if not issue.body.strip():
        return EMPTY_ISSUE_BODY
    summary = summarize_files(candidate.files)
    docs_meta_only = summary.total > 0 and summary.docs_files + summary.meta_files == summary.total
    if docs_meta_only:
        # Checked before the test-file rule so it is reported as its own exclusion (the spec
        # lists docs-only separately) instead of as "no tests".
        return DOCS_ONLY
    if summary.test_files == 0:
        return NO_TEST_FILES
    return check_size_limits(summary, ctx.limits)
