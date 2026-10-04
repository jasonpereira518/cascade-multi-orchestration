"""Builders for the Phase 3 tests (split, audit, manifest, orchestration). All SYNTHETIC."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from cascade.config import ResolvedRepo, SelectionLimits
from cascade.taskgen.schema import Features, Task
from tests.taskgen_helpers import (
    Upstream,
    build_upstream,
    default_files,
    make_issue,
    make_pr,
    repo_response,
    search_response,
    to_node,
)

GOLD_PATCH = """\
diff --git a/src/pkg/core.py b/src/pkg/core.py
--- a/src/pkg/core.py
+++ b/src/pkg/core.py
@@ -1,3 +1,3 @@ def parse_option(flag):
     x = 1
-    return flag
+    return flag.strip()
"""

TEST_PATCH = """\
diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1,2 +1,4 @@
 def test_a():
     pass
+def test_strip():
+    assert parse_option(" x ") == "x"
"""


def make_task(repo: str = "o/a", n: int = 1, **over: Any) -> Task:
    """A schema-valid synthetic task. ``over`` replaces any Task field."""
    base: dict[str, Any] = {
        "task_id": f"{repo.replace('/', '__')}-{n}",
        "repo": repo,
        "base_commit": "a" * 40,
        "issue_title": "parse_option keeps whitespace",
        "issue_body": "parse_option(' x ') returns ' x ' but I expected 'x'.",
        "gold_patch": GOLD_PATCH,
        "test_patch": TEST_PATCH,
        "fail_to_pass": ["tests/test_core.py::test_strip"],
        "pass_to_pass": ["tests/test_core.py::test_a"],
        "image": "cascade-env/synthetic:abc",
        "features": Features(src_files=1, src_lines=2, issue_chars=60, test_runtime_s=1.5),
    }
    return Task(**{**base, **over})


def make_tasks(counts: dict[str, int], **over: Any) -> list[Task]:
    return [make_task(repo, i, **over) for repo, n in counts.items() for i in range(1, n + 1)]


def resolved_repo(name: str, **over: Any) -> ResolvedRepo:
    base: dict[str, Any] = {
        "name": name,
        "license": "MIT",
        "python_version": "3.12.13",
        "install_cmd": "uv pip install --group tests -e .",
        "test_cmd": "python -m pytest",
        "window_start": dt.date(2025, 1, 1),
        "window_end": dt.date(2025, 1, 31),
        "limits": SelectionLimits(),
    }
    return ResolvedRepo(**{**base, **over})


class MultiRepoGitHub:
    """A fake GitHub transport serving one good PR per repo, keyed by repo name."""

    def __init__(self, merge_shas: dict[str, str]) -> None:
        self.nodes = {
            r: [
                to_node(
                    make_pr(
                        url=f"https://github.com/{r}/pull/101",
                        merge_commit=sha,
                        closing_issues=[make_issue(repo=r)],
                        files=default_files(),
                    )
                )
            ]
            for r, sha in merge_shas.items()
        }
        self.search_calls = 0

    def __call__(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        if "repository(" in query:
            name = f"{variables['owner']}/{variables['name']}"
            resp = repo_response("MIT", "main")
            resp["data"]["repository"]["nameWithOwner"] = name
            return resp
        self.search_calls += 1
        q: str = variables["q"]
        for repo, nodes in self.nodes.items():
            if f"repo:{repo} " in q and variables["after"] is None:
                return search_response(nodes)
        return search_response([])


def build_upstreams(root: Path, repos: list[str]) -> dict[str, Upstream]:
    return {r: build_upstream(root / r.replace("/", "__")) for r in repos}
