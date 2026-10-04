"""Builders shared by the taskgen tests. All SYNTHETIC (see tests/fixtures/README.md)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cascade.taskgen.github import ChangedFile, ClosingIssue, PRCandidate
from tests.conftest import git

REPO = "synthetic/tinylib"
MERGE_OID = "1" * 40


def changed(path: str, additions: int = 1, deletions: int = 0) -> ChangedFile:
    return ChangedFile(path=path, additions=additions, deletions=deletions, change_type="MODIFIED")


def default_files() -> list[ChangedFile]:
    return [
        changed("src/tinylib/clamp.py", 1, 1),
        changed("tests/test_clamp.py", 4, 0),
        changed("CHANGES.md", 1, 0),
    ]


def make_issue(**over: Any) -> ClosingIssue:
    base: dict[str, Any] = {
        "number": 99,
        "title": "clamp() returns the lower bound above the range",
        "body": "clamp(42, 0, 10) returns 0 but I expected 10.",
        "state": "CLOSED",
        "created_at": dt.datetime(2025, 1, 5, 9, 0, tzinfo=dt.UTC),
        "repo": REPO,
    }
    return ClosingIssue(**{**base, **over})


def make_pr(**over: Any) -> PRCandidate:
    """A PR that passes every filter by default; override fields to break one rule."""
    files = over.pop("files", None) or default_files()
    issues = over.pop("closing_issues", None)
    if issues is None:
        issues = [make_issue()]
    base: dict[str, Any] = {
        "number": 101,
        "title": "Fix clamp for values above the range",
        "url": f"https://github.com/{REPO}/pull/101",
        "merged_at": dt.datetime(2025, 1, 10, 12, 0, tzinfo=dt.UTC),
        "base_ref": "main",
        "author_login": "alice",
        "author_type": "User",
        "labels": [],
        "merge_commit": MERGE_OID,
        "commit_count": 1,
        "files": files,
        "files_total": len(files),
        "closing_issues": issues,
        "closing_issues_total": len(issues),
    }
    return PRCandidate(**{**base, **over})


def to_node(pr: PRCandidate) -> dict[str, Any]:
    """Render a PRCandidate as the GraphQL PullRequest node it would be parsed from."""
    return {
        "number": pr.number,
        "title": pr.title,
        "url": pr.url,
        "mergedAt": pr.merged_at.isoformat().replace("+00:00", "Z"),
        "baseRefName": pr.base_ref,
        "author": None
        if pr.author_login is None
        else {"__typename": pr.author_type, "login": pr.author_login},
        "labels": {"nodes": [{"name": n} for n in pr.labels]},
        "mergeCommit": None if pr.merge_commit is None else {"oid": pr.merge_commit},
        "commits": {"totalCount": pr.commit_count},
        "files": {
            "totalCount": pr.files_total,
            "nodes": [
                {
                    "path": f.path,
                    "additions": f.additions,
                    "deletions": f.deletions,
                    "changeType": f.change_type,
                }
                for f in pr.files
            ],
        },
        "closingIssuesReferences": {
            "totalCount": pr.closing_issues_total,
            "nodes": [
                {
                    "number": i.number,
                    "title": i.title,
                    "body": i.body,
                    "state": i.state,
                    "createdAt": i.created_at.isoformat().replace("+00:00", "Z"),
                    "repository": {"nameWithOwner": i.repo},
                }
                for i in pr.closing_issues
            ],
        },
    }


def search_response(
    nodes: list[dict[str, Any]], *, has_next: bool = False, cursor: str | None = None,
    count: int | None = None,
) -> dict[str, Any]:  # fmt: skip
    return {
        "data": {
            "rateLimit": {"cost": 1, "remaining": 4990, "resetAt": "2025-02-01T00:00:00Z"},
            "search": {
                "issueCount": len(nodes) if count is None else count,
                "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                "nodes": nodes,
            },
        }
    }


def repo_response(license_spdx: str | None = "MIT", branch: str = "main") -> dict[str, Any]:
    return {
        "data": {
            "rateLimit": {"cost": 1, "remaining": 4990, "resetAt": "2025-02-01T00:00:00Z"},
            "repository": {
                "nameWithOwner": REPO,
                "isArchived": False,
                "stargazerCount": 1234,
                "pushedAt": "2025-01-31T00:00:00Z",
                "licenseInfo": None if license_spdx is None else {"spdxId": license_spdx},
                "defaultBranchRef": {"name": branch},
            },
        }
    }


# --- a throwaway upstream git repository -------------------------------------------------------

BASE_FILES = {
    "src/tinylib/__init__.py": "from tinylib.clamp import clamp\n\n__all__ = ['clamp']\n",
    "src/tinylib/clamp.py": (
        "def clamp(value, low, high):\n"
        "    if value < low:\n        return low\n"
        "    if value > high:\n        return low\n"
        "    return value\n"
    ),
    "tests/test_clamp.py": (
        "from tinylib import clamp\n\n\n"
        "def test_inside_range():\n    assert clamp(5, 0, 10) == 5\n\n\n"
        "def test_below_range():\n    assert clamp(-3, 0, 10) == 0\n"
    ),
    "CHANGES.md": "# Changes\n\n- initial\n",
}


@dataclass(frozen=True)
class Upstream:
    path: Path
    base_sha: str
    merge_sha: str


def commit_files(repo: Path, files: dict[str, str | None], message: str) -> str:
    """Write (or delete, when the value is None) files and commit them; return the new sha."""
    for rel, content in files.items():
        target = repo / rel
        if content is None:
            target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def build_upstream(root: Path) -> Upstream:
    """Base commit, then one squash-style commit: fixes clamp, adds a test, edits CHANGES."""
    repo = root / "upstream"
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    base = commit_files(repo, dict(BASE_FILES), "base")
    fixed = BASE_FILES["src/tinylib/clamp.py"].replace(
        "    if value > high:\n        return low\n", "    if value > high:\n        return high\n"
    )
    tests = (
        BASE_FILES["tests/test_clamp.py"]
        + "\n\ndef test_above_range():\n    assert clamp(42, 0, 10) == 10\n"
    )
    merge = commit_files(
        repo,
        {
            "src/tinylib/clamp.py": fixed,
            "tests/test_clamp.py": tests,
            "CHANGES.md": BASE_FILES["CHANGES.md"] + "- fix clamp above range (#101)\n",
        },
        "Fix clamp for values above the range (#101)",
    )
    return Upstream(repo, base, merge)
