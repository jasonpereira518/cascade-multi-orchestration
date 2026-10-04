"""GitHub GraphQL client and the PR candidate model.

Field names were checked against GitHub's published schema (docs.github.com/public/fpt/
schema.docs.graphql): PullRequest.{closingIssuesReferences, files, mergeCommit, commits, labels,
author, baseRefName}, PullRequestChangedFile.{path, additions, deletions, changeType: PatchStatus},
Query.search(type: ISSUE), RateLimit.{cost, remaining, resetAt}.

The token comes from the GITHUB_TOKEN environment variable only and is never logged or recorded.
Tests inject a transport and never touch the network.
"""

# pyright: basic
# The GraphQL response is untyped external JSON; strict mode would only add casts. Every value
# is validated into the pydantic models below before use.
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from cascade.errors import CascadeError

GRAPHQL_URL = "https://api.github.com/graphql"
TOKEN_ENV = "GITHUB_TOKEN"  # noqa: S105  (name of an env var, not a secret)

SEARCH_QUERY = """
query($q: String!, $after: String, $first: Int!) {
  rateLimit { cost remaining resetAt }
  search(query: $q, type: ISSUE, first: $first, after: $after) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest {
        number
        title
        url
        mergedAt
        baseRefName
        author { __typename login }
        labels(first: 20) { nodes { name } }
        mergeCommit { oid }
        commits { totalCount }
        files(first: 100) { totalCount nodes { path additions deletions changeType } }
        closingIssuesReferences(first: 10) {
          totalCount
          nodes { number title body state createdAt repository { nameWithOwner } }
        }
      }
    }
  }
}
"""

REPO_QUERY = """
query($owner: String!, $name: String!) {
  rateLimit { cost remaining resetAt }
  repository(owner: $owner, name: $name) {
    nameWithOwner
    isArchived
    stargazerCount
    pushedAt
    licenseInfo { spdxId }
    defaultBranchRef { name }
  }
}
"""

# A transport takes (query, variables) and returns the decoded JSON response body.
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


class GitHubError(CascadeError):
    """A GitHub API call failed."""


class MissingTokenError(GitHubError):
    """GITHUB_TOKEN is not set."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChangedFile(_Strict):
    path: str
    additions: Annotated[int, Field(ge=0)]
    deletions: Annotated[int, Field(ge=0)]
    change_type: str


class ClosingIssue(_Strict):
    number: int
    title: str
    body: str
    state: str
    created_at: dt.datetime
    repo: str  # nameWithOwner


class PRCandidate(_Strict):
    number: int
    title: str
    url: str
    merged_at: dt.datetime
    base_ref: str
    author_login: str | None
    author_type: str | None
    labels: list[str]
    merge_commit: str | None
    commit_count: int
    files: list[ChangedFile]
    files_total: int
    closing_issues: list[ClosingIssue]
    closing_issues_total: int


class RepoMeta(_Strict):
    name_with_owner: str
    license_spdx: str | None
    default_branch: str
    archived: bool
    stars: int
    pushed_at: dt.datetime | None


def _ts(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_pr(node: dict[str, Any]) -> PRCandidate:
    """Convert one GraphQL PullRequest node into a PRCandidate."""
    author = node.get("author") or {}
    merge = node.get("mergeCommit") or {}
    files = node.get("files") or {"totalCount": 0, "nodes": []}
    closing = node.get("closingIssuesReferences") or {"totalCount": 0, "nodes": []}
    return PRCandidate(
        number=node["number"],
        title=node["title"],
        url=node["url"],
        merged_at=_ts(node["mergedAt"]),
        base_ref=node["baseRefName"],
        author_login=author.get("login"),
        author_type=author.get("__typename"),
        labels=[n["name"] for n in (node.get("labels") or {}).get("nodes", [])],
        merge_commit=merge.get("oid"),
        commit_count=(node.get("commits") or {}).get("totalCount", 0),
        files=[
            ChangedFile(
                path=f["path"],
                additions=f["additions"],
                deletions=f["deletions"],
                change_type=f["changeType"],
            )
            for f in files["nodes"]
        ],
        files_total=files["totalCount"],
        closing_issues=[
            ClosingIssue(
                number=i["number"],
                title=i["title"],
                body=i.get("body") or "",
                state=i["state"],
                created_at=_ts(i["createdAt"]),
                repo=i["repository"]["nameWithOwner"],
            )
            for i in closing["nodes"]
        ],
        closing_issues_total=closing["totalCount"],
    )


def parse_repo(data: dict[str, Any]) -> RepoMeta:
    repo = data["repository"]
    if repo is None:
        raise GitHubError("repository not found or not visible to this token")
    pushed = repo.get("pushedAt")
    return RepoMeta(
        name_with_owner=repo["nameWithOwner"],
        license_spdx=(repo.get("licenseInfo") or {}).get("spdxId"),
        default_branch=(repo.get("defaultBranchRef") or {}).get("name", ""),
        archived=repo["isArchived"],
        stars=repo["stargazerCount"],
        pushed_at=_ts(pushed) if pushed else None,
    )


def _urllib_transport(token: str) -> Transport:
    def send(query: str, variables: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps({"query": query, "variables": variables}).encode()
        req = urllib.request.Request(
            GRAPHQL_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "cascade-taskgen",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            decoded: dict[str, Any] = json.loads(resp.read())
            return decoded

    return send


class GitHubClient:
    """Runs GraphQL queries with retry on transient failures. Primary rate-limit exhaustion raises
    (with the reset time) instead of sleeping for hours; the pipeline is resumable, so just rerun.
    """

    def __init__(
        self,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 4,
        record_dir: Path | None = None,
    ) -> None:
        if transport is None:
            token = os.environ.get(TOKEN_ENV)
            if not token:
                raise MissingTokenError(
                    f"{TOKEN_ENV} is not set; export it (read-only, public-repo scope is enough)"
                )
            transport = _urllib_transport(token)
        self._transport = transport
        self._sleep = sleep
        self._max_retries = max_retries
        self._record_dir = record_dir

    def query(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Return the response's ``data`` object, raising GitHubError on any GraphQL error."""
        for attempt in range(self._max_retries + 1):
            try:
                body = self._transport(query, variables)
                break
            except urllib.error.HTTPError as e:
                transient = e.code in (429, 500, 502, 503, 504) or (
                    e.code == 403 and e.headers.get("Retry-After") is not None
                )
                if not transient or attempt == self._max_retries:
                    raise GitHubError(f"HTTP {e.code} from GitHub GraphQL") from e
                retry_after = e.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 2.0**attempt
                self._sleep(min(delay, 120.0))
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self._max_retries:
                    raise GitHubError(f"network error talking to GitHub: {e}") from e
                self._sleep(min(2.0**attempt, 60.0))
        else:  # pragma: no cover - loop always breaks or raises
            raise GitHubError("retries exhausted")

        self._record(query, variables, body)
        if body.get("errors"):
            messages = "; ".join(str(e.get("message", e)) for e in body["errors"])
            raise GitHubError(f"GraphQL errors: {messages}")
        data: dict[str, Any] = body["data"]
        limit = data.get("rateLimit")
        if limit and limit["remaining"] < 10:
            raise GitHubError(f"GitHub rate limit nearly exhausted; resets at {limit['resetAt']}")
        return data

    def _record(self, query: str, variables: dict[str, Any], body: dict[str, Any]) -> None:
        """Save raw responses for building recorded test fixtures. The token is never included."""
        if self._record_dir is None:
            return
        self._record_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256((query + json.dumps(variables, sort_keys=True)).encode()).hexdigest()
        payload = {"variables": variables, "response": body}
        (self._record_dir / f"{key[:16]}.json").write_text(json.dumps(payload, indent=2))
