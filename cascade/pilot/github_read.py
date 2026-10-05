"""Read-only access to a GitHub issue, by REST. This module cannot write.

The pilot reads one issue per draft and nothing else from the API. Reads are plain HTTP GETs of
``/repos/{owner}/{repo}/issues/{n}``; the optional token comes from GITHUB_TOKEN and should be a
fine-grained token with read-only access to issues (no write scopes are needed or requested). The
transport takes only a URL, so there is no code path here that can send a request body or use any
method but GET. Tests inject a transport and never touch the network.
"""

# pyright: basic
# The REST response is untyped external JSON; every value is validated into IssueData before use.
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from cascade.errors import CascadeError

API = "https://api.github.com"
TOKEN_ENV = "GITHUB_TOKEN"  # noqa: S105  (name of an env var, not a secret)
Transport = Callable[[str], dict[str, Any]]  # GET this URL, return the decoded JSON object


class IssueReadError(CascadeError):
    """The issue could not be read."""


class IssueData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    number: int
    title: str
    body: str
    state: str
    url: str
    labels: list[str]
    is_pull_request: bool


class IssueReader(Protocol):
    def issue(self, repo: str, number: int) -> IssueData: ...


def _urllib_transport(token: str | None) -> Transport:
    def get(url: str) -> dict[str, Any]:
        if not url.startswith(API + "/"):
            raise IssueReadError(f"refusing to fetch a non-GitHub-API URL: {url}")
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers, method="GET")  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=30) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            hint = (
                " (a private repo needs GITHUB_TOKEN with read access to issues)"
                if e.code in (401, 403, 404)
                else ""
            )
            raise IssueReadError(f"HTTP {e.code} reading {url}{hint}") from e
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise IssueReadError(f"could not read {url}: {e}") from e

    return get


class GitHubIssueReader:
    def __init__(self, *, transport: Transport | None = None) -> None:
        self._get = transport or _urllib_transport(os.environ.get(TOKEN_ENV))

    def issue(self, repo: str, number: int) -> IssueData:
        if "/" not in repo or number < 1:
            raise IssueReadError(f"bad issue reference {repo!r} #{number}")
        data = self._get(f"{API}/repos/{repo}/issues/{number}")
        try:
            return IssueData(
                number=int(data["number"]),
                title=str(data["title"]),
                body=str(data.get("body") or ""),
                state=str(data.get("state", "")),
                url=str(data["html_url"]),
                labels=[str(x["name"]) for x in data.get("labels", []) if isinstance(x, dict)],
                is_pull_request="pull_request" in data,
            )
        except (KeyError, TypeError, ValueError) as e:
            raise IssueReadError(f"unexpected response shape for {repo}#{number}: {e}") from e
