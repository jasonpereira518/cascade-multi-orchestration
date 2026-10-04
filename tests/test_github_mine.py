from __future__ import annotations

import datetime as dt
import json
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

from cascade.config import ResolvedRepo, SelectionLimits
from cascade.taskgen.github import (
    TOKEN_ENV,
    GitHubClient,
    GitHubError,
    MissingTokenError,
    parse_pr,
    parse_repo,
)
from cascade.taskgen.mine import ConfigMismatchError, check_repo, mine, month_slices
from cascade.taskgen.store import StageStore
from tests.conftest import SYNTHETIC
from tests.taskgen_helpers import REPO, make_pr, repo_response, search_response, to_node

GQL = SYNTHETIC / "graphql"


def load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((GQL / name).read_text())
    return data


def resolved(start: str = "2025-01-01", end: str = "2025-02-28") -> ResolvedRepo:
    return ResolvedRepo(
        name=REPO,
        license="BSD-3-Clause",
        python_version="3.12",
        install_cmd="x",
        test_cmd="y",
        window_start=dt.date.fromisoformat(start),
        window_end=dt.date.fromisoformat(end),
        limits=SelectionLimits(),
    )


class FakeGitHub:
    """A transport that serves canned responses and records every call."""

    def __init__(self, pages: dict[tuple[str, str | None], dict[str, Any]], repo: dict[str, Any]):
        self.pages = pages
        self.repo = repo
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((query, variables))
        if "repository(" in query:
            return self.repo
        key = (variables["q"], variables["after"])
        return self.pages.get(key, search_response([]))

    @property
    def searches(self) -> list[dict[str, Any]]:
        return [v for q, v in self.calls if "search(" in q]


def january_fake() -> FakeGitHub:
    q = "repo:synthetic/tinylib is:pr is:merged merged:2025-01-01..2025-01-31"
    return FakeGitHub(
        {(q, None): load("search_p1.json"), (q, "CURSOR_PAGE_1"): load("search_p2.json")},
        load("repo.json"),
    )


# --- slices ----------------------------------------------------------------------------------


def d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def test_month_slices_clip_to_the_window() -> None:
    assert month_slices(d("2025-01-15"), d("2025-03-10")) == [
        (d("2025-01-15"), d("2025-01-31")),
        (d("2025-02-01"), d("2025-02-28")),
        (d("2025-03-01"), d("2025-03-10")),
    ]


def test_month_slices_handle_leap_years_and_single_days() -> None:
    assert month_slices(d("2024-02-01"), d("2024-02-29")) == [(d("2024-02-01"), d("2024-02-29"))]
    assert month_slices(d("2025-05-05"), d("2025-05-05")) == [(d("2025-05-05"), d("2025-05-05"))]
    assert month_slices(d("2024-12-20"), d("2025-01-02")) == [
        (d("2024-12-20"), d("2024-12-31")),
        (d("2025-01-01"), d("2025-01-02")),
    ]


# --- parsing ---------------------------------------------------------------------------------


def test_parse_pr_from_the_fixture_file() -> None:
    node = load("search_p1.json")["data"]["search"]["nodes"][0]
    pr = parse_pr(node)
    assert pr.number == 101 and pr.merge_commit == "1" * 40 and pr.base_ref == "main"
    assert [f.path for f in pr.files] == [
        "src/tinylib/clamp.py",
        "tests/test_clamp.py",
        "CHANGES.md",
    ]
    assert pr.closing_issues[0].number == 99 and pr.closing_issues[0].repo == REPO
    assert pr.merged_at == dt.datetime(2025, 1, 10, 12, 0, tzinfo=dt.UTC)


def test_parse_handles_null_author_null_merge_commit_and_null_body() -> None:
    pr = make_pr(author_login=None, author_type=None, merge_commit=None)
    node = to_node(pr)
    node["closingIssuesReferences"]["nodes"][0]["body"] = None
    parsed = parse_pr(node)
    assert parsed.author_login is None and parsed.merge_commit is None
    assert parsed.closing_issues[0].body == ""


def test_helper_node_roundtrips_through_the_parser() -> None:
    pr = make_pr()
    assert parse_pr(to_node(pr)) == pr


def test_parse_repo() -> None:
    meta = parse_repo(load("repo.json")["data"])
    assert (meta.license_spdx, meta.default_branch, meta.archived) == (
        "BSD-3-Clause",
        "main",
        False,
    )
    with pytest.raises(GitHubError, match="not found"):
        parse_repo({"repository": None})


# --- client ----------------------------------------------------------------------------------


def http_error(code: int, retry_after: str | None = None) -> HTTPError:
    headers = Message()
    if retry_after:
        headers["Retry-After"] = retry_after
    return HTTPError("https://api.github.com/graphql", code, "x", headers, None)


def test_client_requires_a_token_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with pytest.raises(MissingTokenError, match=TOKEN_ENV):
        GitHubClient()
    monkeypatch.setenv(TOKEN_ENV, "ghp_not_a_real_token")
    GitHubClient()  # constructs fine; no request is made until query() is called


def test_client_retries_transient_http_errors_then_succeeds() -> None:
    attempts: list[int] = []
    sleeps: list[float] = []

    def transport(query: str, variables: dict[str, Any]) -> dict[str, Any]:
        attempts.append(1)
        if len(attempts) < 3:
            raise http_error(502)
        return repo_response()

    client = GitHubClient(transport=transport, sleep=sleeps.append)
    assert client.query("q", {})["repository"]["nameWithOwner"] == REPO
    assert len(attempts) == 3 and sleeps == [1.0, 2.0]  # exponential backoff


def test_client_honours_retry_after_and_gives_up_eventually() -> None:
    sleeps: list[float] = []

    def transport(query: str, variables: dict[str, Any]) -> dict[str, Any]:
        raise http_error(403, retry_after="7")

    client = GitHubClient(transport=transport, sleep=sleeps.append, max_retries=2)
    with pytest.raises(GitHubError, match="HTTP 403"):
        client.query("q", {})
    assert sleeps == [7.0, 7.0]


def test_client_does_not_retry_permanent_errors() -> None:
    calls: list[int] = []

    def transport(query: str, variables: dict[str, Any]) -> dict[str, Any]:
        calls.append(1)
        raise http_error(401)

    with pytest.raises(GitHubError, match="HTTP 401"):
        GitHubClient(transport=transport, sleep=lambda _s: None).query("q", {})
    assert calls == [1]


def test_client_raises_on_graphql_errors_and_exhausted_rate_limit() -> None:
    err = GitHubClient(transport=lambda q, v: {"errors": [{"message": "Field 'x' doesn't exist"}]})
    with pytest.raises(GitHubError, match="doesn't exist"):
        err.query("q", {})
    low = repo_response()
    low["data"]["rateLimit"]["remaining"] = 3
    with pytest.raises(GitHubError, match="rate limit"):
        GitHubClient(transport=lambda q, v: low).query("q", {})


def test_client_records_responses_for_fixture_capture(tmp_path: Path) -> None:
    client = GitHubClient(transport=lambda q, v: repo_response(), record_dir=tmp_path / "rec")
    client.query("query A", {"owner": "o"})
    [saved] = list((tmp_path / "rec").glob("*.json"))
    payload = json.loads(saved.read_text())
    assert payload["variables"] == {"owner": "o"}
    assert payload["response"]["data"]["repository"]["nameWithOwner"] == REPO


# --- repo check and mining -------------------------------------------------------------------


def test_check_repo_accepts_matching_license_and_rejects_mismatch() -> None:
    ok = GitHubClient(transport=FakeGitHub({}, load("repo.json")))
    assert check_repo(ok, resolved()).default_branch == "main"
    wrong = GitHubClient(transport=FakeGitHub({}, repo_response("Apache-2.0")))
    with pytest.raises(ConfigMismatchError, match=r"Apache-2\.0"):
        check_repo(wrong, resolved())
    none = GitHubClient(transport=FakeGitHub({}, repo_response(None)))
    with pytest.raises(ConfigMismatchError):
        check_repo(none, resolved())


def test_mine_pages_through_slices_and_stores_only_prs() -> None:
    fake = january_fake()
    store = StageStore(":memory:")
    stats = mine(GitHubClient(transport=fake), store, resolved(), today=dt.date(2025, 6, 1))
    assert [n for n, _ in store.candidates()] == [101, 102, 103]  # the empty node is skipped
    assert (stats.slices_fetched, stats.slices_skipped, stats.prs_seen) == (2, 0, 3)
    queries = [v["q"] for v in fake.searches]
    assert queries[0].endswith("merged:2025-01-01..2025-01-31")
    assert (
        queries[1].endswith("merged:2025-01-01..2025-01-31")
        and fake.searches[1]["after"] == "CURSOR_PAGE_1"
    )
    assert queries[2].endswith("merged:2025-02-01..2025-02-28")
    assert all("repo:synthetic/tinylib is:pr is:merged" in q for q in queries)


def test_mine_is_resumable_finished_slices_are_not_refetched() -> None:
    fake = january_fake()
    store = StageStore(":memory:")
    client = GitHubClient(transport=fake)
    mine(client, store, resolved(), today=dt.date(2025, 6, 1))
    before = len(fake.searches)
    again = mine(client, store, resolved(), today=dt.date(2025, 6, 1))
    assert len(fake.searches) == before  # zero new requests
    assert (again.slices_fetched, again.slices_skipped) == (0, 2)
    assert store.candidate_count() == 3  # nothing duplicated


def test_mine_refetches_the_slice_that_contains_today() -> None:
    fake = january_fake()
    store = StageStore(":memory:")
    client = GitHubClient(transport=fake)
    mine(client, store, resolved(), today=dt.date(2025, 2, 15))  # February is still open
    again = mine(client, store, resolved(), today=dt.date(2025, 2, 15))
    assert (again.slices_fetched, again.slices_skipped) == (1, 1)


def test_mine_refuses_to_silently_truncate_at_the_search_ceiling() -> None:
    q = "repo:synthetic/tinylib is:pr is:merged merged:2025-01-01..2025-01-31"
    full = search_response([to_node(make_pr())], count=1000)
    fake = FakeGitHub({(q, None): full}, load("repo.json"))
    with pytest.raises(GitHubError, match="ceiling"):
        mine(
            GitHubClient(transport=fake),
            StageStore(":memory:"),
            resolved("2025-01-01", "2025-01-31"),
        )


def test_mine_makes_no_network_calls_with_an_injected_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket

    def no_network(*_a: object, **_k: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", no_network)
    store = StageStore(":memory:")
    mine(GitHubClient(transport=january_fake()), store, resolved(), today=dt.date(2025, 6, 1))
    assert store.candidate_count() == 3


def dual_licensed(**over: Any) -> ResolvedRepo:
    return resolved().model_copy(update={"license": "MIT OR Apache-2.0", **over})


def test_dual_licensed_repos_pass_when_the_github_id_is_configured() -> None:
    """GitHub reports dual-licensed repos as NOASSERTION (verified for hynek/structlog)."""
    client = GitHubClient(transport=FakeGitHub({}, repo_response("NOASSERTION")))
    repo = dual_licensed(github_license_id="NOASSERTION")
    assert check_repo(client, repo).license_spdx == "NOASSERTION"


def test_a_dual_licensed_repo_without_the_github_id_is_refused_which_is_why_the_field_exists() -> (
    None
):
    client = GitHubClient(transport=FakeGitHub({}, repo_response("NOASSERTION")))
    with pytest.raises(ConfigMismatchError, match="NOASSERTION"):
        check_repo(client, dual_licensed())


def test_a_configured_github_id_is_still_checked() -> None:
    client = GitHubClient(transport=FakeGitHub({}, repo_response("MIT")))
    with pytest.raises(
        ConfigMismatchError, match=r"expects GitHub to report license 'NOASSERTION'"
    ):
        check_repo(client, dual_licensed(github_license_id="NOASSERTION"))
