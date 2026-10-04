"""Mine merged PRs that close issues, in monthly slices, resumably.

Why slices: GitHub search returns at most 1000 results per query, so a long date window is split
into calendar months. A slice is recorded as done only once it lies entirely in the past, so a
slice that includes today is refetched on the next run and picks up newly merged PRs.
"""

from __future__ import annotations

import calendar
import datetime as dt
from dataclasses import dataclass

from cascade.config import ResolvedRepo
from cascade.taskgen.github import (
    REPO_QUERY,
    SEARCH_QUERY,
    GitHubClient,
    GitHubError,
    PRCandidate,
    RepoMeta,
    parse_pr,
    parse_repo,
)
from cascade.taskgen.store import StageStore

PAGE_SIZE = 50
SEARCH_RESULT_CEILING = 1000  # GitHub search cannot return more than this per query


class ConfigMismatchError(GitHubError):
    """repos.yaml disagrees with what GitHub reports for the repository."""


@dataclass(frozen=True)
class MineStats:
    slices_fetched: int
    slices_skipped: int
    prs_seen: int


def month_slices(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    """Split [start, end] into calendar-month slices, clipped to the window."""
    slices: list[tuple[dt.date, dt.date]] = []
    cursor = start
    while cursor <= end:
        last_day = calendar.monthrange(cursor.year, cursor.month)[1]
        slice_end = min(dt.date(cursor.year, cursor.month, last_day), end)
        slices.append((cursor, slice_end))
        cursor = slice_end + dt.timedelta(days=1)
    return slices


def search_string(repo: str, start: dt.date, end: dt.date) -> str:
    return f"repo:{repo} is:pr is:merged merged:{start.isoformat()}..{end.isoformat()}"


def check_repo(client: GitHubClient, repo: ResolvedRepo) -> RepoMeta:
    """Fetch repo metadata and fail if the configured license does not match GitHub's."""
    owner, name = repo.name.split("/", 1)
    meta = parse_repo(client.query(REPO_QUERY, {"owner": owner, "name": name}))
    if meta.license_spdx != repo.license:
        raise ConfigMismatchError(
            f"{repo.name}: repos.yaml says license {repo.license!r}, "
            f"GitHub reports {meta.license_spdx!r}"
        )
    if not meta.default_branch:
        raise GitHubError(f"{repo.name}: GitHub reported no default branch")
    return meta


def mine(
    client: GitHubClient,
    store: StageStore,
    repo: ResolvedRepo,
    *,
    today: dt.date | None = None,
) -> MineStats:
    today = today or dt.datetime.now(dt.UTC).date()
    fetched = skipped = seen = 0
    for start, end in month_slices(repo.window_start, repo.window_end):
        if store.slice_done(start, end):
            skipped += 1
            continue
        after: str | None = None
        while True:
            data = client.query(
                SEARCH_QUERY,
                {"q": search_string(repo.name, start, end), "after": after, "first": PAGE_SIZE},
            )
            search = data["search"]
            if search["issueCount"] >= SEARCH_RESULT_CEILING:
                raise GitHubError(
                    f"{start}..{end} matched {search['issueCount']} PRs, at GitHub's search "
                    "ceiling; results would be silently truncated"
                )
            for node in search["nodes"]:
                if not node:  # non-PR search hits come back as empty objects
                    continue
                candidate: PRCandidate = parse_pr(node)
                store.upsert_candidate(candidate.number, candidate.model_dump(mode="json"))
                seen += 1
            page = search["pageInfo"]
            if not page["hasNextPage"]:
                break
            after = page["endCursor"]
        fetched += 1
        if end < today:  # a slice that includes today may still gain PRs; refetch it next time
            store.mark_slice_done(start, end)
    return MineStats(slices_fetched=fetched, slices_skipped=skipped, prs_seen=seen)
