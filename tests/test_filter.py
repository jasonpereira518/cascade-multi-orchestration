from __future__ import annotations

import datetime as dt

import pytest

from cascade.config import SelectionLimits
from cascade.taskgen import filter as f
from cascade.taskgen.filter import FilterContext, classify_path, filter_metadata, summarize_files
from tests.taskgen_helpers import REPO, changed, make_issue, make_pr

CTX = FilterContext(
    window_start=dt.date(2025, 1, 1),
    window_end=dt.date(2025, 1, 31),
    default_branch="main",
    repo=REPO,
    limits=SelectionLimits(),
)


def src(n: int, lines: int = 2) -> list[object]:
    return [changed(f"src/pkg/mod{i}.py", lines, 0) for i in range(n)]


def files_with(source: int, lines_each: int = 2, tests: int = 1) -> list:
    return [
        *[changed(f"src/pkg/mod{i}.py", lines_each, 0) for i in range(source)],
        *[changed(f"tests/test_{i}.py", 3, 0) for i in range(tests)],
    ]


def test_a_good_pr_passes_every_rule() -> None:
    assert filter_metadata(make_pr(), CTX) is None


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("src/click/core.py", "source"),
        ("click/core.py", "source"),
        ("tests/test_core.py", "test"),
        ("tests/typing/typing_x.py", "test"),
        ("tests/data/sample.json", "test"),
        ("pkg/test_thing.py", "test"),
        ("pkg/thing_test.py", "test"),
        ("pkg/conftest.py", "test"),
        ("CHANGES.md", "docs"),
        ("docs/conf.py", "docs"),
        ("docs/api.rst", "docs"),
        ("examples/app/main.py", "docs"),
        ("README.md", "docs"),
        ("uv.lock", "meta"),
        (".github/workflows/tests.yaml", "meta"),
        ("requirements/dev.txt", "meta"),
        ("requirements.txt", "meta"),
        ("notes.txt", "docs"),
        (".pre-commit-config.yaml", "meta"),
        ("pyproject.toml", "other"),
        ("src/click/py.typed", "other"),
    ],
)
def test_classify_path(path: str, kind: str) -> None:
    assert classify_path(path) == kind


def test_summarize_counts_source_lines_only() -> None:
    s = summarize_files(
        [changed("src/a.py", 10, 5), changed("tests/test_a.py", 99, 1), changed("CHANGES.md", 3, 0)]
    )
    assert (s.src_files, s.src_lines, s.test_files, s.docs_files) == (1, 15, 1, 1)


# One case per reason code. Each PR is a good PR with exactly one thing broken.
REASON_CASES = [
    (f.OUTSIDE_DATE_WINDOW, {"merged_at": dt.datetime(2025, 2, 2, tzinfo=dt.UTC)}),
    (f.NOT_DEFAULT_BRANCH, {"base_ref": "stable"}),
    (f.NO_MERGE_COMMIT, {"merge_commit": None}),
    (f.BOT_AUTHOR, {"author_type": "Bot", "author_login": "some-app"}),
    (f.DEPENDENCY_PR, {"title": "Bump pytest from 8 to 9"}),
    (f.FILES_TRUNCATED, {"files_total": 250}),
    (f.NO_CLOSING_ISSUE, {"closing_issues": [], "closing_issues_total": 0}),
    (
        f.MULTIPLE_CLOSING_ISSUES,
        {"closing_issues": [make_issue(), make_issue(number=98)], "closing_issues_total": 2},
    ),
    (f.CROSS_REPO_ISSUE, {"closing_issues": [make_issue(repo="other/repo")]}),
    (f.EMPTY_ISSUE_BODY, {"closing_issues": [make_issue(body="  \n")]}),
    (f.NO_TEST_FILES, {"files": [changed("src/pkg/a.py", 2, 0)]}),
    (f.DOCS_ONLY, {"files": [changed("docs/a.rst"), changed("CHANGES.md")]}),
    (f.NO_SOURCE_FILES, {"files": [changed("pyproject.toml"), changed("tests/test_a.py")]}),
    (f.TOO_MANY_SOURCE_FILES, {"files": files_with(6)}),
    (f.TOO_MANY_SOURCE_LINES, {"files": files_with(2, lines_each=76)}),
]


@pytest.mark.parametrize(("reason", "override"), REASON_CASES, ids=[r for r, _ in REASON_CASES])
def test_each_rule_fires_with_its_own_reason(reason: str, override: dict) -> None:
    assert filter_metadata(make_pr(**override), CTX) == reason


def test_every_reason_code_is_covered_by_a_case() -> None:
    covered = {r for r, _ in REASON_CASES} | {f.TOO_FEW_SOURCE_FILES}
    declared = {
        v for k, v in vars(f).items()
        if k.isupper() and isinstance(v, str) and v.islower() and "_" in v
        and k not in {"TODO_SENTINEL"}
    }  # fmt: skip
    assert declared <= covered, declared - covered


def test_size_boundaries_are_inclusive() -> None:
    assert filter_metadata(make_pr(files=files_with(5)), CTX) is None  # 5 source files
    assert filter_metadata(make_pr(files=files_with(6)), CTX) == f.TOO_MANY_SOURCE_FILES
    assert filter_metadata(make_pr(files=files_with(2, lines_each=75)), CTX) is None  # 150 lines
    assert (
        filter_metadata(
            make_pr(files=[*files_with(2, lines_each=75), changed("src/pkg/x.py", 1)]), CTX
        )
        == f.TOO_MANY_SOURCE_LINES
    )  # 151 lines


def test_min_source_files_is_configurable() -> None:
    ctx = FilterContext(**{**CTX.__dict__, "limits": SelectionLimits(min_source_files=2)})
    assert filter_metadata(make_pr(files=files_with(1)), ctx) == f.TOO_FEW_SOURCE_FILES
    assert filter_metadata(make_pr(files=files_with(2)), ctx) is None


def test_window_is_inclusive_and_uses_utc() -> None:
    first = dt.datetime(2025, 1, 1, 0, 0, tzinfo=dt.UTC)
    last = dt.datetime(2025, 1, 31, 23, 59, tzinfo=dt.UTC)
    assert filter_metadata(make_pr(merged_at=first), CTX) is None
    assert filter_metadata(make_pr(merged_at=last), CTX) is None
    # 23:30 on Jan 31 in UTC-5 is already Feb 1 in UTC: outside the window.
    est = dt.timezone(dt.timedelta(hours=-5))
    late_local = dt.datetime(2025, 1, 31, 23, 30, tzinfo=est)
    assert filter_metadata(make_pr(merged_at=late_local), CTX) == f.OUTSIDE_DATE_WINDOW
    assert (
        filter_metadata(make_pr(merged_at=dt.datetime(2024, 12, 31, 23, 59, tzinfo=dt.UTC)), CTX)
        == f.OUTSIDE_DATE_WINDOW
    )


@pytest.mark.parametrize(
    ("login", "kind", "is_bot"),
    [
        ("dependabot", "Bot", True),
        ("dependabot[bot]", "User", True),
        ("pre-commit-ci", "User", True),
        ("renovate", "User", True),
        ("some-app", "Bot", True),
        ("alice", "User", False),
        (None, None, False),  # deleted accounts come back as null
    ],
)
def test_bot_detection(login: str | None, kind: str | None, is_bot: bool) -> None:
    pr = make_pr(author_login=login, author_type=kind)
    assert f.is_bot(pr) is is_bot


@pytest.mark.parametrize(
    "title",
    [
        "Bump pytest from 8 to 9",
        "bump version to 8.1.1",
        "[pre-commit.ci] pre-commit autoupdate",
        "pre-commit autoupdate",
        "Update all dependencies",
    ],
)
def test_dependency_titles(title: str) -> None:
    assert filter_metadata(make_pr(title=title), CTX) == f.DEPENDENCY_PR


def test_dependency_by_label_or_by_files() -> None:
    assert filter_metadata(make_pr(labels=["Dependencies"]), CTX) == f.DEPENDENCY_PR
    only_lock = make_pr(files=[changed("uv.lock", 4, 4)])
    assert f.is_dependency_pr(only_lock)
    assert not f.is_dependency_pr(make_pr())


def test_first_failing_rule_wins() -> None:
    # Outside the window AND a bot AND no issue: the earliest rule in the order is reported.
    pr = make_pr(
        merged_at=dt.datetime(2020, 1, 1, tzinfo=dt.UTC), author_type="Bot", closing_issues=[],
        closing_issues_total=0,
    )  # fmt: skip
    assert filter_metadata(pr, CTX) == f.OUTSIDE_DATE_WINDOW


def test_docs_plus_tests_without_source_is_no_source_files_not_docs_only() -> None:
    pr = make_pr(files=[changed("docs/a.rst"), changed("tests/test_a.py", 2, 0)])
    assert filter_metadata(pr, CTX) == f.NO_SOURCE_FILES
