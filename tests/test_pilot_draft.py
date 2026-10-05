"""The draft-only live pilot, on the mock provider and the tiny fixture repo. All SYNTHETIC.

The fixture repo is a local git repo; the GitHub reader is a stub that records every call. Nothing
here touches the network, and no test spends money.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from cascade.agents.schemas import PatchOutput
from cascade.cli import app
from cascade.config import PilotRepo, ReposConfig, ResolvedBudget, load_repos
from cascade.harness.providers.mock import MockModelSource, final_step
from cascade.pilot import draft as dr
from cascade.pilot import github_read as gh
from cascade.pilot.github_read import GitHubIssueReader, IssueData, IssueReadError
from cascade.taskgen.gitrepo import run_git
from tests.agent_helpers import (
    CLAMP,
    approve_step,
    localize_steps,
    native_factory,
    patch_steps,
    pipeline_config,
    repro_steps,
    triage_step,
)
from tests.conftest import FixtureRepo
from tests.matrix_helpers import matrix_budget, matrix_models, new_ledger

runner = CliRunner()
REPO = "acme/tinylib"

REPOS_YAML = """\
date_window: {{start: TODO, end: TODO}}
repos:
  - {{name: {repo}, license: MIT, python_version: "3.12", install_cmd: pip install -e ., test_cmd: python -m pytest}}
pilot_repos:
{pilot}
"""


def repos_cfg(pilot: str = f"  - {{name: {REPO}, status: owned}}") -> ReposConfig:
    return ReposConfig.model_validate(yaml.safe_load(REPOS_YAML.format(repo=REPO, pilot=pilot)))


class FakeGitHub:
    def __init__(self, *, is_pr: bool = False) -> None:
        self.calls: list[tuple[str, int]] = []
        self.is_pr = is_pr

    def issue(self, repo: str, number: int) -> IssueData:
        self.calls.append((repo, number))
        return IssueData(
            number=number, title="clamp returns low for big values",
            body="clamp(99, 0, 10) returns 0 but should return 10.", state="open",
            url=f"https://github.com/{repo}/issues/{number}", labels=["bug"],
            is_pull_request=self.is_pr,
        )  # fmt: skip


def good_plans() -> dict[str, Any]:
    return {
        "triager": [triage_step()], "reproducer": repro_steps(), "localizer": localize_steps(),
        "patcher": patch_steps(), "reviewer": [approve_step()],
    }  # fmt: skip


class Harness:
    def __init__(self, tmp: Path, fx: FixtureRepo, **budget_over: Any) -> None:
        over = {"buckets": {"live_pilot": 5.0}, **budget_over}
        self.tmp, self.fx = tmp, fx
        self.budget: ResolvedBudget = matrix_budget(tmp, **over)
        self.models = matrix_models()
        self.ledger = new_ledger(tmp, **over)
        self.github = FakeGitHub()
        self.clones: list[tuple[str, Path]] = []
        self.drafts = tmp / "drafts"

    def clone(self, repo: str, dest: Path) -> Path:
        self.clones.append((repo, dest))
        # a bare clone whose default branch sits at the commit BEFORE the fix, like a live repo
        # whose bug is still present
        if not dest.exists():  # like ensure_clone: reuse an existing clone
            dest.parent.mkdir(parents=True, exist_ok=True)
            run_git(None, "clone", "--quiet", "--bare", str(self.fx.path), str(dest))
            run_git(dest, "update-ref", "refs/heads/main", self.fx.base_sha)
        return dest

    def go(self, **kw: Any) -> dr.DraftResult:
        args: dict[str, Any] = {
            "cfg": repos_cfg(), "models": self.models, "budget": self.budget,
            "ledger": self.ledger, "source": MockModelSource(good_plans()),
            "github": self.github, "repo": REPO, "issue_number": 7, "policy": "always_cheap",
            "drafts_root": self.drafts, "work_root": self.tmp / "work",
            "pipeline_config": pipeline_config(), "clone": self.clone,
            "executor_factory_for": lambda _clone, _commit: native_factory,
        }  # fmt: skip
        args.update(kw)
        return dr.run_draft(**args)


@pytest.fixture
def h(tmp_path: Path, fixture_repo: FixtureRepo) -> Harness:
    return Harness(tmp_path, fixture_repo)


# --- a successful draft -------------------------------------------------------------------------


def test_a_draft_writes_a_patch_pr_text_with_disclosure_and_metadata(h: Harness) -> None:
    result = h.go()
    d = h.drafts / "acme__tinylib" / "7"
    assert result.directory == d and result.verified and result.patch_written
    assert "+        return high\n" in (d / "patch.diff").read_text()
    pr = (d / "PR_DESCRIPTION.md").read_text()
    assert "## AI-assistance disclosure" in pr and "drafted by an AI system" in pr
    assert "Refs #7" in pr and "**confirmed**" in pr and CLAMP in pr
    meta = json.loads((d / "draft.json").read_text())
    assert meta["repo"] == REPO and meta["issue"] == 7 and meta["policy"] == "always_cheap"
    assert meta["verified"] is True and meta["purpose"] == "live_pilot"
    assert meta["bucket"] == "live_pilot" and meta["provider"] == "mock"
    assert meta["arms_tried"] == ["mock-luna@low"] and len(meta["runs"]) == 1
    assert meta["base_commit"] == h.fx.base_sha


def test_every_ledger_row_is_tagged_live_pilot_in_the_live_pilot_bucket(h: Harness) -> None:
    result = h.go()
    rows = h.ledger.rows()
    assert rows and {r.purpose for r in rows} == {"live_pilot"}
    assert {r.bucket for r in rows} == {"live_pilot"}
    meta = json.loads((result.directory / "draft.json").read_text())
    assert meta["cost_usd"] > 0
    assert meta["cost_usd"] == pytest.approx(sum(r.cost_usd for r in rows))
    assert result.cost_usd == pytest.approx(meta["cost_usd"])


def test_the_live_pilot_bucket_caps_the_draft(tmp_path: Path, fixture_repo: FixtureRepo) -> None:
    h = Harness(tmp_path, fixture_repo, buckets={"live_pilot": 1e-9})
    result = h.go()
    assert not result.verified and not result.patch_written
    meta = json.loads((result.directory / "draft.json").read_text())
    assert "bucket cap" in (meta["runs"][0]["error"] or "")
    assert sum(r.cost_usd for r in h.ledger.rows()) <= 1e-9
    assert "did **NOT** confirm" in (result.directory / "PR_DESCRIPTION.md").read_text()
    assert (result.directory / "patch.diff").read_text() == ""


def test_an_unverified_patch_is_labelled_unverified(h: Harness) -> None:
    plans = good_plans()
    plans["patcher"] = [final_step(PatchOutput(summary="x", files_changed=[], confidence="low"))]
    result = h.go(source=MockModelSource(plans), pipeline_config=pipeline_config(max_repairs=0))
    assert not result.verified
    assert "did **NOT** confirm" in (result.directory / "PR_DESCRIPTION.md").read_text()


def test_the_cascade_policy_stops_at_the_first_verified_arm(h: Harness) -> None:
    result = h.go(policy="cascade")
    assert [a.label for a in result.arms] == ["cheap"] and result.verified


def test_the_cascade_policy_escalates_until_an_arm_is_verified(h: Harness) -> None:
    plans = {
        "triager": [triage_step()] * 3,
        "reproducer": repro_steps() + repro_steps(),
        "localizer": localize_steps() + localize_steps(),
        "patcher": [
            final_step(PatchOutput(summary="x", files_changed=[], confidence="low")),
            *patch_steps(),
        ],
        "reviewer": [approve_step()] * 2,
    }
    result = h.go(
        policy="cascade", source=MockModelSource(plans),
        pipeline_config=pipeline_config(max_repairs=0),
    )  # fmt: skip
    assert [a.label for a in result.arms] == ["cheap", "mid"] and result.verified
    meta = json.loads((result.directory / "draft.json").read_text())
    assert meta["arms_tried"] == ["mock-luna@low", "mock-luna@high"]
    assert meta["cost_usd"] == pytest.approx(sum(r.cost_usd for r in h.ledger.rows()))


def test_a_fixed_policy_uses_its_arm(h: Harness) -> None:
    assert [a.label for a in h.go(policy="always_max").arms] == ["max"]


def test_the_predictive_policy_uses_the_chosen_arm(h: Harness) -> None:
    seen: list[dr.AgentView] = []

    def choose(view: dr.AgentView) -> int:
        seen.append(view)
        return 1

    result = h.go(policy="predictive_issue", choose_arm=choose)
    assert [a.label for a in result.arms] == ["mid"] and seen[0].repo == REPO


def test_the_predictive_chooser_sees_only_issue_side_features() -> None:
    captured: list[Any] = []

    class StubRouter:
        def choose(self, records: Any) -> list[int]:
            captured.extend(records)
            return [2]

    view = dr.AgentView(
        task_id="live:x#1", repo=REPO, base_commit="a" * 40, issue_title="boom",
        issue_body="Traceback (most recent call last):\n  File x\nValueError",
    )  # fmt: skip
    assert dr.predictive_chooser(StubRouter())(view) == 2  # type: ignore[arg-type]
    task = captured[0].task
    assert task.features.issue_has_traceback is True
    assert task.features.src_files == 0 and task.features.f2p_count == 0  # nothing gold-derived


# --- refusals: before any network, any model call, or any file ----------------------------------


def refused(h: Harness, match: str, **kw: Any) -> None:
    with pytest.raises(dr.DraftRefusal, match=match):
        h.go(**kw)
    assert h.github.calls == [] and h.clones == [] and h.ledger.rows() == []
    assert not h.drafts.exists()


def test_a_repo_not_listed_under_pilot_repos_is_refused(h: Harness) -> None:
    refused(h, "not listed under pilot_repos", repo="someone/else")
    refused(h, "not listed under pilot_repos", cfg=repos_cfg(pilot="  []"))


def test_a_listed_pilot_repo_must_also_define_its_commands(h: Harness) -> None:
    cfg = repos_cfg(pilot="  - {name: other/repo, status: owned}")
    refused(h, "not defined under `repos:`", cfg=cfg, repo="other/repo")


def test_an_opt_in_repo_needs_a_maintainer_contact_note() -> None:
    with pytest.raises(ValidationError, match="maintainer_contact"):
        PilotRepo(name="o/r", status="opt_in")
    with pytest.raises(ValidationError, match="maintainer_contact"):
        PilotRepo(name="o/r", status="opt_in", maintainer_contact="   ")
    assert PilotRepo(name="o/r", status="opt_in", maintainer_contact="agreed by email").status


def test_an_opt_in_repo_with_a_contact_note_is_allowed(h: Harness) -> None:
    pilot = f'  - {{name: {REPO}, status: opt_in, maintainer_contact: "Jo agreed, issue 3"}}'
    assert h.go(cfg=repos_cfg(pilot)).verified


def test_a_status_other_than_owned_or_opt_in_is_not_valid() -> None:
    with pytest.raises(ValidationError):
        PilotRepo(name="o/r", status="third_party")  # type: ignore[arg-type]


def test_without_a_live_pilot_bucket_the_draft_is_refused(
    tmp_path: Path, fixture_repo: FixtureRepo
) -> None:
    h = Harness(tmp_path, fixture_repo, buckets={"pilot": 1.0})
    refused(h, "buckets.live_pilot")


def test_an_unknown_policy_is_refused(h: Harness) -> None:
    refused(h, "unknown policy", policy="vibes")
    refused(h, "needs a router", policy="predictive_issue")


def test_a_pull_request_is_refused_and_an_existing_draft_needs_force(h: Harness) -> None:
    h.github.is_pr = True
    with pytest.raises(dr.DraftRefusal, match="pull request"):
        h.go()
    h.github.is_pr = False
    first = h.go()
    with pytest.raises(dr.DraftRefusal, match="already has a draft"):
        h.go()
    calls_before = len(h.ledger.rows())
    again = h.go(force=True)
    assert again.directory == first.directory and len(h.ledger.rows()) > calls_before


# --- it can only read ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = json.dumps(body).encode()

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *_a: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


ISSUE_JSON = {
    "number": 7, "title": "t", "body": None, "state": "open", "labels": [{"name": "bug"}, "x"],
    "html_url": "https://github.com/acme/tinylib/issues/7",
}  # fmt: skip


def test_the_reader_only_issues_a_get_for_one_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[urllib.request.Request] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float = 0) -> _Resp:
        sent.append(req)
        return _Resp(ISSUE_JSON)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    issue = GitHubIssueReader().issue(REPO, 7)
    assert issue.body == "" and issue.labels == ["bug"] and not issue.is_pull_request
    [req] = sent
    assert req.get_method() == "GET" and req.data is None
    assert req.full_url == "https://api.github.com/repos/acme/tinylib/issues/7"
    assert not req.has_header("Authorization")


def test_the_token_comes_from_the_environment_and_is_sent_only_as_a_bearer_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[urllib.request.Request] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float = 0) -> _Resp:
        sent.append(req)
        return _Resp(ISSUE_JSON)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GITHUB_TOKEN", "tok-synthetic")
    GitHubIssueReader().issue(REPO, 7)
    assert sent[0].get_header("Authorization") == "Bearer tok-synthetic"
    assert "tok-synthetic" not in sent[0].full_url


def test_the_reader_refuses_bad_references_and_odd_responses() -> None:
    reader = GitHubIssueReader(transport=lambda _url: {"number": 1})
    with pytest.raises(IssueReadError, match="response shape"):
        reader.issue(REPO, 1)
    with pytest.raises(IssueReadError, match="bad issue reference"):
        reader.issue("noslash", 1)
    with pytest.raises(IssueReadError, match="bad issue reference"):
        reader.issue(REPO, 0)
    pr = {**ISSUE_JSON, "pull_request": {"url": "x"}}
    assert GitHubIssueReader(transport=lambda _u: pr).issue(REPO, 7).is_pull_request


def test_the_urllib_transport_refuses_a_non_api_url() -> None:
    with pytest.raises(IssueReadError, match="non-GitHub-API"):
        gh._urllib_transport(None)("https://evil.example/x")  # pyright: ignore[reportPrivateUsage]


def test_the_pilot_code_has_no_way_to_write_to_github() -> None:
    root = Path(dr.__file__).parent
    code = "\n".join(
        re.sub(r'""".*?"""', "", p.read_text(), flags=re.S)
        for p in (root / "draft.py", root / "github_read.py", root / "reviews.py")
    )
    for forbidden in (
        r"method=\"(?!GET\")", r"\.post\(", r"\.put\(", r"\.patch\(", r"\.delete\(",
        r"git push", r"\"push\"", r"create_pull", r"add_issue_comment", r"issue_write",
        r"/pulls", r"/comments", r"gh pr", r"urlopen\(.*data=",
    ):  # fmt: skip
        assert not re.search(forbidden, code), forbidden


# --- the CLI ------------------------------------------------------------------------------------


def write_configs(tmp: Path, pilot: str) -> Path:
    configs = tmp / "configs"
    configs.mkdir()
    (configs / "repos.yaml").write_text(REPOS_YAML.format(repo=REPO, pilot=pilot))
    return configs


def test_the_cli_refuses_an_unlisted_repo_before_reading_anything_else(tmp_path: Path) -> None:
    configs = write_configs(tmp_path, "  []")  # no models.yaml or budget.yaml either
    runs, drafts = tmp_path / "runs", tmp_path / "drafts"
    r = runner.invoke(
        app,
        ["draft", "--repo", REPO, "--issue", "7", "--configs-dir", str(configs),
         "--runs-dir", str(runs), "--drafts-dir", str(drafts)],
    )  # fmt: skip
    assert r.exit_code == 2 and "not listed under pilot_repos" in r.output
    assert not runs.exists() and not drafts.exists()  # no ledger, no draft, nothing


def test_the_cli_without_a_policy_and_without_a_final_evaluation_stops(tmp_path: Path) -> None:
    configs = write_configs(tmp_path, f"  - {{name: {REPO}, status: owned}}")
    r = runner.invoke(
        app,
        ["draft", "--repo", REPO, "--issue", "7", "--configs-dir", str(configs),
         "--final-dir", str(tmp_path / "none"), "--lock-path", str(tmp_path / "none.lock"),
         "--runs-dir", str(tmp_path / "runs"), "--drafts-dir", str(tmp_path / "drafts")],
    )  # fmt: skip
    assert r.exit_code == 2 and "no final evaluation" in r.output
    assert not (tmp_path / "runs").exists()


def test_the_repos_file_in_the_repository_loads_and_has_the_pilot_off() -> None:
    cfg = load_repos(Path("configs/repos.yaml"))
    assert cfg.pilot_repos == [] and cfg.pilot_repo(REPO) is None


def test_check_allowed_itself_refuses_an_opt_in_repo_without_a_contact_note() -> None:
    # the config schema rejects this at load time; check_allowed must not rely on that alone
    cfg = repos_cfg()
    bad = PilotRepo.model_construct(name=REPO, status="opt_in", maintainer_contact="", notes="")
    with pytest.raises(dr.DraftRefusal, match="no maintainer_contact"):
        dr.check_allowed(cfg.model_copy(update={"pilot_repos": [bad]}), REPO)
    ok = PilotRepo(name=REPO, status="opt_in", maintainer_contact="Jo agreed")
    assert dr.check_allowed(cfg.model_copy(update={"pilot_repos": [ok]}), REPO) is ok


def test_the_cascade_stops_at_a_bucket_cap_instead_of_trying_the_next_arm(
    tmp_path: Path, fixture_repo: FixtureRepo
) -> None:
    h = Harness(tmp_path, fixture_repo, buckets={"live_pilot": 1e-9})
    result = h.go(policy="cascade")
    assert [a.label for a in result.arms] == ["cheap"] and not result.verified
    assert "bucket cap" in (result.arms[0].error or "")
