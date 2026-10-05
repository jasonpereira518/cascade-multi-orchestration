"""The draft-only live pilot: run a routing policy on one real issue and write a DRAFT for a human.

What it does: read one issue (read-only), run the chosen policy's arm(s) through the same agent
pipeline the experiments use, and write ``patch.diff``, ``PR_DESCRIPTION.md`` (with an AI-assistance
disclosure), and ``draft.json`` into ``pilot/drafts/<owner__name>/<issue>/``.

What it never does (CLAUDE.md hard rule 6): push, open a pull request, or comment on an issue. The
only network it uses is the read-only issue fetch and a read-only ``git clone``/``fetch``; this
module imports nothing that can write to GitHub, and a test scans it for write verbs.

Guards, in order, each BEFORE any network call or model call:
1. the repo must be listed under ``pilot_repos`` (owned, or opt_in with a maintainer_contact note);
2. the repo must be defined under ``repos`` with filled install/test commands (it needs an image);
3. ``buckets.live_pilot`` must be set in budget.yaml (the draft is capped by it, and every call is
   tagged ``purpose=live_pilot``);
4. the policy name must be valid.

There are no hidden tests on a live issue, so nothing is graded: the draft carries the Verifier's
verdict (the agent's reproducer plus the repo's visible tests) and says plainly when the Verifier
did NOT confirm the patch. Untrusted issue text reaches the agents (sandboxed, network off); the
draft is for human review, never automatic submission.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cascade.agents.pipeline import AgentPipeline, PipelineConfig
from cascade.agents.runner import ExecutorFactory
from cascade.config import (
    ArmSpec,
    ModelsConfig,
    PilotRepo,
    ReposConfig,
    ResolvedBudget,
    ResolvedRepo,
    SelectionLimits,
)
from cascade.errors import CascadeError
from cascade.harness.ledger import LIVE_PILOT_PURPOSE, STANDARD_TIER, Ledger
from cascade.harness.models import AttemptRef
from cascade.harness.providers.base import ModelSource
from cascade.harness.sandbox import LeakGuard, SandboxRunner, prepare_workspace
from cascade.harness.trace import TraceWriter
from cascade.pilot.github_read import IssueData, IssueReader
from cascade.routing.matrix import interruption_reason
from cascade.routing.policies import ARMS, Cell, FinalRouter, TaskRecord
from cascade.taskgen.features import compute_features, split_patch
from cascade.taskgen.gitrepo import run_git
from cascade.taskgen.schema import AgentView, Task

LIVE_BUCKET = "live_pilot"
POLICIES = ("always_cheap", "always_mid", "always_max", "cascade", "predictive_issue")
_FIXED = {"always_cheap": "cheap", "always_mid": "mid", "always_max": "max"}
DRAFT_FILES = ("patch.diff", "PR_DESCRIPTION.md", "draft.json")
CloneFn = Callable[[str, Path], Path]  # (owner/name, destination) -> path of a (bare) clone
# (clone path, base commit) -> factory building the test executor around a container runner
ExecutorFactoryFor = Callable[[Path, str], ExecutorFactory]


class DraftRefusal(CascadeError):
    """The draft was refused before anything was fetched or run."""


def repo_slug(repo: str) -> str:
    return repo.replace("/", "__")


def draft_dir(root: Path, repo: str, issue: int) -> Path:
    return root / repo_slug(repo) / str(issue)


def check_allowed(cfg: ReposConfig, repo: str) -> PilotRepo:
    """Refuse any repo that is not an owned or opt-in pilot repo. Runs before any network call."""
    pilot = cfg.pilot_repo(repo)
    if pilot is None:
        raise DraftRefusal(
            f"{repo} is not listed under pilot_repos in configs/repos.yaml, so no draft will be "
            "made. The pilot runs only on repos you own or whose maintainers opted in."
        )
    if pilot.status == "opt_in" and not pilot.maintainer_contact.strip():
        raise DraftRefusal(f"{repo} is opt_in but has no maintainer_contact note; refusing")
    return pilot


def resolved_for_build(name: str, cfg: ReposConfig) -> ResolvedRepo:
    """A ResolvedRepo for the image build: needs filled commands, not the mining date window."""
    spec = cfg.spec(name)
    if spec is None:
        raise DraftRefusal(
            f"{name} is a pilot repo but is not defined under `repos:` in configs/repos.yaml; "
            "add its license, python_version, install_cmd and test_cmd (enabled: false is fine)"
        )
    license_, python, install, test = (
        spec.license,
        spec.python_version,
        spec.install_cmd,
        spec.test_cmd,
    )
    if any(f.startswith("TODO") for f in (license_, python, install, test)):
        raise DraftRefusal(f"repos.yaml has unfilled TODO fields for {name}")
    placeholder = dt.date(2000, 1, 1)  # the mining window is irrelevant to a live draft
    return ResolvedRepo(
        name=spec.name, license=license_, python_version=python, install_cmd=install,
        test_cmd=test, system_packages=tuple(spec.system_packages),
        github_license_id=spec.github_license_id, window_start=placeholder,
        window_end=placeholder, limits=SelectionLimits(),
    )  # fmt: skip


@dataclass(frozen=True)
class ArmRun:
    arm_id: str
    label: str
    claimed: bool
    calls: int
    cost_usd: float
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    error: str | None
    patch_lines: int
    reasons: list[str]


@dataclass(frozen=True)
class DraftResult:
    directory: Path
    policy: str
    arms: list[ArmRun]
    verified: bool
    cost_usd: float
    patch_written: bool


def _patch_stats(patch: str) -> tuple[list[str], int, int]:
    files: list[str] = []
    added = removed = 0
    for path, lines in split_patch(patch):
        files.append(path)
        added += sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
        removed += sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))
    return files, added, removed


def render_pr_description(
    *, repo: str, issue: IssueData, policy: str, arms: Sequence[ArmRun],
    models: Sequence[str], patch: str, verified: bool, base_commit: str,
) -> str:  # fmt: skip
    files, added, removed = _patch_stats(patch)
    changed = (
        "\n".join(f"- `{f}`" for f in files) + f"\n\n+{added} / -{removed} lines"
        if files
        else "_No patch was produced._"
    )
    last = arms[-1] if arms else None
    status = (
        "The tool's Verifier **confirmed** this patch: the agent's reproducer test failed on the "
        "original code and passes now, and the repository's visible tests did not regress. This is "
        "NOT a substitute for review, and no hidden tests were involved."
        if verified
        else "The tool's Verifier did **NOT** confirm this patch. Treat it as an unverified "
        "suggestion that probably needs substantial work."
    )
    reasons = "\n".join(f"- {r}" for r in (last.reasons if last else [])) or "- (none recorded)"
    tried = ", ".join(f"`{a.arm_id}`" for a in arms) or "none"
    used = ", ".join(f"`{m}`" for m in models) or "none"
    return f"""# {issue.title}

Refs #{issue.number} ({issue.url})

## What this changes

{changed}

## AI-assistance disclosure

**This change was drafted by an AI system, not written by a person.** It was produced by Cascade, an
automated issue-to-patch pipeline, using the model(s) {used} (policy `{policy}`; arms tried:
{tried}), from the text of {repo}#{issue.number} at commit `{base_commit[:12]}`.

{status}

Verifier notes:

{reasons}

## Human review (to be completed by the submitter before this is sent anywhere)

- [ ] I read every changed line and understand it.
- [ ] I ran the project's full test suite locally.
- [ ] I am submitting this myself and take responsibility for it, with this disclosure kept.
"""


def predictive_chooser(router: FinalRouter) -> Callable[[AgentView], int]:
    """Arm index from the ISSUE-side features of a live issue (no gold-derived feature is used)."""

    def choose(view: AgentView) -> int:
        features = compute_features(
            issue_title=view.issue_title, issue_body=view.issue_body, gold_patch="",
            test_patch="", fail_to_pass=[], pass_to_pass=[], test_runtime_s=0.0,
        )  # fmt: skip
        task = Task(
            task_id=view.task_id, repo=view.repo, base_commit=view.base_commit,
            issue_title=view.issue_title, issue_body=view.issue_body, gold_patch="live",
            test_patch="live", fail_to_pass=["live"], pass_to_pass=[], image="live",
            features=features,
        )  # fmt: skip
        zero = Cell(False, False, 0.0, 0.0)
        return router.choose([TaskRecord(task, (zero, zero, zero))])[0]

    return choose


def _always_max(_view: AgentView) -> int:
    return len(ARMS) - 1


def _arm(models: ModelsConfig, label: str) -> ArmSpec:
    try:
        return models.arm_by_label_or_id(label)
    except KeyError:
        raise DraftRefusal(f"models.yaml has no arm labelled {label!r}") from None


def run_draft(
    *,
    cfg: ReposConfig,
    models: ModelsConfig,
    budget: ResolvedBudget,
    ledger: Ledger,
    source: ModelSource,
    github: IssueReader,
    repo: str,
    issue_number: int,
    policy: str,
    drafts_root: Path,
    work_root: Path,
    pipeline_config: PipelineConfig,
    clone: CloneFn,
    executor_factory_for: ExecutorFactoryFor,
    choose_arm: Callable[[AgentView], int] | None = None,
    force: bool = False,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    cascade_commit: str = "unknown",
) -> DraftResult:
    check_allowed(cfg, repo)  # 1. the allowlist, before anything else
    resolved = resolved_for_build(repo, cfg)  # 2. the repo's install/test commands
    if LIVE_BUCKET not in budget.buckets:  # 3. the live_pilot bucket
        raise DraftRefusal(
            "budget.yaml has no buckets.live_pilot: set a cap for the live pilot first (every "
            "draft is capped by it)"
        )
    if policy not in POLICIES:  # 4. the policy
        raise DraftRefusal(f"unknown policy {policy!r}; choose one of {', '.join(POLICIES)}")
    if policy == "predictive_issue" and choose_arm is None:
        raise DraftRefusal("policy predictive_issue needs a router fitted on the train split")
    out_dir = draft_dir(drafts_root, repo, issue_number)
    if (out_dir / "draft.json").exists() and not force:
        raise DraftRefusal(f"{out_dir} already has a draft; pass --force to replace it")

    issue = github.issue(repo, issue_number)  # read-only
    if issue.is_pull_request:
        raise DraftRefusal(f"{repo}#{issue_number} is a pull request, not an issue")
    clone_path = clone(repo, work_root / repo_slug(repo) / "clone.git")
    base_commit = run_git(clone_path, "rev-parse", "HEAD").strip()
    view = AgentView(
        task_id=f"live:{repo}#{issue_number}", repo=repo, base_commit=base_commit,
        issue_title=issue.title, issue_body=issue.body,
    )  # fmt: skip
    factory = executor_factory_for(clone_path, base_commit)

    if policy in _FIXED:
        plan = [_arm(models, _FIXED[policy])]
    elif policy == "cascade":
        plan = [_arm(models, label) for label in ARMS]
    else:
        plan = [_arm(models, ARMS[(choose_arm or _always_max)(view)])]

    runs: list[ArmRun] = []
    patch = ""
    verified = False
    for arm in plan:
        run_id = uuid.uuid4().hex[:12]
        attempt_dir = work_root / repo_slug(repo) / str(issue_number) / f"{arm.arm_id}-{run_id}"
        shutil.rmtree(attempt_dir, ignore_errors=True)
        base_dir = prepare_workspace(clone_path, base_commit, attempt_dir / "base")
        work_dir = prepare_workspace(clone_path, base_commit, attempt_dir / "work")
        scratch = attempt_dir / "scratch"
        scratch.mkdir()
        attempt = AttemptRef(
            run_id=run_id, task_id=view.task_id, attempt_id=f"{arm.arm_id}|live|{run_id}",
            arm_id=arm.arm_id, bucket=LIVE_BUCKET, tier=STANDARD_TIER, purpose=LIVE_PILOT_PURPOSE,
        )  # fmt: skip
        guard = LeakGuard(forbidden_host_paths=(drafts_root.resolve(),))
        executor = factory(SandboxRunner(guard, sandbox_root=attempt_dir))
        outcome = AgentPipeline(
            source=source, arm=arm, ledger=ledger, executor=executor, config=pipeline_config,
            attempt=attempt, trace=TraceWriter(attempt_dir / "trace.jsonl"), scratch=scratch,
        ).run(view, base_dir, work_dir)  # fmt: skip
        totals = ledger.attempt_totals(task_id=view.task_id, attempt_id=attempt.attempt_id)
        reasons = [] if outcome.report is None else list(outcome.report.reasons)
        runs.append(
            ArmRun(
                arm.arm_id, arm.label or arm.arm_id, outcome.claimed, totals.calls, totals.cost_usd,
                totals.input_tokens, totals.output_tokens, totals.reasoning_tokens, outcome.error,
                len(outcome.src_patch.splitlines()), reasons,
            )
        )  # fmt: skip
        patch, verified = outcome.src_patch, outcome.claimed
        if outcome.claimed or interruption_reason(outcome.error) is not None:
            break  # verified, or a global stop (kill switch, global or bucket cap)

    out_dir.mkdir(parents=True, exist_ok=True)
    patch_written = bool(patch.strip())
    (out_dir / "patch.diff").write_text(patch if patch_written else "", encoding="utf-8")
    used_models = sorted({a.model for a in plan[: len(runs)]})
    (out_dir / "PR_DESCRIPTION.md").write_text(
        render_pr_description(
            repo=repo, issue=issue, policy=policy, arms=runs, models=used_models, patch=patch,
            verified=verified, base_commit=base_commit,
        ),
        encoding="utf-8",
    )  # fmt: skip
    total = sum(r.cost_usd for r in runs)
    meta: dict[str, Any] = {
        "repo": repo, "issue": issue_number, "issue_url": issue.url, "issue_title": issue.title,
        "issue_state": issue.state, "base_commit": base_commit, "policy": policy,
        "arms_tried": [r.arm_id for r in runs], "verified": verified,
        "patch_written": patch_written,
        "cost_usd": total, "provider": source.name, "purpose": LIVE_PILOT_PURPOSE,
        "bucket": LIVE_BUCKET, "tier": STANDARD_TIER, "cascade_commit": cascade_commit,
        "created_at": now().isoformat(), "runs": [asdict(r) for r in runs],
        "build_repo": resolved.name,
    }  # fmt: skip
    (out_dir / "draft.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return DraftResult(out_dir, policy, runs, verified, total, patch_written)
