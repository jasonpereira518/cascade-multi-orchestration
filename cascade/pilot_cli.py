"""`cascade draft`, `cascade review`, `cascade pilot-metrics`: the draft-only live pilot.

``draft`` produces a patch file and PR text for a human; it never pushes, never opens a PR, never
comments on an issue (see ``cascade.pilot.draft``). ``review`` records the human's honest verdict;
``pilot-metrics`` computes the pilot numbers from the review log alone.

The default provider is mock ($0). A real draft needs ``--provider openai``, CASCADE_ALLOW_SPEND=1,
a ``buckets.live_pilot`` cap in budget.yaml, and a reachable Docker daemon.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from cascade.agents.runner import ExecutorFactory
from cascade.config import load_budget, load_models, load_repos
from cascade.errors import CascadeError
from cascade.eval_cli import arms_for
from cascade.evals import bootstrap as bs
from cascade.evals.data import load_dataset
from cascade.evals.designate import designated_policy, load_final
from cascade.harness.ledger import Ledger
from cascade.matrix_cli import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    _make_source,  # pyright: ignore[reportPrivateUsage]
    _provider_name,  # pyright: ignore[reportPrivateUsage]
    default_pipeline_config,
)
from cascade.pilot import draft as draft_mod
from cascade.pilot import reviews as rev
from cascade.pilot.github_read import GitHubIssueReader
from cascade.routing import policies as pol
from cascade.taskgen.build_env import DockerTestExecutor, build_image, docker_available
from cascade.taskgen.gitrepo import ensure_clone
from cascade.taskgen.schema import AgentView

DEFAULT_REVIEWS = Path("tasks/pilot_reviews.jsonl")
DEFAULT_DRAFTS = Path("pilot/drafts")


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(2)


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip()


def _github_clone(repo: str, dest: Path) -> Path:
    ensure_clone(f"https://github.com/{repo}.git", dest)  # read-only clone/fetch
    return dest


def _fit_chooser(
    data_dir: Path, manifest: Path, audit_file: Path, configs_dir: Path
) -> Callable[[AgentView], int]:
    """The issue-features router, fitted on TRAIN tasks only (as in the final evaluation)."""
    train = load_dataset(
        data_dir=data_dir, manifest=manifest, audit_file=audit_file,
        arms=arms_for(data_dir, configs_dir), splits=("train",), repo_root=Path("."),
    )  # fmt: skip
    n_test = train.audited_by_split.get("test", 0)
    margin = bs.margin_for(n_test if n_test else len(train.records))
    router = pol.fit_final_router(train.records, "issue", margin / 2.0)
    return draft_mod.predictive_chooser(router)


def draft_command(
    repo: Annotated[str, typer.Option(help="owner/name; must be listed under pilot_repos.")],
    issue: Annotated[int, typer.Option(help="Issue number.")],
    policy: Annotated[
        str | None, typer.Option(help="Default: the policy the final evaluation designated.")
    ] = None,
    provider: Annotated[str | None, typer.Option(help="mock (default) or openai.")] = None,
    force: Annotated[bool, typer.Option(help="Replace an existing draft.")] = False,
    configs_dir: Annotated[Path, typer.Option(help="Config dir.")] = Path("configs"),
    runs_dir: Annotated[Path, typer.Option(help="Ledger and attempt dirs.")] = Path("runs"),
    drafts_dir: Annotated[Path, typer.Option(help="Where drafts are written.")] = DEFAULT_DRAFTS,
    final_dir: Annotated[Path, typer.Option(help="Final-evaluation outputs.")] = Path(
        "results/final"
    ),
    lock_path: Annotated[Path, typer.Option(help="Final-evaluation lock.")] = Path(
        "FINAL.lock.json"
    ),
    data_dir: Annotated[Path, typer.Option(help="Matrix export (predictive policy).")] = Path(
        "runs/export"
    ),
    manifest: Annotated[Path, typer.Option(help="Task manifest.")] = Path("tasks/manifest.jsonl"),
    audit_file: Annotated[Path, typer.Option(help="Audit log.")] = Path("tasks/audit.jsonl"),
) -> None:
    """Draft a patch and PR text for one issue. Never pushes, opens a PR, or comments."""
    try:
        repos = load_repos(configs_dir / "repos.yaml")
        draft_mod.check_allowed(repos, repo)  # refuse before anything else is read or run
        if policy is None:
            try:
                policy = designated_policy(load_final(final_dir, lock_path).results)
            except CascadeError as e:
                raise _fail(
                    f"no --policy given and no final evaluation to take one from: {e}"
                ) from e
            typer.echo(f"policy: {policy} (designated by the final evaluation)")
        if policy not in draft_mod.POLICIES:
            raise draft_mod.DraftRefusal(
                f"unknown policy {policy!r}; choose one of {', '.join(draft_mod.POLICIES)}"
            )
        resolved = draft_mod.resolved_for_build(repo, repos)
        models = load_models(configs_dir / "models.yaml")
        budget = load_budget(configs_dir / "budget.yaml").resolve()
        chooser = (
            _fit_chooser(data_dir, manifest, audit_file, configs_dir)
            if policy == "predictive_issue"
            else None
        )
        source = _make_source(_provider_name(provider))
        if not docker_available():
            raise draft_mod.DraftRefusal("the docker daemon is not reachable; start Docker")

        def executor_factory_for(clone: Path, base_commit: str) -> ExecutorFactory:
            tag = build_image(resolved, clone, base_commit)
            return lambda runner: DockerTestExecutor(tag, runner, resolved.test_cmd)

        runs_dir.mkdir(parents=True, exist_ok=True)
        ledger = Ledger(runs_dir / "ledger.sqlite", budget=budget, models=models)
        try:
            result = draft_mod.run_draft(
                cfg=repos, models=models, budget=budget, ledger=ledger, source=source,
                github=GitHubIssueReader(), repo=repo, issue_number=issue, policy=policy,
                drafts_root=drafts_dir, work_root=runs_dir / "pilot",
                pipeline_config=default_pipeline_config(repos, DEFAULT_MAX_OUTPUT_TOKENS),
                clone=_github_clone, executor_factory_for=executor_factory_for,
                choose_arm=chooser, force=force, cascade_commit=_git_head(),
            )  # fmt: skip
        finally:
            ledger.close()
    except CascadeError as e:
        raise _fail(str(e)) from e
    arms = ", ".join(a.arm_id for a in result.arms)
    typer.echo(
        f"draft written to {result.directory}/ (policy {result.policy}; arms tried: {arms}; "
        f"verified: {result.verified}; ${result.cost_usd:.4f})"
    )
    typer.echo("Nothing was pushed, opened, or commented. Review the draft, then `cascade review`.")


def review_command(
    repo: Annotated[str, typer.Option(help="owner/name.")],
    issue: Annotated[int, typer.Option(help="Issue number.")],
    outcome: Annotated[str, typer.Option(help="accepted_as_is | minor_edits | rejected.")],
    lines_edited: Annotated[
        int | None, typer.Option(help="Lines you changed (required for minor_edits).")
    ] = None,
    merged: Annotated[
        bool | None,
        typer.Option("--merged/--not-merged", help="Merged upstream? Omit if not applicable."),
    ] = None,
    notes: Annotated[str, typer.Option(help="Free-text notes.")] = "",
    reviews_file: Annotated[Path, typer.Option(help="Review log.")] = DEFAULT_REVIEWS,
    drafts_dir: Annotated[Path, typer.Option(help="Where drafts live.")] = DEFAULT_DRAFTS,
) -> None:
    """Record your honest verdict on a draft (appends to tasks/pilot_reviews.jsonl)."""
    try:
        row = rev.record_review(
            reviews_path=reviews_file, drafts_root=drafts_dir, repo=repo, issue=issue,
            outcome=outcome, lines_edited=lines_edited, merged_upstream=merged, notes=notes,
        )  # fmt: skip
    except CascadeError as e:
        raise _fail(str(e)) from e
    typer.echo(
        f"recorded {row.repo}#{row.issue}: {row.outcome} (arms: {', '.join(row.arms_tried)}; "
        f"${row.cost_usd:.4f})"
    )


def pilot_metrics_command(
    reviews_file: Annotated[Path, typer.Option(help="Review log.")] = DEFAULT_REVIEWS,
    drafts_dir: Annotated[Path, typer.Option(help="Where drafts live.")] = DEFAULT_DRAFTS,
) -> None:
    """Pilot accept rate, merge rate, median lines edited, cost per accepted draft."""
    try:
        rows = rev.latest(rev.read_reviews(reviews_file))
    except CascadeError as e:
        raise _fail(str(e)) from e
    if not rows:
        raise _fail(f"no reviews in {reviews_file}; run `cascade review` for each draft first")
    typer.echo(rev.format_metrics(rev.pilot_metrics(rows), rev.unreviewed_drafts(drafts_dir, rows)))


def register(app: typer.Typer) -> None:
    app.command("draft")(draft_command)
    app.command("review")(review_command)
    app.command("pilot-metrics")(pilot_metrics_command)
