"""Command-line entry point (`cascade`)."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Annotated

import typer

from cascade import eval_cli, matrix_cli, pilot_cli, resume_cli
from cascade.agents import runner as agent_runner
from cascade.agents.pipeline import PipelineConfig
from cascade.agents.triage import format_triage, triage_runs
from cascade.config import load_budget, load_models, load_repos
from cascade.errors import CascadeError
from cascade.harness.ledger import Ledger, summarize
from cascade.harness.providers import resolve_provider_name
from cascade.harness.providers.base import ModelSource
from cascade.harness.providers.mock import MockModelSource
from cascade.harness.providers.openai import OpenAIModelSource
from cascade.routing.matrix import MatrixStore
from cascade.taskgen import attrition
from cascade.taskgen import audit as audit_mod
from cascade.taskgen import split as split_mod
from cascade.taskgen.build_env import DockerUnavailableError, docker_available
from cascade.taskgen.cutoffs import arm_cutoffs
from cascade.taskgen.github import TOKEN_ENV
from cascade.taskgen.manifest import finalize_manifest, read_manifest
from cascade.taskgen.orchestrate import (
    DEFAULT_JOBS,
    DEFAULT_WORKERS,
    format_report,
    repo_funnel,
    run_all,
)
from cascade.taskgen.pipeline import (
    DEFAULT_PR_TIMEOUT_S,
    STAGE_ORDER,
    PipelineDeps,
    run_pipeline,
    state_dir,
)
from cascade.taskgen.runtime import ProgressPrinter, check_docker, check_github
from cascade.taskgen.schema import Task
from cascade.taskgen.store import StageStore

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Cascade CLI.")
taskgen_app = typer.Typer(no_args_is_help=True, help="Build tasks from a repo's merged PRs ($0).")
app.add_typer(taskgen_app, name="taskgen")
split_app = typer.Typer(no_args_is_help=True, help="The frozen repo-level train/test split.")
app.add_typer(split_app, name="split")
audit_app = typer.Typer(no_args_is_help=True, help="Human audit of tasks.")
app.add_typer(audit_app, name="audit")
tasks_app = typer.Typer(no_args_is_help=True, help="Operations on the task manifest.")
app.add_typer(tasks_app, name="tasks")


@app.callback()
def _main() -> None:
    """Cost-aware routing of models and reasoning effort for issue resolution."""


@app.command("config-status")
def config_status(
    configs_dir: Annotated[Path, typer.Option(help="Directory holding the config YAMLs.")] = Path(
        "configs"
    ),
) -> None:
    """List every TODO field still unfilled in configs/*.yaml. Exits 1 if any remain."""
    unfilled = {
        "models.yaml": load_models(configs_dir / "models.yaml").unfilled(),
        "budget.yaml": load_budget(configs_dir / "budget.yaml").unfilled(),
        "repos.yaml": load_repos(configs_dir / "repos.yaml").unfilled(),
    }
    total = 0
    for filename, paths in unfilled.items():
        for path in paths:
            typer.echo(f"{filename}: {path}")
        total += len(paths)
    if total:
        typer.echo(f"{total} unfilled TODO field(s).")
        raise typer.Exit(1)
    typer.echo("All config fields are filled.")


_ConfigsDir = Annotated[Path, typer.Option(help="Directory holding the config YAMLs.")]
_RunsDir = Annotated[Path, typer.Option(help="Where per-repo state, clones and workspaces live.")]
_StatusMd = Annotated[Path, typer.Option(help="File whose attrition block is refreshed.")]
_Relaxed = Annotated[
    bool,
    typer.Option(
        "--relaxed-window",
        help="Use relaxed_date_window from repos.yaml (more tasks, more contamination risk).",
    ),
]
_ReplaceWindow = Annotated[
    bool,
    typer.Option(help="Replace manifest tasks mined under a different window (discards audits)."),
]
_PrTimeout = Annotated[
    float,
    typer.Option(help="Seconds per PR per stage before it is skipped and logged (0 = no limit)."),
]


def _pipeline_deps(
    configs_dir: Path, *, pr_timeout: float, replace_window: bool, printer: ProgressPrinter
) -> PipelineDeps:
    """Deps for a real run: live progress, per-PR timeouts, and knowledge-cutoff tags."""
    try:
        cutoffs = arm_cutoffs(load_models(configs_dir / "models.yaml"))
    except (CascadeError, ValueError, OSError) as e:
        printer.note(f"warning: models.yaml unreadable ({e}); tasks get no cutoff tags")
        cutoffs = {}
    unknown = sorted(arm for arm, c in cutoffs.items() if c is None)
    if unknown:
        printer.note(
            "warning: knowledge_cutoff is still TODO for "
            + ", ".join(unknown)
            + "; those arms are tagged unknown (null). `cascade tasks finalize` retags later."
        )
    return PipelineDeps(
        pr_timeout_s=pr_timeout or None, progress=printer, cutoffs=cutoffs,
        replace_window=replace_window,
    )  # fmt: skip


def _window_line(r: object) -> str:
    start, end, kind = r.window_start, r.window_end, r.window_kind  # type: ignore[attr-defined]
    return f"window: {start} to {end} ({kind}); state in w-{start}_{end}/"


@taskgen_app.command("run")
def taskgen_run(
    repo: Annotated[str, typer.Option(help="owner/name, as listed in repos.yaml.")],
    configs_dir: _ConfigsDir = Path("configs"),
    runs_dir: _RunsDir = Path("runs/taskgen"),
    manifest: Annotated[Path, typer.Option(help="Validated tasks are written here.")] = Path(
        "tasks/manifest.jsonl"
    ),
    status_md: _StatusMd = Path("STATUS.md"),
    stop_after: Annotated[
        str, typer.Option(help=f"Last stage to run: {', '.join(STAGE_ORDER)}.")
    ] = "validate",
    relaxed_window: _Relaxed = False,
    replace_window: _ReplaceWindow = False,
    pr_timeout: _PrTimeout = DEFAULT_PR_TIMEOUT_S,
    jobs: Annotated[int, typer.Option(help="PRs in parallel.")] = DEFAULT_JOBS,
) -> None:
    """Run the task pipeline for one repo. Safe to rerun: finished PRs are skipped."""
    printer = ProgressPrinter()
    try:
        resolved = load_repos(configs_dir / "repos.yaml").resolve(repo, relaxed=relaxed_window)
        typer.echo(_window_line(resolved), err=True)
        summary = run_pipeline(
            resolved,
            runs_dir=runs_dir,
            manifest_path=manifest,
            status_md=status_md,
            stop_after=stop_after,
            deps=_pipeline_deps(
                configs_dir, pr_timeout=pr_timeout, replace_window=replace_window, printer=printer
            ),
            jobs=jobs,
        )
    except CascadeError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    typer.echo(f"reached stage: {summary.stage_reached}; manifest tasks: {summary.manifest_tasks}")
    for message in summary.errors:
        typer.echo(f"retry on rerun: {message}", err=True)


@taskgen_app.command("check")
def taskgen_check() -> None:
    """Check that GITHUB_TOKEN works (one read-only query) and that Docker is running."""
    results = [check_github(), check_docker()]
    for r in results:
        typer.echo(f"{'OK  ' if r.ok else 'FAIL'} {r.name}: {r.detail}")
    if not all(r.ok for r in results):
        raise typer.Exit(1)


@taskgen_app.command("attrition")
def taskgen_attrition(
    repo: Annotated[str, typer.Option(help="owner/name, as listed in repos.yaml.")],
    configs_dir: _ConfigsDir = Path("configs"),
    runs_dir: _RunsDir = Path("runs/taskgen"),
    status_md: Annotated[
        Path | None, typer.Option(help="If given, refresh the attrition block in this file.")
    ] = None,
    relaxed_window: _Relaxed = False,
) -> None:
    """Print the attrition table computed from the saved state (no network, no docker)."""
    try:
        resolved = load_repos(configs_dir / "repos.yaml").resolve(repo, relaxed=relaxed_window)
    except CascadeError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    state = state_dir(runs_dir, resolved) / "state.sqlite"
    if not state.exists():
        typer.echo(f"error: no state at {state}; run `cascade taskgen run` first", err=True)
        raise typer.Exit(2)
    store = StageStore(state)
    try:
        block = attrition.render_markdown(
            attrition.compute(store),
            repo=resolved.name,
            window=(resolved.window_start, resolved.window_end),
            generated_at=dt.datetime.now(dt.UTC),
            state_path=str(state),
        )
        typer.echo(block)
        if status_md is not None:
            attrition.update_status_md(status_md, resolved.name, block)
    finally:
        store.close()


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(2)


def _load_task(manifest: Path, task_id: str) -> Task:
    if not manifest.exists():
        raise _fail(f"no manifest at {manifest}; build tasks first (cascade taskgen run)")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            task = Task.model_validate_json(line)
            if task.task_id == task_id:
                return task
    raise _fail(f"task {task_id!r} is not in {manifest}")


def _make_source(provider: str) -> ModelSource:
    if provider == "mock":
        return MockModelSource()
    if provider == "openai":
        return OpenAIModelSource()  # refuses unless CASCADE_ALLOW_SPEND=1 and OPENAI_API_KEY is set
    raise _fail(f"unknown provider {provider!r}; expected 'mock' or 'openai'")


@app.command("run")
def run_command(
    task: Annotated[str, typer.Option("--task", help="Task id from the manifest.")],
    arm: Annotated[str, typer.Option("--arm", help="Arm id, e.g. model@effort (see models.yaml).")],
    seed: Annotated[int, typer.Option(help="Attempt seed (part of the matrix key).")] = 0,
    reproducer: Annotated[bool, typer.Option(help="Enable the Reproducer role.")] = True,
    reviewer: Annotated[bool, typer.Option(help="Enable the Reviewer role.")] = True,
    max_repairs: Annotated[int, typer.Option(help="Extra Patcher rounds after the first.")] = 2,
    max_output_tokens: Annotated[
        int, typer.Option(help="Per model call, reasoning included.")
    ] = 4096,
    cap_usd: Annotated[
        float | None, typer.Option(help="Per-attempt dollar cap; never above budget.yaml's.")
    ] = None,
    provider: Annotated[
        str | None, typer.Option(help="mock (default) or openai. $CASCADE_PROVIDER also works.")
    ] = None,
    force: Annotated[
        bool, typer.Option(help="Rerun even if the matrix already has this row.")
    ] = False,
    configs_dir: _ConfigsDir = Path("configs"),
    manifest: Annotated[Path, typer.Option(help="Task manifest.")] = Path("tasks/manifest.jsonl"),
    runs_dir: Annotated[Path, typer.Option(help="Ledger, matrix, attempts, traces.")] = Path(
        "runs"
    ),
    taskgen_runs_dir: _RunsDir = Path("runs/taskgen"),
) -> None:
    """Run one attempt of one arm on one task and record both outcomes. Real spend needs
    --provider openai AND CASCADE_ALLOW_SPEND=1."""
    try:
        models = load_models(configs_dir / "models.yaml")
        arm_spec = next((a for a in models.arms if a.arm_id == arm), None)
        if arm_spec is None:
            known = ", ".join(a.arm_id for a in models.arms)
            raise _fail(f"unknown arm {arm!r}; models.yaml defines: {known}")
        budget = load_budget(configs_dir / "budget.yaml").resolve()
        if cap_usd is not None:
            budget = budget.model_copy(
                update={"per_attempt_cap_usd": min(cap_usd, budget.per_attempt_cap_usd)}
            )
        provider_name = resolve_provider_name(provider)
        if provider_name == "openai" and models.model_spec(arm_spec.model).provider != "openai":
            raise _fail(f"model {arm_spec.model!r} is not an openai model in models.yaml")
        source = _make_source(provider_name)
        task_obj = _load_task(manifest, task)
        repos = load_repos(configs_dir / "repos.yaml")
        repo_spec = next((r for r in repos.repos if r.name == task_obj.repo), None)
        if repo_spec is None or repo_spec.test_cmd.startswith("TODO"):
            raise _fail(f"repos.yaml has no filled entry for {task_obj.repo}")
        clone = taskgen_runs_dir / task_obj.repo.replace("/", "__") / "clone.git"
        if not clone.exists():
            raise _fail(f"no clone at {clone}; run `cascade taskgen run` for {task_obj.repo} first")
        if not docker_available():
            raise DockerUnavailableError("the docker daemon is not reachable; start Docker")
        ledger = Ledger(runs_dir / "ledger.sqlite", budget=budget, models=models)
        matrix = MatrixStore(runs_dir / "matrix.sqlite")
        try:
            report = agent_runner.run_attempt(
                task=task_obj, arm=arm_spec, seed=seed,
                config=PipelineConfig(
                    reproducer=reproducer, reviewer=reviewer, max_repairs=max_repairs,
                    max_output_tokens=max_output_tokens,
                    test_timeout_s=float(repos.limits.max_test_runtime_s),
                ),
                ledger=ledger, matrix=matrix, source=source, clone=clone, runs_dir=runs_dir,
                executor_factory=agent_runner.docker_executor_factory(task_obj, repo_spec.test_cmd),
                forbidden_host_paths=[manifest.resolve().parent, taskgen_runs_dir.resolve()],
                grading_timeout_s=float(repos.limits.max_test_runtime_s), force=force,
            )  # fmt: skip
            total_spent = ledger.spent_usd()
        finally:
            ledger.close()
            matrix.close()
    except CascadeError as e:
        raise _fail(str(e)) from e

    row = report.row
    reused = " | (existing row reused; use --force to rerun)" if report.reused else ""
    typer.echo(f"task {row.task_id} | arm {row.arm_id} | seed {row.seed}{reused}")
    typer.echo(f"true_success={row.true_success}  verifier_claimed={row.verifier_claimed}")
    tokens = f"{row.input_tokens}/{row.output_tokens}/{row.reasoning_tokens}"
    typer.echo(
        f"cost ${row.cost_usd:.4f}  calls {row.turns}  wall {row.wall_s:.1f}s  "
        f"tokens in/out/reasoning {tokens}"
    )
    if row.error:
        typer.echo(f"attempt ended early: {row.error}")
    if report.grade is not None and not report.grade.true_success:
        typer.echo(f"grading: {report.grade.reason}")
    for violation in report.leak_violations:
        typer.echo(f"LEAK AUDIT: {violation}", err=True)
    if report.attempt_dir is not None:
        typer.echo(f"artifacts: {report.attempt_dir}")
    typer.echo(f"ledger total spent so far: ${total_spent:.4f}")


@app.command("ledger-summary")
def ledger_summary_command(
    runs_dir: Annotated[
        Path, typer.Option(help="Where ledger.sqlite and matrix.sqlite live.")
    ] = Path("runs"),
) -> None:
    """Summarize logged calls and attempt outcomes. Paste this back after a paid run."""
    db = runs_dir / "ledger.sqlite"
    if not db.exists():
        raise _fail(f"no ledger at {db}")
    s = summarize(db)
    typer.echo("LEDGER SUMMARY")
    typer.echo(f"calls: {s.calls} (errors: {s.errors})  total cost: ${s.cost_usd:.4f}")
    typer.echo(
        f"tokens: input {s.input_tokens}, output {s.output_tokens}, reasoning {s.reasoning_tokens}"
    )
    typer.echo("by arm:")
    for arm_id, calls, cost in s.by_arm:
        typer.echo(f"  {arm_id}: {calls} calls, ${cost:.4f}")
    typer.echo("by role:")
    for role, calls, cost in s.by_role:
        typer.echo(f"  {role}: {calls} calls, ${cost:.4f}")
    if s.error_rows:
        typer.echo("failed calls:")
        for task_id, arm_id, role, error in s.error_rows:
            typer.echo(f"  {task_id} | {arm_id} | {role}: {error[:200]}")
    matrix_path = runs_dir / "matrix.sqlite"
    if matrix_path.exists():
        store = MatrixStore(matrix_path)
        rows = store.rows()
        store.close()
        typer.echo("OUTCOMES (one row per attempt)")
        typer.echo("task | arm | seed | true | claimed | cost | calls | wall_s | error")
        for r in rows:
            typer.echo(
                f"{r.task_id} | {r.arm_id} | {r.seed} | {r.true_success} | {r.verifier_claimed} | "
                f"${r.cost_usd:.4f} | {r.turns} | {r.wall_s:.0f} | {r.error or '-'}"
            )


# --- multi-repo task generation, split, audit, finalize, triage -----------------------------

_Manifest = Annotated[Path, typer.Option(help="Task manifest.")]
_SplitFile = Annotated[Path, typer.Option(help="The frozen split file.")]
_AuditFile = Annotated[Path, typer.Option(help="The append-only audit verdict log.")]


@taskgen_app.command("run-all")
def taskgen_run_all(
    repo: Annotated[
        list[str] | None,
        typer.Option("--repo", help="Repo to run (repeatable). Default: enabled repos."),
    ] = None,
    include_disabled: Annotated[
        bool, typer.Option(help="Also run repos marked enabled: false.")
    ] = False,
    workers: Annotated[int, typer.Option(help="Repos in parallel.")] = DEFAULT_WORKERS,
    jobs: Annotated[int, typer.Option(help="PRs in parallel within each repo.")] = DEFAULT_JOBS,
    stop_after: Annotated[
        str, typer.Option(help=f"Last stage to run: {', '.join(STAGE_ORDER)}.")
    ] = "validate",
    configs_dir: _ConfigsDir = Path("configs"),
    runs_dir: _RunsDir = Path("runs/taskgen"),
    manifest: _Manifest = Path("tasks/manifest.jsonl"),
    status_md: _StatusMd = Path("STATUS.md"),
    relaxed_window: _Relaxed = False,
    replace_window: _ReplaceWindow = False,
    pr_timeout: _PrTimeout = DEFAULT_PR_TIMEOUT_S,
) -> None:
    """Run the task pipeline over many repos in parallel. Resumable: rerun to continue."""
    if not os.environ.get(TOKEN_ENV):
        raise _fail(f"{TOKEN_ENV} is not set; mining needs it (read-only, public repos is enough)")
    try:
        cfg = load_repos(configs_dir / "repos.yaml")
        if repo:
            names = list(repo)
        elif include_disabled:
            names = [r.name for r in cfg.repos]
        else:
            names = cfg.enabled_repos()
        resolved = [cfg.resolve(n, relaxed=relaxed_window) for n in names]
        if resolved:
            typer.echo(_window_line(resolved[0]), err=True)
        printer = ProgressPrinter()
        deps = _pipeline_deps(
            configs_dir, pr_timeout=pr_timeout, replace_window=replace_window, printer=printer
        )
        outcomes = run_all(
            resolved,
            runs_dir=runs_dir,
            manifest_path=manifest,
            status_md=status_md,
            stop_after=stop_after,
            workers=workers,
            jobs=jobs,
            deps_for=lambda _repo: deps,
        )
        funnels = [f for r in resolved if (f := repo_funnel(r, runs_dir)) is not None]
        typer.echo(format_report(funnels, outcomes=outcomes))
    except (CascadeError, ValueError) as e:
        raise _fail(str(e)) from e
    if any(o.error for o in outcomes):
        raise typer.Exit(1)


@taskgen_app.command("report")
def taskgen_report(
    configs_dir: _ConfigsDir = Path("configs"),
    runs_dir: _RunsDir = Path("runs/taskgen"),
    manifest: _Manifest = Path("tasks/manifest.jsonl"),
    relaxed_window: _Relaxed = False,
) -> None:
    """Per-stage attrition and per-repo task counts from saved state (no network, no Docker)."""
    try:
        cfg = load_repos(configs_dir / "repos.yaml")
        resolved = [cfg.resolve(r.name, relaxed=relaxed_window) for r in cfg.repos]
        tasks = read_manifest(manifest)
    except (CascadeError, ValueError) as e:
        raise _fail(str(e)) from e
    funnels = [f for r in resolved if (f := repo_funnel(r, runs_dir)) is not None]
    have = {f.repo for f in funnels}
    split_counts: dict[str, dict[str, int]] = {}
    for t in tasks:
        side = split_counts.setdefault(t.repo, {})
        key = t.split or "unassigned"
        side[key] = side.get(key, 0) + 1
    missing = [r.name for r in resolved if r.name not in have]
    if resolved:
        typer.echo(_window_line(resolved[0]) + "\n")
    typer.echo(format_report(funnels, missing=missing, split_counts=split_counts))


@split_app.command("freeze")
def split_freeze(
    manifest: _Manifest = Path("tasks/manifest.jsonl"),
    split_file: _SplitFile = Path("tasks/split.json"),
    salt: Annotated[
        str, typer.Option(help="Hash salt (part of the frozen record).")
    ] = split_mod.DEFAULT_SALT,
    target: Annotated[
        float, typer.Option(help="Target fraction of tasks in the test split.")
    ] = split_mod.DEFAULT_TARGET,
    force: Annotated[
        bool, typer.Option(help="Overwrite a frozen split (invalidates results).")
    ] = False,
) -> None:
    """Freeze the repo-level split from per-repo task counts. Refuses to overwrite."""
    try:
        counts: dict[str, int] = {}
        for t in read_manifest(manifest):
            counts[t.repo] = counts.get(t.repo, 0) + 1
        result = split_mod.freeze(
            counts, path=split_file, salt=salt, target_test_fraction=target, force=force
        )
    except (CascadeError, ValueError) as e:
        raise _fail(str(e)) from e
    _echo_split(result, counts)


def _echo_split(result: split_mod.SplitFile, counts: dict[str, int]) -> None:
    per_side = {"train": 0, "test": 0}
    for repo_name, side in sorted(result.assignments.items()):
        typer.echo(f"{side:5s} {repo_name} ({counts.get(repo_name, 0)} tasks)")
        per_side[side] += counts.get(repo_name, 0)
    total = sum(per_side.values()) or 1
    typer.echo(
        f"train {per_side['train']} tasks, test {per_side['test']} tasks "
        f"({per_side['test'] / total:.0%} test; target {result.target_test_fraction:.0%})"
    )


@split_app.command("show")
def split_show(
    manifest: _Manifest = Path("tasks/manifest.jsonl"),
    split_file: _SplitFile = Path("tasks/split.json"),
) -> None:
    """Print the frozen split and the task counts it gives."""
    try:
        result = split_mod.read_split(split_file)
        tasks = read_manifest(manifest)
    except (CascadeError, ValueError) as e:
        raise _fail(str(e)) from e
    counts: dict[str, int] = {}
    for t in tasks:
        counts[t.repo] = counts.get(t.repo, 0) + 1
    typer.echo(f"frozen {result.frozen_at} with salt {result.salt!r}")
    if result.late_assignments:
        typer.echo("assigned after the freeze: " + ", ".join(sorted(result.late_assignments)))
    _echo_split(result, counts)


@tasks_app.command("finalize")
def tasks_finalize(
    manifest: _Manifest = Path("tasks/manifest.jsonl"),
    split_file: _SplitFile = Path("tasks/split.json"),
    audit_file: _AuditFile = Path("tasks/audit.jsonl"),
    assign_new: Annotated[
        bool, typer.Option(help="Assign repos added after the freeze by hash bucket.")
    ] = True,
    configs_dir: _ConfigsDir = Path("configs"),
) -> None:
    """Stamp split, audit verdicts, features and knowledge-cutoff tags on the manifest."""
    try:
        try:
            cutoffs = arm_cutoffs(load_models(configs_dir / "models.yaml"))
        except (CascadeError, ValueError, OSError):
            cutoffs = None  # no readable models.yaml: leave existing tags untouched
        result = finalize_manifest(
            manifest, split_file, audit_file, assign_new=assign_new, cutoffs=cutoffs
        )
    except (CascadeError, ValueError) as e:
        raise _fail(str(e)) from e
    typer.echo(f"{result.tasks} task(s) finalized")
    typer.echo("split: " + ", ".join(f"{k} {v}" for k, v in sorted(result.by_split.items())))
    typer.echo("audit: " + ", ".join(f"{k} {v}" for k, v in sorted(result.by_audit.items())))
    if result.new_repos:
        typer.echo("assigned by hash bucket (after the freeze): " + ", ".join(result.new_repos))


@audit_app.command("review")
def audit_review(
    manifest: _Manifest = Path("tasks/manifest.jsonl"),
    audit_file: _AuditFile = Path("tasks/audit.jsonl"),
    repo: Annotated[str | None, typer.Option(help="Only this repo.")] = None,
    split: Annotated[str | None, typer.Option(help="Only this split: train or test.")] = None,
    task: Annotated[str | None, typer.Option(help="Only this task id.")] = None,
    redo: Annotated[
        bool, typer.Option(help="Also show tasks that already have a verdict.")
    ] = False,
    full: Annotated[bool, typer.Option(help="Do not truncate long issues and patches.")] = False,
) -> None:
    """Review tasks one at a time and record a verdict for each. Safe to quit and resume."""
    try:
        all_tasks = read_manifest(manifest)
    except ValueError as e:
        raise _fail(f"unreadable manifest: {e}") from e
    tasks = [
        t
        for t in all_tasks
        if (repo is None or t.repo == repo)
        and (split is None or t.split == split)
        and (task is None or t.task_id == task)
    ]
    if not tasks:
        raise _fail("no tasks match; build the manifest first, or loosen --repo/--split/--task")
    try:
        stats = audit_mod.run_review(
            tasks,
            audit_file,
            ask=lambda q: typer.prompt(q, default="", show_default=False),
            echo=typer.echo,
            redo=redo,
            full=full,
        )
    except (typer.Abort, EOFError):
        typer.echo("\ninput ended; every verdict so far is saved")
        return
    typer.echo(
        f"recorded {stats.recorded}, skipped {stats.skipped}, left {stats.remaining}"
        + (" (quit early)" if stats.quit else "")
    )


@audit_app.command("status")
def audit_status_command(
    manifest: _Manifest = Path("tasks/manifest.jsonl"),
    audit_file: _AuditFile = Path("tasks/audit.jsonl"),
) -> None:
    """Progress against the audit targets (60 ok, minimum 40, about 25-30 in the test split)."""
    try:
        status = audit_mod.audit_status(
            read_manifest(manifest), audit_mod.load_verdicts(audit_file)
        )
    except (CascadeError, ValueError) as e:
        raise _fail(str(e)) from e
    typer.echo(audit_mod.format_status(status))


@app.command("triage")
def triage_command(
    runs_dir: Annotated[Path, typer.Option(help="Where matrix.sqlite and attempts/ live.")] = Path(
        "runs"
    ),
) -> None:
    """Classify finished attempts: pipeline bug, Verifier issue, or genuine model failure."""
    typer.echo(format_triage(triage_runs(runs_dir)))


matrix_cli.register(app)  # preflight, pilot, size, calibrate, matrix, ablate, ...
eval_cli.register(app)  # eval (dev: train/CV only; --final: once, guarded)
pilot_cli.register(app)  # draft, review, pilot-metrics (the draft-only live pilot)
resume_cli.register(app)  # resume-stats, resume-bullets (numbers come from artifacts only)
