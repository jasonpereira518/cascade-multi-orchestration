"""Command-line entry point (`cascade`)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated

import typer

from cascade.config import load_budget, load_models, load_repos
from cascade.errors import CascadeError
from cascade.taskgen import attrition
from cascade.taskgen.pipeline import STAGE_ORDER, run_pipeline
from cascade.taskgen.store import StageStore

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Cascade CLI.")
taskgen_app = typer.Typer(no_args_is_help=True, help="Build tasks from a repo's merged PRs ($0).")
app.add_typer(taskgen_app, name="taskgen")


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
) -> None:
    """Run the task pipeline for one repo. Safe to rerun: finished PRs are skipped."""
    try:
        resolved = load_repos(configs_dir / "repos.yaml").resolve(repo)
        summary = run_pipeline(
            resolved,
            runs_dir=runs_dir,
            manifest_path=manifest,
            status_md=status_md,
            stop_after=stop_after,
        )
    except CascadeError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    typer.echo(f"reached stage: {summary.stage_reached}; manifest tasks: {summary.manifest_tasks}")
    for message in summary.errors:
        typer.echo(f"retry on rerun: {message}", err=True)


@taskgen_app.command("attrition")
def taskgen_attrition(
    repo: Annotated[str, typer.Option(help="owner/name, as listed in repos.yaml.")],
    configs_dir: _ConfigsDir = Path("configs"),
    runs_dir: _RunsDir = Path("runs/taskgen"),
    status_md: Annotated[
        Path | None, typer.Option(help="If given, refresh the attrition block in this file.")
    ] = None,
) -> None:
    """Print the attrition table computed from the saved state (no network, no docker)."""
    try:
        resolved = load_repos(configs_dir / "repos.yaml").resolve(repo)
    except CascadeError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2) from e
    state = runs_dir / resolved.slug / "state.sqlite"
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
