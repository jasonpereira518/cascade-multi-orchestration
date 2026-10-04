"""Command-line entry point (`cascade`)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from cascade.config import load_budget, load_models, load_repos

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Cascade CLI.")


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
