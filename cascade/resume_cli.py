"""`cascade resume-stats` and `cascade resume-bullets`.

``resume-stats`` writes ``artifacts/resume_stats.json`` from run artifacts only (see
``cascade.resume.stats``) and fails loudly when an input is missing. ``resume-bullets`` fills
``docs/resume_templates.yaml`` from that file and writes ``docs/resume_bullets.md`` with each
bullet's character count and any flags (see ``cascade.resume.bullets``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from cascade.errors import CascadeError
from cascade.resume import bullets as bl
from cascade.resume.stats import DEFAULT_PATH, StatsInputs, build_stats, load_stats, write_stats


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(2)


def resume_stats_command(
    out: Annotated[Path, typer.Option(help="Where the stats go.")] = DEFAULT_PATH,
    final_dir: Annotated[Path, typer.Option(help="Final-evaluation outputs.")] = Path(
        "results/final"
    ),
    lock_path: Annotated[Path, typer.Option(help="Final-evaluation lock.")] = Path(
        "FINAL.lock.json"
    ),
    data_dir: Annotated[Path, typer.Option(help="Parquet export (`cascade export`).")] = Path(
        "runs/export"
    ),
    manifest: Annotated[Path, typer.Option(help="Task manifest.")] = Path("tasks/manifest.jsonl"),
    audit_file: Annotated[Path, typer.Option(help="Audit log.")] = Path("tasks/audit.jsonl"),
    reviews_file: Annotated[Path, typer.Option(help="Pilot review log.")] = Path(
        "tasks/pilot_reviews.jsonl"
    ),
    dev_dir: Annotated[Path, typer.Option(help="Dev-analysis outputs (noise estimate).")] = Path(
        "results/dev"
    ),
    configs_dir: Annotated[Path, typer.Option(help="Config dir (arm labels).")] = Path("configs"),
    allow_synthetic: Annotated[
        bool, typer.Option(help="Tests only: accept a labeled-synthetic final evaluation.")
    ] = False,
) -> None:
    """Write artifacts/resume_stats.json; every number is read from an artifact."""
    inputs = StatsInputs(
        final_dir=final_dir, lock_path=lock_path, data_dir=data_dir, manifest=manifest,
        audit_file=audit_file, reviews_file=reviews_file, dev_dir=dev_dir, configs_dir=configs_dir,
    )  # fmt: skip
    try:
        stats = build_stats(inputs, allow_synthetic=allow_synthetic)
    except CascadeError as e:
        raise _fail(str(e)) from e
    write_stats(stats, out)
    typer.echo(f"wrote {out} (data source: {stats['data_source']})")


def resume_bullets_command(
    stats_file: Annotated[Path, typer.Option(help="resume_stats.json.")] = DEFAULT_PATH,
    templates: Annotated[Path, typer.Option(help="Bullet templates.")] = bl.TEMPLATES_PATH,
    out: Annotated[Path, typer.Option(help="Where the bullets go.")] = bl.BULLETS_PATH,
    allow_synthetic: Annotated[
        bool, typer.Option(help="Tests only: render synthetic stats, marked SYNTHETIC.")
    ] = False,
) -> None:
    """Fill the resume templates from resume_stats.json; flag bullets outside their limits."""
    try:
        stats = load_stats(stats_file)
        bullets = bl.render_bullets(
            stats, bl.load_templates(templates), allow_synthetic=allow_synthetic
        )
    except CascadeError as e:
        raise _fail(str(e)) from e
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(bl.format_markdown(bullets, stats), encoding="utf-8")
    for b in bullets:
        if b.text is None:
            typer.echo(f"{b.name}: {b.reason}")
            continue
        typer.echo(f"{b.name} ({b.chars}/{b.limit} chars): {b.text}")
        for flag in b.flags:
            typer.echo(f"  FLAG: {flag}")
    typer.echo(f"wrote {out}")


def register(app: typer.Typer) -> None:
    app.command("resume-stats")(resume_stats_command)
    app.command("resume-bullets")(resume_bullets_command)
