"""`cascade eval`: the development analysis (train/CV) and, once, the final test-split evaluation.

Without ``--final`` this reads TRAIN tasks only; the test split's outcomes are never loaded. With
``--final`` it runs the guarded single evaluation (``evals.final``): the plan must be committed, the
tree clean, and no lock present. Real data comes from the Phase 4 exports (``cascade export``);
the labeled-synthetic matrix from ``tests/synthetic`` is for development and is refused by --final.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from cascade.config import load_models
from cascade.errors import CascadeError
from cascade.evals import bootstrap as bs
from cascade.evals import report
from cascade.evals.data import SYNTHETIC_MARKER, Dataset, load_dataset
from cascade.evals.final import run_final
from cascade.evals.run import EvalResult, evaluate_dataset
from cascade.routing.policies import ARMS


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(2)


def arms_for(data_dir: Path, configs_dir: Path) -> dict[str, str]:
    """label -> arm id: from the synthetic dataset's arms.json, else from models.yaml labels."""
    if (data_dir / SYNTHETIC_MARKER).exists():
        raw = json.loads((data_dir / "arms.json").read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in raw.items()}
    try:
        models = load_models(configs_dir / "models.yaml")
    except (OSError, ValueError) as e:
        raise CascadeError(
            f"cannot read {configs_dir / 'models.yaml'} to find the arm labels ({e}); real data "
            "needs arms labelled cheap, mid, and max there"
        ) from e
    by_label = {a.label: a.arm_id for a in models.matrix_arms() if a.label}
    missing = [a for a in ARMS if a not in by_label]
    if missing:
        raise CascadeError(f"models.yaml has no enabled arm labelled {missing}")
    return {a: by_label[a] for a in ARMS}


def dev_result(train: Dataset, resamples: int) -> EvalResult:
    n_test = train.audited_by_split.get("test", 0)  # a count from the manifest, no outcomes
    basis_n = n_test if n_test else len(train.records)
    basis = (
        f"rule of docs/EVAL_PLAN.md for {n_test} audited test tasks"
        if n_test
        else f"no test tasks in the manifest; rule applied to the {basis_n} train tasks"
    )
    return evaluate_dataset(
        mode="dev", train=train, target=train, margin=bs.margin_for(basis_n),
        margin_basis=basis, b=resamples,
    )  # fmt: skip


def eval_command(
    final: Annotated[
        bool, typer.Option("--final", help="Run the single test-split evaluation (once).")
    ] = False,
    data_dir: Annotated[
        Path, typer.Option(help="Dir with matrix.parquet and ledger.parquet.")
    ] = Path("runs/export"),
    manifest: Annotated[Path, typer.Option(help="Task manifest.")] = Path("tasks/manifest.jsonl"),
    audit_file: Annotated[Path, typer.Option(help="Audit log.")] = Path("tasks/audit.jsonl"),
    configs_dir: Annotated[Path, typer.Option(help="Config dir (arm labels).")] = Path("configs"),
    out_dir: Annotated[Path | None, typer.Option(help="Where reports go.")] = None,
    resamples: Annotated[
        int, typer.Option(help="Bootstrap resamples (dev only; --final always uses 10,000).")
    ] = bs.B_DEFAULT,
    repo_root: Annotated[Path, typer.Option(help="The git repository root.")] = Path("."),
) -> None:
    """Evaluate routing policies. Dev (default): train/CV only. --final: once, guarded."""
    try:
        arms = arms_for(data_dir, configs_dir)
        if final:
            outcome = run_final(
                repo_root=repo_root, data_dir=data_dir, manifest=manifest, audit_file=audit_file,
                arms=arms, out_dir=out_dir or Path("results/final"),
            )  # fmt: skip
            typer.echo(report.to_markdown(outcome.result))
            typer.echo(f"wrote {outcome.out_dir}/ and {outcome.lock_path}")
            typer.echo("The final evaluation is locked. Paste this output into the chat first.")
            return
        train = load_dataset(
            data_dir=data_dir, manifest=manifest, audit_file=audit_file, arms=arms,
            splits=("train",), repo_root=repo_root,
        )  # fmt: skip
        result = dev_result(train, resamples)
    except CascadeError as e:
        raise _fail(str(e)) from e
    name = f"{train.provenance.data_source}-{train.provenance.matrix_hash[:8]}"
    target = out_dir or Path("results/dev") / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.md").write_text(report.to_markdown(result), encoding="utf-8")
    (target / "results.json").write_text(report.to_json(result), encoding="utf-8")
    (target / "pareto.svg").write_text(report.to_svg(result), encoding="utf-8")
    typer.echo(report.to_markdown(result))
    typer.echo(f"wrote {target}/")


def register(app: typer.Typer) -> None:
    app.command("eval")(eval_command)
