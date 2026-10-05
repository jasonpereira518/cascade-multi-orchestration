"""CLI for the experiment phase: preflight, pilot, size, calibrate, matrix, ablate, repeat, stats.

Everything here is $0 on the mock provider (the default). A real run needs ``--provider openai``,
``CASCADE_ALLOW_SPEND=1``, and (for real runs only) a passing ``cascade preflight`` that covers
every task about to run. The commands are thin: selection, ordering, caps, sizing, calibration, and
statistics live in ``cascade.routing``; running one attempt lives in ``cascade.agents.runner``.

One processing tier per matrix store (see routing.matrix); the pilot is always standard tier.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from cascade.agents import runner as agent_runner
from cascade.agents.pipeline import PipelineConfig
from cascade.config import (
    ArmSpec,
    ModelsConfig,
    ReposConfig,
    ResolvedBudget,
    load_budget,
    load_models,
    load_repos,
)
from cascade.errors import CascadeError
from cascade.harness.ledger import FLEX_TIER, STANDARD_TIER, Ledger
from cascade.harness.providers.base import ModelSource
from cascade.harness.providers.mock import MockModelSource
from cascade.harness.providers.openai import OpenAIModelSource
from cascade.routing import analysis, calibrate, sizing
from cascade.routing.export import export_parquet
from cascade.routing.matrix import (
    Cell,
    MatrixRow,
    MatrixStatus,
    MatrixStore,
    config_id_of,
    plan_cells,
    run_matrix,
)
from cascade.routing.preflight import format_preflight, run_preflight
from cascade.routing.selection import audited_ok, draw_train
from cascade.routing.stats import collect_stats, format_stats
from cascade.taskgen.audit import AuditVerdict, load_verdicts
from cascade.taskgen.build_env import DockerUnavailableError, docker_available
from cascade.taskgen.manifest import read_manifest
from cascade.taskgen.schema import Task

DEFAULT_MAX_OUTPUT_TOKENS = 8000  # part of the pipeline config, so pilot and matrix must agree
PILOT_N = 10
ABLATE_N = 20
REPEAT_N = 10
SHUFFLE_SEED = 0
REPEAT_SEED = 1

matrix_app = typer.Typer(no_args_is_help=True, help="The resumable outcome matrix.")

_Configs = Annotated[Path, typer.Option(help="Directory holding the config YAMLs.")]
_Manifest = Annotated[Path, typer.Option(help="Task manifest.")]
_Audit = Annotated[Path, typer.Option(help="Audit verdict log.")]
_Runs = Annotated[Path, typer.Option(help="Ledger, matrix, attempts, status files.")]
_TaskgenRuns = Annotated[Path, typer.Option(help="Per-repo clones from the task pipeline.")]
_Provider = Annotated[str | None, typer.Option(help="mock (default) or openai.")]
_MaxOut = Annotated[int, typer.Option(help="Per model call, reasoning included.")]
_MatrixDb = Annotated[
    Path | None, typer.Option(help="Matrix store (default: <runs-dir>/matrix.sqlite).")
]


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(2)


# --- the environment every run command shares ---------------------------------------------------


@dataclass
class Env:
    models: ModelsConfig
    budget: ResolvedBudget
    repos: ReposConfig
    tasks: list[Task]
    verdicts: dict[str, AuditVerdict]
    ledger: Ledger
    store: MatrixStore
    source: ModelSource
    provider: str
    runs_dir: Path
    taskgen_runs_dir: Path
    manifest: Path
    config: PipelineConfig

    def clone_for(self, task: Task) -> Path:
        return self.taskgen_runs_dir / task.repo.replace("/", "__") / "clone.git"


def default_pipeline_config(repos: ReposConfig, max_output_tokens: int) -> PipelineConfig:
    """THE pipeline config of the matrix. Pilot, matrix, size, and calibrate must all build it the
    same way, because it is part of the matrix key: a mismatch silently finds no rows."""
    return PipelineConfig(
        max_output_tokens=max_output_tokens,
        test_timeout_s=float(repos.limits.max_test_runtime_s),
    )


def _make_source(provider: str, tier: str = STANDARD_TIER) -> ModelSource:
    if provider == "mock":
        return MockModelSource()
    if provider == "openai":
        # refuses unless CASCADE_ALLOW_SPEND=1 and OPENAI_API_KEY is set
        return OpenAIModelSource(flex=tier == FLEX_TIER)
    raise _fail(f"unknown provider {provider!r}; expected 'mock' or 'openai'")


def _provider_name(provider: str | None) -> str:
    return provider or os.environ.get("CASCADE_PROVIDER") or "mock"


def _load_tasks(manifest: Path, audit_file: Path) -> tuple[list[Task], dict[str, AuditVerdict]]:
    return read_manifest(manifest), dict(load_verdicts(audit_file))


@contextmanager
def open_env(
    *,
    configs_dir: Path,
    manifest: Path,
    audit_file: Path,
    runs_dir: Path,
    taskgen_runs_dir: Path,
    provider: str | None,
    max_output_tokens: int,
    matrix_db: Path | None,
    tier: str = STANDARD_TIER,
) -> Generator[Env]:
    models = load_models(configs_dir / "models.yaml")
    budget = load_budget(configs_dir / "budget.yaml").resolve()
    repos = load_repos(configs_dir / "repos.yaml")
    tasks, verdicts = _load_tasks(manifest, audit_file)
    name = _provider_name(provider)
    source = _make_source(name, tier)
    runs_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(runs_dir / "ledger.sqlite", budget=budget, models=models)
    store = MatrixStore(matrix_db or runs_dir / "matrix.sqlite")
    try:
        yield Env(
            models=models, budget=budget, repos=repos, tasks=tasks, verdicts=verdicts,
            ledger=ledger, store=store, source=source, provider=name, runs_dir=runs_dir,
            taskgen_runs_dir=taskgen_runs_dir, manifest=manifest,
            config=default_pipeline_config(repos, max_output_tokens),
        )  # fmt: skip
    finally:
        ledger.close()
        store.close()


def require_preflight(runs_dir: Path, tasks: list[Task], provider: str) -> None:
    """A real run needs a passing preflight record covering every task about to run."""
    if provider == "mock":
        return
    path = runs_dir / "preflight.json"
    if not path.exists():
        raise _fail("no preflight record; run `cascade preflight` first (real runs require it)")
    record = json.loads(path.read_text(encoding="utf-8"))
    passed = set(record.get("passed", []))
    missing = sorted(t.task_id for t in tasks if t.task_id not in passed)
    if missing:
        raise _fail(
            f"{len(missing)} task(s) have no passing preflight (e.g. {missing[0]}); "
            "rerun `cascade preflight` and fix or exclude them"
        )


def _check_tier(env: Env, arms: list[ArmSpec], tier: str) -> None:
    if tier not in (STANDARD_TIER, FLEX_TIER):
        raise _fail(f"unknown tier {tier!r}; expected 'standard' or 'flex'")
    if tier == FLEX_TIER:
        for arm in arms:
            if env.models.model_spec(arm.model).flex_multiplier is None:
                raise _fail(
                    f"model {arm.model!r} has no flex_multiplier in models.yaml, so it cannot "
                    "run on flex (check OpenAI's pricing page for whether it supports flex)"
                )
    other = sorted({r.processing_tier for r in env.store.rows() if r.processing_tier != tier})
    if other:
        raise _fail(
            f"the matrix store already has rows on tier {other}; one matrix, one tier. "
            "Use --matrix-db with a fresh file to run on a different tier"
        )


def run_batch(
    env: Env,
    *,
    tasks: list[Task],
    arms: list[ArmSpec],
    seeds: list[int],
    config: PipelineConfig,
    bucket: str,
    tier: str,
    status_name: str,
    max_cells: int | None = None,
    echo: Callable[[str], None] = typer.echo,
    executor_factory_for: Callable[[Task], agent_runner.ExecutorFactory] | None = None,
) -> MatrixStatus:
    """Run (tasks x arms x seeds) task-major through the resumable runner."""
    if executor_factory_for is None and not docker_available():
        raise DockerUnavailableError("the docker daemon is not reachable; start Docker")
    by_id = {t.task_id: t for t in tasks}
    arm_by_id = {a.arm_id: a for a in arms}
    cells = plan_cells(tasks, [a.arm_id for a in arms], seeds, shuffle_seed=SHUFFLE_SEED)
    total = len(cells)
    counter = {"n": 0}

    def execute(cell: Cell) -> MatrixRow:
        task = by_id[cell.task_id]
        spec = next((r for r in env.repos.repos if r.name == task.repo), None)
        if spec is None or spec.test_cmd.startswith("TODO"):
            raise CascadeError(f"repos.yaml has no filled entry for {task.repo}")
        clone = env.clone_for(task)
        if not clone.exists():
            raise CascadeError(f"no clone at {clone}; run `cascade taskgen run-all` first")
        factory = (
            executor_factory_for(task)
            if executor_factory_for
            else agent_runner.docker_executor_factory(task, spec.test_cmd)
        )
        report = agent_runner.run_attempt(
            task=task, arm=arm_by_id[cell.arm_id], seed=cell.seed, config=config,
            ledger=env.ledger, matrix=env.store, source=env.source, clone=clone,
            runs_dir=env.runs_dir, executor_factory=factory,
            forbidden_host_paths=[env.manifest.resolve().parent, env.taskgen_runs_dir.resolve()],
            grading_timeout_s=float(env.repos.limits.max_test_runtime_s),
            bucket=bucket, tier=tier,
        )  # fmt: skip
        counter["n"] += 1
        r = report.row
        echo(
            f"[{counter['n']}/{total}] {r.task_id} | {r.arm_id} | seed {r.seed} | "
            f"true={r.true_success} claimed={r.verifier_claimed} | ${r.cost_usd:.4f} | "
            f"{r.turns} calls" + (f" | {r.error}" if r.error else "")
        )
        for violation in report.leak_violations:
            typer.echo(f"LEAK AUDIT: {r.task_id} {r.arm_id}: {violation}", err=True)
        return r

    status = run_matrix(
        cells=cells, config_id=config_id_of(config.canonical()), store=env.store,
        ledger=env.ledger, budget=env.budget, execute=execute, bucket=bucket, tier=tier,
        status_path=env.runs_dir / status_name, max_cells=max_cells,
    )  # fmt: skip
    echo(
        f"{status.state}: {status.done}/{status.planned} cells done "
        f"({status.executed} this run), reason: {status.reason}; spent ${status.spent_usd:.4f}"
    )
    return status


# --- commands -----------------------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    app.add_typer(matrix_app, name="matrix")
    app.command("preflight")(preflight_command)
    app.command("pilot")(pilot_command)
    app.command("size")(size_command)
    app.command("calibrate")(calibrate_command)
    app.command("ablate")(ablate_command)
    app.command("repeat-subset")(repeat_command)
    app.command("stats")(stats_command)
    app.command("export")(export_command)


def preflight_command(
    manifest: _Manifest = Path("tasks/manifest.jsonl"),
    audit_file: _Audit = Path("tasks/audit.jsonl"),
    runs_dir: _Runs = Path("runs"),
    taskgen_runs_dir: _TaskgenRuns = Path("runs/taskgen"),
) -> None:
    """Leakage pre-flight over every audited-ok task. No spend. Writes runs/preflight.json."""
    try:
        tasks, verdicts = _load_tasks(manifest, audit_file)
    except (CascadeError, ValueError) as e:
        raise _fail(str(e)) from e
    ok = audited_ok(tasks, verdicts)
    if not ok:
        raise _fail("no audited-ok tasks to check; build the manifest and finish the audit first")
    results = run_preflight(
        ok, lambda t: taskgen_runs_dir / t.repo.replace("/", "__") / "clone.git"
    )
    typer.echo(format_preflight(results))
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "preflight.json").write_text(
        json.dumps(
            {
                "at": dt.datetime.now(dt.UTC).isoformat(),
                "checked": len(results),
                "passed": [r.task_id for r in results if r.ok],
                "failed": {r.task_id: r.violations for r in results if not r.ok},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if any(not r.ok for r in results):
        raise typer.Exit(1)


def pilot_command(
    n: Annotated[int, typer.Option(help="Train tasks in the pilot.")] = PILOT_N,
    seed: Annotated[int, typer.Option(help="Stratified-draw seed.")] = 0,
    max_cells: Annotated[int | None, typer.Option(help="Stop after this many cells.")] = None,
    provider: _Provider = None,
    max_output_tokens: _MaxOut = DEFAULT_MAX_OUTPUT_TOKENS,
    configs_dir: _Configs = Path("configs"),
    manifest: _Manifest = Path("tasks/manifest.jsonl"),
    audit_file: _Audit = Path("tasks/audit.jsonl"),
    runs_dir: _Runs = Path("runs"),
    taskgen_runs_dir: _TaskgenRuns = Path("runs/taskgen"),
    matrix_db: _MatrixDb = None,
) -> None:
    """Pilot: N TRAIN tasks (stratified by repo and size), every arm, standard tier."""
    try:
        with open_env(
            configs_dir=configs_dir, manifest=manifest, audit_file=audit_file, runs_dir=runs_dir,
            taskgen_runs_dir=taskgen_runs_dir, provider=provider,
            max_output_tokens=max_output_tokens, matrix_db=matrix_db,
        ) as env:  # fmt: skip
            arms = env.models.matrix_arms()
            drawn = draw_train(env.tasks, env.verdicts, n, seed=seed, purpose="pilot")
            require_preflight(runs_dir, drawn, env.provider)
            _check_tier(env, arms, STANDARD_TIER)
            typer.echo(
                f"pilot: {len(drawn)} train task(s), {len(arms)} arm(s), standard tier, "
                f"provider {env.provider}"
            )
            run_batch(
                env, tasks=drawn, arms=arms, seeds=[0], config=env.config, bucket="pilot",
                tier=STANDARD_TIER, status_name="pilot_status.json", max_cells=max_cells,
            )  # fmt: skip
            typer.echo(_pilot_table(env, drawn, arms))
    except (CascadeError, ValueError) as e:
        raise _fail(str(e)) from e


def _pilot_table(env: Env, tasks: list[Task], arms: list[ArmSpec]) -> str:
    cfg_id = config_id_of(env.config.canonical())
    lines = [
        "PILOT RESULTS",
        "task | arm | true | claimed | cost | tokens in/out/reasoning | calls",
    ]
    for t in sorted(tasks, key=lambda x: x.task_id):
        subtotal = 0.0
        for a in arms:
            r = env.store.get(t.task_id, a.arm_id, 0, cfg_id)
            if r is None:
                lines.append(f"{t.task_id} | {a.arm_id} | (not run)")
                continue
            subtotal += r.cost_usd
            lines.append(
                f"{t.task_id} | {a.arm_id} | {r.true_success} | {r.verifier_claimed} | "
                f"${r.cost_usd:.4f} | {r.input_tokens}/{r.output_tokens}/{r.reasoning_tokens} | "
                f"{r.turns}" + (f" | {r.error}" if r.error else "")
            )
        lines.append(f"{t.task_id} | all arms | ${subtotal:.4f}")
    return "\n".join(lines)


def size_command(
    configs_dir: _Configs = Path("configs"),
    manifest: _Manifest = Path("tasks/manifest.jsonl"),
    audit_file: _Audit = Path("tasks/audit.jsonl"),
    runs_dir: _Runs = Path("runs"),
    max_output_tokens: _MaxOut = DEFAULT_MAX_OUTPUT_TOKENS,
    matrix_db: _MatrixDb = None,
) -> None:
    """Size the full matrix from the pilot ledger. Prints options; chooses none."""
    try:
        models = load_models(configs_dir / "models.yaml")
        budget = load_budget(configs_dir / "budget.yaml").resolve()
        repos = load_repos(configs_dir / "repos.yaml")
        ledger_db = runs_dir / "ledger.sqlite"
        if not ledger_db.exists():
            raise _fail(f"no ledger at {ledger_db}; run `cascade pilot` first")
        tasks, verdicts = _load_tasks(manifest, audit_file)
        stats = sizing.arm_cost_stats(sizing.attempt_costs(ledger_db, "pilot"), models)
        store = MatrixStore(matrix_db or runs_dir / "matrix.sqlite")
        ledger = Ledger(ledger_db, budget=budget, models=models)
        try:
            cfg_id = config_id_of(default_pipeline_config(repos, max_output_tokens).canonical())
            arm_ids = [a.arm_id for a in models.matrix_arms()]
            complete = sum(
                all(store.get(t, a, 0, cfg_id) is not None for a in arm_ids)
                for t in sizing.bucket_task_ids(ledger_db, "pilot")
            )
            report = sizing.build_report(
                stats, models, budget,
                matrix_spent_usd=ledger.bucket_spent_usd("matrix"),
                global_spent_usd=ledger.spent_usd(),
                audited_tasks=len(audited_ok(tasks, verdicts)),
                pilot_tasks=complete,
            )  # fmt: skip
        finally:
            ledger.close()
            store.close()
    except (CascadeError, ValueError) as e:
        raise _fail(str(e)) from e
    typer.echo(sizing.format_report(report))


def calibrate_command(
    configs_dir: _Configs = Path("configs"),
    runs_dir: _Runs = Path("runs"),
    max_output_tokens: _MaxOut = DEFAULT_MAX_OUTPUT_TOKENS,
    matrix_db: _MatrixDb = None,
) -> None:
    """Per-arm pass rate, Verifier claimed rate and FP/FN on the pilot, with flags."""
    try:
        models = load_models(configs_dir / "models.yaml")
        repos = load_repos(configs_dir / "repos.yaml")
        ledger_db = runs_dir / "ledger.sqlite"
        if not ledger_db.exists():
            raise _fail(f"no ledger at {ledger_db}; run `cascade pilot` first")
        store = MatrixStore(matrix_db or runs_dir / "matrix.sqlite")
        try:
            rows = store.rows()
        finally:
            store.close()
        report = calibrate.calibrate(
            rows, models, task_ids=sizing.bucket_task_ids(ledger_db, "pilot"),
            pipeline_config=default_pipeline_config(repos, max_output_tokens).canonical(),
        )  # fmt: skip
    except (CascadeError, ValueError) as e:
        raise _fail(str(e)) from e
    typer.echo(calibrate.format_calibration(report))


def _label_arm(models: ModelsConfig, label: str) -> ArmSpec:
    try:
        return models.arm_by_label_or_id(label)
    except KeyError:
        raise _fail(f"models.yaml has no arm labelled {label!r} (add `label: {label}`)") from None


def ablate_command(
    n: Annotated[int, typer.Option(help="Train tasks to ablate on.")] = ABLATE_N,
    seed: Annotated[int, typer.Option(help="Stratified-draw seed.")] = 0,
    tier: Annotated[str, typer.Option(help="standard (default) or flex.")] = STANDARD_TIER,
    max_cells: Annotated[int | None, typer.Option(help="Stop after this many cells.")] = None,
    provider: _Provider = None,
    max_output_tokens: _MaxOut = DEFAULT_MAX_OUTPUT_TOKENS,
    configs_dir: _Configs = Path("configs"),
    manifest: _Manifest = Path("tasks/manifest.jsonl"),
    audit_file: _Audit = Path("tasks/audit.jsonl"),
    runs_dir: _Runs = Path("runs"),
    taskgen_runs_dir: _TaskgenRuns = Path("runs/taskgen"),
    matrix_db: _MatrixDb = None,
) -> None:
    """Ablations (no reproducer, no reviewer) with the mid arm on N TRAIN tasks."""
    try:
        with open_env(
            configs_dir=configs_dir, manifest=manifest, audit_file=audit_file, runs_dir=runs_dir,
            taskgen_runs_dir=taskgen_runs_dir, provider=provider,
            max_output_tokens=max_output_tokens, matrix_db=matrix_db, tier=tier,
        ) as env:  # fmt: skip
            mid = _label_arm(env.models, "mid")
            drawn = draw_train(env.tasks, env.verdicts, n, seed=seed, purpose="ablate")
            require_preflight(runs_dir, drawn, env.provider)
            _check_tier(env, [mid], tier)
            typer.echo(
                f"ablate: {len(drawn)} train task(s), arm {mid.arm_id}, tier {tier}, "
                f"provider {env.provider}"
            )
            base = env.config
            results: list[analysis.Paired] = []
            ids = {t.task_id for t in drawn}
            for name, cfg in analysis.ablation_configs(base).items():
                run_batch(
                    env, tasks=drawn, arms=[mid], seeds=[0], config=cfg, bucket="ablate",
                    tier=tier, status_name=f"ablate_{name}_status.json", max_cells=max_cells,
                )  # fmt: skip
                results.append(
                    analysis.pair(
                        env.store.rows(),
                        arm_id=mid.arm_id,
                        label=f"{name} vs full",
                        task_ids=ids,
                        base=(0, base.canonical()),
                        other=(0, cfg.canonical()),
                    )
                )
            typer.echo(analysis.format_paired("ABLATIONS (mid arm)", "full", results))
    except (CascadeError, ValueError) as e:
        raise _fail(str(e)) from e


def repeat_command(
    n: Annotated[int, typer.Option(help="Train tasks to repeat.")] = REPEAT_N,
    seed: Annotated[int, typer.Option(help="Seed for the repeat attempts.")] = REPEAT_SEED,
    draw_seed: Annotated[int, typer.Option(help="Stratified-draw seed.")] = 0,
    tier: Annotated[str, typer.Option(help="standard (default) or flex.")] = STANDARD_TIER,
    max_cells: Annotated[int | None, typer.Option(help="Stop after this many cells.")] = None,
    provider: _Provider = None,
    max_output_tokens: _MaxOut = DEFAULT_MAX_OUTPUT_TOKENS,
    configs_dir: _Configs = Path("configs"),
    manifest: _Manifest = Path("tasks/manifest.jsonl"),
    audit_file: _Audit = Path("tasks/audit.jsonl"),
    runs_dir: _Runs = Path("runs"),
    taskgen_runs_dir: _TaskgenRuns = Path("runs/taskgen"),
    matrix_db: _MatrixDb = None,
) -> None:
    """Rerun the cheap and mid arms on N TRAIN tasks with a second seed, to estimate noise."""
    if seed == 0:
        raise _fail("the repeat seed must differ from the base seed 0")
    try:
        with open_env(
            configs_dir=configs_dir, manifest=manifest, audit_file=audit_file, runs_dir=runs_dir,
            taskgen_runs_dir=taskgen_runs_dir, provider=provider,
            max_output_tokens=max_output_tokens, matrix_db=matrix_db, tier=tier,
        ) as env:  # fmt: skip
            arms = [_label_arm(env.models, "cheap"), _label_arm(env.models, "mid")]
            drawn = draw_train(env.tasks, env.verdicts, n, seed=draw_seed, purpose="repeat")
            require_preflight(runs_dir, drawn, env.provider)
            _check_tier(env, arms, tier)
            typer.echo(
                f"repeat-subset: {len(drawn)} train task(s), arms "
                f"{', '.join(a.arm_id for a in arms)}, seed {seed}, tier {tier}"
            )
            run_batch(
                env, tasks=drawn, arms=arms, seeds=[seed], config=env.config, bucket="repeat",
                tier=tier, status_name="repeat_status.json", max_cells=max_cells,
            )  # fmt: skip
            ids = {t.task_id for t in drawn}
            cfg = env.config.canonical()
            results = [
                analysis.pair(
                    env.store.rows(),
                    arm_id=a.arm_id,
                    label=f"{a.label}: seed 0 vs seed {seed}",
                    task_ids=ids,
                    base=(0, cfg),
                    other=(seed, cfg),
                )
                for a in arms
            ]
            typer.echo(analysis.format_paired("REPEAT (noise estimate)", "seed 0", results))
            for p in results:
                if p.agreement is not None:
                    typer.echo(
                        f"  {p.arm_id}: outcome agreement {p.agreement:.0%} over "
                        f"{p.tasks_with_both} paired task(s)"
                    )
    except (CascadeError, ValueError) as e:
        raise _fail(str(e)) from e


@matrix_app.command("run")
def matrix_run_command(
    tier: Annotated[str, typer.Option(help="standard (default) or flex; one per matrix.")] = (
        STANDARD_TIER
    ),
    seed: Annotated[int, typer.Option(help="Attempt seed (base seed is 0).")] = 0,
    max_cells: Annotated[int | None, typer.Option(help="Stop after this many cells.")] = None,
    provider: _Provider = None,
    max_output_tokens: _MaxOut = DEFAULT_MAX_OUTPUT_TOKENS,
    configs_dir: _Configs = Path("configs"),
    manifest: _Manifest = Path("tasks/manifest.jsonl"),
    audit_file: _Audit = Path("tasks/audit.jsonl"),
    runs_dir: _Runs = Path("runs"),
    taskgen_runs_dir: _TaskgenRuns = Path("runs/taskgen"),
    matrix_db: _MatrixDb = None,
) -> None:
    """Run (audited-ok task x arm) cells, task-major and resumable. Stop file: the kill switch."""
    try:
        with open_env(
            configs_dir=configs_dir, manifest=manifest, audit_file=audit_file, runs_dir=runs_dir,
            taskgen_runs_dir=taskgen_runs_dir, provider=provider,
            max_output_tokens=max_output_tokens, matrix_db=matrix_db, tier=tier,
        ) as env:  # fmt: skip
            ok = audited_ok(env.tasks, env.verdicts)
            if not ok:
                raise _fail("no audited-ok tasks; finish the audit first")
            unsplit = [t.task_id for t in ok if t.split is None]
            if unsplit:
                raise _fail(
                    f"{len(unsplit)} audited-ok task(s) have no split; run `tasks finalize`"
                )
            arms = env.models.matrix_arms()
            require_preflight(runs_dir, ok, env.provider)
            _check_tier(env, arms, tier)
            typer.echo(
                f"matrix: {len(ok)} task(s) x {len(arms)} arm(s), tier {tier}, "
                f"provider {env.provider}; stop file: {env.budget.kill_switch_path}"
            )
            run_batch(
                env, tasks=ok, arms=arms, seeds=[seed], config=env.config, bucket="matrix",
                tier=tier, status_name="matrix_status.json", max_cells=max_cells,
            )  # fmt: skip
    except (CascadeError, ValueError) as e:
        raise _fail(str(e)) from e


@matrix_app.command("status")
def matrix_status_command(runs_dir: _Runs = Path("runs")) -> None:
    """Print matrix_status.json."""
    path = runs_dir / "matrix_status.json"
    if not path.exists():
        raise _fail(f"no {path}; the matrix has not run")
    typer.echo(path.read_text(encoding="utf-8").rstrip())


def stats_command(
    configs_dir: _Configs = Path("configs"),
    manifest: _Manifest = Path("tasks/manifest.jsonl"),
    audit_file: _Audit = Path("tasks/audit.jsonl"),
    runs_dir: _Runs = Path("runs"),
    matrix_db: _MatrixDb = None,
    repo_root: Annotated[Path, typer.Option(help="Where to run pytest --collect-only.")] = Path(
        "."
    ),
) -> None:
    """Resume-ready numbers, each read from an artifact."""
    try:
        models = load_models(configs_dir / "models.yaml")
        tasks, verdicts = _load_tasks(manifest, audit_file)
        db = matrix_db or runs_dir / "matrix.sqlite"
        rows: list[MatrixRow] = []
        if db.exists():
            store = MatrixStore(db)
            try:
                rows = store.rows()
            finally:
                store.close()
        stats = collect_stats(
            tasks=tasks, verdicts=verdicts, models=models, matrix_rows=rows,
            ledger_db=runs_dir / "ledger.sqlite", repo_root=repo_root,
        )  # fmt: skip
    except (CascadeError, ValueError) as e:
        raise _fail(str(e)) from e
    typer.echo(format_stats(stats))


def export_command(
    runs_dir: _Runs = Path("runs"),
    out_dir: Annotated[
        Path | None, typer.Option(help="Where the parquet files go (default: <runs-dir>/export).")
    ] = None,
    matrix_db: _MatrixDb = None,
) -> None:
    """Export the matrix and the ledger to parquet (a straight copy; nothing derived)."""
    target = out_dir or runs_dir / "export"
    counts = export_parquet(
        matrix_db or runs_dir / "matrix.sqlite", runs_dir / "ledger.sqlite", target
    )
    typer.echo(f"wrote {target}/matrix.parquet ({counts['matrix']} rows)")
    typer.echo(f"wrote {target}/ledger.parquet ({counts['ledger']} rows)")
