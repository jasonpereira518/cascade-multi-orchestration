"""The agent pipeline: triage -> reproduce -> localize -> patch -> review -> verify, with repair.

Flow for one attempt (one arm on one task):
  1. Triager reads the issue (no tools).
  2. Reproducer (toggle) writes one failing pytest file. The pipeline itself then runs it on the
     untouched code and records whether it truly fails; the agent's word is not trusted.
  3. Localizer finds files and functions with BM25/grep/read tools.
  4. Patcher edits source. The source-only patch is built deterministically from the edits.
  5. Verifier (no LLM) runs the evidence checks and yields verifier_claimed_success.
  6. Reviewer (toggle), a separate critic, reads the patch once the Verifier is satisfied.
  If the Verifier fails or the Reviewer requests changes, the Patcher gets the feedback and tries
  again, at most ``max_repairs`` more times.

Caps: every model call is authorized by the ledger first (hard global/task/attempt dollar caps and
the per-attempt token cap). When a cap or the kill switch trips, the attempt stops calling models,
but its current edits are still turned into a patch and verified, so the outcome is recorded
honestly rather than discarded.

The pipeline receives an ``AgentView`` only. Grading against hidden tests happens elsewhere
(``agents.runner``) and never feeds back into any prompt.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import openai
from agents.exceptions import AgentsException
from pydantic import BaseModel

from cascade.agents import prompts
from cascade.agents.failures import summarize_run
from cascade.agents.patching import PatchError, is_source_path, make_patch
from cascade.agents.roles import ROLE_SPECS, build_agent, run_role
from cascade.agents.schemas import (
    LOCALIZER,
    PATCHER,
    REPRODUCER,
    REVIEWER,
    TRIAGER,
    VERIFIER,
    LocalizationOutput,
    ReproOutput,
    ReviewOutput,
    StageRecord,
    TriageOutput,
    VerifierReport,
    clamp_difficulty,
)
from cascade.agents.tools import REPRO_PATH, Toolbox
from cascade.agents.verifier import Verifier
from cascade.config import ArmSpec
from cascade.errors import BudgetExceededError, KillSwitchEngagedError
from cascade.harness.ledger import Ledger
from cascade.harness.models import AttemptRef, MeteredModel
from cascade.harness.providers.base import ModelSource
from cascade.harness.trace import TraceWriter
from cascade.taskgen.schema import AgentView
from cascade.taskgen.validate import TestExecutor


@dataclass(frozen=True)
class PipelineConfig:
    reproducer: bool = True
    reviewer: bool = True
    max_repairs: int = 2
    max_output_tokens: int = 4096  # per model call, reasoning included
    test_timeout_s: float = 120.0
    max_test_runs_per_role: int = 6

    def __post_init__(self) -> None:
        if self.max_repairs < 0 or self.max_output_tokens <= 0 or self.test_timeout_s <= 0:
            raise ValueError("invalid pipeline config")

    def canonical(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def config_id(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:10]


@dataclass
class AttemptOutcome:
    src_patch: str = ""  # source-only patch (no test files): what gets graded
    full_patch: str = ""  # everything the agent changed, tests and reproducer included
    report: VerifierReport | None = None
    rounds: int = 0  # Patcher invocations
    error: str | None = None
    stages: list[StageRecord] = field(default_factory=lambda: [])
    triage: TriageOutput | None = None
    repro_valid: bool = False
    review_verdicts: list[str] = field(default_factory=lambda: [])

    @property
    def claimed(self) -> bool:
        return self.report is not None and self.report.claimed


class _Abort(Exception):
    """A cap or the kill switch stopped the attempt; carries the reason for the outcome row."""


class AgentPipeline:
    def __init__(
        self,
        *,
        source: ModelSource,
        arm: ArmSpec,
        ledger: Ledger,
        executor: TestExecutor,
        config: PipelineConfig,
        attempt: AttemptRef,
        trace: TraceWriter,
        scratch: Path,
    ) -> None:
        self._source = source
        self._arm = arm
        self._ledger = ledger
        self._executor = executor
        self._cfg = config
        self._attempt = attempt
        self._trace = trace
        self._scratch = scratch

    # --- helpers ---------------------------------------------------------------------------

    def _stage(
        self, out: AttemptOutcome, stage: str, ok: bool, started: float, detail: str = ""
    ) -> None:
        out.stages.append(
            StageRecord(stage=stage, ok=ok, detail=detail[:300], wall_s=time.monotonic() - started)
        )
        self._trace.write("stage", stage=stage, ok=ok, detail=detail[:300])

    def _call_role(
        self, out: AttemptOutcome, role: str, input_text: str, toolbox: Toolbox | None
    ) -> BaseModel | None:
        """Run one LLM role. Returns None if it failed in a way the attempt can survive."""
        spec = ROLE_SPECS[role]
        model = MeteredModel(
            self._source.model_for(self._arm.model, role),
            ledger=self._ledger,
            attempt=self._attempt,
            role=role,
            model_name=self._arm.model,
            provider_name=self._source.name,
            real=self._source.is_real,
            trace=self._trace,
        )
        agent = build_agent(
            spec, model=model, toolbox=toolbox, arm=self._arm,
            max_output_tokens=self._cfg.max_output_tokens, tier=self._attempt.tier,
        )  # fmt: skip
        started = time.monotonic()
        try:
            result = run_role(agent, spec, input_text)
        except (BudgetExceededError, KillSwitchEngagedError) as e:
            self._stage(out, role, False, started, f"{type(e).__name__}: {e}")
            raise _Abort(f"{type(e).__name__}: {e}") from e
        except (AgentsException, openai.OpenAIError) as e:
            self._stage(out, role, False, started, f"{type(e).__name__}: {e}")
            return None
        self._stage(out, role, True, started)
        return result

    def _toolbox(self, role: str, work_dir: Path, *, tests: bool) -> Toolbox:
        return Toolbox(
            work_dir, role=role, trace=self._trace,
            executor=self._executor if tests else None,
            test_timeout_s=self._cfg.test_timeout_s,
            max_test_runs=self._cfg.max_test_runs_per_role,
        )  # fmt: skip

    def _patches(self, base_dir: Path, work_dir: Path) -> tuple[str, str]:
        return (
            make_patch(base_dir, work_dir, include=is_source_path),
            make_patch(base_dir, work_dir),
        )

    # --- stages ----------------------------------------------------------------------------

    def _reproduce(
        self, out: AttemptOutcome, view: AgentView, triage: TriageOutput, work_dir: Path
    ) -> tuple[ReproOutput | None, str | None]:
        role_out = self._call_role(
            out, REPRODUCER, prompts.repro_input(view, triage, REPRO_PATH),
            self._toolbox(REPRODUCER, work_dir, tests=True),
        )  # fmt: skip
        repro = role_out if isinstance(role_out, ReproOutput) else None
        started = time.monotonic()
        if not (work_dir / REPRO_PATH).is_file():
            self._stage(out, "repro_check", False, started, "no reproducer file was written")
            return repro, None
        # Independent check: run the reproducer on the untouched code. The agent's claim that it
        # fails is not trusted; only a real failing test counts as a reproduction.
        run = self._executor.run(
            workspace=work_dir, pytest_args=["--tb=short", REPRO_PATH],
            timeout_s=self._cfg.test_timeout_s,
        )  # fmt: skip
        prefix = REPRO_PATH + "::"
        mine = {n: o for n, o in run.outcomes.items() if n.startswith(prefix)}
        has_failed = any(o == "failed" for o in mine.values())
        has_error = (
            any(o == "error" for o in mine.values()) or run.outcomes.get(REPRO_PATH) == "error"
        )
        out.repro_valid = has_failed and not has_error
        summary = summarize_run(run)
        self._stage(
            out,
            "repro_check",
            out.repro_valid,
            started,
            "fails on the base code" if out.repro_valid else "does not fail cleanly",
        )
        return repro, summary

    def run(self, view: AgentView, base_dir: Path, work_dir: Path) -> AttemptOutcome:
        if not isinstance(view, AgentView):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("the pipeline accepts an AgentView only, never a full Task")
        out = AttemptOutcome()
        verifier = Verifier(
            executor=self._executor, base_dir=base_dir, scratch=self._scratch,
            repro_path=REPRO_PATH, timeout_s=self._cfg.test_timeout_s,
        )  # fmt: skip
        patcher_started = False
        try:
            self._run_stages(out, view, base_dir, work_dir, verifier)
            patcher_started = True  # reached the end normally
        except _Abort as stop:
            out.error = str(stop)
            patcher_started = any(s.stage == PATCHER for s in out.stages)
        # Finalize: whatever state the workspace is in, record the patch and what the Verifier says.
        try:
            out.src_patch, out.full_patch = self._patches(base_dir, work_dir)
        except PatchError as e:
            out.error = out.error or f"patch_error: {e}"
        if out.report is None and patcher_started:
            result = verifier.verify(
                work_dir=work_dir, patch_nonempty=bool(out.src_patch.strip()),
                repro_enabled=self._cfg.reproducer, repro_failed_before=out.repro_valid,
            )  # fmt: skip
            out.report = result.report
        return out

    def _run_stages(
        self,
        out: AttemptOutcome,
        view: AgentView,
        base_dir: Path,
        work_dir: Path,
        verifier: Verifier,
    ) -> None:
        cfg = self._cfg
        triage_out = self._call_role(out, TRIAGER, prompts.triage_input(view), None)
        triage = triage_out if isinstance(triage_out, TriageOutput) else TriageOutput(
            summary=view.issue_title, issue_type="other", suspected_areas=[], search_terms=[],
            difficulty=3, reproducible_with_unit_test=True,
        )  # fmt: skip
        triage = triage.model_copy(update={"difficulty": clamp_difficulty(triage.difficulty)})
        out.triage = triage

        repro: ReproOutput | None = None
        repro_summary: str | None = None
        if cfg.reproducer:
            repro, repro_summary = self._reproduce(out, view, triage, work_dir)

        loc_out = self._call_role(
            out, LOCALIZER, prompts.localize_input(view, triage),
            self._toolbox(LOCALIZER, work_dir, tests=False),
        )  # fmt: skip
        localization = None
        if isinstance(loc_out, LocalizationOutput):
            kept = [f for f in loc_out.files if (work_dir / f.path).is_file()][:5]
            localization = loc_out.model_copy(update={"files": kept})

        feedback: str | None = None
        for round_index in range(cfg.max_repairs + 1):
            self._call_role(
                out, PATCHER,
                prompts.patch_input(
                    view, triage, localization, repro, repro_summary, REPRO_PATH, feedback
                ),
                self._toolbox(PATCHER, work_dir, tests=True),
            )  # fmt: skip
            out.rounds += 1
            started = time.monotonic()
            try:
                out.src_patch, out.full_patch = self._patches(base_dir, work_dir)
            except PatchError as e:
                out.src_patch = ""
                self._stage(out, "diff", False, started, str(e))
            result = verifier.verify(
                work_dir=work_dir, patch_nonempty=bool(out.src_patch.strip()),
                repro_enabled=cfg.reproducer, repro_failed_before=out.repro_valid,
            )  # fmt: skip
            out.report = result.report
            self._stage(
                out, VERIFIER, result.report.claimed, started, "; ".join(result.report.reasons)
            )
            more_rounds = round_index < cfg.max_repairs

            if not result.report.claimed:
                if not more_rounds:
                    return
                feedback = prompts.feedback_from_verifier(result.report, result.feedback)
                continue
            if not cfg.reviewer:
                return
            review_out = self._call_role(
                out, REVIEWER, prompts.review_input(view, out.src_patch, result.report),
                self._toolbox(REVIEWER, work_dir, tests=False),
            )  # fmt: skip
            if not isinstance(review_out, ReviewOutput):
                return  # a reviewer that fails to answer does not block a verified patch
            out.review_verdicts.append(review_out.verdict)
            if review_out.verdict == "approve" or not more_rounds:
                return
            feedback = prompts.feedback_from_review(review_out)
