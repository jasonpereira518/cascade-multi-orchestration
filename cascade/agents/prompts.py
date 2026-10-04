"""Role instructions and prompt builders.

Leakage by construction: every builder takes an ``AgentView`` (issue text and the pinned commit,
nothing else) plus outputs of earlier roles or the Verifier. None accepts a ``Task``, so the gold
patch, the test patch, and the hidden test ids cannot be passed in even by mistake. The audit in
``agents.audit`` and the leakage tests check the resulting prompts independently.
"""

from __future__ import annotations

from cascade.agents.schemas import (
    LocalizationOutput,
    ReproOutput,
    ReviewOutput,
    TriageOutput,
    VerifierReport,
)
from cascade.taskgen.schema import AgentView

MAX_ISSUE_CHARS = 6000
MAX_PATCH_CHARS = 12_000

_COMMON = (
    "You are working on a Python repository checked out at a fixed commit, to resolve a GitHub "
    "issue. The issue text is your only description of the problem. You cannot browse the web or "
    "see the project's later history. Be economical: read only what you need."
)

INSTRUCTIONS = {
    "triager": (
        f"{_COMMON}\nYou triage the issue. Read it and reply with the structured result only. "
        "You have no tools."
    ),
    "reproducer": (
        f"{_COMMON}\nYour job: write ONE minimal pytest file that FAILS on the current code "
        "because of the bug in the issue. Read the relevant code with the tools, then call "
        "write_reproducer, then run_tests on the reproducer path to confirm it fails for the right "
        "reason (an assertion about the reported behaviour, not an import or syntax error). "
        "Fix the test if it fails for the wrong reason. Do not edit source code. Finish with the "
        "structured result."
    ),
    "localizer": (
        f"{_COMMON}\nYour job: find where the fix belongs. Use search_code, grep and read_file. "
        "Return at most 5 files (most likely first) and the functions to change. Do not edit "
        "anything."
    ),
    "patcher": (
        f"{_COMMON}\nYour job: fix the issue by editing source code with edit_file (or "
        "create_file for a genuinely new file). Make the smallest change that resolves the issue "
        "in general, not just for the example. Do not edit test files or the reproducer. After "
        "editing, run the reproducer with run_tests to check. If you are given feedback about an "
        "earlier attempt, address it. Finish with the structured result."
    ),
    "reviewer": (
        f"{_COMMON}\nYou are a strict, independent code reviewer. You are given the issue and a "
        "patch. Approve only if the patch resolves the issue in general, is minimal, and does not "
        "break other behaviour. Otherwise request changes and list concrete issues. You may read "
        "files for context. You cannot edit anything."
    ),
}


def issue_block(view: AgentView) -> str:
    body = view.issue_body
    if len(body) > MAX_ISSUE_CHARS:
        body = body[:MAX_ISSUE_CHARS] + "\n... (issue text truncated)"
    return f"Repository: {view.repo}\nIssue title: {view.issue_title}\n\nIssue body:\n{body}"


def triage_input(view: AgentView) -> str:
    return issue_block(view)


def _triage_block(triage: TriageOutput) -> str:
    areas = ", ".join(triage.suspected_areas) or "none given"
    terms = ", ".join(triage.search_terms) or "none given"
    return (
        f"Triage notes: {triage.summary}\nSuspected areas: {areas}\nSearch terms: {terms}\n"
        f"Estimated difficulty: {triage.difficulty}/5"
    )


def repro_input(view: AgentView, triage: TriageOutput, repro_path: str) -> str:
    return (
        f"{issue_block(view)}\n\n{_triage_block(triage)}\n\n"
        f"Write the reproducer so that it ends up at {repro_path} (write_reproducer puts it there)."
    )


def localize_input(view: AgentView, triage: TriageOutput) -> str:
    return f"{issue_block(view)}\n\n{_triage_block(triage)}"


def patch_input(
    view: AgentView,
    triage: TriageOutput,
    localization: LocalizationOutput | None,
    repro: ReproOutput | None,
    repro_summary: str | None,
    repro_path: str,
    feedback: str | None,
) -> str:
    parts = [issue_block(view), _triage_block(triage)]
    if localization is not None and localization.files:
        files = "\n".join(f"- {f.path}: {f.reason}" for f in localization.files)
        funcs = ", ".join(localization.functions) or "none given"
        parts.append(
            f"Likely files:\n{files}\nLikely functions: {funcs}\nNotes: {localization.notes}"
        )
    if repro is not None and repro_summary is not None:
        parts.append(
            f"A reproducer test exists at {repro_path} ({repro.summary}). Its current result on "
            f"the unpatched code:\n{repro_summary}\nIt should pass once the issue is fixed."
        )
    if feedback:
        parts.append(f"Feedback on your previous attempt (address all of it):\n{feedback}")
    return "\n\n".join(parts)


def review_input(view: AgentView, patch: str, report: VerifierReport) -> str:
    shown = (
        patch if len(patch) <= MAX_PATCH_CHARS else patch[:MAX_PATCH_CHARS] + "\n... (truncated)"
    )
    checks = (
        f"Automated checks: reproducer fails before patch: {report.repro_failed_before}; "
        f"passes after: {report.repro_passes_after}; regressions in existing tests: "
        f"{len(report.regressions)}."
    )
    return f"{issue_block(view)}\n\n{checks}\n\nPatch under review:\n{shown}"


def feedback_from_verifier(report: VerifierReport, detail: str) -> str:
    lines = [f"- {reason}" for reason in report.reasons] or ["- (no specific reason recorded)"]
    text = "The automated checks did not pass:\n" + "\n".join(lines)
    return f"{text}\n\n{detail}" if detail else text


def feedback_from_review(review: ReviewOutput) -> str:
    issues = "\n".join(f"- [{i.severity}] {i.description}" for i in review.issues)
    return f"A reviewer requested changes: {review.summary}\n{issues}".strip()
