"""Render an EvalResult as Markdown, JSON, and an SVG figure. Every artifact carries the stamp.

The stamp is ``data_source | commit (clean/DIRTY) | matrix hash | tier | splits``. Synthetic results
get a banner saying so, dev results get a banner saying they are development numbers, and small
samples get the plain statement that the confidence intervals are wide.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import Any

from cascade.evals import bootstrap as bs
from cascade.evals import svg
from cascade.evals.run import EvalResult


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        items: dict[Any, Any] = dict(value)  # pyright: ignore[reportUnknownArgumentType]
        return {str(k): _clean(v) for k, v in items.items()}
    if isinstance(value, (list, tuple)):
        seq: list[Any] = list(value)  # pyright: ignore[reportUnknownArgumentType]
        return [_clean(v) for v in seq]
    return value


def to_json(r: EvalResult) -> str:
    body = {k: v for k, v in asdict(r).items() if k != "pareto"}
    body["stamp"] = r.stamp
    return json.dumps(_clean(body), indent=2, sort_keys=True) + "\n"


def to_svg(r: EvalResult) -> str:
    kind = "TRAIN / CV (dev)" if r.mode == "dev" else "TEST (final)"
    return svg.pareto_svg(r.pareto, title=f"{kind}: cost vs resolve rate", stamp=r.stamp)


def _iv(i: bs.Interval, *, pct: bool = False, money: bool = False) -> str:
    if i.lo != i.lo:  # NaN: undefined in every resample
        return "undefined"

    def f(v: float) -> str:
        if pct:
            return f"{v:+.1%}"
        return f"${v:.4f}" if money else f"{v:.3f}"

    extra = f" [{i.n_undefined} undefined]" if i.n_undefined else ""
    return f"{f(i.point)} [{f(i.lo)}, {f(i.hi)}]{extra}"


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        *["| " + " | ".join(r) + " |" for r in rows],
    ]


def to_markdown(r: EvalResult) -> str:
    dev = r.mode == "dev"
    title = "TRAIN / CV (dev): development numbers, not evidence" if dev else "TEST (final)"
    lines = [f"# Policy evaluation: {title}", "", f"**{r.stamp}**", ""]
    lines += [
        f"- tasks: {r.n_tasks} in {r.n_repos} repos; incomplete tasks excluded: "
        f"{r.provenance['n_incomplete_tasks']}",
        f"- cap-exceeded cells counted as failures: {r.provenance['n_cap_normalized']}",
        f"- non-inferiority margin: {r.margin:.2f} ({r.margin_basis})",
        f"- intervals: 95% percentile bootstrap, {r.resamples:,} resamples, "
        f"seed {r.seed}; paired where policies are compared",
    ]
    if r.provenance["data_source"] == "synthetic":
        lines += ["", "> **SYNTHETIC DATA.** Invented numbers for developing the code. No claim."]
    if r.n_tasks < 40:
        lines += [
            "",
            f"> **Confidence intervals are wide: only {r.n_tasks} tasks.** Differences smaller "
            "than the intervals cannot be told from noise.",
        ]
    if r.n_repos < 5:
        lines += [
            "",
            f"> **Repo-clustered intervals rest on only {r.n_repos} repo(s)** and are unreliable "
            "(a resample draws few distinct repos); read them as a rough check, not an interval.",
        ]
    head = [
        "policy",
        "resolve rate",
        "cost / task",
        "cost / resolved",
        "median latency (s)",
        "arms",
    ]
    rows = [
        [
            m.name, _iv(m.resolve_rate), _iv(m.cost_per_task, money=True),
            _iv(m.cost_per_resolved, money=True), _iv(m.median_latency), f"{m.mean_arms_tried:.2f}",
        ]
        for m in r.metrics
    ]  # fmt: skip
    lines += ["", "## Policies", "", *_table(head, rows)]

    lines += [
        "",
        "## Headline: cost reduction vs always-max at equal-or-better resolve rate",
        "",
        f"Margin {r.margin:.2f}. MET means non-inferior (the CI lower bound of the resolve-rate "
        "difference is above -margin) and a cost saving whose CI lower bound is above 0.",
        "",
    ]
    head = [
        "policy", "resolve-rate diff", "cost reduction", "oracle savings captured", "verdict",
        "repo-clustered",
    ]  # fmt: skip
    rows = [
        [
            c.policy, _iv(c.resolve_rate_diff, pct=True), _iv(c.cost_reduction, pct=True),
            _iv(c.oracle_savings_share, pct=True), f"**{c.verdict}**", c.verdict_clustered,
        ]
        for c in r.comparisons
    ]  # fmt: skip
    lines += _table(head, rows)
    lines += [
        "",
        "`oracle` is an unachievable bound (it picks the cheapest successful arm with hindsight). "
        "A policy cheaper than the oracle can capture more than 100% of its savings by failing "
        "more often.",
    ]

    rows = [
        [x.repo, x.policy, str(x.n), f"{x.resolve_rate:.2f}", f"${x.cost_per_task:.4f}"]
        for x in r.per_repo
    ]
    head = ["repo", "policy", "n", "resolve rate", "cost / task"]
    lines += ["", "## Per repo (point estimates)", "", *_table(head, rows)]

    rows = [
        [
            v.arm, str(v.n), str(v.passed), str(v.claimed), str(v.false_pos), str(v.false_neg),
            "n/a" if v.fp_rate is None else _iv(v.fp_rate),
            "n/a" if v.fn_rate is None else _iv(v.fn_rate),
        ]
        for v in r.verifier
    ]  # fmt: skip
    head = ["arm", "n", "passed", "claimed", "FP", "FN", "FP rate", "FN rate"]
    lines += ["", "## Verifier errors per arm", "", *_table(head, rows)]

    if r.ablation:
        rows = [
            [
                a.variant, str(a.n_shared), f"{a.resolve_rate:.2f}", f"${a.cost_per_task:.4f}",
                "-" if a.variant == "default" else _iv(a.resolve_rate_diff, pct=True),
                "-" if a.variant == "default" else _iv(a.cost_per_task_diff, money=True),
            ]
            for a in r.ablation
        ]  # fmt: skip
        head = ["variant", "shared n", "resolve rate", "cost / task", "resolve diff", "cost diff"]
        title = "## Ablations (mid arm, paired on shared train tasks)"
        lines += ["", title, "", *_table(head, rows)]
    if r.noise:
        rows = [[n.arm, str(n.n_pairs), str(n.flips), _iv(n.flip_rate)] for n in r.noise]
        head = ["arm", "pairs", "flips", "flip rate"]
        lines += [
            "",
            "## Noise: cells whose outcome flipped on a repeat seed",
            "",
            *_table(head, rows),
        ]

    lines += ["", "## Predictive router selection (leave-one-repo-out CV on train)", ""]
    lines += [
        f"- `{fs}`: {json.dumps(_clean(s), sort_keys=True)}" for fs, s in r.selections.items()
    ]
    lines += [f"\n> {note}" for note in r.notes]
    lines += ["", f"_{r.stamp}_", ""]
    return "\n".join(lines)
