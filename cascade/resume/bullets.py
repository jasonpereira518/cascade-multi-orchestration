"""Resume bullets: fill the templates from resume_stats.json, count characters, flag misfits.

Rules (from the Phase 6 brief, and what this module enforces):
- A template renders only if EVERY placeholder it uses has a value in the stats. Otherwise it is
  listed as not rendered, with the missing metrics named. Nothing is invented or defaulted.
- Wording is never edited to make a bullet fit. A rendered bullet longer than its limit, or more
  than 15 characters under it, is FLAGGED with its count. (one_line <= 119, two_line <= 239
  characters, spaces included.)
- ``router_supported`` is used only when the final evaluation supports the claim: the designated
  policy's verdict is MET, i.e. the cost-reduction interval excludes zero AND the lower end of the
  resolve-rate difference is inside the margin of docs/EVAL_PLAN.md. Otherwise ``router_neutral``
  (about cascade vs always-max) is used, and its wording flips when the resolve-rate gap has the
  opposite sign ("higher"), or reads "the same" when the gap is exactly zero.
- Numbers are never flattered. A figure that makes the result look better (cost cut, share of oracle
  savings, a resolve-rate advantage) is TRUNCATED toward zero at one decimal; one that makes it look
  worse (resolve-rate gap, false-positive rate, spend) is rounded UP at one decimal (spend: at the
  cent). Counts are exact. Nothing is ever rounded in the favorable direction.
- Synthetic stats are refused unless ``allow_synthetic``, and every bullet is then marked SYNTHETIC.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any

import yaml

from cascade.errors import CascadeError

ONE_LINE_MAX = 119
TWO_LINE_MAX = 239
UNDER_SLACK = 15
TEMPLATES_PATH = Path("docs/resume_templates.yaml")
BULLETS_PATH = Path("docs/resume_bullets.md")


class BulletsError(CascadeError):
    """The templates or the stats cannot be used."""


def _dec(x: float) -> Decimal:
    return Decimal(repr(float(x)))


def floor_to(x: float, places: int, scale: int = 1) -> str:
    """``x * scale`` rounded toward negative infinity at ``places`` decimals, in exact decimal
    arithmetic (so 0.29 as a percentage is 29.0, never 28.9, and nothing is ever rounded up)."""
    q = Decimal(1).scaleb(-places)
    return f"{(_dec(x) * scale).quantize(q, rounding=ROUND_FLOOR):.{places}f}"


def ceil_to(x: float, places: int, scale: int = 1) -> str:
    """``x * scale`` rounded toward positive infinity at ``places`` decimals."""
    q = Decimal(1).scaleb(-places)
    return f"{(_dec(x) * scale).quantize(q, rounding=ROUND_CEILING):.{places}f}"


@dataclass(frozen=True)
class Template:
    name: str
    kind: str  # "one_line" | "two_line"
    text: str

    @property
    def limit(self) -> int:
        return ONE_LINE_MAX if self.kind == "one_line" else TWO_LINE_MAX

    @property
    def fields(self) -> list[str]:
        return [f for _, f, _, _ in string.Formatter().parse(self.text) if f]


@dataclass
class Bullet:
    name: str
    kind: str
    text: str | None  # None: not rendered
    limit: int
    reason: str = ""  # why it was not rendered, or not used
    flags: list[str] = field(default_factory=lambda: [])
    notes: list[str] = field(default_factory=lambda: [])

    @property
    def chars(self) -> int | None:
        return None if self.text is None else len(self.text)


def load_templates(path: Path = TEMPLATES_PATH) -> list[Template]:
    if not path.exists():
        raise BulletsError(f"missing templates file {path}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: list[Template] = []
    for kind in ("two_line", "one_line"):
        section: dict[str, Any] = raw.get(kind) or {}
        for name, text in section.items():
            out.append(Template(str(name), kind, " ".join(str(text).split())))
    if not out:
        raise BulletsError(f"{path} defines no templates")
    return out


# --- values, each None when the stats do not have it --------------------------------------------


def _pct_floor(x: float | None) -> str | None:
    return None if x is None else floor_to(x, 1, 100)


def _point(d: dict[str, Any] | None) -> float | None:
    return None if not d or d.get("point") is None else float(d["point"])


def _signed(x: float | None) -> str:
    return "n/a" if x is None else f"{100.0 * x:+.1f}"


def comparison_values(stats: dict[str, Any], policy: str) -> dict[str, Any]:
    c = stats["comparisons"].get(policy)
    if c is None:
        return {}
    return {
        "cost_cut": c["cost_per_resolved_reduction_point"],
        "oracle_share": _point(c["oracle_savings_share"]),
        "rr_diff": _point(c["resolve_rate_diff"]),
    }


def base_values(stats: dict[str, Any]) -> dict[str, str | None]:
    """Placeholders that do not depend on which router template is used."""
    pooled = stats["verifier"]["pooled"]
    tasks = stats["tasks"]
    pilot = stats["pilot"]
    fp = None if pooled is None else ceil_to(pooled["fp_rate"]["point"], 1, 100)
    return {
        "n_tasks": str(tasks["evaluated"]["total"]),
        "n_repos": None if tasks["repos"]["n"] is None else str(tasks["repos"]["n"]),
        "n_arms": str(stats["arms"]["n"]),
        "n_attempts": str(stats["attempts"]["n_matrix_rows"]),
        "spend_usd": ceil_to(stats["spend_usd"]["experiment"], 2),
        "verifier_fp_pct": fp,
        "n_pilot": str(pilot["n_reviewed"]),
        "n_pilot_repos": str(pilot["n_repos"]),
        "n_accepted": str(pilot["n_accepted"]),
    }


def _router_values(
    t: Template, stats: dict[str, Any], bullet: Bullet, vals: dict[str, str | None], text: str
) -> str | None:
    """Fill the router-specific placeholders; return the (possibly re-worded) text, or None when
    this template is not used or cannot be rendered (``bullet.reason`` then says why)."""
    designated: str = stats["designated_policy"]
    supported = bool(stats["claim_supported"])
    if t.name == "router_supported":
        if not supported:
            bullet.reason = (
                "not used: the final evaluation does not support the claim (no primary policy "
                f"met the headline; the designated policy is {designated})"
            )
            return None
        cv = comparison_values(stats, designated)
        cut = cv.get("cost_cut")
        if cut is None or cut <= 0:
            bullet.reason = "not rendered: cost per resolved issue did not fall"
            return None
        vals["cost_cut_pct"] = _pct_floor(cut)
        vals["oracle_share_pct"] = _pct_floor(cv.get("oracle_share"))
        bullet.notes.append(
            f"policy: {designated}; headline MET. 'at equal resolve rate' means non-inferior "
            f"within a margin of {stats['final']['margin']} (point difference "
            f"{_signed(cv.get('rr_diff'))} points), not identical"
        )
        return text
    if supported:
        bullet.reason = "not used: router_supported applies"
        return None
    cv = comparison_values(stats, "cascade")
    cut, diff = cv.get("cost_cut"), cv.get("rr_diff")
    if cut is None or diff is None:
        bullet.reason = "not rendered: the cascade vs always-max figures are missing"
        return None
    if cut <= 0:
        bullet.reason = "not rendered: cascade did not cut cost per resolved issue"
        return None
    vals["cost_cut_pct"] = _pct_floor(cut)
    gap = -diff  # always_max minus cascade, in resolve-rate units
    if gap > 0:
        vals["rr_gap_pts"] = ceil_to(gap, 1, 100)  # a bigger gap is the unflattering direction
    elif gap < 0:  # cascade resolved MORE: the wording flips, and the figure is the favorable one
        vals["rr_gap_pts"] = floor_to(-gap, 1, 100)
        text = text.replace("lower resolve rate", "higher resolve rate")
    else:
        text = text.replace(
            "at a {rr_gap_pts}-point lower resolve rate", "at the same resolve rate"
        )
    bullet.notes.append(
        f"cascade verdict vs always-max: {stats['comparisons']['cascade']['verdict']}"
    )
    return text


def render_bullets(
    stats: dict[str, Any], templates: list[Template], *, allow_synthetic: bool = False
) -> list[Bullet]:
    if stats.get("data_source") != "real" and not allow_synthetic:
        raise BulletsError(
            f"the stats are from {stats.get('data_source')} data; resume bullets are rendered "
            "from real data only (--allow-synthetic is for tests)"
        )
    base = base_values(stats)
    out: list[Bullet] = []
    for t in templates:
        bullet = Bullet(t.name, t.kind, None, t.limit)
        out.append(bullet)
        vals: dict[str, str | None] = dict(base)
        text: str | None = t.text
        if t.name in ("router_supported", "router_neutral"):
            text = _router_values(t, stats, bullet, vals, t.text)
        elif t.name == "pilot" and int(stats["pilot"]["n_accepted"]) == 0:
            bullet.reason = (
                "not rendered: no pilot draft was accepted, so there is nothing to claim"
            )
            text = None
        if text is None:
            continue
        needed = [f for f in t.fields if "{" + f + "}" in text]
        missing = [f for f in needed if vals.get(f) is None]
        if missing:
            bullet.reason = f"not rendered: no value for {', '.join(missing)}"
            continue
        rendered = text.format(**{f: vals[f] for f in needed})
        if stats.get("data_source") != "real":
            rendered = "[SYNTHETIC] " + rendered
        bullet.text = rendered
        n = len(rendered)
        if n > t.limit:
            bullet.flags.append(f"OVER the {t.limit}-character limit by {n - t.limit}")
        elif n < t.limit - UNDER_SLACK:
            bullet.flags.append(
                f"UNDER the {t.limit}-character limit by {t.limit - n} (more than {UNDER_SLACK})"
            )
    return out


WORDING_CHECKS = """\
- Cascade drafts patches and PR text for a human; it never opens a PR. The pipeline bullet says
  "test-verified PRs" and the pilot bullet says "AI-drafted PRs accepted": only claim what
  matches how you submitted the accepted drafts (you opened those PRs yourself, with disclosure).
- "cost per resolved issue" in the router bullets is the point-estimate reduction in cost per
  resolved issue; the plan's headline (and its interval) is cost per task. Both are in
  `artifacts/resume_stats.json`.
- "equal resolve rate" means non-inferior within the plan's margin, not identical.
- Intervals are wide on a small test split; the README states them. A bullet carries no interval.
"""


def format_markdown(bullets: list[Bullet], stats: dict[str, Any]) -> str:
    banner = (
        ""
        if stats.get("data_source") == "real"
        else "> **SYNTHETIC DATA: nothing below is a result.**\n\n"
    )
    lines = [
        "# Resume bullets",
        "",
        banner + "Generated by `cascade resume-bullets` from `artifacts/resume_stats.json`. Do not "
        "edit by hand; rerun the command. Character counts include spaces. Limits: one-line "
        f"<= {ONE_LINE_MAX}, two-line <= {TWO_LINE_MAX}; a bullet outside its limit, or more than "
        f"{UNDER_SLACK} characters under it, is flagged and its wording is NOT changed.",
        "",
    ]
    for b in bullets:
        lines += [f"## {b.name} ({b.kind.replace('_', '-')})", ""]
        if b.text is None:
            lines.append(f"_{b.reason}_")
        else:
            lines += [b.text, "", f"**{b.chars} characters** (limit {b.limit})"]
            lines += [f"\n**FLAG: {flag}.**" for flag in b.flags]
            lines += [f"\nNote: {note}" for note in b.notes]
        lines.append("")
    lines += ["## Check the wording before you use these", "", WORDING_CHECKS]
    return "\n".join(lines).rstrip() + "\n"
