"""A dependency-free, deterministic SVG Pareto plot: cost per task (x) against resolve rate (y).

Hand-written SVG instead of a plotting library keeps the output byte-for-byte reproducible and adds
no dependency. Each policy is a point with 95% bootstrap whiskers; cross-validated sweep points of
the predictive router are drawn small and faint; the provenance stamp (data source, commit, matrix
hash) is printed on the figure itself so a screenshot cannot lose it.
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.sax.saxutils import escape

W, H = 760, 520
LEFT, RIGHT, TOP, BOTTOM = 70, 30, 60, 90
_COLORS = {"fixed": "#555555", "cascade": "#1f77b4", "oracle": "#2ca02c", "predictive": "#d62728"}


@dataclass(frozen=True)
class Point:
    label: str
    kind: str  # fixed | cascade | oracle | predictive | sweep
    x: float  # cost per task
    y: float  # resolve rate
    x_lo: float | None = None
    x_hi: float | None = None
    y_lo: float | None = None
    y_hi: float | None = None


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


def pareto_svg(points: list[Point], *, title: str, stamp: str) -> str:
    xs = [v for p in points for v in (p.x, p.x_lo, p.x_hi) if v is not None]
    xmax = max(xs) * 1.08 if xs else 1.0
    pw, ph = W - LEFT - RIGHT, H - TOP - BOTTOM

    def sx(v: float) -> float:
        return LEFT + (v / xmax) * pw

    def sy(v: float) -> float:
        return TOP + (1.0 - v) * ph

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        'font-family="sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{LEFT}" y="24" font-size="15" font-weight="bold">{escape(title)}</text>',
        f'<rect x="{LEFT}" y="{TOP}" width="{pw}" height="{ph}" fill="none" stroke="#999"/>',
    ]
    for t in _ticks(0.0, 1.0):
        y = sy(t)
        out.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + pw}" y2="{y:.1f}" stroke="#eee"/>')
        out.append(f'<text x="{LEFT - 8}" y="{y + 4:.1f}" text-anchor="end">{t:.0%}</text>')
    for t in _ticks(0.0, xmax):
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{TOP + ph}" stroke="#eee"/>')
        out.append(f'<text x="{x:.1f}" y="{TOP + ph + 16}" text-anchor="middle">${t:.3f}</text>')
    out.append(
        f'<text x="{LEFT + pw / 2:.0f}" y="{TOP + ph + 36}" text-anchor="middle">'
        "mean cost per task (USD)</text>"
    )
    mid_y = TOP + ph / 2
    out.append(
        f'<text x="16" y="{mid_y:.0f}" text-anchor="middle" '
        f'transform="rotate(-90 16 {mid_y:.0f})">resolve rate</text>'
    )
    for p in sorted(points, key=lambda q: (q.kind != "sweep", q.label)):
        color = _COLORS.get(p.kind, "#bbbbbb")
        cx, cy = sx(p.x), sy(p.y)
        if p.kind == "sweep":
            out.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.5" fill="{color}" opacity="0.6"/>'
            )
            continue
        if p.y_lo is not None and p.y_hi is not None:
            out.append(
                f'<line x1="{cx:.1f}" y1="{sy(p.y_lo):.1f}" x2="{cx:.1f}" '
                f'y2="{sy(p.y_hi):.1f}" stroke="{color}"/>'
            )
        if p.x_lo is not None and p.x_hi is not None:
            out.append(
                f'<line x1="{sx(p.x_lo):.1f}" y1="{cy:.1f}" x2="{sx(p.x_hi):.1f}" '
                f'y2="{cy:.1f}" stroke="{color}"/>'
            )
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}"/>')
        out.append(
            f'<text x="{cx + 8:.1f}" y="{cy - 8:.1f}" fill="{color}">{escape(p.label)}</text>'
        )
    out.append(f'<text x="{LEFT}" y="{H - 24}" font-size="10" fill="#444">{escape(stamp)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
