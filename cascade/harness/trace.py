"""Local JSONL traces. Every model call and tool call of an attempt is appended here.

These are the project's own traces (CLAUDE.md: "write local JSONL traces too"). They are also what
the leak audit reads afterwards, so they store full prompts. The Agents SDK's built-in tracing is
disabled everywhere because it uploads prompts to a third-party backend.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **fields: Any) -> None:
        if self.path is None:
            return
        row = {"ts": dt.datetime.now(dt.UTC).isoformat(), "kind": kind, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")

    def text(self) -> str:
        """The whole trace as text (empty if nothing was written)."""
        if self.path is None or not self.path.exists():
            return ""
        return self.path.read_text(encoding="utf-8")
