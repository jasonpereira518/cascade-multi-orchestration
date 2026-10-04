"""Deterministic code search: BM25 over line-window chunks, plus a regex grep. No LLM, no network.

Why this exists: a model that can ask "where is X handled?" and get a handful of ranked snippets
back needs far fewer prompt tokens than one that reads whole files. Output is deterministic (ties
break by path and line) so runs are reproducible.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = frozenset(
    {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".venv", "venv",
     "node_modules", "build", "dist", ".eggs"}
)  # fmt: skip
TEXT_SUFFIXES = frozenset(
    {".py", ".pyi", ".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json"}
)
MAX_FILE_BYTES = 300_000
WINDOW = 40
STRIDE = 20
K1 = 1.5
B = 0.75

_WORD = re.compile(r"[A-Za-z0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercased tokens. Identifiers contribute both whole and split (camelCase, snake_case)."""
    tokens: list[str] = []
    for word in _WORD.findall(text):
        lowered = word.lower()
        parts = [p.lower() for p in _CAMEL.findall(word)]
        tokens.append(lowered)
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


def is_hidden(name: str) -> bool:
    """Dot-files and Cascade's own scratch files are never indexed or shown to agents."""
    return name.startswith(".")


def iter_text_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not is_hidden(d))
        for name in sorted(filenames):
            if is_hidden(name):
                continue
            path = Path(dirpath) / name
            if path.suffix not in TEXT_SUFFIXES or path.is_symlink():
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            yield path


@dataclass(frozen=True)
class Hit:
    path: str
    start: int  # 1-based, inclusive
    end: int
    score: float
    snippet: str


@dataclass(frozen=True)
class _Chunk:
    path: str
    start: int
    end: int
    lines: tuple[str, ...]
    counts: Counter[str]
    length: int


class Bm25Index:
    def __init__(self, root: Path) -> None:
        self._chunks: list[_Chunk] = []
        for path in iter_text_files(root):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            rel = path.relative_to(root).as_posix()
            for start in range(0, max(len(lines), 1), STRIDE):
                window = lines[start : start + WINDOW]
                if not window:
                    break
                tokens = tokenize("\n".join(window) + "\n" + rel)
                self._chunks.append(
                    _Chunk(
                        rel,
                        start + 1,
                        start + len(window),
                        tuple(window),
                        Counter(tokens),
                        len(tokens),
                    )
                )
                if start + WINDOW >= len(lines):
                    break
        n = len(self._chunks)
        self._avg_len = (sum(c.length for c in self._chunks) / n) if n else 1.0
        df: Counter[str] = Counter()
        for chunk in self._chunks:
            df.update(chunk.counts.keys())
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, top_k: int, *, max_per_file: int = 2) -> list[Hit]:
        terms = sorted(set(tokenize(query)))
        scored: list[tuple[float, _Chunk]] = []
        for chunk in self._chunks:
            score = 0.0
            for term in terms:
                freq = chunk.counts.get(term)
                if not freq:
                    continue
                norm = freq * (K1 + 1) / (freq + K1 * (1 - B + B * chunk.length / self._avg_len))
                score += self._idf.get(term, 0.0) * norm
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda sc: (-sc[0], sc[1].path, sc[1].start))
        hits: list[Hit] = []
        per_file: Counter[str] = Counter()
        for score, chunk in scored:
            if per_file[chunk.path] >= max_per_file:
                continue
            per_file[chunk.path] += 1
            snippet = "\n".join(
                f"{chunk.start + i}: {line}" for i, line in enumerate(chunk.lines[:12])
            )
            hits.append(Hit(chunk.path, chunk.start, chunk.end, round(score, 3), snippet))
            if len(hits) >= top_k:
                break
        return hits


def grep(root: Path, pattern: str, path_glob: str, *, max_matches: int = 60) -> list[str]:
    """Regex search, case-sensitive, over indexable text files matching ``path_glob``."""
    regex = re.compile(pattern)
    out: list[str] = []
    for path in iter_text_files(root):
        rel = path.relative_to(root).as_posix()
        if not Path(rel).match(path_glob or "*"):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(lines, start=1):
            if regex.search(line):
                out.append(f"{rel}:{number}: {line.strip()[:200]}")
                if len(out) >= max_matches:
                    return out
    return out
