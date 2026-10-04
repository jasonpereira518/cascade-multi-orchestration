"""The deterministic tools the roles call, bound to one workspace.

Safety properties, each covered by tests:
- Paths are relative to the workspace root. Absolute paths, ``..``, dot-files and dot-directories
  (so ``.git`` and Cascade's own scratch files), and symlinks that resolve outside the root are all
  refused. Together with the workspace being a base-commit export, an agent has no route to the
  gold patch, hidden tests, or later history.
- Every tool takes all its parameters (no defaults): the SDK's strict schema marks every parameter
  required anyway, and explicit is better for models.
- Output is bounded (line caps, character caps), which is what keeps prompts small.
- Mistakes the model can fix (bad path, ambiguous edit) raise ToolError and come back to it as the
  tool's text result. Anything else is a bug or infrastructure failure and propagates.

Editing and searching happen on the host: they do not execute code. Running tests goes through the
injected executor, which in real runs is the network-off Docker sandbox.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from agents import RunContextWrapper, function_tool
from agents.tool import Tool

from cascade.agents.failures import summarize_run
from cascade.agents.search import SKIP_DIRS, Bm25Index, grep, is_hidden
from cascade.errors import CascadeError
from cascade.harness.trace import TraceWriter
from cascade.taskgen.results import RunResult
from cascade.taskgen.validate import TestExecutor

MAX_READ_LINES = 400
MAX_OUTPUT_CHARS = 6000
MAX_LIST_ENTRIES = 200
MAX_WRITE_CHARS = 60_000
REPRO_PATH = "tests/test_cascade_repro.py"

ALL_TOOL_NAMES = (
    "list_files", "read_file", "search_code", "grep", "edit_file", "create_file",
    "write_reproducer", "run_tests",
)  # fmt: skip


class ToolError(CascadeError):
    """A mistake the model can correct; reported back to it as the tool result."""


def _failure(_ctx: RunContextWrapper[Any], error: Exception) -> str:
    if isinstance(error, ToolError):
        return f"Error: {error}"
    raise error


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[: MAX_OUTPUT_CHARS - 40].rstrip() + "\n... (output truncated)"


class Toolbox:
    def __init__(
        self,
        root: Path,
        *,
        role: str,
        trace: TraceWriter,
        executor: TestExecutor | None = None,
        test_timeout_s: float = 120.0,
        max_test_runs: int = 6,
        repro_path: str = REPRO_PATH,
    ) -> None:
        self.root = root.resolve()
        self.role = role
        self._trace = trace
        self._executor = executor
        self._timeout = test_timeout_s
        self._max_runs = max_test_runs
        self._repro_path = repro_path
        self._index: Bm25Index | None = None
        self.test_runs = 0
        self.last_run: RunResult | None = None

    # --- path handling ---------------------------------------------------------------------

    def resolve(self, rel: str, *, must_exist: bool = True) -> Path:
        if not rel or "\x00" in rel or rel.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", rel):
            raise ToolError("path must be relative to the repository root")
        posix = PurePosixPath(rel.replace("\\", "/"))
        if ".." in posix.parts:
            raise ToolError("path may not contain '..'")
        if any(is_hidden(part) and part != "." for part in posix.parts):
            raise ToolError("hidden files and directories are not accessible")
        full = (self.root / posix).resolve()
        if full != self.root and not full.is_relative_to(self.root):
            raise ToolError("path resolves outside the repository")
        if must_exist and not full.exists():
            raise ToolError(f"no such file or directory: {rel}")
        return full

    def _log(self, tool: str, args: dict[str, Any], result: str) -> str:
        self._trace.write("tool", role=self.role, tool=tool, args=args, result=result[:2000])
        return result

    def _read_text(self, path: Path, rel: str) -> str:
        if not path.is_file():
            raise ToolError(f"not a file: {rel}")
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ToolError(f"{rel} is not a UTF-8 text file") from e

    # --- read-only tools -------------------------------------------------------------------

    def list_files(self, directory: str) -> str:
        target = self.resolve(directory)
        if not target.is_dir():
            raise ToolError(f"not a directory: {directory}")
        entries: list[str] = []
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
            if is_hidden(child.name) or child.name in SKIP_DIRS:
                continue
            entries.append(child.name + ("/" if child.is_dir() else ""))
        shown = entries[:MAX_LIST_ENTRIES]
        extra = f"\n... and {len(entries) - len(shown)} more" if len(entries) > len(shown) else ""
        return self._log(
            "list_files", {"directory": directory}, "\n".join(shown) + extra or "(empty)"
        )

    def read_file(self, path: str, start_line: int, end_line: int) -> str:
        full = self.resolve(path)
        lines = self._read_text(full, path).splitlines()
        total = len(lines)
        start = max(start_line, 1)
        end = end_line if end_line >= start else start + MAX_READ_LINES - 1
        end = min(end, start + MAX_READ_LINES - 1, total)
        if start > total:
            raise ToolError(f"{path} has only {total} lines")
        body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
        result = _clip(f"{path} (lines {start}-{end} of {total})\n{body}")
        return self._log("read_file", {"path": path, "start": start_line, "end": end_line}, result)

    def search_code(self, query: str, top_k: int) -> str:
        if self._index is None:
            self._index = Bm25Index(self.root)
        hits = self._index.search(query, max(1, min(top_k, 10)))
        if not hits:
            return self._log("search_code", {"query": query}, "no matches")
        blocks = [f"{h.path}:{h.start}-{h.end} (score {h.score})\n{h.snippet}" for h in hits]
        return self._log(
            "search_code", {"query": query, "top_k": top_k}, _clip("\n\n".join(blocks))
        )

    def grep(self, pattern: str, path_glob: str) -> str:
        try:
            matches = grep(self.root, pattern, path_glob.strip() or "*")
        except re.error as e:
            raise ToolError(f"invalid regular expression: {e}") from e
        result = "\n".join(matches) if matches else "no matches"
        return self._log("grep", {"pattern": pattern, "glob": path_glob}, _clip(result))

    # --- editing tools ---------------------------------------------------------------------

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        if not old_text:
            raise ToolError("old_text must not be empty; use create_file for new files")
        full = self.resolve(path)
        content = self._read_text(full, path)
        count = content.count(old_text)
        if count == 0:
            raise ToolError("old_text was not found; read the file again and copy it exactly")
        if count > 1:
            raise ToolError(f"old_text matches {count} places; include more surrounding lines")
        full.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        self._index = None
        added, removed = new_text.count("\n") + 1, old_text.count("\n") + 1
        return self._log("edit_file", {"path": path}, f"edited {path} (-{removed} +{added} lines)")

    def create_file(self, path: str, content: str) -> str:
        if len(content) > MAX_WRITE_CHARS:
            raise ToolError("content is too large")
        full = self.resolve(path, must_exist=False)
        if full.exists():
            raise ToolError(f"{path} already exists; use edit_file")
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        self._index = None
        return self._log("create_file", {"path": path}, f"created {path}")

    def write_reproducer(self, code: str) -> str:
        if len(code) > MAX_WRITE_CHARS:
            raise ToolError("code is too large")
        if "def test_" not in code:
            raise ToolError("the reproducer must define at least one function named test_...")
        full = self.resolve(self._repro_path, must_exist=False)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(code, encoding="utf-8")
        self._index = None
        return self._log("write_reproducer", {}, f"wrote the reproducer to {self._repro_path}")

    # --- test execution --------------------------------------------------------------------

    def run_tests(self, test_paths: list[str]) -> str:
        if self._executor is None:
            raise ToolError("running tests is not available in this role")
        if self.test_runs >= self._max_runs:
            raise ToolError(f"test-run budget used up ({self._max_runs} runs)")
        paths = [self.resolve(p).relative_to(self.root).as_posix() for p in test_paths]
        if not paths:
            raise ToolError("give at least one test file or test id to run")
        self.test_runs += 1
        run = self._executor.run(
            workspace=self.root, pytest_args=["--tb=short", *paths], timeout_s=self._timeout
        )
        self.last_run = run
        return self._log("run_tests", {"paths": paths}, summarize_run(run))

    # --- SDK wrappers ----------------------------------------------------------------------

    def tools(self, names: Sequence[str]) -> list[Tool]:
        box = self

        @function_tool(failure_error_function=_failure)
        def list_files(directory: str) -> str:
            """List the files and folders in a directory.

            Args:
                directory: Directory relative to the repository root. Use "." for the root.
            """
            return box.list_files(directory)

        @function_tool(failure_error_function=_failure)
        def read_file(path: str, start_line: int, end_line: int) -> str:
            """Read part of a file, with line numbers (at most 400 lines per call).

            Args:
                path: File path relative to the repository root.
                start_line: First line to read, 1-based.
                end_line: Last line to read, inclusive. Use 0 to read 400 lines from start_line.
            """
            return box.read_file(path, start_line, end_line)

        @function_tool(failure_error_function=_failure)
        def search_code(query: str, top_k: int) -> str:
            """Find the most relevant code snippets for a query (BM25 ranking over the repo).

            Args:
                query: Identifiers or words likely to appear in the code, e.g. "parse option flag".
                top_k: How many snippets to return, from 1 to 10.
            """
            return box.search_code(query, top_k)

        @function_tool(failure_error_function=_failure)
        def grep(pattern: str, path_glob: str) -> str:
            """Search for a regular expression in text files, case-sensitive.

            Args:
                pattern: A Python regular expression.
                path_glob: Only search files whose path matches this glob. Use "*" for all files.
            """
            return box.grep(pattern, path_glob)

        @function_tool(failure_error_function=_failure)
        def edit_file(path: str, old_text: str, new_text: str) -> str:
            """Replace one exact occurrence of text in a file.

            Args:
                path: File path relative to the repository root.
                old_text: Exact existing text to replace. It must match exactly one place.
                new_text: The replacement text.
            """
            return box.edit_file(path, old_text, new_text)

        @function_tool(failure_error_function=_failure)
        def create_file(path: str, content: str) -> str:
            """Create a new file. Fails if the file already exists.

            Args:
                path: File path relative to the repository root.
                content: The full file content.
            """
            return box.create_file(path, content)

        @function_tool(failure_error_function=_failure)
        def write_reproducer(code: str) -> str:
            """Write (or overwrite) the single pytest file that reproduces the issue.

            Args:
                code: Complete Python source of the test file. It must define test_ functions.
            """
            return box.write_reproducer(code)

        @function_tool(failure_error_function=_failure)
        def run_tests(test_paths: list[str]) -> str:
            """Run pytest on the given test files or node ids and summarize the result.

            Args:
                test_paths: Test file paths or pytest node ids relative to the repository root.
            """
            return box.run_tests(test_paths)

        registry: dict[str, Tool] = {
            "list_files": list_files, "read_file": read_file, "search_code": search_code,
            "grep": grep, "edit_file": edit_file, "create_file": create_file,
            "write_reproducer": write_reproducer, "run_tests": run_tests,
        }  # fmt: skip
        unknown = [n for n in names if n not in registry]
        if unknown:
            raise ValueError(f"unknown tools: {unknown}")
        return [registry[n] for n in names]
