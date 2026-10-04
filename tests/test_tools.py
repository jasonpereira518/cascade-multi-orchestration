from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from agents import Agent, FunctionTool, RunConfig, Runner
from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call

from cascade.agents.tools import (
    ALL_TOOL_NAMES,
    MAX_READ_LINES,
    REPRO_PATH,
    Toolbox,
    ToolError,
)
from cascade.harness.providers.mock import sdk_usage
from cascade.harness.trace import TraceWriter
from cascade.taskgen.results import RunResult


class FakeExecutor:
    def __init__(self, result: RunResult | None = None) -> None:
        self.result = result or RunResult(1, 1.0, False, {"t.py::a": "failed", "t.py::b": "passed"},
                                          "", {"t.py::a": "assert 1 == 2"})  # fmt: skip
        self.calls: list[tuple[Path, list[str], float]] = []

    def run(self, *, workspace: Path, pytest_args: Sequence[str], timeout_s: float) -> RunResult:
        self.calls.append((workspace, list(pytest_args), timeout_s))
        return self.result


def build(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    build(
        root,
        {
            "pkg/mod.py": "\n".join(f"line_{i} = {i}" for i in range(1, 31)) + "\n",
            "pkg/other.py": "def helper():\n    return 'help'\n",
            "tests/test_mod.py": "def test_mod():\n    assert True\n",
            "README.md": "readme\n",
            ".env": "SECRET=1\n",
            ".git/config": "[core]\n",
            "pkg/.hidden.py": "x = 1\n",
        },
    )
    return root


@pytest.fixture
def box(ws: Path, tmp_path: Path) -> Toolbox:
    return Toolbox(
        ws, role="patcher", trace=TraceWriter(tmp_path / "trace.jsonl"), executor=FakeExecutor()
    )


# --- path safety ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../outside.py", "pkg/../../outside.py", "..\\outside.py", ".git/config",
     ".env", "pkg/.hidden.py", "C:\\Windows\\win.ini", "", "a\x00b.py", ".git", "pkg/./../../x"],
)  # fmt: skip
def test_unsafe_paths_are_refused_by_every_path_taking_tool(box: Toolbox, path: str) -> None:
    with pytest.raises(ToolError):
        box.read_file(path, 1, 5)
    with pytest.raises(ToolError):
        box.edit_file(path, "x", "y")
    with pytest.raises(ToolError):
        box.create_file(path, "x")
    with pytest.raises(ToolError):
        box.list_files(path)


def test_symlinks_that_escape_the_workspace_are_refused(
    box: Toolbox, ws: Path, tmp_path: Path
) -> None:
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("top secret\n")
    (ws / "link.py").symlink_to(secret)
    (ws / "linkdir").symlink_to(tmp_path)
    with pytest.raises(ToolError, match="outside"):
        box.read_file("link.py", 1, 5)
    with pytest.raises(ToolError, match="outside"):
        box.read_file("linkdir/outside-secret.txt", 1, 5)
    with pytest.raises(ToolError, match="outside"):
        box.edit_file("link.py", "top", "x")
    assert secret.read_text() == "top secret\n"  # untouched


def test_the_absolute_path_of_the_workspace_itself_is_refused(box: Toolbox, ws: Path) -> None:
    with pytest.raises(ToolError):
        box.read_file(str(ws / "pkg/mod.py"), 1, 3)


# --- read-only tools -----------------------------------------------------------------------


def test_read_file_numbers_lines_and_reports_totals(box: Toolbox) -> None:
    out = box.read_file("pkg/mod.py", 3, 5)
    assert out.splitlines()[0] == "pkg/mod.py (lines 3-5 of 30)"
    assert out.splitlines()[1:] == ["3: line_3 = 3", "4: line_4 = 4", "5: line_5 = 5"]


def test_read_file_end_zero_means_a_capped_window(tmp_path: Path) -> None:
    root = tmp_path / "w"
    build(root, {"big.py": "\n".join(f"v{i} = 1" for i in range(1000)) + "\n"})
    out = Toolbox(root, role="r", trace=TraceWriter(None)).read_file("big.py", 10, 0)
    assert f"lines 10-{10 + MAX_READ_LINES - 1} of 1000" in out.splitlines()[0]


def test_read_file_clamps_and_validates_ranges(box: Toolbox) -> None:
    assert "lines 1-30 of 30" in box.read_file("pkg/mod.py", -5, 999)
    with pytest.raises(ToolError, match="only 30 lines"):
        box.read_file("pkg/mod.py", 31, 40)
    with pytest.raises(ToolError, match="not a file"):
        box.read_file("pkg", 1, 5)
    with pytest.raises(ToolError, match="no such file"):
        box.read_file("pkg/missing.py", 1, 5)


def test_read_file_rejects_binary_and_truncates_long_output(box: Toolbox, ws: Path) -> None:
    (ws / "blob.bin").write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(ToolError, match="not a UTF-8"):
        box.read_file("blob.bin", 1, 5)
    (ws / "wide.py").write_text("\n".join("x" * 200 for _ in range(100)) + "\n")
    out = box.read_file("wide.py", 1, 100)
    assert len(out) <= 6000 and out.endswith("(output truncated)")


def test_list_files_hides_hidden_and_cache_entries_and_marks_directories(
    box: Toolbox, ws: Path
) -> None:
    (ws / "__pycache__").mkdir()
    out = box.list_files(".").splitlines()
    assert out == ["pkg/", "tests/", "README.md"]  # directories first, hidden and caches omitted
    assert box.list_files("pkg").splitlines() == ["mod.py", "other.py"]


def test_list_files_caps_entries_and_checks_the_target(box: Toolbox, ws: Path) -> None:
    for i in range(250):
        (ws / "pkg" / f"f{i:03}.py").write_text("")
    out = box.list_files("pkg")
    assert "... and" in out.splitlines()[-1]
    with pytest.raises(ToolError, match="not a directory"):
        box.list_files("pkg/mod.py")
    with pytest.raises(ToolError, match="no such"):
        box.list_files("nope")


def test_search_code_finds_ranks_and_clamps_top_k(box: Toolbox) -> None:
    out = box.search_code("helper return help", 3)
    assert out.startswith("pkg/other.py:") and "def helper" in out
    assert box.search_code("zzzzqqq", 3) == "no matches"
    assert box.search_code("line", 0).count("(score") == 1  # clamped up to 1
    assert box.search_code("line", 99).count("(score") <= 10  # clamped down to 10


def test_search_index_is_rebuilt_after_edits(box: Toolbox) -> None:
    assert box.search_code("zorblaxone", 3) == "no matches"
    box.create_file("pkg/new.py", "zorblaxone = 1\n")
    assert box.search_code("zorblaxone", 3).startswith("pkg/new.py")
    box.edit_file("pkg/new.py", "zorblaxone", "quuxifytwo")
    assert box.search_code("zorblaxone", 3) == "no matches"
    assert box.search_code("quuxifytwo", 3).startswith("pkg/new.py")


def test_grep(box: Toolbox) -> None:
    out = box.grep(r"def \w+", "*.py")
    assert "pkg/other.py:1: def helper():" in out and "tests/test_mod.py:1: def test_mod():" in out
    assert box.grep("nothing_like_this", "*") == "no matches"
    assert "pkg/other.py" in box.grep("helper", "")  # an empty glob means every file
    with pytest.raises(ToolError, match="invalid regular expression"):
        box.grep("([", "*")


# --- editing tools -------------------------------------------------------------------------


def test_edit_file_replaces_exactly_one_occurrence(box: Toolbox, ws: Path) -> None:
    msg = box.edit_file("pkg/other.py", "return 'help'", "return 'helped'")
    assert "edited pkg/other.py" in msg
    assert (ws / "pkg/other.py").read_text() == "def helper():\n    return 'helped'\n"


def test_edit_file_refuses_missing_ambiguous_and_empty_matches(box: Toolbox, ws: Path) -> None:
    before = (ws / "pkg/mod.py").read_text()
    with pytest.raises(ToolError, match="not found"):
        box.edit_file("pkg/mod.py", "no such text", "x")
    with pytest.raises(ToolError, match="matches 30 places"):
        box.edit_file("pkg/mod.py", " = ", "=")
    with pytest.raises(ToolError, match="must not be empty"):
        box.edit_file("pkg/mod.py", "", "x")
    with pytest.raises(ToolError, match="no such"):
        box.edit_file("pkg/absent.py", "a", "b")
    assert (ws / "pkg/mod.py").read_text() == before


def test_create_file(box: Toolbox, ws: Path) -> None:
    box.create_file("pkg/deep/er/new.py", "X = 1\n")
    assert (ws / "pkg/deep/er/new.py").read_text() == "X = 1\n"
    with pytest.raises(ToolError, match="already exists"):
        box.create_file("pkg/mod.py", "x")
    with pytest.raises(ToolError, match="too large"):
        box.create_file("pkg/huge.py", "x" * 70_000)


def test_write_reproducer_goes_to_the_fixed_path_and_needs_a_test(box: Toolbox, ws: Path) -> None:
    box.write_reproducer("def test_it():\n    assert False\n")
    assert (ws / REPRO_PATH).read_text() == "def test_it():\n    assert False\n"
    box.write_reproducer("def test_other():\n    assert 1 == 2\n")  # overwrites
    assert "test_other" in (ws / REPRO_PATH).read_text()
    with pytest.raises(ToolError, match="test_"):
        box.write_reproducer("x = 1\n")
    with pytest.raises(ToolError, match="too large"):
        box.write_reproducer("def test_x():\n    pass\n" + "#" * 70_000)


# --- running tests -------------------------------------------------------------------------


def test_run_tests_passes_normalized_paths_and_returns_a_summary(box: Toolbox, ws: Path) -> None:
    out = box.run_tests(["tests/test_mod.py", "./tests/test_mod.py::test_mod"][:1])
    executor = box._executor
    assert isinstance(executor, FakeExecutor)
    root, args, timeout = executor.calls[0]
    assert root == ws.resolve() and args == ["--tb=short", "tests/test_mod.py"] and timeout == 120.0
    assert "1 passed, 1 failed" in out and "FAILED t.py::a" in out and "assert 1 == 2" in out
    assert box.last_run is executor.result


def test_run_tests_validates_input_and_enforces_a_run_budget(ws: Path) -> None:
    executor = FakeExecutor()
    box = Toolbox(ws, role="r", trace=TraceWriter(None), executor=executor, max_test_runs=2)
    with pytest.raises(ToolError, match="at least one"):
        box.run_tests([])
    with pytest.raises(ToolError, match="no such"):
        box.run_tests(["tests/missing.py"])
    with pytest.raises(ToolError, match="relative"):
        box.run_tests(["/etc/passwd"])
    assert executor.calls == [] and box.test_runs == 0  # nothing ran for bad input
    box.run_tests(["tests/test_mod.py"])
    box.run_tests(["tests/test_mod.py"])
    with pytest.raises(ToolError, match="budget used up"):
        box.run_tests(["tests/test_mod.py"])
    assert len(executor.calls) == 2


def test_run_tests_is_unavailable_without_an_executor(ws: Path) -> None:
    box = Toolbox(ws, role="reviewer", trace=TraceWriter(None))
    with pytest.raises(ToolError, match="not available"):
        box.run_tests(["tests/test_mod.py"])


# --- tracing -------------------------------------------------------------------------------


def test_every_tool_call_is_traced_with_role_and_arguments(box: Toolbox, tmp_path: Path) -> None:
    box.read_file("pkg/other.py", 1, 2)
    box.grep("helper", "*")
    lines = [json.loads(ln) for ln in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [(ln["kind"], ln["role"], ln["tool"]) for ln in lines] == [
        ("tool", "patcher", "read_file"),
        ("tool", "patcher", "grep"),
    ]
    assert lines[0]["args"] == {"path": "pkg/other.py", "start": 1, "end": 2}


# --- SDK wrappers --------------------------------------------------------------------------


def test_sdk_tools_have_strict_schemas_with_every_parameter_required(box: Toolbox) -> None:
    tools = box.tools(ALL_TOOL_NAMES)
    assert [t.name for t in tools if isinstance(t, FunctionTool)] == list(ALL_TOOL_NAMES)
    for tool in tools:
        assert isinstance(tool, FunctionTool) and tool.strict_json_schema
        schema = tool.params_json_schema
        assert sorted(schema["required"]) == sorted(schema["properties"])  # no optional params
        assert schema.get("additionalProperties") is False


def test_unknown_tool_names_are_rejected(box: Toolbox) -> None:
    with pytest.raises(ValueError, match="unknown tools"):
        box.tools(["read_file", "format_disk"])


def drive(box: Toolbox, calls: list[tuple[str, dict[str, Any]]]) -> ScriptedModel:
    steps = [
        ModelStep(output=[function_call(n, a, call_id=f"c{i}")], usage=sdk_usage(10, 5))
        for i, (n, a) in enumerate(calls)
    ]
    steps.append(ModelStep(output=[assistant_message("done")], usage=sdk_usage(10, 5)))
    model = ScriptedModel(steps)
    agent = Agent(name="t", instructions="x", tools=box.tools(ALL_TOOL_NAMES), model=model)
    Runner.run_sync(agent, "go", run_config=RunConfig(tracing_disabled=True))
    return model


def tool_outputs(model: ScriptedModel) -> list[str]:
    last = model.calls[-1].input
    return [
        str(i["output"])
        for i in last
        if isinstance(i, dict) and i.get("type") == "function_call_output"
    ]


def test_tool_mistakes_come_back_to_the_model_as_text(box: Toolbox) -> None:
    model = drive(
        box,
        [
            ("read_file", {"path": "../etc/passwd", "start_line": 1, "end_line": 5}),
            ("read_file", {"path": "pkg/other.py", "start_line": 1, "end_line": 2}),
            ("edit_file", {"path": "pkg/mod.py", "old_text": " = ", "new_text": "="}),
        ],
    )
    out = tool_outputs(model)
    assert out[0] == "Error: path may not contain '..'"
    assert out[1].startswith("pkg/other.py (lines 1-2 of 2)")
    assert out[2].startswith("Error: old_text matches 30 places")


def test_genuine_bugs_in_a_tool_propagate_instead_of_being_swallowed(box: Toolbox) -> None:
    def boom(*_a: object) -> str:
        raise RuntimeError("bug inside the tool")

    box.read_file = boom  # type: ignore[method-assign]
    with pytest.raises(Exception, match="bug inside the tool"):
        drive(box, [("read_file", {"path": "pkg/other.py", "start_line": 1, "end_line": 2})])
