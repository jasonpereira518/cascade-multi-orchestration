from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cascade.cli import app
from tests.conftest import ROOT, SYNTHETIC

runner = CliRunner()

_FILLED_REPOS = """\
date_window: {start: 2025-01-01, end: 2025-06-30}
repos:
  - name: example/example
    license: MIT
    python_version: "3.11"
    install_cmd: pip install -e .
    test_cmd: pytest -x
"""


def test_config_status_lists_todos_in_shipped_configs() -> None:
    result = runner.invoke(app, ["config-status", "--configs-dir", str(ROOT / "configs")])
    assert result.exit_code == 1
    assert "models.yaml: models[0].input_usd_per_mtok" in result.output
    assert "budget.yaml: global_cap_usd" in result.output
    assert "unfilled TODO field(s)" in result.output


def test_config_status_passes_when_everything_is_filled(tmp_path: Path) -> None:
    shutil.copy(SYNTHETIC / "models.yaml", tmp_path / "models.yaml")
    shutil.copy(SYNTHETIC / "budget.yaml", tmp_path / "budget.yaml")
    (tmp_path / "repos.yaml").write_text(_FILLED_REPOS)
    result = runner.invoke(app, ["config-status", "--configs-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "All config fields are filled." in result.output


_TAGGED_REPOS = """\
date_window: {start: 2025-01-01, end: 2025-03-31}
repos:
  - name: o/r
    license: MIT
    python_version: "3.12"
    install_cmd: x
    test_cmd: y
"""


def test_taskgen_run_refuses_unfilled_date_window_with_a_clear_message(tmp_path: Path) -> None:
    text = (ROOT / "configs" / "repos.yaml").read_text()
    text = text.replace("start: 2026-06-17", "start: TODO").replace(
        "end: 2026-09-20", "end: TODO", 1
    )
    (tmp_path / "repos.yaml").write_text(text)
    result = runner.invoke(
        app, ["taskgen", "run", "--repo", "pallets/click", "--configs-dir", str(tmp_path)]
    )
    assert result.exit_code == 2
    assert "date_window.start" in result.output and "unfilled TODO" in result.output


def test_taskgen_run_requires_a_token_before_touching_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    (tmp_path / "repos.yaml").write_text(_TAGGED_REPOS)
    result = runner.invoke(
        app,
        ["taskgen", "run", "--repo", "o/r", "--configs-dir", str(tmp_path),
         "--runs-dir", str(tmp_path / "runs"), "--status-md", str(tmp_path / "S.md")],
    )  # fmt: skip
    assert result.exit_code == 2 and "GITHUB_TOKEN" in result.output


def test_taskgen_attrition_needs_existing_state(tmp_path: Path) -> None:
    (tmp_path / "repos.yaml").write_text(_TAGGED_REPOS)
    result = runner.invoke(
        app, ["taskgen", "attrition", "--repo", "o/r", "--configs-dir", str(tmp_path),
              "--runs-dir", str(tmp_path / "runs")]
    )  # fmt: skip
    assert result.exit_code == 2 and "no state" in result.output


def test_taskgen_attrition_prints_and_updates_status(tmp_path: Path) -> None:
    from cascade.taskgen.store import OK, StageStore

    (tmp_path / "repos.yaml").write_text(_TAGGED_REPOS)
    store = StageStore(tmp_path / "runs" / "o__r" / "w-2025-01-01_2025-03-31" / "state.sqlite")
    store.upsert_candidate(1, {"n": 1})
    store.put_stage(1, "filter", OK)
    store.close()
    status = tmp_path / "STATUS.md"
    result = runner.invoke(
        app, ["taskgen", "attrition", "--repo", "o/r", "--configs-dir", str(tmp_path),
              "--runs-dir", str(tmp_path / "runs"), "--status-md", str(status)]
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "1 candidates -> 0 after filters" in result.output
    assert "1 candidates -> 0 after filters" in status.read_text()


def test_relaxed_window_flag_needs_the_relaxed_window_in_the_config(tmp_path: Path) -> None:
    text = (ROOT / "configs" / "repos.yaml").read_text()
    head = text.split("# Used ONLY with")[0]  # drop relaxed_date_window
    tail = text[text.index("repos:\n") :]
    (tmp_path / "repos.yaml").write_text(head + tail)
    result = runner.invoke(
        app,
        ["taskgen", "report", "--relaxed-window", "--configs-dir", str(tmp_path),
         "--manifest", str(tmp_path / "m.jsonl"), "--runs-dir", str(tmp_path / "runs")],
    )  # fmt: skip
    assert result.exit_code == 2 and "needs relaxed_date_window" in result.output


def test_report_names_the_window_and_the_relaxed_flag_switches_it(tmp_path: Path) -> None:
    common = ["--configs-dir", str(ROOT / "configs"), "--manifest", str(tmp_path / "m.jsonl"),
              "--runs-dir", str(tmp_path / "runs")]  # fmt: skip
    std = runner.invoke(app, ["taskgen", "report", *common])
    rel = runner.invoke(app, ["taskgen", "report", "--relaxed-window", *common])
    assert std.exit_code == 0 and rel.exit_code == 0
    assert "window: 2026-06-17 to 2026-09-20 (standard)" in std.output
    assert "window: 2026-05-19 to 2026-09-20 (relaxed)" in rel.output
