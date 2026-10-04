from __future__ import annotations

import shutil
from pathlib import Path

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
    assert "models.yaml: models[0].name" in result.output
    assert "budget.yaml: global_cap_usd" in result.output
    assert "repos.yaml: date_window.start" in result.output
    assert "unfilled TODO field(s)" in result.output


def test_config_status_passes_when_everything_is_filled(tmp_path: Path) -> None:
    shutil.copy(SYNTHETIC / "models.yaml", tmp_path / "models.yaml")
    shutil.copy(SYNTHETIC / "budget.yaml", tmp_path / "budget.yaml")
    (tmp_path / "repos.yaml").write_text(_FILLED_REPOS)
    result = runner.invoke(app, ["config-status", "--configs-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "All config fields are filled." in result.output
