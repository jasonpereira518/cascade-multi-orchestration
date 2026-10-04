"""Shared fixtures. All data used here is SYNTHETIC (see tests/fixtures/README.md)."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from cascade.config import ModelsConfig, ResolvedBudget, load_budget, load_models
from cascade.harness.ledger import Ledger
from cascade.taskgen.schema import Task

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
SYNTHETIC = FIXTURES / "synthetic"
TINY_REPO = FIXTURES / "tiny_repo"

_GIT_ID = ["-c", "user.name=fixture", "-c", "user.email=fixture@example.com"]


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *_GIT_ID, "-c", "commit.gpgsign=false", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must never inherit the developer's spend gate or provider choice."""
    monkeypatch.delenv("CASCADE_ALLOW_SPEND", raising=False)
    monkeypatch.delenv("CASCADE_PROVIDER", raising=False)


@pytest.fixture
def models() -> ModelsConfig:
    return load_models(SYNTHETIC / "models.yaml")


@pytest.fixture
def budget(tmp_path: Path) -> ResolvedBudget:
    resolved = load_budget(SYNTHETIC / "budget.yaml").resolve()
    return resolved.model_copy(update={"kill_switch_path": tmp_path / "KILL_SWITCH"})


@pytest.fixture
def ledger(tmp_path: Path, budget: ResolvedBudget, models: ModelsConfig) -> Iterator[Ledger]:
    ledger = Ledger(tmp_path / "ledger.sqlite", budget=budget, models=models)
    yield ledger
    ledger.close()


@dataclass(frozen=True)
class FixtureRepo:
    path: Path
    base_sha: str
    head_sha: str  # a later commit that contains the gold and test patches
    task: Task  # the synthetic task with base_commit set to base_sha


@pytest.fixture
def fixture_repo(tmp_path: Path) -> FixtureRepo:
    repo = tmp_path / "upstream"
    shutil.copytree(TINY_REPO, repo, ignore=shutil.ignore_patterns("__pycache__"))
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    base_sha = git(repo, "rev-parse", "HEAD")

    git(repo, "apply", str(SYNTHETIC / "gold.patch"))
    git(repo, "apply", str(SYNTHETIC / "test.patch"))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "fix clamp (gold + test)")
    head_sha = git(repo, "rev-parse", "HEAD")

    task = Task.model_validate_json((SYNTHETIC / "task.json").read_text())
    task = task.model_copy(update={"base_commit": base_sha})
    return FixtureRepo(repo, base_sha, head_sha, task)


@pytest.fixture
def task(fixture_repo: FixtureRepo) -> Task:
    return fixture_repo.task
