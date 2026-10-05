from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from cascade.config import (
    ArmSpec,
    BudgetConfig,
    ModelsConfig,
    load_budget,
    load_models,
    load_repos,
)
from cascade.errors import ConfigNotFilledError
from tests.conftest import ROOT, SYNTHETIC

CONFIGS = ROOT / "configs"


def _models_dict() -> dict[str, Any]:
    return {
        "models": [
            {
                "name": "m1",
                "provider": "mock",
                "input_usd_per_mtok": 1.0,
                "output_usd_per_mtok": 2.0,
                "knowledge_cutoff": "2024-01-01",
                "reasoning_efforts": ["low", "high"],
            }
        ],
        "arms": [{"model": "m1", "reasoning_effort": "low"}],
    }


def test_shipped_configs_load_and_report_todos() -> None:
    models = load_models(CONFIGS / "models.yaml")
    budget = load_budget(CONFIGS / "budget.yaml")
    repos = load_repos(CONFIGS / "repos.yaml")
    assert "models[0].input_usd_per_mtok" in models.unfilled()
    assert "models[0].input_usd_per_mtok" in models.unfilled()
    assert "global_cap_usd" in budget.unfilled()
    assert "date_window.start" in repos.unfilled()
    # click's repo-specific fields are filled; only the human-owned date window remains.
    assert [u for u in repos.unfilled() if u.startswith("repos[")] == []


def test_unfilled_prices_fail_closed() -> None:
    models = load_models(CONFIGS / "models.yaml")
    with pytest.raises(ConfigNotFilledError):
        models.prices_for(models.models[0].name)


def test_unfilled_budget_cannot_resolve() -> None:
    with pytest.raises(ConfigNotFilledError) as exc:
        load_budget(CONFIGS / "budget.yaml").resolve()
    assert "global_cap_usd" in exc.value.paths


def test_synthetic_prices_and_default_reasoning_price(models: ModelsConfig) -> None:
    small, large = models.models
    assert models.prices_for(small.name).reasoning_usd_per_mtok == small.output_usd_per_mtok
    assert models.prices_for(large.name).reasoning_usd_per_mtok == 50.0
    assert models.unfilled() == []


def test_arm_ids_are_unique_and_derived(models: ModelsConfig) -> None:
    ids = [a.arm_id for a in models.arms]
    assert len(set(ids)) == len(ids) == 3
    assert ArmSpec(model="x", reasoning_effort=None).arm_id == "x@default"


def test_arm_must_reference_known_model_and_effort() -> None:
    bad_model = _models_dict()
    bad_model["arms"] = [{"model": "nope", "reasoning_effort": "low"}]
    with pytest.raises(ValidationError, match="unknown model"):
        ModelsConfig.model_validate(bad_model)

    bad_effort = _models_dict()
    bad_effort["arms"] = [{"model": "m1", "reasoning_effort": "extreme"}]
    with pytest.raises(ValidationError, match="reasoning_efforts"):
        ModelsConfig.model_validate(bad_effort)

    missing_effort = _models_dict()
    missing_effort["arms"] = [{"model": "m1"}]
    with pytest.raises(ValidationError, match="must set a reasoning_effort"):
        ModelsConfig.model_validate(missing_effort)


def test_duplicate_arms_and_models_rejected() -> None:
    dup_arm = _models_dict()
    dup_arm["arms"] = dup_arm["arms"] * 2
    with pytest.raises(ValidationError, match="duplicate arm"):
        ModelsConfig.model_validate(dup_arm)

    dup_model = _models_dict()
    dup_model["models"] = dup_model["models"] * 2
    with pytest.raises(ValidationError, match="duplicate model"):
        ModelsConfig.model_validate(dup_model)


def test_unknown_fields_are_rejected() -> None:
    data = _models_dict()
    data["models"][0]["typo_field"] = 1
    with pytest.raises(ValidationError):
        ModelsConfig.model_validate(data)


def test_budget_ordering_enforced() -> None:
    cfg = BudgetConfig(
        global_cap_usd=1.0,
        per_task_cap_usd=2.0,  # larger than global
        per_attempt_cap_usd=0.1,
        per_attempt_max_tokens=1000,
    )
    with pytest.raises(ValidationError, match="per_attempt <= per_task <= global"):
        cfg.resolve()


def test_synthetic_budget_resolves() -> None:
    resolved = load_budget(SYNTHETIC / "budget.yaml").resolve()
    assert resolved.per_attempt_cap_usd <= resolved.per_task_cap_usd <= resolved.global_cap_usd


def test_negative_price_rejected() -> None:
    data = _models_dict()
    data["models"][0]["input_usd_per_mtok"] = -1.0
    with pytest.raises(ValidationError):
        ModelsConfig.model_validate(data)


def test_load_models_from_path_roundtrip(tmp_path: Path) -> None:
    import yaml

    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(_models_dict()))
    assert load_models(path).arms[0].model == "m1"


def test_repo_resolve_requires_window_and_named_repo() -> None:
    repos = load_repos(CONFIGS / "repos.yaml")
    with pytest.raises(ConfigNotFilledError) as exc:
        repos.resolve("pallets/click")  # date window is still TODO in the shipped config
    assert "date_window.start" in exc.value.paths
    with pytest.raises(ValueError, match="not listed"):
        repos.resolve("nobody/nothing")


def test_repo_resolve_returns_filled_parameters() -> None:
    from cascade.config import ReposConfig

    cfg = ReposConfig.model_validate(
        {
            "date_window": {"start": "2025-01-01", "end": "2025-03-31"},
            "repos": [
                {
                    "name": "o/r",
                    "license": "MIT",
                    "python_version": "3.12.1",
                    "install_cmd": "uv pip install -e .",
                    "test_cmd": "python -m pytest",
                }
            ],
        }
    )
    resolved = cfg.resolve("o/r")
    assert resolved.slug == "o__r"
    assert (resolved.window_start.isoformat(), resolved.window_end.isoformat()) == (
        "2025-01-01",
        "2025-03-31",
    )
    assert resolved.limits.max_source_files == 5


def test_repo_resolve_rejects_inverted_window() -> None:
    from cascade.config import ReposConfig

    cfg = ReposConfig.model_validate(
        {
            "date_window": {"start": "2025-06-01", "end": "2025-01-01"},
            "repos": [
                {
                    "name": "o/r", "license": "MIT", "python_version": "3.12",
                    "install_cmd": "x", "test_cmd": "y",
                }
            ],
        }
    )  # fmt: skip
    with pytest.raises(ValueError, match="on or before"):
        cfg.resolve("o/r")


def test_the_shipped_repos_all_load_and_the_license_override_reaches_the_resolved_repo() -> None:
    repos = load_repos(CONFIGS / "repos.yaml")
    names = [r.name for r in repos.repos]
    assert names == [
        "pallets/click", "pallets/jinja", "python-attrs/attrs", "hynek/structlog",
        "marshmallow-code/marshmallow",
    ]  # fmt: skip
    by_name = {r.name: r for r in repos.repos}
    assert by_name["hynek/structlog"].github_license_id == "NOASSERTION"
    assert all(r.github_license_id is None for n, r in by_name.items() if n != "hynek/structlog")
    assert all(r.python_version == "3.12.13" for r in repos.repos)  # one pinned interpreter
    assert [u for u in repos.unfilled() if u.startswith("repos[")] == []

    from datetime import date

    filled = repos.model_copy(
        update={
            "date_window": repos.date_window.model_copy(
                update={"start": date(2025, 1, 1), "end": date(2025, 6, 1)}
            )
        }
    )
    assert filled.resolve("hynek/structlog").github_license_id == "NOASSERTION"
    assert filled.resolve("pallets/click").github_license_id is None
