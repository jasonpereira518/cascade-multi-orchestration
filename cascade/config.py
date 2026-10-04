"""Config schemas for configs/{models,budget,repos}.yaml.

Fields the human must supply hold the literal string ``TODO`` until filled. Configs load with
TODOs present (so ``cascade config-status`` can list them), but anything that needs a real value
(prices, caps) calls a resolver that raises ConfigNotFilledError. Nothing is defaulted silently.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cascade.errors import ConfigNotFilledError

TODO_SENTINEL = "TODO"
Todo = Literal["TODO"]
Price = Annotated[float, Field(ge=0)]
Cap = Annotated[float, Field(ge=0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _is_todo(text: str) -> bool:
    # "TODO" for typed fields; "TODO-<hint>" where names must stay distinct (e.g. TODO-small).
    return text == TODO_SENTINEL or text.startswith(f"{TODO_SENTINEL}-")


def find_todos(value: Any, path: str = "") -> list[str]:
    """Return dotted paths of every field still set to a TODO placeholder."""
    if isinstance(value, str):
        return [path] if _is_todo(value) else []
    if isinstance(value, BaseModel):
        found: list[str] = []
        for name in type(value).model_fields:
            found += find_todos(getattr(value, name), f"{path}.{name}" if path else name)
        return found
    if isinstance(value, (list, tuple)):
        items: list[Any] = list(value)  # pyright: ignore[reportUnknownArgumentType]
        found = []
        for i, item in enumerate(items):
            found += find_todos(item, f"{path}[{i}]")
        return found
    return []


# --- models.yaml -------------------------------------------------------------------------------


class Prices(_Strict):
    """Resolved USD prices per million tokens."""

    input_usd_per_mtok: Price
    output_usd_per_mtok: Price
    reasoning_usd_per_mtok: Price


class ModelSpec(_Strict):
    name: str
    provider: Literal["mock", "openai"]
    input_usd_per_mtok: Price | Todo
    output_usd_per_mtok: Price | Todo
    # None means reasoning tokens are billed at the output price.
    reasoning_usd_per_mtok: Price | None = None
    knowledge_cutoff: dt.date | Todo
    # Valid reasoning-effort values for this model. Empty means the model has no effort setting.
    reasoning_efforts: list[str] = Field(default_factory=list)

    def prices(self) -> Prices:
        input_price, output_price = self.input_usd_per_mtok, self.output_usd_per_mtok
        if isinstance(input_price, str) or isinstance(output_price, str):
            raise ConfigNotFilledError(find_todos(self, f"models[{self.name}]"))
        reasoning = self.reasoning_usd_per_mtok
        return Prices(
            input_usd_per_mtok=input_price,
            output_usd_per_mtok=output_price,
            reasoning_usd_per_mtok=output_price if reasoning is None else reasoning,
        )


class ArmSpec(_Strict):
    """An arm is (model, reasoning effort) applied to the whole pipeline."""

    model: str
    reasoning_effort: str | None = None

    @property
    def arm_id(self) -> str:
        return f"{self.model}@{self.reasoning_effort or 'default'}"


class ModelsConfig(_Strict):
    models: list[ModelSpec]
    arms: list[ArmSpec]

    @model_validator(mode="after")
    def _check_references(self) -> ModelsConfig:
        names = [m.name for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError("duplicate model names")
        by_name = {m.name: m for m in self.models}
        arm_ids: set[str] = set()
        for arm in self.arms:
            spec = by_name.get(arm.model)
            if spec is None:
                raise ValueError(f"arm references unknown model {arm.model!r}")
            if arm.reasoning_effort is None:
                if spec.reasoning_efforts:
                    raise ValueError(f"arm {arm.arm_id} must set a reasoning_effort")
            elif arm.reasoning_effort not in spec.reasoning_efforts:
                raise ValueError(
                    f"arm {arm.arm_id}: effort not in model's reasoning_efforts "
                    f"{spec.reasoning_efforts}"
                )
            if arm.arm_id in arm_ids:
                raise ValueError(f"duplicate arm {arm.arm_id}")
            arm_ids.add(arm.arm_id)
        return self

    def model_spec(self, name: str) -> ModelSpec:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(f"unknown model {name!r}")

    def prices_for(self, name: str) -> Prices:
        return self.model_spec(name).prices()

    def unfilled(self) -> list[str]:
        return find_todos(self)


# --- budget.yaml -------------------------------------------------------------------------------


class ResolvedBudget(_Strict):
    global_cap_usd: Cap
    per_task_cap_usd: Cap
    per_attempt_cap_usd: Cap
    per_attempt_max_tokens: Annotated[int, Field(gt=0)]
    kill_switch_path: Path

    @model_validator(mode="after")
    def _check_ordering(self) -> ResolvedBudget:
        if not self.per_attempt_cap_usd <= self.per_task_cap_usd <= self.global_cap_usd:
            raise ValueError("caps must satisfy per_attempt <= per_task <= global")
        return self


class BudgetConfig(_Strict):
    global_cap_usd: Cap | Todo
    per_task_cap_usd: Cap | Todo
    per_attempt_cap_usd: Cap | Todo
    per_attempt_max_tokens: Annotated[int, Field(gt=0)] | Todo
    # Touch this file to halt all spend. Relative paths resolve against the working directory.
    kill_switch_path: str = "KILL_SWITCH"

    def resolve(self) -> ResolvedBudget:
        todos = find_todos(self)
        if todos:
            raise ConfigNotFilledError(todos)
        return ResolvedBudget.model_validate(self.model_dump())

    def unfilled(self) -> list[str]:
        return find_todos(self)


# --- repos.yaml --------------------------------------------------------------------------------


class DateWindow(_Strict):
    # Must start after the newest arm's knowledge cutoff (see models.yaml).
    start: dt.date | Todo
    end: dt.date | Todo


class SelectionLimits(_Strict):
    min_source_files: int = 1
    max_source_files: int = 5
    max_source_lines: int = 150
    max_test_runtime_s: int = 120


class RepoSpec(_Strict):
    name: str  # owner/name
    license: str | Todo
    python_version: str | Todo
    install_cmd: str | Todo
    test_cmd: str | Todo
    notes: str = ""


class ReposConfig(_Strict):
    date_window: DateWindow
    limits: SelectionLimits = Field(default_factory=SelectionLimits)
    repos: list[RepoSpec]

    def unfilled(self) -> list[str]:
        return find_todos(self)


# --- loading -----------------------------------------------------------------------------------


def load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models(path: Path) -> ModelsConfig:
    return ModelsConfig.model_validate(load_yaml(path))


def load_budget(path: Path) -> BudgetConfig:
    return BudgetConfig.model_validate(load_yaml(path))


def load_repos(path: Path) -> ReposConfig:
    return ReposConfig.model_validate(load_yaml(path))
