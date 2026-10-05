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
    if isinstance(value, dict):
        mapping: dict[Any, Any] = dict(value)  # pyright: ignore[reportUnknownArgumentType]
        found = []
        for key, item in mapping.items():
            found += find_todos(item, f"{path}[{key}]")
        return found
    return []


# --- models.yaml -------------------------------------------------------------------------------


class Prices(_Strict):
    """Resolved USD prices per million tokens, plus the optional pricing rules a model has."""

    input_usd_per_mtok: Price
    output_usd_per_mtok: Price
    reasoning_usd_per_mtok: Price
    # Billed instead of the input price for cached / cache-write input tokens (None: input price).
    cached_input_usd_per_mtok: Price | None = None
    cache_write_usd_per_mtok: Price | None = None
    # A call whose input exceeds the threshold bills input (and output) at these multiples.
    long_context_threshold_tokens: Annotated[int, Field(gt=0)] | None = None
    long_context_input_multiplier: Annotated[float, Field(ge=1)] = 1.0
    long_context_output_multiplier: Annotated[float, Field(ge=1)] = 1.0
    # Multiplier for service_tier="flex" calls; None means flex is not available for this model.
    flex_multiplier: Annotated[float, Field(gt=0, le=1)] | None = None


class ModelSpec(_Strict):
    name: str
    provider: Literal["mock", "openai"]
    input_usd_per_mtok: Price | Todo
    output_usd_per_mtok: Price | Todo
    # None means reasoning tokens are billed at the output price.
    reasoning_usd_per_mtok: Price | None = None
    # Optional pricing rules, all from the provider's pricing page (see Prices). Leave unset where
    # the provider has no such rule; nothing is assumed.
    cached_input_usd_per_mtok: Price | None = None
    cache_write_usd_per_mtok: Price | None = None
    long_context_threshold_tokens: Annotated[int, Field(gt=0)] | None = None
    long_context_input_multiplier: Annotated[float, Field(ge=1)] = 1.0
    long_context_output_multiplier: Annotated[float, Field(ge=1)] = 1.0
    flex_multiplier: Annotated[float, Field(gt=0, le=1)] | None = None
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
            cached_input_usd_per_mtok=self.cached_input_usd_per_mtok,
            cache_write_usd_per_mtok=self.cache_write_usd_per_mtok,
            long_context_threshold_tokens=self.long_context_threshold_tokens,
            long_context_input_multiplier=self.long_context_input_multiplier,
            long_context_output_multiplier=self.long_context_output_multiplier,
            flex_multiplier=self.flex_multiplier,
        )


class ArmSpec(_Strict):
    """An arm is (model, reasoning effort) applied to the whole pipeline."""

    model: str
    reasoning_effort: str | None = None
    # A short role name ("cheap", "mid", "max") used by budget.yaml and the CLI.
    label: str | None = None
    # An optional arm never runs unless the human sets `enabled: true` on it explicitly.
    optional: bool = False
    enabled: bool | None = None

    @property
    def is_enabled(self) -> bool:
        if self.enabled is None:
            return not self.optional
        return self.enabled

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
        labels = [a.label for a in self.arms if a.label is not None]
        if len(set(labels)) != len(labels):
            raise ValueError("duplicate arm labels")
        return self

    def matrix_arms(self) -> list[ArmSpec]:
        """The arms the matrix runs, in file order (cheapest first, by convention of the file).

        Optional arms are included only if the human set ``enabled: true`` on them.
        """
        return [a for a in self.arms if a.is_enabled]

    def arm_by_label_or_id(self, key: str) -> ArmSpec:
        for a in self.arms:
            if key in (a.label, a.arm_id):
                return a
        raise KeyError(f"unknown arm {key!r}")

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
    # Named spending buckets (e.g. pilot, matrix, ablate, repeat). A call tagged with a bucket
    # that has a cap is refused once the bucket's spend plus the call's worst case would exceed it.
    buckets: dict[str, Cap] = Field(default_factory=dict)
    # Per-arm per-attempt caps, keyed by arm label or arm id. Never above per_attempt_cap_usd.
    per_arm_attempt_cap_usd: dict[str, Cap] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_ordering(self) -> ResolvedBudget:
        if not self.per_attempt_cap_usd <= self.per_task_cap_usd <= self.global_cap_usd:
            raise ValueError("caps must satisfy per_attempt <= per_task <= global")
        if sum(self.buckets.values()) > self.global_cap_usd + 1e-9:
            raise ValueError("bucket caps must sum to at most global_cap_usd")
        for arm, cap in self.per_arm_attempt_cap_usd.items():
            if cap > self.per_attempt_cap_usd + 1e-12:
                raise ValueError(f"per-arm cap for {arm!r} exceeds per_attempt_cap_usd")
        return self

    def attempt_cap_for(self, arm_id: str, label: str | None = None) -> float:
        """The per-attempt dollar cap that applies to an arm (its own, else the general one)."""
        for key in (arm_id, label):
            if key is not None and key in self.per_arm_attempt_cap_usd:
                return self.per_arm_attempt_cap_usd[key]
        return self.per_attempt_cap_usd


class BudgetConfig(_Strict):
    global_cap_usd: Cap | Todo
    per_task_cap_usd: Cap | Todo
    per_attempt_cap_usd: Cap | Todo
    per_attempt_max_tokens: Annotated[int, Field(gt=0)] | Todo
    # Touch this file to halt all spend. Relative paths resolve against the working directory.
    kill_switch_path: str = "KILL_SWITCH"
    buckets: dict[str, Cap | Todo] = Field(default_factory=dict)
    per_arm_attempt_cap_usd: dict[str, Cap | Todo] = Field(default_factory=dict)

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
    # Debian packages the tests need that the slim base image lacks (e.g. `less` for pager tests).
    system_packages: list[str] = Field(default_factory=list)
    # What GitHub's API reports as the license id, when it differs from ``license``. Dual-licensed
    # repos are reported as NOASSERTION, which would otherwise fail the license check at mine time.
    github_license_id: str | None = None
    # Disabled repos are skipped by `taskgen run-all` unless named explicitly (e.g. a dormant repo).
    enabled: bool = True
    notes: str = ""


PilotStatus = Literal["owned", "opt_in"]


class PilotRepo(_Strict):
    """A repo the draft-only live pilot may run on. Anything not listed is refused."""

    name: str  # owner/name
    status: PilotStatus  # owned: the human owns it; opt_in: a maintainer agreed to AI-drafted fixes
    # For opt_in repos: who agreed, when, and how (an email thread, an issue comment). Required.
    maintainer_contact: str = ""
    notes: str = ""

    @model_validator(mode="after")
    def _opt_in_needs_a_contact(self) -> PilotRepo:
        if self.status == "opt_in" and not self.maintainer_contact.strip():
            raise ValueError(
                f"pilot repo {self.name!r} is opt_in and needs a maintainer_contact note "
                "(who agreed to AI-drafted fixes, and where that is recorded)"
            )
        return self


class ResolvedRepo(_Strict):
    """One repo's job parameters with every TODO filled in."""

    name: str
    license: str
    python_version: str
    install_cmd: str
    test_cmd: str
    system_packages: tuple[str, ...] = ()
    github_license_id: str | None = None
    window_start: dt.date
    window_end: dt.date
    limits: SelectionLimits

    @property
    def slug(self) -> str:
        return self.name.replace("/", "__")


class ReposConfig(_Strict):
    date_window: DateWindow
    limits: SelectionLimits = Field(default_factory=SelectionLimits)
    repos: list[RepoSpec]
    # Repos the live pilot may draft fixes for (owned or opt_in). Empty means the pilot is off.
    pilot_repos: list[PilotRepo] = Field(default_factory=lambda: [])

    def pilot_repo(self, name: str) -> PilotRepo | None:
        return next((r for r in self.pilot_repos if r.name == name), None)

    def spec(self, name: str) -> RepoSpec | None:
        return next((r for r in self.repos if r.name == name), None)

    def unfilled(self) -> list[str]:
        return find_todos(self)

    def enabled_repos(self) -> list[str]:
        return [r.name for r in self.repos if r.enabled]

    def resolve(self, name: str) -> ResolvedRepo:
        """Return the named repo's parameters, or raise if any field it needs is still TODO.

        Only the date window and the named repo are checked; other repos may stay unfilled.
        """
        spec = next((r for r in self.repos if r.name == name), None)
        if spec is None:
            raise ValueError(f"repo {name!r} is not listed in repos.yaml")
        todos = find_todos(self.date_window, "date_window") + find_todos(spec, f"repos[{name}]")
        if todos:
            raise ConfigNotFilledError(todos)
        start, end = self.date_window.start, self.date_window.end
        if isinstance(start, str) or isinstance(end, str):  # unreachable: TODOs raised above
            raise ConfigNotFilledError(["date_window"])
        if start > end:
            raise ValueError("date_window.start must be on or before date_window.end")
        return ResolvedRepo(
            name=spec.name,
            license=spec.license,
            python_version=spec.python_version,
            install_cmd=spec.install_cmd,
            test_cmd=spec.test_cmd,
            system_packages=tuple(spec.system_packages),
            github_license_id=spec.github_license_id,
            window_start=start,
            window_end=end,
            limits=self.limits,
        )


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
