"""Exception types shared across Cascade. Every guard fails closed with one of these."""

from __future__ import annotations


class CascadeError(Exception):
    """Base class for all Cascade errors."""


class ConfigNotFilledError(CascadeError):
    """A config still contains TODO fields the human must fill in."""

    def __init__(self, paths: list[str]) -> None:
        self.paths = paths
        super().__init__("unfilled TODO fields: " + ", ".join(paths))


class SpendNotAllowedError(CascadeError):
    """A real (billable) provider was used without CASCADE_ALLOW_SPEND=1."""


class BudgetExceededError(CascadeError):
    """A call would exceed a global, per-task, or per-attempt cap."""


class KillSwitchEngagedError(CascadeError):
    """The kill-switch file exists; all calls are refused."""


class DuplicateCallError(CascadeError):
    """A call_id was already recorded; re-running it would make the ledger undercount spend."""


class ProviderError(CascadeError):
    """A provider call failed."""


class LeakError(CascadeError):
    """A sandbox mount or workspace would expose gold/test patches, hidden tests, or git history."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("leak guard violations: " + "; ".join(violations))


class SandboxError(CascadeError):
    """The sandbox could not be prepared or launched safely."""
