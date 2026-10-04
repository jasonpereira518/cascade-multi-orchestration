"""The spend gate (hard rule 2): real LLM calls need CASCADE_ALLOW_SPEND=1 *and* ledger approval."""

from __future__ import annotations

import os

from cascade.errors import SpendNotAllowedError

SPEND_ENV = "CASCADE_ALLOW_SPEND"


def spend_allowed() -> bool:
    """True only when the env var is exactly "1". Anything else (unset, "true", "0") is closed."""
    return os.environ.get(SPEND_ENV) == "1"


def require_spend_allowed(what: str) -> None:
    if not spend_allowed():
        raise SpendNotAllowedError(
            f"{what} would incur real spend; refusing because {SPEND_ENV}=1 is not set"
        )
