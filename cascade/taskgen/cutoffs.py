"""Which arms' knowledge cutoffs a task clears, from models.yaml.

A PR "clears" an arm when it was merged strictly AFTER that arm's knowledge cutoff, so the arm
cannot have seen it in training (as far as the stated cutoff is believed). An arm whose cutoff is
still TODO gets ``None``: unknown is never reported as cleared or failed.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping

from cascade.config import ModelsConfig


def arm_cutoffs(models: ModelsConfig) -> dict[str, dt.date | None]:
    """arm id -> knowledge cutoff of its model (None while still TODO), for every arm listed."""
    out: dict[str, dt.date | None] = {}
    for arm in models.arms:
        cutoff = models.model_spec(arm.model).knowledge_cutoff
        out[arm.arm_id] = None if isinstance(cutoff, str) else cutoff
    return out


def clearance(
    merged_at: dt.datetime | None, cutoffs: Mapping[str, dt.date | None]
) -> dict[str, bool | None]:
    return {
        arm: None if cutoff is None or merged_at is None else merged_at.date() > cutoff
        for arm, cutoff in cutoffs.items()
    }
