"""End-to-end smoke test (`make smoke`): fixture repo -> full agent pipeline -> graded matrix row.

Mock models and synthetic data only; no Docker, no network, $0. This is the acceptance run for the
walking skeleton: all six roles, the repair loop machinery, both outcomes, ledger-derived costs,
and the leak audit, on the synthetic fixture task (no live validated task exists yet).
"""

from __future__ import annotations

import pytest

from tests.agent_helpers import AttemptEnv, good_plans

pytestmark = pytest.mark.smoke


def test_full_pipeline_on_the_fixture_with_the_mock_provider(attempt_env: AttemptEnv) -> None:
    source = attempt_env.source(good_plans())
    report = attempt_env.run(source=source)

    row = report.row
    assert row.true_success and row.verifier_claimed and row.error is None

    # All five LLM roles ran, and the sixth (the Verifier) produced the report that claimed success.
    assert {role for role in good_plans() if source.calls(role)} == set(good_plans())
    assert report.outcome is not None and report.outcome.report is not None
    assert report.outcome.report.claimed and report.outcome.repro_valid

    # Every number in the row is computed from the logged calls, none estimated.
    rows = attempt_env.ledger.rows()
    assert row.turns == len(rows) and row.cost_usd == pytest.approx(sum(r.cost_usd for r in rows))
    assert row.cost_usd > 0 and all(r.provider == "mock" for r in rows)

    # Nothing hidden reached any prompt, and the row is durable.
    assert report.leak_violations == []
    assert attempt_env.matrix.rows() == [row]
