from __future__ import annotations

from pathlib import Path

import pytest

from cascade.config import ModelsConfig, ResolvedBudget, load_models
from cascade.errors import (
    BudgetExceededError,
    ConfigNotFilledError,
    DuplicateCallError,
    KillSwitchEngagedError,
    ProviderError,
    SpendNotAllowedError,
)
from cascade.harness.ledger import CallContext, Ledger
from cascade.harness.metered import MeteredClient, estimate_input_tokens
from cascade.harness.providers.base import LLMRequest, LLMResponse, Message, Provider, Usage
from cascade.harness.providers.mock import MockProvider, MockReply
from tests.conftest import ROOT


def ctx(
    call_id: str, *, task: str = "t1", attempt: str = "a1", role: str = "triager"
) -> CallContext:
    return CallContext(
        call_id=call_id, run_id="run", task_id=task, attempt_id=attempt, arm_id="arm", role=role
    )


def record(ledger: Ledger, call_id: str, model: str, usage: Usage, **kw: str) -> None:
    ledger.record(ctx(call_id, **kw), provider="mock", model=model, usage=usage, latency_s=0.1)


def small(models: ModelsConfig) -> str:
    return models.models[0].name


def large(models: ModelsConfig) -> str:
    return models.models[1].name


# --- pricing and persistence -------------------------------------------------------------------


def test_cost_uses_models_yaml_prices(ledger: Ledger, models: ModelsConfig) -> None:
    p = models.prices_for(large(models))
    usage = Usage(input_tokens=1000, output_tokens=500, reasoning_tokens=200)
    expected = (1000 * p.input_usd_per_mtok + 500 * p.output_usd_per_mtok
                + 200 * p.reasoning_usd_per_mtok) / 1e6  # fmt: skip
    assert ledger.cost_of(large(models), usage) == pytest.approx(expected)
    assert expected > 0


def test_record_roundtrip_and_all_fields(ledger: Ledger, models: ModelsConfig) -> None:
    usage = Usage(input_tokens=10, output_tokens=20, reasoning_tokens=30)
    rec = ledger.record(
        ctx("c1"), provider="mock", model=small(models), usage=usage, latency_s=1.25
    )
    [row] = ledger.rows()
    assert row == rec
    assert (row.input_tokens, row.output_tokens, row.reasoning_tokens) == (10, 20, 30)
    assert row.latency_s == 1.25
    assert (row.task_id, row.attempt_id, row.arm_id, row.role) == ("t1", "a1", "arm", "triager")
    assert row.status == "ok" and row.error is None


def test_ledger_survives_reopen(
    tmp_path: Path, budget: ResolvedBudget, models: ModelsConfig
) -> None:
    db = tmp_path / "persist.sqlite"
    first = Ledger(db, budget=budget, models=models)
    record(first, "c1", large(models), Usage(input_tokens=1000, output_tokens=1000))
    spent = first.spent_usd()
    first.close()

    second = Ledger(db, budget=budget, models=models)
    assert second.spent_usd() == pytest.approx(spent) and spent > 0
    assert second.has_call("c1")
    second.close()


def test_duplicate_call_id_is_rejected(ledger: Ledger, models: ModelsConfig) -> None:
    record(ledger, "dup", small(models), Usage(input_tokens=1, output_tokens=1))
    with pytest.raises(DuplicateCallError):
        record(ledger, "dup", small(models), Usage(input_tokens=1, output_tokens=1))
    assert len(ledger.rows()) == 1


def test_unfilled_prices_fail_closed(tmp_path: Path, budget: ResolvedBudget) -> None:
    template = load_models(ROOT / "configs" / "models.yaml")
    ledger = Ledger(tmp_path / "x.sqlite", budget=budget, models=template)
    try:
        with pytest.raises(ConfigNotFilledError):
            ledger.cost_of(template.models[0].name, Usage(input_tokens=1, output_tokens=1))
    finally:
        ledger.close()


# --- caps --------------------------------------------------------------------------------------


def auth(
    ledger: Ledger, models: ModelsConfig, call: CallContext, *, est: int = 0, max_out: int = 100,
    model: str | None = None, real: bool = False,
) -> None:  # fmt: skip
    ledger.authorize(
        call,
        model=model or large(models),
        est_input_tokens=est,
        max_output_tokens=max_out,
        real=real,
    )


def test_authorize_passes_under_all_caps(ledger: Ledger, models: ModelsConfig) -> None:
    auth(ledger, models, ctx("ok"))


def test_per_attempt_cap(ledger: Ledger, models: ModelsConfig) -> None:
    # mock-large worst case 3000 output tokens at 50/Mtok = $0.15 > $0.10 attempt cap
    with pytest.raises(BudgetExceededError, match="attempt cap"):
        auth(ledger, models, ctx("big"), max_out=3000)


def test_per_attempt_cap_counts_prior_spend(ledger: Ledger, models: ModelsConfig) -> None:
    # 2000 output tokens at $40/Mtok = $0.08 recorded; a further $0.05 worst case breaches $0.10
    record(ledger, "c1", large(models), Usage(input_tokens=0, output_tokens=2000))
    with pytest.raises(BudgetExceededError, match="attempt cap"):
        auth(ledger, models, ctx("c2"), max_out=1000)
    auth(ledger, models, ctx("c3", attempt="a2"), max_out=1000)  # another attempt is unaffected


def test_per_task_cap(ledger: Ledger, models: ModelsConfig) -> None:
    for i in range(5):  # five attempts at $0.09 = $0.45 on one task
        record(ledger, f"c{i}", large(models), Usage(input_tokens=0, output_tokens=2250),
               attempt=f"a{i}")  # fmt: skip
    with pytest.raises(BudgetExceededError, match="task cap"):
        auth(ledger, models, ctx("new", attempt="a-new"), max_out=2000)  # worst $0.10 -> $0.55
    auth(ledger, models, ctx("other", task="t2"), max_out=2000)  # a different task is unaffected


def test_global_cap(ledger: Ledger, models: ModelsConfig) -> None:
    for i in range(11):  # eleven tasks at $0.09 = $0.99 against a $1.00 global cap
        record(ledger, f"c{i}", large(models), Usage(input_tokens=0, output_tokens=2250),
               task=f"t{i}")  # fmt: skip
    with pytest.raises(BudgetExceededError, match="global cap"):
        auth(ledger, models, ctx("new", task="t-new"), max_out=1000)  # worst $0.05 -> $1.04


def test_per_attempt_token_cap(
    ledger: Ledger, models: ModelsConfig, budget: ResolvedBudget
) -> None:
    record(ledger, "c1", small(models), Usage(input_tokens=15_000, output_tokens=0))
    # Cheap in dollars, but 15000 used + 3000 + 3000 requested > 20000 tokens.
    with pytest.raises(BudgetExceededError, match="token cap"):
        auth(ledger, models, ctx("c2"), model=small(models), est=3000, max_out=3000)
    assert budget.per_attempt_max_tokens == 20_000


# --- kill switch and spend gate ----------------------------------------------------------------


def test_kill_switch_blocks_then_releases(
    ledger: Ledger, models: ModelsConfig, budget: ResolvedBudget
) -> None:
    budget.kill_switch_path.touch()
    with pytest.raises(KillSwitchEngagedError):
        auth(ledger, models, ctx("k1"))
    budget.kill_switch_path.unlink()
    auth(ledger, models, ctx("k2"))


def test_real_provider_needs_spend_gate_even_with_budget_room(
    ledger: Ledger, models: ModelsConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SpendNotAllowedError):
        auth(ledger, models, ctx("r1"), real=True)
    monkeypatch.setenv("CASCADE_ALLOW_SPEND", "1")
    auth(ledger, models, ctx("r2"), real=True)


def test_real_provider_still_subject_to_caps_when_gate_open(
    ledger: Ledger, models: ModelsConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CASCADE_ALLOW_SPEND", "1")
    with pytest.raises(BudgetExceededError):
        auth(ledger, models, ctx("r3"), max_out=3000, real=True)


# --- metered client ----------------------------------------------------------------------------


def request(model: str, text: str = "hello", max_out: int = 200) -> LLMRequest:
    return LLMRequest(
        model=model, messages=[Message(role="user", content=text)], max_output_tokens=max_out
    )


def test_metered_call_is_recorded_with_provider_usage(ledger: Ledger, models: ModelsConfig) -> None:
    provider = MockProvider(
        [MockReply(input_tokens=100, output_tokens=50, reasoning_tokens=25, latency_s=2.0)]
    )
    MeteredClient(provider, ledger).complete(ctx("m1"), request(small(models)))
    [row] = ledger.rows()
    assert (row.input_tokens, row.output_tokens, row.reasoning_tokens) == (100, 50, 25)
    assert row.latency_s == 2.0 and row.provider == "mock"
    assert row.cost_usd == pytest.approx(
        ledger.cost_of(
            small(models), Usage(input_tokens=100, output_tokens=50, reasoning_tokens=25)
        )
    )


def test_metered_refuses_repeat_call_id_without_calling_provider(
    ledger: Ledger, models: ModelsConfig
) -> None:
    provider = MockProvider()
    client = MeteredClient(provider, ledger)
    client.complete(ctx("same"), request(small(models)))
    with pytest.raises(DuplicateCallError):
        client.complete(ctx("same"), request(small(models)))
    assert len(provider.calls) == 1 and len(ledger.rows()) == 1


def test_metered_blocks_before_provider_when_kill_switch_set(
    ledger: Ledger, models: ModelsConfig, budget: ResolvedBudget
) -> None:
    provider = MockProvider()
    budget.kill_switch_path.touch()
    with pytest.raises(KillSwitchEngagedError):
        MeteredClient(provider, ledger).complete(ctx("k"), request(small(models)))
    assert provider.calls == [] and ledger.rows() == []


def test_metered_blocks_before_provider_when_over_cap(ledger: Ledger, models: ModelsConfig) -> None:
    provider = MockProvider()
    with pytest.raises(BudgetExceededError):
        MeteredClient(provider, ledger).complete(ctx("big"), request(large(models), max_out=3000))
    assert provider.calls == [] and ledger.rows() == []


def test_metered_records_provider_errors_as_zero_cost_rows(
    ledger: Ledger, models: ModelsConfig
) -> None:
    provider = MockProvider([MockReply(error="upstream 500")])
    with pytest.raises(ProviderError):
        MeteredClient(provider, ledger).complete(ctx("e1"), request(small(models)))
    [row] = ledger.rows()
    assert row.status == "error" and row.error == "upstream 500" and row.cost_usd == 0.0


class _FakeRealProvider(Provider):
    name = "fake-real"
    is_real = True

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text="x", usage=Usage(input_tokens=1, output_tokens=1), latency_s=0.0, model="m"
        )


def test_metered_never_reaches_a_real_provider_without_spend_gate(
    ledger: Ledger, models: ModelsConfig
) -> None:
    provider = _FakeRealProvider()
    with pytest.raises(SpendNotAllowedError):
        MeteredClient(provider, ledger).complete(ctx("r"), request(small(models)))
    assert provider.calls == 0 and ledger.rows() == []


def test_input_estimate_is_conservative() -> None:
    req = request("m", text="x" * 300)
    assert estimate_input_tokens(req) == 100
