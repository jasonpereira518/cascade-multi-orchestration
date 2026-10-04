from __future__ import annotations

import pytest

from cascade.errors import ProviderError, SpendNotAllowedError
from cascade.harness.providers import get_provider
from cascade.harness.providers.base import LLMRequest, Message
from cascade.harness.providers.mock import MockProvider, MockReply, MockScriptExhaustedError
from cascade.harness.providers.openai import OpenAIProvider


def _req(text: str = "hello world!", max_out: int = 100) -> LLMRequest:
    return LLMRequest(
        model="any", messages=[Message(role="user", content=text)], max_output_tokens=max_out
    )


def test_mock_default_reply_and_estimated_tokens() -> None:
    p = MockProvider()
    r = p.complete(_req("x" * 40))
    assert r.text == "mock reply"
    assert r.usage.input_tokens == 10  # 40 chars / 4
    assert r.usage.output_tokens == len("mock reply") // 4
    assert r.usage.reasoning_tokens == 0
    assert not p.is_real
    assert len(p.calls) == 1


def test_mock_script_consumed_in_order_then_exhausted() -> None:
    p = MockProvider([MockReply(text="a"), MockReply(text="b")])
    assert [p.complete(_req()).text, p.complete(_req()).text] == ["a", "b"]
    with pytest.raises(MockScriptExhaustedError):
        p.complete(_req())


def test_mock_script_falls_back_to_default() -> None:
    p = MockProvider([MockReply(text="a")], default=MockReply(text="fallback"))
    assert [p.complete(_req()).text, p.complete(_req()).text] == ["a", "fallback"]


def test_mock_callable_script_sees_request_and_index() -> None:
    p = MockProvider(lambda req, i: MockReply(text=f"{i}:{req.messages[0].content}"))
    assert p.complete(_req("q")).text == "0:q"
    assert p.complete(_req("q")).text == "1:q"


def test_mock_explicit_usage_and_latency() -> None:
    p = MockProvider(
        [MockReply(input_tokens=7, output_tokens=3, reasoning_tokens=5, latency_s=1.5)]
    )
    r = p.complete(_req())
    assert (r.usage.input_tokens, r.usage.output_tokens, r.usage.reasoning_tokens) == (7, 3, 5)
    assert r.latency_s == 1.5


def test_mock_error_reply_raises_provider_error() -> None:
    p = MockProvider([MockReply(error="boom")])
    with pytest.raises(ProviderError, match="boom"):
        p.complete(_req())


def test_mock_clamps_to_max_output_tokens() -> None:
    p = MockProvider([MockReply(output_tokens=90, reasoning_tokens=40)])
    r = p.complete(_req(max_out=100))
    assert r.usage.reasoning_tokens == 40
    assert r.usage.output_tokens == 60
    assert r.finish_reason == "length"


def test_openai_stub_refuses_without_spend_gate() -> None:
    with pytest.raises(SpendNotAllowedError):
        OpenAIProvider()


@pytest.mark.parametrize("value", ["", "0", "true", "yes", "2"])
def test_spend_gate_requires_exactly_one(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("CASCADE_ALLOW_SPEND", value)
    with pytest.raises(SpendNotAllowedError):
        OpenAIProvider()


def test_openai_stub_rechecks_gate_on_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CASCADE_ALLOW_SPEND", "1")
    p = OpenAIProvider()
    assert p.is_real
    with pytest.raises(NotImplementedError):  # gate open, but no network code exists yet
        p.complete(_req())
    monkeypatch.delenv("CASCADE_ALLOW_SPEND")
    with pytest.raises(SpendNotAllowedError):
        p.complete(_req())


def test_get_provider_defaults_to_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    assert get_provider().name == "mock"
    monkeypatch.setenv("CASCADE_PROVIDER", "openai")
    with pytest.raises(SpendNotAllowedError):
        get_provider()
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("nope")
