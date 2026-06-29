"""Tests for the Claude-backed feedback generation wiring."""

import json

from app.services import ai_feedback
from app.services.ai_feedback import (
    FeedbackContext,
    generate_feedback,
    normalize_language,
)


_CONTEXT = FeedbackContext(
    product_title="Wireless Earbuds",
    trust_score=72,
    risk_level="Medium Risk",
    confidence=0.6,
    component_scores={
        "review_authenticity": 70,
        "seller_reliability": 65,
        "sentiment": 80,
        "return_policy_clarity": 75,
        "price_safety": 60,
        "user_feedback_history": 50,
    },
    reasons=["Price should be compared with similar products."],
    missing_inputs=["seller tenure"],
    language="ja",
)


class _Block:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _Message:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


class _FakeMessages:
    def __init__(self, captured: dict) -> None:
        self._captured = captured

    def create(self, **kwargs):
        self._captured.update(kwargs)
        return _Message(
            "Here you go:\n"
            + json.dumps(
                {
                    "recommendation": "購入前に価格と販売者を確認してください。",
                    "reasons": ["価格を類似商品と比較してください。"],
                },
                ensure_ascii=False,
            )
        )


class _FakeClient:
    def __init__(self, captured: dict) -> None:
        self.messages = _FakeMessages(captured)


def test_generate_feedback_returns_none_without_client(monkeypatch) -> None:
    monkeypatch.setattr(ai_feedback, "_client", lambda: None)
    assert generate_feedback(_CONTEXT) is None


def test_generate_feedback_uses_claude_response(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(ai_feedback, "_client", lambda: _FakeClient(captured))

    result = generate_feedback(_CONTEXT)

    assert result is not None
    assert result.recommendation == "購入前に価格と販売者を確認してください。"
    assert result.reasons == ["価格を類似商品と比較してください。"]
    assert captured["model"] == ai_feedback.settings.anthropic_model
    content = captured["messages"][0]["content"]
    assert "Wireless Earbuds" in content
    assert "Japanese (ja)" in content


def test_normalize_language_reduces_to_primary_subtag() -> None:
    assert normalize_language("ja-JP") == "ja"
    assert normalize_language("EN_us") == "en"
    assert normalize_language(None) == "en"
    assert normalize_language("") == "en"


def test_generate_feedback_swallows_errors(monkeypatch) -> None:
    class _Boom:
        @property
        def messages(self):
            raise RuntimeError("api down")

    monkeypatch.setattr(ai_feedback, "_client", lambda: _Boom())
    assert generate_feedback(_CONTEXT) is None
