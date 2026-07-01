"""Claude-generated, localized shopping-safety feedback for a scored product page.

The TrustScore itself is always produced by the ML pipeline. This module turns
that structured result into a short, natural-language recommendation AND
translates the rule-based reasons into the shopper's page language, in a single
Claude call. It is strictly best-effort: when no ANTHROPIC_API_KEY is set, the
`anthropic` package is missing, or the call fails, `generate_feedback` returns
None and the caller falls back to the deterministic English text.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import logging

from app.core.config import settings


logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are AI TrustScore, a careful online-shopping safety assistant. "
    "You are given a precomputed trust analysis for one product listing and a "
    "target language (BCP-47 code). Respond with ONLY a JSON object, no prose, "
    "no markdown fences, of the exact shape: "
    '{"recommendation": string, "reasons": [string, ...]}. '
    "Write ALL text in the target language. "
    "recommendation: concise non-accusatory shopper guidance, 2 sentences max, "
    "under 300 characters, plain text, never invent facts, never restate the "
    "numeric score. reasons: translate each provided English reason faithfully "
    "into the target language, keeping the same number and order. Keep currency "
    "amounts and product names exactly as given."
)

# Common BCP-47 primary subtags -> human names, to steer translation quality.
# Japanese and English are the primary targets; others are best-effort.
_LANG_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
}


@dataclass(frozen=True)
class FeedbackContext:
    """Structured analysis passed to Claude to ground and localize the output."""

    product_title: str
    trust_score: int
    risk_level: str
    confidence: float
    component_scores: dict[str, int]
    reasons: list[str]
    missing_inputs: list[str]
    language: str = "en"


@dataclass(frozen=True)
class AiFeedback:
    """Localized recommendation and reasons returned from Claude."""

    recommendation: str
    reasons: list[str]


def normalize_language(locale: str | None) -> str:
    """Reduce a locale like 'mn-MN' or 'EN_us' to a primary subtag like 'mn'."""
    if not locale:
        return "en"
    primary = locale.strip().lower().replace("_", "-").split("-")[0]
    return primary or "en"


@lru_cache(maxsize=1)
def _client():
    """Return a cached Anthropic client, or None when unavailable."""
    if not settings.ai_feedback_active:
        return None
    try:
        import anthropic
    except Exception as exc:  # package not installed
        logger.warning("anthropic_import_failed", extra={"error_type": type(exc).__name__})
        return None
    try:
        return anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.ai_feedback_timeout_seconds,
            max_retries=0,
        )
    except Exception as exc:  # pragma: no cover - construction failure only
        logger.warning("anthropic_client_failed", extra={"error_type": type(exc).__name__})
        return None


def generate_feedback(context: FeedbackContext) -> AiFeedback | None:
    """Generate localized guidance + translated reasons, or None for the fallback."""
    client = _client()
    if client is None:
        return None

    try:
        message = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=max(settings.ai_feedback_max_tokens, 600),
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _user_prompt(context)}],
        )
        text = _first_text(message)
        if not text:
            return None
        return _parse(text, context)
    except Exception as exc:
        logger.warning("ai_feedback_failed", extra={"error_type": type(exc).__name__})
        return None


def _user_prompt(context: FeedbackContext) -> str:
    lang = context.language
    lang_name = _LANG_NAMES.get(lang, lang)
    component_lines = "\n".join(
        f"- {key.replace('_', ' ')}: {value}/100"
        for key, value in context.component_scores.items()
        if key != "user_feedback_history"
    )
    reasons_json = json.dumps(context.reasons, ensure_ascii=False)
    missing = ", ".join(context.missing_inputs) or "none"
    return (
        f"Target language: {lang_name} ({lang})\n"
        f"Product: {context.product_title}\n"
        f"Trust score: {context.trust_score}/100 ({context.risk_level})\n"
        f"Confidence: {context.confidence:.0%}\n"
        f"Component scores:\n{component_lines}\n"
        f"Missing evidence: {missing}\n"
        f"English reasons to translate (keep order and count): {reasons_json}\n\n"
        "Return the JSON object now."
    )


def _parse(text: str, context: FeedbackContext) -> AiFeedback | None:
    payload = _extract_json(text)
    if payload is None:
        return None
    recommendation = payload.get("recommendation")
    if not isinstance(recommendation, str) or not recommendation.strip():
        return None
    raw_reasons = payload.get("reasons")
    if (
        isinstance(raw_reasons, list)
        and len(raw_reasons) == len(context.reasons)
        and all(isinstance(item, str) and item.strip() for item in raw_reasons)
    ):
        reasons = [_tidy(item, 200) for item in raw_reasons]
    else:
        # Translation of reasons was unusable; keep the original English reasons.
        reasons = list(context.reasons)
    return AiFeedback(recommendation=_tidy(recommendation, 480), reasons=reasons)


def _extract_json(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _first_text(message) -> str | None:
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            return getattr(block, "text", None)
    return None


def _tidy(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) > limit:
        compact = compact[: limit - 3].rstrip() + "..."
    return compact
