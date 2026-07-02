"""Product-level ML inference orchestration for the TrustScore API."""

from __future__ import annotations

from uuid import uuid4

from app.core.config import settings
from app.ml.fake_review_service import fake_review_service
from app.ml.preprocessing import extract_review_features
from app.ml.risk_service import risk_model_service, score_risk_signals
from app.ml.sentiment_service import sentiment_service
from app.services.ai_feedback import (
    FeedbackContext,
    generate_feedback,
    normalize_language,
)
from app.services.market_reference import market_reference_count_from_signals
from app.schemas.product_analysis import (
    ComponentEvidence,
    ComponentScores,
    MarketContext,
    ProductMetadata,
    ProductPageData,
    ProductAnalysisResponse,
    ScoreTraceItem,
)
from app.services.market_context import (
    expected_currency_for_market,
    market_country_from_url,
    market_from_url,
    normalize_target_market,
    resolve_target_market,
)
from app.services.trustscore_engine import (
    calculate_trustscore,
    classify_risk,
    recommendation_for_risk,
    top_reasons,
)


def build_trustscore(
    payload: ProductPageData,
    *,
    fetch_mode: str = "unknown",
    extraction_signals: list[str] | None = None,
    locale: str | None = None,
    target_market: str | None = "auto",
    page_type: str = "product",
    product_identity_confidence: float = 1.0,
    canonical_product_url: str | None = None,
) -> ProductAnalysisResponse:
    """Run preprocessing, inference fallbacks, and final TrustScore calculation."""
    signals = extraction_signals or []
    review_features = extract_review_features(payload.reviews)
    sentiment = sentiment_service.score_product(review_features, payload.rating)
    fake_reviews = fake_review_service.score_product(review_features)
    risk_scores = score_risk_signals(payload)
    active_components = _active_components(payload)

    feedback_applied = payload.feedback_score is not None
    component_scores = ComponentScores(
        review_authenticity=fake_reviews.authenticity_score,
        seller_reliability=risk_scores.seller_reliability,
        sentiment=sentiment.score,
        return_policy_clarity=risk_scores.return_policy_clarity,
        price_safety=risk_scores.price_safety,
        user_feedback_history=payload.feedback_score if feedback_applied else 50,
    )
    trust_score = calculate_trustscore(
        component_scores,
        active_components=active_components,
        include_feedback=feedback_applied,
    )
    risk_level = classify_risk(trust_score)
    missing_inputs = _missing_inputs(payload, review_features.total_reviews, signals)
    english_reasons = top_reasons(component_scores, active_components=active_components)
    language = normalize_language(locale)
    confidence = _confidence(
        review_count=review_features.total_reviews,
        fake_review_confidence=fake_reviews.confidence,
        sentiment_confidence=sentiment.confidence,
        seller_completeness=risk_scores.seller_completeness,
        price_completeness=risk_scores.price_completeness,
        policy_completeness=risk_scores.policy_completeness,
    )
    recommendation, recommendation_source = _feedback(
        payload=payload,
        component_scores=component_scores,
        trust_score=trust_score,
        risk_level=risk_level,
        confidence=confidence,
        reasons=english_reasons,
        missing_inputs=missing_inputs,
        language=language,
    )
    evidence = _component_evidence(
        payload=payload,
        component_scores=component_scores,
        review_count=review_features.total_reviews,
        unique_review_count=review_features.unique_review_count,
        duplicate_review_rate=review_features.duplicate_review_rate,
        fake_review_confidence=fake_reviews.confidence,
        sentiment_confidence=sentiment.confidence,
        seller_completeness=risk_scores.seller_completeness,
        price_completeness=risk_scores.price_completeness,
        policy_completeness=risk_scores.policy_completeness,
        active_components=active_components,
        extraction_signals=signals,
    )
    model_modes = {
        "fake_review": fake_reviews.mode,
        "sentiment": sentiment.mode,
        "seller_reliability": risk_scores.seller_mode,
        "price_safety": _price_mode(payload, risk_scores.price_mode, signals),
        "return_policy_clarity": _policy_mode(payload, risk_scores.policy_mode),
        "user_feedback_history": "applied_feedback_history" if feedback_applied else "not_applied",
    }
    normalized_identity_confidence = round(max(0.0, min(1.0, product_identity_confidence)), 2)
    score_status = (
        "low_evidence_triage"
        if confidence < 0.55 or normalized_identity_confidence < 0.75
        else "scored"
    )

    return ProductAnalysisResponse(
        scan_id=uuid4(),
        product=ProductMetadata(
            url=payload.url,
            site=payload.site,
            product_title=payload.product_title,
            product_image_url=payload.product_image_url,
            price=payload.price,
            currency=payload.currency,
            seller_name=payload.seller.name if payload.seller else None,
        ),
        trust_score=trust_score,
        risk_level=risk_level,
        confidence=confidence,
        component_scores=component_scores,
        top_reasons=english_reasons,
        evidence=evidence,
        missing_inputs=missing_inputs,
        score_semantics=(
            "TrustScore is normalized over the visible evidence. Prior user feedback for "
            "the same product is applied with a small weight when available; otherwise it "
            "is not scored. Use confidence and missing inputs before relying on the score."
            if feedback_applied
            else "TrustScore is normalized over non-feedback evidence. User feedback is stored "
            "for evaluation and is applied with a small weight only when prior feedback "
            "exists for the product. Use confidence and missing inputs before relying on the score."
        ),
        recommendation=recommendation,
        recommendation_source=recommendation_source,
        language=language,
        model_version=settings.model_version,
        fetch_mode=fetch_mode,
        extraction_signals=signals,
        model_modes=model_modes,
        model_artifact_status={
            "fake_review": fake_review_service.artifact_status,
            "sentiment": sentiment_service.artifact_status,
            "risk": risk_model_service.artifact_status,
        },
        model_versions={
            "trustscore": settings.model_version,
            "fake_review": settings.model_version,
            "sentiment": "0.2.0" if settings.use_v3_risk else settings.model_version,
            "risk": settings.model_version,
        },
        score_status=score_status,
        page_type=_normalized_page_type(page_type),
        product_identity_confidence=normalized_identity_confidence,
        canonical_product_url=canonical_product_url,
        market_context=_market_context(
            payload,
            target_market=target_market,
            locale=locale,
        ),
        score_trace=_score_trace(
            component_scores=component_scores,
            evidence=evidence,
            model_modes=model_modes,
            active_components=active_components,
        ),
        is_mock=False,
    )


def _feedback(
    *,
    payload: ProductPageData,
    component_scores: ComponentScores,
    trust_score: int,
    risk_level: str,
    confidence: float,
    reasons: list[str],
    missing_inputs: list[str],
    language: str,
) -> tuple[str, str]:
    """Localized Claude guidance + reasons; fall back to the English rule text.

    Returns ``(recommendation, source)``. AI feedback may localize shopping
    guidance, but deterministic reasons stay owned by the scoring pipeline.
    """
    fallback = recommendation_for_risk(risk_level)
    if not settings.ai_feedback_active:
        return fallback, "rule"

    ai = generate_feedback(
        FeedbackContext(
            product_title=payload.product_title,
            trust_score=trust_score,
            risk_level=risk_level,
            confidence=confidence,
            component_scores=component_scores.model_dump(),
            reasons=reasons,
            missing_inputs=missing_inputs,
            language=language,
        )
    )
    if ai is not None:
        return ai.recommendation, "ai"
    return fallback, "rule"


def _confidence(
    *,
    review_count: int,
    fake_review_confidence: float,
    sentiment_confidence: float,
    seller_completeness: float,
    price_completeness: float,
    policy_completeness: float,
) -> float:
    review_data_completeness = min(1.0, review_count / 10) if review_count else 0.0
    model_confidence = (fake_review_confidence + sentiment_confidence) / 2
    confidence = (
        review_data_completeness
        + seller_completeness
        + price_completeness
        + policy_completeness
        + model_confidence
    ) / 5
    return round(max(0.05, min(0.98, confidence)), 2)


def _active_components(payload: ProductPageData) -> set[str]:
    active = {"review_authenticity", "seller_reliability", "sentiment"}
    if payload.price is not None and payload.average_market_price is not None:
        active.add("price_safety")
    if payload.return_policy:
        active.add("return_policy_clarity")
    if payload.feedback_score is not None:
        active.add("user_feedback_history")
    return active


def _price_mode(payload: ProductPageData, default_mode: str, signals: list[str]) -> str:
    ignored_currency = _ignored_price_currency(signals) if payload.price is None else None
    if ignored_currency:
        return f"not_scored_localized_currency:{ignored_currency}"
    if payload.price is None:
        return "not_scored_missing_price"
    if payload.average_market_price is None:
        return "not_scored_missing_market_reference"
    return default_mode


def _policy_mode(payload: ProductPageData, default_mode: str) -> str:
    if not payload.return_policy:
        return "not_scored_missing_policy"
    return default_mode


def _normalized_page_type(value: str | None) -> str:
    allowed = {"product", "review_page", "search", "category", "cart", "account", "unknown"}
    normalized = (value or "unknown").strip().lower()
    return normalized if normalized in allowed else "unknown"


def _market_context(
    payload: ProductPageData,
    *,
    target_market: str | None,
    locale: str | None,
) -> MarketContext:
    requested = normalize_target_market(target_market)
    resolved = resolve_target_market(requested, url=payload.url, locale=locale)
    country = market_country_from_url(payload.url) or market_from_url(payload.url) or resolved
    return MarketContext(
        requested_market=requested,
        resolved_market=resolved,
        resolved_country=country,
        expected_currency=expected_currency_for_market(resolved),
        listed_currency=payload.currency,
    )


def _score_trace(
    *,
    component_scores: ComponentScores,
    evidence: list[ComponentEvidence],
    model_modes: dict[str, str],
    active_components: set[str],
) -> list[ScoreTraceItem]:
    evidence_by_component = {item.component: item for item in evidence}
    scores = component_scores.model_dump()
    trace: list[ScoreTraceItem] = []
    for component, score in scores.items():
        item = evidence_by_component.get(component)
        mode = model_modes.get(component) or model_modes.get(_model_key_for_component(component)) or "unknown"
        is_active = component in active_components
        if component == "user_feedback_history" and not is_active:
            status = "not_scored"
            public_score = None
        elif not is_active or mode.startswith("not_scored"):
            status = "not_scored"
            public_score = None
        elif item and item.missing_inputs and not item.evidence:
            status = "missing_evidence"
            public_score = score
        elif "fallback" in mode:
            status = "fallback"
            public_score = score
        else:
            status = "scored"
            public_score = score
        trace.append(
            ScoreTraceItem(
                component=component,
                score=public_score,
                status=status,
                mode=mode,
                confidence=item.confidence if item else 0.0,
                evidence=item.evidence if item else [],
                missing_inputs=item.missing_inputs if item else [],
            )
        )
    return trace


def _model_key_for_component(component: str) -> str:
    return {
        "review_authenticity": "fake_review",
        "sentiment": "sentiment",
        "seller_reliability": "seller_reliability",
        "price_safety": "price_safety",
        "return_policy_clarity": "return_policy_clarity",
        "user_feedback_history": "user_feedback_history",
    }.get(component, component)


def _missing_inputs(payload: ProductPageData, review_count: int, signals: list[str]) -> list[str]:
    missing: list[str] = []
    if review_count == 0:
        missing.append("visible review text")
    if payload.seller is None:
        missing.append("seller profile")
    if payload.price is None and not _ignored_price_currency(signals):
        missing.append("product price")
    return missing


def _component_evidence(
    *,
    payload: ProductPageData,
    component_scores: ComponentScores,
    review_count: int,
    unique_review_count: int,
    duplicate_review_rate: float,
    fake_review_confidence: float,
    sentiment_confidence: float,
    seller_completeness: float,
    price_completeness: float,
    policy_completeness: float,
    active_components: set[str],
    extraction_signals: list[str],
) -> list[ComponentEvidence]:
    seller_evidence: list[str] = []
    seller_missing: list[str] = []
    if payload.seller:
        if payload.seller.name:
            seller_evidence.append(f"Seller: {payload.seller.name}")
        if payload.seller.is_official_store or payload.seller.brand_store_name:
            seller_evidence.append("Seller identity: official brand store signal")
        if payload.seller.is_platform_seller:
            seller_evidence.append("Seller identity: marketplace/platform seller")
        if payload.seller.is_platform_fulfilled:
            seller_evidence.append("Fulfillment: marketplace fulfilled")
        if payload.seller.rating is not None:
            seller_evidence.append(f"Seller rating: {payload.seller.rating:.1f}/5")
        if payload.seller.review_count is not None:
            seller_evidence.append(f"Seller reviews: {payload.seller.review_count}")
        if payload.seller.years_active is not None:
            seller_evidence.append(f"Seller tenure: {payload.seller.years_active} years")
    else:
        seller_missing.append("seller profile")

    if payload.review_count is not None:
        seller_evidence.append(f"Marketplace ratings: {payload.review_count:,}")
    if payload.units_bought_recent is not None:
        seller_evidence.append(
            f"Recent demand: {_units_label(payload.units_bought_recent)} bought in past month"
        )

    price_evidence = []
    price_missing = []
    ignored_currency = _ignored_price_currency(extraction_signals) if payload.price is None else None
    market_reference_count = payload.market_reference_count or market_reference_count_from_signals(
        extraction_signals
    )
    on_sale = (
        payload.price is not None
        and payload.list_price is not None
        and payload.list_price > payload.price
    )
    if payload.price is not None:
        if payload.average_market_price is not None:
            price_evidence.append(f"Listed price: {_money(payload.price, payload.currency)}")
        else:
            price_evidence.append(f"Listed price only: {_money(payload.price, payload.currency)}")
        if on_sale:
            price_evidence.append(
                f"On sale: {_money(payload.price, payload.currency)} "
                f"(was {_money(payload.list_price, payload.currency)})"
            )
    else:
        if ignored_currency:
            price_evidence.append(f"Price not used: localized marketplace currency ({ignored_currency})")
        else:
            price_missing.append("product price")
    if payload.average_market_price is not None and not ignored_currency:
        source = payload.market_reference_source or "market"
        count_label = (
            f" from {market_reference_count} comparable listings"
            if market_reference_count
            else ""
        )
        price_evidence.append(
            f"Market reference found{count_label}: {_money(payload.average_market_price, payload.currency)}"
            f" ({source})"
        )
        if "market_reference:serper:query=category" in extraction_signals:
            price_evidence.append("Comparables matched by category keywords.")
        if (
            payload.market_reference_original_currency
            and payload.market_reference_exchange_rate is not None
            and payload.market_reference_exchange_rate_source
        ):
            date_label = (
                f", {payload.market_reference_exchange_rate_date}"
                if payload.market_reference_exchange_rate_date
                else ""
            )
            price_evidence.append(
                "Converted market reference from "
                f"{payload.market_reference_original_currency}: "
                f"1 {payload.market_reference_original_currency} = "
                f"{_money(payload.market_reference_exchange_rate, payload.currency)} "
                f"({payload.market_reference_exchange_rate_source}{date_label})"
            )
    elif payload.price is not None and not ignored_currency:
        failure = _market_reference_failure_label(extraction_signals)
        price_evidence.append(failure or "No verified market reference found.")
        price_missing.append("verified market reference or FX conversion")

    policy_evidence = []
    policy_missing = []
    if payload.return_policy:
        policy_evidence.append(f"Policy snippet: {_short(payload.return_policy, 120)}")
    else:
        policy_evidence.append("Return policy not visible on this page.")

    review_evidence = []
    review_missing = []
    if review_count:
        review_evidence.append(f"{review_count} visible reviews; {unique_review_count} unique after cleanup")
        review_evidence.append(f"Duplicate review rate: {duplicate_review_rate:.0%}")
    else:
        review_missing.append("visible review text")

    sentiment_evidence = []
    sentiment_missing = []
    if review_count:
        sentiment_evidence.append(f"Sentiment based on {review_count} visible reviews")
    elif payload.rating is not None:
        sentiment_evidence.append(f"Fallback from product rating: {payload.rating:.1f}/5")
        sentiment_missing.append("review text for sentiment")
    else:
        sentiment_missing.append("review text or product rating")

    feedback_evidence: list[str] = []
    feedback_missing: list[str] = []
    if payload.feedback_score is not None:
        feedback_evidence.append(
            f"Applied prior user feedback for this product ({payload.feedback_score}/100, small weight)"
        )
    else:
        feedback_missing.append("reviewed feedback aggregate")

    return [
        ComponentEvidence(
            component="review_authenticity",
            summary="Review authenticity uses visible review text and duplicate-pattern signals.",
            evidence=review_evidence[:5],
            missing_inputs=review_missing,
            confidence=fake_review_confidence,
        ),
        ComponentEvidence(
            component="seller_reliability",
            summary="Seller reliability uses seller identity and marketplace popularity when direct reputation is unavailable.",
            evidence=seller_evidence[:5],
            missing_inputs=seller_missing,
            confidence=seller_completeness,
        ),
        ComponentEvidence(
            component="sentiment",
            summary="Sentiment reflects visible customer-review language or a rating fallback.",
            evidence=sentiment_evidence[:5],
            missing_inputs=sentiment_missing,
            confidence=sentiment_confidence,
        ),
        ComponentEvidence(
            component="return_policy_clarity",
            summary=(
                "Return-policy clarity checks visible policy wording when available; "
                "missing policy text is not scored."
            ),
            evidence=policy_evidence[:5],
            missing_inputs=policy_missing,
            confidence=policy_completeness if "return_policy_clarity" in active_components else 0.0,
        ),
        ComponentEvidence(
            component="price_safety",
            summary=(
                "Price safety is scored with verified market references in the listed currency "
                "or converted through a supported FX rate."
            ),
            evidence=price_evidence[:5],
            missing_inputs=price_missing,
            confidence=price_completeness if "price_safety" in active_components else 0.0,
        ),
        ComponentEvidence(
            component="user_feedback_history",
            summary=(
                "Prior user feedback for this product, applied with a small weight."
                if payload.feedback_score is not None
                else "Feedback is collected for evaluation but is not applied to this score."
            ),
            evidence=feedback_evidence[:5],
            missing_inputs=feedback_missing,
            confidence=0.5 if payload.feedback_score is not None else 0.0,
        ),
    ]


def _market_reference_failure_label(signals: list[str]) -> str | None:
    prefix = "market_reference:unavailable:"
    for signal in signals:
        if not signal.startswith(prefix):
            continue
        reason = signal.removeprefix(prefix)
        if reason == "provider_inactive":
            return "No verified market reference found: provider inactive."
        if reason == "missing_listed_price":
            return "No verified market reference found: listed price missing."
        if reason.startswith("currency_mismatch"):
            return "No verified market reference found: currency mismatch."
        if reason == "no_verified_comparables":
            return "No verified market reference found: no comparable listings passed checks."
        return f"No verified market reference found: {reason.replace('_', ' ')}."
    return None


_ZERO_DECIMAL_CURRENCIES = {"JPY", "KRW", "VND", "CLP", "ISK"}


def _units_label(units: int) -> str:
    if units >= 1_000_000:
        return f"{units / 1_000_000:.0f}M+"
    if units >= 1_000:
        return f"{units / 1_000:.0f}K+"
    return str(units)


def _money(value: float, currency: str | None) -> str:
    prefix = f"{currency} " if currency else ""
    if currency and currency.upper() in _ZERO_DECIMAL_CURRENCIES:
        return f"{prefix}{value:,.0f}"
    return f"{prefix}{value:,.2f}"


def _ignored_price_currency(signals: list[str]) -> str | None:
    prefix = "price_ignored_localized_currency:"
    for signal in signals:
        if signal.startswith(prefix):
            return signal.removeprefix(prefix)
    return None


def _short(value: str, max_length: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_length:
        return compact
    return f"{compact[: max_length - 1]}..."
