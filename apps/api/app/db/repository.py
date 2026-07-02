"""Optional PostgreSQL repository for scan and feedback persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse, urlunparse

from app.core.config import settings
from app.db.session import engine_in_cooldown, get_engine, report_engine_connection_failure
from app.schemas.product_analysis import (
    FeedbackRequest,
    ProductAnalysisResponse,
    ProductPageData,
)


logger = logging.getLogger(__name__)
FeedbackPersistenceStatus = Literal["saved", "accepted", "missing", "error"]


@dataclass(frozen=True)
class FeedbackPersistenceResult:
    """Result from attempting to persist user feedback."""

    status: FeedbackPersistenceStatus


def save_scan(
    *,
    product: ProductPageData,
    response: ProductAnalysisResponse,
    browser_id: str | None,
    fetch_mode: str,
    extraction_signals: list[str],
) -> bool:
    """Persist one scan if a database is configured.

    Persistence is intentionally best-effort so local demos still return a
    TrustScore even when no database is configured or a write fails.
    """
    engine = get_engine() if not engine_in_cooldown() else None
    if engine is None:
        return _save_scan_local(
            product=product,
            response=response,
            browser_id=browser_id,
            fetch_mode=fetch_mode,
            extraction_signals=extraction_signals,
        )

    try:
        from sqlalchemy import text

        normalized_url = _normalize_url(product.url)
        browser_id_hash = _hash_browser_id(browser_id)
        with engine.begin() as connection:
            seller_id = _upsert_seller(connection, product)
            product_id = connection.execute(
                text(
                    """
                    insert into products (
                        seller_id,
                        url,
                        normalized_url,
                        site,
                        title,
                        description,
                        product_image_url,
                        return_policy,
                        price,
                        currency,
                        average_market_price,
                        rating,
                        review_count
                    )
                    values (
                        :seller_id,
                        :url,
                        :normalized_url,
                        :site,
                        :title,
                        :description,
                        :product_image_url,
                        :return_policy,
                        :price,
                        :currency,
                        :average_market_price,
                        :rating,
                        :review_count
                    )
                    on conflict (normalized_url) do update set
                        seller_id = excluded.seller_id,
                        site = excluded.site,
                        title = excluded.title,
                        description = excluded.description,
                        product_image_url = excluded.product_image_url,
                        return_policy = excluded.return_policy,
                        price = excluded.price,
                        currency = excluded.currency,
                        average_market_price = excluded.average_market_price,
                        rating = excluded.rating,
                        review_count = excluded.review_count
                    returning id
                    """
                ),
                {
                    "seller_id": seller_id,
                    "url": product.url,
                    "normalized_url": normalized_url,
                    "site": product.site,
                    "title": product.product_title,
                    "description": product.description,
                    "product_image_url": product.product_image_url,
                    "return_policy": product.return_policy,
                    "price": product.price,
                    "currency": product.currency,
                    "average_market_price": product.average_market_price,
                    "rating": product.rating,
                    "review_count": product.review_count,
                },
            ).scalar_one()

            model_version_id = _upsert_model_version(connection, response)
            _insert_reviews(connection, product_id, product)
            connection.execute(
                text(
                    """
                    insert into prediction_runs (
                        id,
                        product_id,
                        model_version_id,
                        browser_id_hash,
                        fetch_mode,
                        extraction_signals,
                        trust_score,
                        risk_level,
                        confidence,
                        component_scores,
                        top_reasons,
                        recommendation
                    )
                    values (
                        :id,
                        :product_id,
                        :model_version_id,
                        :browser_id_hash,
                        :fetch_mode,
                        cast(:extraction_signals as jsonb),
                        :trust_score,
                        :risk_level,
                        :confidence,
                        cast(:component_scores as jsonb),
                        cast(:top_reasons as jsonb),
                        :recommendation
                    )
                    """
                ),
                {
                    "id": str(response.scan_id),
                    "product_id": product_id,
                    "model_version_id": model_version_id,
                    "browser_id_hash": browser_id_hash,
                    "fetch_mode": fetch_mode,
                    "extraction_signals": json.dumps(extraction_signals),
                    "trust_score": response.trust_score,
                    "risk_level": response.risk_level,
                    "confidence": response.confidence,
                    "component_scores": response.component_scores.model_dump_json(),
                    "top_reasons": json.dumps(response.top_reasons),
                    "recommendation": response.recommendation,
                },
            )
            # After the prediction_runs insert: these rows carry a foreign key
            # to the run.
            _insert_scan_review_samples(connection, response, product)
            _insert_model_predictions(connection, response)
            _insert_audit_snapshot(
                connection,
                product=product,
                response=response,
                fetch_mode=fetch_mode,
                extraction_signals=extraction_signals,
            )
        return True
    except Exception as exc:
        logger.warning(
            "scan_persistence_failed",
            extra={"error_type": type(exc).__name__, "error": str(exc)},
            exc_info=True,
        )
        if "OperationalError" in type(exc).__name__:
            report_engine_connection_failure()
        # Keep a durable local record even when the database write fails.
        return _save_scan_local(
            product=product,
            response=response,
            browser_id=browser_id,
            fetch_mode=fetch_mode,
            extraction_signals=extraction_signals,
        )


def save_feedback(payload: FeedbackRequest) -> FeedbackPersistenceResult:
    """Persist feedback or return accepted when no database is configured."""
    engine = get_engine() if not engine_in_cooldown() else None
    if engine is None:
        _save_feedback_local(payload)
        return FeedbackPersistenceResult(status="accepted")

    try:
        from sqlalchemy import text

        browser_id_hash = _hash_browser_id(payload.browser_id)
        with engine.begin() as connection:
            exists = connection.execute(
                text("select 1 from prediction_runs where id = :scan_id"),
                {"scan_id": str(payload.scan_id)},
            ).scalar_one_or_none()
            if exists is None:
                return FeedbackPersistenceResult(status="missing")

            connection.execute(
                text(
                    """
                    insert into user_feedback (
                        prediction_run_id,
                        browser_id_hash,
                        helpful,
                        issue_category,
                        corrected_component,
                        expected_risk_level,
                        comment
                    )
                    values (
                        :scan_id,
                        :browser_id_hash,
                        :helpful,
                        :issue_category,
                        :corrected_component,
                        :expected_risk_level,
                        :comment
                    )
                    """
                ),
                {
                    "scan_id": str(payload.scan_id),
                    "browser_id_hash": browser_id_hash,
                    "helpful": payload.helpful,
                    "issue_category": payload.issue_category,
                    "corrected_component": payload.corrected_component,
                    "expected_risk_level": payload.expected_risk_level,
                    "comment": payload.comment,
                },
            )
        return FeedbackPersistenceResult(status="saved")
    except Exception as exc:
        logger.warning(
            "feedback_persistence_failed",
            extra={"error_type": type(exc).__name__, "error": str(exc)},
            exc_info=True,
        )
        if "OperationalError" in type(exc).__name__:
            report_engine_connection_failure()
            _save_feedback_local(payload)
            return FeedbackPersistenceResult(status="accepted")
        return FeedbackPersistenceResult(status="error")


def _upsert_seller(connection, product: ProductPageData) -> str | None:
    if product.seller is None:
        return None

    from sqlalchemy import text

    seller = product.seller
    if seller.name:
        existing = connection.execute(
            text("select id from sellers where lower(name) = lower(:name) limit 1"),
            {"name": seller.name},
        ).scalar_one_or_none()
        if existing is not None:
            connection.execute(
                text(
                    """
                    update sellers
                    set rating = :rating,
                        review_count = :review_count,
                        years_active = :years_active
                    where id = :id
                    """
                ),
                {
                    "id": existing,
                    "rating": seller.rating,
                    "review_count": seller.review_count,
                    "years_active": seller.years_active,
                },
            )
            return existing

    return connection.execute(
        text(
            """
            insert into sellers (name, rating, review_count, years_active)
            values (:name, :rating, :review_count, :years_active)
            returning id
            """
        ),
        {
            "name": seller.name,
            "rating": seller.rating,
            "review_count": seller.review_count,
            "years_active": seller.years_active,
        },
    ).scalar_one()


def _upsert_model_version(connection, response: ProductAnalysisResponse) -> str:
    from sqlalchemy import text

    return connection.execute(
        text(
            """
            insert into model_versions (
                version,
                sentiment_model_name,
                fake_review_model_name,
                weights,
                metrics
            )
            values (
                :version,
                :sentiment_model_name,
                :fake_review_model_name,
                cast(:weights as jsonb),
                cast(:metrics as jsonb)
            )
            on conflict (version) do update set
                sentiment_model_name = excluded.sentiment_model_name,
                fake_review_model_name = excluded.fake_review_model_name,
                weights = excluded.weights
            returning id
            """
        ),
        {
            "version": response.model_version,
            "sentiment_model_name": settings.sentiment_model_name,
            "fake_review_model_name": response.model_modes.get("fake_review"),
            "weights": json.dumps(
                {
                    "review_authenticity": settings.trust_weight_review_authenticity,
                    "seller_reliability": settings.trust_weight_seller_reliability,
                    "sentiment": settings.trust_weight_sentiment,
                    "return_policy_clarity": settings.trust_weight_policy,
                    "price_safety": settings.trust_weight_price,
                    "user_feedback_history": settings.trust_weight_feedback,
                }
            ),
            "metrics": json.dumps({}),
        },
    ).scalar_one()


def _insert_reviews(connection, product_id: str, product: ProductPageData) -> None:
    from sqlalchemy import text

    for review in product.reviews[:50]:
        connection.execute(
            text(
                """
                insert into reviews (
                    product_id,
                    text,
                    rating,
                    verified_purchase,
                    review_date
                )
                values (:product_id, :text, :rating, :verified_purchase, :review_date)
                """
            ),
            {
                "product_id": product_id,
                "text": review.text[:2000],
                "rating": review.rating,
                "verified_purchase": review.verified_purchase,
                "review_date": review.date,
            },
        )


def _insert_scan_review_samples(
    connection,
    response: ProductAnalysisResponse,
    product: ProductPageData,
) -> None:
    from sqlalchemy import text

    for index, review in enumerate(product.reviews[:50]):
        connection.execute(
            text(
                """
                insert into scan_review_samples (
                    prediction_run_id,
                    review_hash,
                    redacted_text,
                    rating,
                    verified_purchase,
                    review_date,
                    source_position
                )
                values (
                    :prediction_run_id,
                    :review_hash,
                    :redacted_text,
                    :rating,
                    :verified_purchase,
                    :review_date,
                    :source_position
                )
                """
            ),
            {
                "prediction_run_id": str(response.scan_id),
                "review_hash": _hash_text(review.text),
                "redacted_text": review.text[:500],
                "rating": review.rating,
                "verified_purchase": review.verified_purchase,
                "review_date": review.date,
                "source_position": index,
            },
        )


def _insert_model_predictions(connection, response: ProductAnalysisResponse) -> None:
    from sqlalchemy import text

    component_scores = response.component_scores.model_dump()
    for component, score in component_scores.items():
        model_key = _model_key_for_component(component)
        connection.execute(
            text(
                """
                insert into model_predictions (
                    prediction_run_id,
                    model_name,
                    input_features,
                    output
                )
                values (
                    :prediction_run_id,
                    :model_name,
                    cast(:input_features as jsonb),
                    cast(:output as jsonb)
                )
                """
            ),
            {
                "prediction_run_id": str(response.scan_id),
                "model_name": component,
                "input_features": json.dumps(
                    {
                        "mode": response.model_modes.get(component)
                        or response.model_modes.get(model_key),
                        "model_version": response.model_versions.get(model_key),
                    }
                ),
                "output": json.dumps({"score": score}),
            },
        )


def _insert_audit_snapshot(
    connection,
    *,
    product: ProductPageData,
    response: ProductAnalysisResponse,
    fetch_mode: str,
    extraction_signals: list[str],
) -> None:
    from sqlalchemy import text

    seller = product.seller.model_dump(mode="json") if product.seller else None
    market_context = response.market_context.model_dump(mode="json") if response.market_context else {}
    connection.execute(
        text(
            """
            insert into prediction_run_audit (
                prediction_run_id,
                requested_target_market,
                resolved_market,
                resolved_country,
                locale,
                language,
                product_snapshot,
                evidence,
                missing_inputs,
                score_trace,
                model_modes,
                model_versions,
                model_artifact_status,
                recommendation_source,
                market_reference,
                extraction_profile
            )
            values (
                :prediction_run_id,
                :requested_target_market,
                :resolved_market,
                :resolved_country,
                :locale,
                :language,
                cast(:product_snapshot as jsonb),
                cast(:evidence as jsonb),
                cast(:missing_inputs as jsonb),
                cast(:score_trace as jsonb),
                cast(:model_modes as jsonb),
                cast(:model_versions as jsonb),
                cast(:model_artifact_status as jsonb),
                :recommendation_source,
                cast(:market_reference as jsonb),
                cast(:extraction_profile as jsonb)
            )
            on conflict (prediction_run_id) do nothing
            """
        ),
        {
            "prediction_run_id": str(response.scan_id),
            "requested_target_market": market_context.get("requested_market"),
            "resolved_market": market_context.get("resolved_market"),
            "resolved_country": market_context.get("resolved_country"),
            "locale": None,
            "language": response.language,
            "product_snapshot": json.dumps(
                {
                    "url": product.url,
                    "site": product.site,
                    "title": product.product_title,
                    "product_image_url": product.product_image_url,
                    "price": product.price,
                    "currency": product.currency,
                    "average_market_price": product.average_market_price,
                    "market_reference_count": product.market_reference_count,
                    "seller": seller,
                    "return_policy": product.return_policy,
                    "rating": product.rating,
                    "review_count": product.review_count,
                    "units_bought_recent": product.units_bought_recent,
                },
                ensure_ascii=False,
            ),
            "evidence": json.dumps([item.model_dump(mode="json") for item in response.evidence]),
            "missing_inputs": json.dumps(response.missing_inputs),
            "score_trace": json.dumps([item.model_dump(mode="json") for item in response.score_trace]),
            "model_modes": json.dumps(response.model_modes),
            "model_versions": json.dumps(response.model_versions),
            "model_artifact_status": json.dumps(response.model_artifact_status),
            "recommendation_source": response.recommendation_source,
            "market_reference": json.dumps(
                {
                    "currency": product.currency,
                    "average_market_price": product.average_market_price,
                    "count": product.market_reference_count,
                    "source": product.market_reference_source,
                    "original_currency": product.market_reference_original_currency,
                    "exchange_rate": product.market_reference_exchange_rate,
                    "exchange_rate_source": product.market_reference_exchange_rate_source,
                    "exchange_rate_date": product.market_reference_exchange_rate_date,
                },
                ensure_ascii=False,
            ),
            "extraction_profile": json.dumps(
                {
                    "fetch_mode": fetch_mode,
                    "extraction_signals": extraction_signals,
                    "page_type": response.page_type,
                    "product_identity_confidence": response.product_identity_confidence,
                    "canonical_product_url": response.canonical_product_url,
                    "score_status": response.score_status,
                }
            ),
        },
    )


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    normalized = parsed._replace(fragment="")
    return urlunparse(normalized)


def _model_key_for_component(component: str) -> str:
    return {
        "review_authenticity": "fake_review",
        "sentiment": "sentiment",
        "seller_reliability": "risk",
        "price_safety": "risk",
        "return_policy_clarity": "risk",
        "user_feedback_history": "trustscore",
    }.get(component, component)


def _hash_browser_id(browser_id: str | None) -> str | None:
    if not browser_id:
        return None
    value = f"{settings.browser_id_hash_salt}:{browser_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _save_scan_local(
    *,
    product: ProductPageData,
    response: ProductAnalysisResponse,
    browser_id: str | None,
    fetch_mode: str,
    extraction_signals: list[str],
) -> bool:
    """Persist one scored page to the local JSONL store when no DB is configured."""
    record = {
        "type": "scan",
        "scan_id": str(response.scan_id),
        "created_at": _now_iso(),
        "browser_id_hash": _hash_browser_id(browser_id),
        "fetch_mode": fetch_mode,
        "extraction_signals": extraction_signals,
        "model_version": response.model_version,
        "product": {
            "url": product.url,
            "normalized_url": _normalize_url(product.url),
            "site": product.site,
            "title": product.product_title,
            "price": product.price,
            "currency": product.currency,
            "seller_name": product.seller.name if product.seller else None,
        },
        "trust_score": response.trust_score,
        "risk_level": response.risk_level,
        "confidence": response.confidence,
        "component_scores": response.component_scores.model_dump(),
        "evidence": [item.model_dump(mode="json") for item in response.evidence],
        "score_trace": [item.model_dump(mode="json") for item in response.score_trace],
        "market_context": response.market_context.model_dump(mode="json")
        if response.market_context
        else None,
        "score_status": response.score_status,
        "page_type": response.page_type,
        "product_identity_confidence": response.product_identity_confidence,
        "top_reasons": response.top_reasons,
        "recommendation": response.recommendation,
        "recommendation_source": response.recommendation_source,
        "missing_inputs": response.missing_inputs,
    }
    return _append_local(record)


def _save_feedback_local(payload: FeedbackRequest) -> bool:
    """Append feedback to the local JSONL store; status stays best-effort accepted."""
    record = {
        "type": "feedback",
        "scan_id": str(payload.scan_id),
        "created_at": _now_iso(),
        "browser_id_hash": _hash_browser_id(payload.browser_id),
        "helpful": payload.helpful,
        "issue_category": payload.issue_category,
        "corrected_component": payload.corrected_component,
        "expected_risk_level": payload.expected_risk_level,
        "comment": payload.comment,
    }
    return _append_local(record)


def _append_local(record: dict) -> bool:
    if not settings.persist_local_scans:
        return False
    try:
        path = Path(settings.local_scan_store_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception as exc:
        logger.warning(
            "local_persistence_failed",
            extra={"error_type": type(exc).__name__, "error": str(exc)},
            exc_info=True,
        )
        return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
