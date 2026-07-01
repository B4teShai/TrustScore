"""Rule-based seller, price, and return-policy scoring."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import json
import logging
import math
from pathlib import Path
import re
from typing import Any

from app.core.config import settings
from app.schemas.product_analysis import ProductPageData, SellerInfo


TIME_PERIOD_RE = re.compile(
    r"\b\d+\s*-?\s*(day|days|week|weeks|month|months|jours?)\b|\d+\s*日以内|30日",
    re.I,
)
RETURN_KEYWORDS = ("return", "refund", "retour", "remboursement", "返品", "返金")
WARRANTY_KEYWORDS = ("warranty", "exchange", "replacement", "garantie", "échange", "echange", "交換", "保証")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskSignalScores:
    """Rule-based component scores plus completeness estimates."""

    seller_reliability: int
    price_safety: int
    return_policy_clarity: int
    seller_completeness: float
    price_completeness: float
    policy_completeness: float
    seller_mode: str
    price_mode: str
    policy_mode: str


class RiskModelService:
    """Score seller, price, and policy signals with artifacts when available."""

    def __init__(
        self,
        *,
        seller_model_path: str | None,
        price_model_path: str | None,
        policy_model_path: str | None,
        price_anomaly_model_path: str | None = None,
        price_anomaly_medians_path: str | None = None,
        use_v3: bool = False,
    ) -> None:
        self.seller_model_path = seller_model_path
        self.price_model_path = price_model_path
        self.policy_model_path = policy_model_path
        self.price_anomaly_model_path = price_anomaly_model_path
        self.price_anomaly_medians_path = price_anomaly_medians_path
        self.use_v3 = use_v3

    @cached_property
    def _seller_loaded(self) -> tuple[Any | None, dict[str, Any] | None]:
        return _load_artifact(self.seller_model_path, "seller_reliability_tfidf_rf.joblib")

    @cached_property
    def _price_loaded(self) -> tuple[Any | None, dict[str, Any] | None]:
        return _load_artifact(self.price_model_path, "price_safety_tfidf_rf.joblib")

    @cached_property
    def _policy_loaded(self) -> tuple[Any | None, dict[str, Any] | None]:
        return _load_artifact(self.policy_model_path, "policy_clarity_tfidf_rf.joblib")

    @cached_property
    def _price_anomaly_loaded(self) -> tuple[Any | None, dict[str, float]]:
        model, _spec = _load_artifact(
            self.price_anomaly_model_path,
            "price_anomaly_iforest.joblib",
        )
        medians = _load_json_artifact(
            self.price_anomaly_medians_path,
            "price_anomaly_medians.json",
        )
        return model, {
            str(key): float(value)
            for key, value in medians.items()
            if isinstance(value, int | float)
        }

    @property
    def seller_model(self) -> Any | None:
        return self._seller_loaded[0]

    @property
    def price_model(self) -> Any | None:
        return self._price_loaded[0]

    @property
    def policy_model(self) -> Any | None:
        return self._policy_loaded[0]

    @property
    def artifact_status(self) -> dict[str, str]:
        if self.use_v3:
            price_model, _medians = self._price_anomaly_loaded
            return {
                "seller": "transparent_rule_v3",
                "price": "v3_anomaly_loaded"
                if price_model is not None
                else "v3_anomaly_missing_rule_fallback",
                "policy": "transparent_rule_v3",
            }
        return {
            "seller": "loaded" if self.seller_model is not None else "missing_or_unavailable",
            "price": "loaded" if self.price_model is not None else "missing_or_unavailable",
            "policy": "loaded" if self.policy_model is not None else "missing_or_unavailable",
        }

    def score_product(self, payload: ProductPageData) -> RiskSignalScores:
        seller_score, seller_completeness, seller_mode = self._score_seller(payload)
        price_score, price_completeness, price_mode = self._score_price(payload)
        policy_score, policy_completeness, policy_mode = self._score_policy(payload)

        return RiskSignalScores(
            seller_reliability=seller_score,
            price_safety=price_score,
            return_policy_clarity=policy_score,
            seller_completeness=seller_completeness,
            price_completeness=price_completeness,
            policy_completeness=policy_completeness,
            seller_mode=seller_mode,
            price_mode=price_mode,
            policy_mode=policy_mode,
        )

    def _score_seller(self, payload: ProductPageData) -> tuple[int, float, str]:
        popularity = _popularity_signal(payload)
        if self.use_v3:
            score, completeness = _score_seller_rules(payload.seller, popularity)
            return score, completeness, "transparent_rule_v3"

        model, spec = self._seller_loaded
        label_scores = {"reliable": 88, "mixed": 60, "weak": 32}
        text = _seller_text(payload)
        if model is not None:
            if spec is not None:
                numeric = {
                    "rating": (payload.seller.rating if payload.seller and payload.seller.rating else 0.0),
                    "log_review_count": _log1p(
                        payload.seller.review_count if payload.seller else None
                    ),
                }
                score = _score_v2_model(model, spec, numeric, text, label_scores)
                if score is not None:
                    return score, 0.75 if payload.seller else 0.45, "artifact_v2"
            elif text:
                score = _score_label_model(model, text, label_scores)
                if score is not None:
                    return score, 0.75 if payload.seller else 0.45, "artifact"
        score, completeness = _score_seller_rules(payload.seller, popularity)
        return score, completeness, "rule_fallback"

    def _score_price(self, payload: ProductPageData) -> tuple[int, float, str]:
        if self.use_v3:
            return _score_price_v3(payload)

        model, spec = self._price_loaded
        label_scores = {"normal": 90, "high_price": 65, "suspicious_low": 35}
        text = _price_text(payload)
        if model is not None and payload.price is not None:
            if spec is not None:
                numeric = {
                    "price": float(payload.price),
                    "price_ratio": _ratio(payload.price, payload.average_market_price),
                }
                score = _score_v2_model(model, spec, numeric, text, label_scores)
                if score is not None:
                    return score, 0.75, "artifact_v2"
            elif text:
                score = _score_label_model(model, text, label_scores)
                if score is not None:
                    return score, 0.75, "artifact"
        score, completeness = _score_price_rules(
            payload.price, payload.average_market_price, payload.list_price
        )
        return score, completeness, "rule_fallback"

    def _score_policy(self, payload: ProductPageData) -> tuple[int, float, str]:
        if self.use_v3:
            score, completeness = _score_policy_rules(payload.return_policy)
            return score, completeness, "transparent_rule_v3"

        model, spec = self._policy_loaded
        label_scores = {"clear": 90, "partial": 65, "unclear": 35}
        text = (payload.return_policy or "").strip()
        if model is not None and text:
            if spec is not None:
                score = _score_v2_model(model, spec, _policy_flags(text), text, label_scores)
                if score is not None:
                    return score, 0.75, "artifact_v2"
            else:
                score = _score_label_model(model, text, label_scores)
                if score is not None:
                    return score, 0.75, "artifact"
        score, completeness = _score_policy_rules(payload.return_policy)
        return score, completeness, "rule_fallback"


def score_risk_signals(payload: ProductPageData) -> RiskSignalScores:
    return risk_model_service.score_product(payload)


def _score_seller_rules(
    seller: SellerInfo | None,
    popularity: tuple[float, float] | None = None,
) -> tuple[int, float]:
    """Score seller reliability from direct seller fields when visible.

    Product pages rarely expose seller rating / review count / tenure, so when
    those direct signals are missing we fall back to a marketplace-popularity
    prior (rating volume + recent sales). A wildly popular, well-rated listing
    is treated as a moderately reliable seller rather than an unknown one.
    """
    parts: list[tuple[float, float]] = []
    completeness = 0.15 if (seller and seller.name) else 0.0
    has_direct_signal = False
    has_popularity_signal = False

    if seller is not None:
        official_store = bool(seller.is_official_store or _looks_like_official_store(seller.name))
        if seller.is_platform_seller or official_store:
            parts.append((90.0 if official_store else 84.0, 0.65))
            completeness += 0.55
            has_direct_signal = True
        elif seller.is_platform_fulfilled:
            parts.append((78.0, 0.35))
            completeness += 0.25
            has_direct_signal = True

        if seller.rating is not None:
            parts.append((seller.rating / 5 * 100, 0.55))
            completeness += 0.45
            has_direct_signal = True

        if seller.review_count is not None:
            review_score = min(100, math.log10(seller.review_count + 1) / 4 * 100)
            parts.append((review_score, 0.25))
            completeness += 0.25
            has_direct_signal = True

        if seller.years_active is not None:
            years_score = min(100, seller.years_active / 5 * 100)
            parts.append((years_score, 0.20))
            completeness += 0.15
            has_direct_signal = True

    if popularity is not None:
        has_popularity_signal = True
        pop_score, pop_completeness = popularity
        if has_direct_signal:
            # Direct seller signals lead; popularity only nudges the result.
            parts.append((pop_score, 0.20))
            completeness += min(0.15, pop_completeness * 0.3)
        else:
            # No direct seller reputation: temper the popularity prior against a
            # neutral baseline so it cannot fully stand in for seller identity.
            parts.append((pop_score, 0.55))
            parts.append((55.0, 0.45))
            completeness += min(0.5, pop_completeness)

    if not parts:
        return (55, min(1.0, completeness)) if seller is not None else (50, 0.0)

    weighted = sum(score * weight for score, weight in parts)
    total_weight = sum(weight for _, weight in parts)
    score = _clamp_score(weighted / total_weight)
    if seller is not None and seller.name and not has_direct_signal and not has_popularity_signal:
        score = min(score, 60)
    return score, min(1.0, completeness)


def _popularity_signal(payload: ProductPageData) -> tuple[float, float] | None:
    """Marketplace-popularity score (0-100) and completeness (0-0.6).

    Combines product rating volume, recent purchase volume, and the average
    rating. These are the strongest legitimacy signals available on a typical
    product page even when seller-specific reputation is not exposed.
    """
    parts: list[tuple[float, float]] = []
    completeness = 0.0

    if payload.review_count is not None:
        # 100k+ ratings saturates the volume signal.
        volume = min(100.0, math.log10(payload.review_count + 1) / 5 * 100)
        parts.append((volume, 0.45))
        completeness += 0.25

    if payload.units_bought_recent is not None:
        # 10k+ recent purchases saturates the recent-sales signal.
        sales = min(100.0, math.log10(payload.units_bought_recent + 1) / 4 * 100)
        parts.append((sales, 0.35))
        completeness += 0.20

    if payload.rating is not None:
        parts.append((payload.rating / 5 * 100, 0.20))
        completeness += 0.10

    if not parts:
        return None

    score = sum(value * weight for value, weight in parts) / sum(weight for _, weight in parts)
    return score, min(0.6, completeness)


def _price_reference(
    price: float,
    list_price: float | None,
    average_market_price: float,
) -> float:
    """Price to compare against the market median for fairness.

    When an item is on sale (a strikethrough / original price above the current
    price), a genuine discount would otherwise read as a suspicious anomaly. If
    the product's regular (list) price is itself market-consistent, score the
    fairness ratio against that regular price so real sales are not penalized.
    A missing list price, or one whose ratio to market is itself implausible
    (e.g. a fake-inflated strikethrough), falls back to the current price so
    scam listings stay flagged.
    """
    if list_price is None or list_price <= price:
        return price
    list_ratio = list_price / average_market_price
    if 0.60 <= list_ratio <= 1.40:
        return list_price
    return price


def _score_price_rules(
    price: float | None,
    average_market_price: float | None,
    list_price: float | None = None,
) -> tuple[int, float]:
    if price is None:
        return 50, 0.0
    if average_market_price is None or average_market_price <= 0:
        return 50, 0.45

    reference = _price_reference(price, list_price, average_market_price)
    ratio = reference / average_market_price
    if ratio < 0.50:
        return 35, 1.0
    if ratio < 0.75:
        return 60, 1.0
    if ratio <= 1.30:
        return 90, 1.0
    return 70, 1.0


def _score_price_v3(payload: ProductPageData) -> tuple[int, float, str]:
    """Leakage-free v3 price scoring from a market reference when available."""
    if payload.price is None:
        return 50, 0.0, "v3_missing_price"
    if payload.average_market_price is None or payload.average_market_price <= 0:
        return 50, 0.45, "v3_missing_market_reference"

    reference = _price_reference(
        payload.price, payload.list_price, payload.average_market_price
    )
    ratio = reference / payload.average_market_price
    if ratio < 0.35:
        return 25, 1.0, "v3_price_ratio_anomaly"
    if ratio < 0.60:
        return 45, 1.0, "v3_price_ratio_anomaly"
    if ratio <= 1.40:
        return 90, 1.0, "v3_price_ratio_anomaly"
    if ratio <= 2.0:
        return 70, 1.0, "v3_price_ratio_anomaly"
    return 50, 1.0, "v3_price_ratio_anomaly"


def _score_policy_rules(return_policy: str | None) -> tuple[int, float]:
    if not return_policy or not return_policy.strip():
        return 50, 0.0

    policy = return_policy.lower()
    score = 30
    completeness = 0.35
    if any(keyword in policy for keyword in RETURN_KEYWORDS):
        score += 25
        completeness += 0.25
    if TIME_PERIOD_RE.search(policy):
        score += 25
        completeness += 0.25
    if any(keyword in policy for keyword in WARRANTY_KEYWORDS):
        score += 10
        completeness += 0.10
    if len(policy.split()) >= 8:
        score += 10
        completeness += 0.05

    return _clamp_score(score), min(1.0, completeness)


def _clamp_score(value: float) -> int:
    return max(0, min(100, round(value)))


def _log1p(value: object) -> float:
    try:
        return float(math.log1p(max(float(value), 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _ratio(price: object, reference: object) -> float:
    try:
        p, ref = float(price), float(reference)
    except (TypeError, ValueError):
        return 1.0
    if p <= 0 or ref <= 0:
        return 1.0
    return p / ref


def _policy_flags(text: str) -> dict[str, float]:
    lowered = text.lower()
    return {
        "has_return": float(any(k in lowered for k in RETURN_KEYWORDS)),
        "has_period": float(bool(TIME_PERIOD_RE.search(lowered))),
        "has_warranty": float(any(k in lowered for k in WARRANTY_KEYWORDS)),
    }


def _score_v2_model(
    model: Any,
    spec: dict[str, Any],
    numeric: dict[str, float],
    text: str,
    label_scores: dict[str, int],
) -> int | None:
    """Score a v2 ColumnTransformer pipeline that consumes a one-row DataFrame."""
    try:
        import pandas as pd

        row: dict[str, Any] = {name: numeric.get(name, 0.0) for name in spec["numeric_features"]}
        row[spec["text_feature"]] = text.lower()
        frame = pd.DataFrame([row])
        labels = list(model.classes_)
        probabilities = model.predict_proba(frame)[0]
    except Exception as exc:
        logger.warning(
            "risk_v2_score_failed", extra={"model": spec.get("model"), "error_type": type(exc).__name__}
        )
        return None
    score = 0.0
    for label, probability in zip(labels, probabilities, strict=True):
        score += label_scores.get(str(label), 50) * float(probability)
    return _clamp_score(score)


def _score_label_model(
    model: Any,
    text: str,
    label_scores: dict[str, int],
) -> int | None:
    try:
        labels = list(model.classes_)
        probabilities = model.predict_proba([text])[0]
    except Exception:
        try:
            prediction = str(model.predict([text])[0])
            return label_scores.get(prediction)
        except Exception:
            return None

    score = 0.0
    for label, probability in zip(labels, probabilities, strict=True):
        score += label_scores.get(str(label), 50) * float(probability)
    return _clamp_score(score)


def _seller_text(payload: ProductPageData) -> str:
    return " ".join(
        value
        for value in [
            payload.seller.name if payload.seller else None,
            payload.seller.sold_by if payload.seller else None,
            payload.seller.ships_from if payload.seller else None,
            payload.seller.fulfilled_by if payload.seller else None,
            payload.seller.brand_store_name if payload.seller else None,
            payload.product_title,
            payload.site,
        ]
        if value
    ).lower()


def _price_text(payload: ProductPageData) -> str:
    return " ".join(
        value
        for value in [payload.product_title, payload.description, payload.site]
        if value
    ).lower()


def _looks_like_official_store(value: str | None) -> bool:
    if not value:
        return False
    return bool(
        re.search(
            r"\b(Apple|Amazon|Logitech|StarTech|Sony|Samsung|Microsoft|Anker|Belkin|Dell|HP)\b.*\b(Store|Official)\b",
            value,
            re.I,
        )
    )


def _load_artifact(raw_path: str | None, filename: str) -> tuple[Any | None, dict[str, Any] | None]:
    """Return (model, feature_spec). feature_spec is non-None only for v2 DataFrame models."""
    if raw_path is None:
        return None, None
    for path in _artifact_candidates(raw_path, filename):
        if not path.exists():
            continue
        try:
            import joblib

            model = joblib.load(path)
        except Exception as exc:
            logger.warning(
                "risk_artifact_load_failed",
                extra={"artifact": filename, "error_type": type(exc).__name__},
            )
            return None, None
        spec = _load_feature_spec(path)
        return model, spec
    return None, None


def _load_feature_spec(model_path: Path) -> dict[str, Any] | None:
    """v2 artifacts ship a sibling <stem>_feature_spec.json describing DataFrame columns."""
    spec_path = model_path.with_name(f"{model_path.stem}_feature_spec.json")
    if not spec_path.exists():
        return None
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if "numeric_features" in spec and "text_feature" in spec:
            return spec
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _load_json_artifact(raw_path: str | None, filename: str) -> dict[str, Any]:
    if raw_path is None:
        return {}
    for path in _artifact_candidates(raw_path, filename):
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _artifact_candidates(raw_path: str | None, filename: str) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[4]
    candidates: list[Path] = []
    if raw_path:
        path = Path(raw_path)
        candidates.append(path if path.is_absolute() else Path.cwd() / path)
        candidates.append(repo_root / "ml" / "artifacts" / path.name)
    candidates.extend(
        [
            repo_root / "ml" / "artifacts" / "v2" / "risk" / filename,
            repo_root / "ml" / "artifacts" / "risk" / filename,
            repo_root / "ml" / "artifacts" / filename,
        ]
    )
    return candidates


risk_model_service = RiskModelService(
    seller_model_path=settings.seller_reliability_model_path,
    price_model_path=settings.price_safety_model_path,
    policy_model_path=settings.policy_clarity_model_path,
    price_anomaly_model_path=settings.price_anomaly_model_path,
    price_anomaly_medians_path=settings.price_anomaly_medians_path,
    use_v3=settings.use_v3_risk,
)
