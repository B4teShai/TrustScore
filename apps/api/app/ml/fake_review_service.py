"""Fake-review probability inference with artifact-aware fallback scoring."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import logging
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.ml.preprocessing import ReviewFeatures


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FakeReviewResult:
    """Product-level fake-review estimate."""

    fake_probability: float
    authenticity_score: int
    confidence: float
    mode: str


class FakeReviewService:
    """Use trained artifacts when present, otherwise score review patterns."""

    def __init__(self, model_path: str | None, vectorizer_path: str | None) -> None:
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path

    @cached_property
    def _artifacts(self) -> tuple[Any, Any] | None:
        if not self.model_path or not self.vectorizer_path:
            return None

        try:
            import joblib
        except Exception as exc:
            logger.info(
                "fake_review_joblib_unavailable",
                extra={"error_type": type(exc).__name__},
            )
            return None

        for model_path in _artifact_candidates(self.model_path, "fake_review_rf.joblib"):
            if not model_path.exists():
                continue
            for vectorizer_path in _artifact_candidates(
                self.vectorizer_path,
                "fake_review_vectorizer.joblib",
            ):
                if not vectorizer_path.exists():
                    continue
                try:
                    return joblib.load(model_path), joblib.load(vectorizer_path)
                except Exception as exc:
                    logger.warning(
                        "fake_review_artifact_load_failed",
                        extra={"error_type": type(exc).__name__},
                    )
                    continue
        return None

    @property
    def artifact_status(self) -> str:
        return "loaded" if self._artifacts is not None else "missing_or_unavailable"

    def score_product(self, features: ReviewFeatures) -> FakeReviewResult:
        if features.total_reviews == 0:
            return FakeReviewResult(
                fake_probability=0.5,
                authenticity_score=50,
                confidence=0.1,
                mode="missing_reviews",
            )

        if self._artifacts is not None and features.cleaned_reviews:
            try:
                return self._score_with_artifacts(features)
            except Exception as exc:
                logger.warning(
                    "fake_review_artifact_scoring_failed",
                    extra={"error_type": type(exc).__name__},
                )

        return self._score_with_heuristics(features)

    def _score_with_artifacts(self, features: ReviewFeatures) -> FakeReviewResult:
        model, vectorizer = self._artifacts
        texts = [review.text for review in features.cleaned_reviews]
        matrix = vectorizer.transform(texts)
        probabilities = model.predict_proba(matrix)
        fake_probabilities = [float(row[-1]) for row in probabilities]
        fake_probability = sum(fake_probabilities) / len(fake_probabilities)
        certainty = sum(max(float(value) for value in row) for row in probabilities) / len(
            probabilities
        )

        return FakeReviewResult(
            fake_probability=round(_clamp_probability(fake_probability), 4),
            authenticity_score=_clamp_score(100 * (1 - fake_probability)),
            confidence=round(certainty, 2),
            mode="calibrated_tfidf_artifact",
        )

    def _score_with_heuristics(self, features: ReviewFeatures) -> FakeReviewResult:
        suspicious_score = (
            features.duplicate_review_rate * 0.32
            + features.short_five_star_rate * 0.24
            + features.rating_sentiment_mismatch_rate * 0.18
            + features.negative_keyword_rate * 0.14
            + features.extreme_rating_ratio * 0.08
            + (1 - features.verified_purchase_ratio) * 0.04
        )
        fake_probability = _clamp_probability(suspicious_score)
        review_volume_factor = min(1.0, features.total_reviews / 12)

        return FakeReviewResult(
            fake_probability=round(fake_probability, 4),
            authenticity_score=_clamp_score(100 * (1 - fake_probability)),
            confidence=round(0.35 + review_volume_factor * 0.25, 2),
            mode="heuristic_fallback",
        )


def _artifact_candidates(raw_path: str, filename: str) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[4]
    path = Path(raw_path)
    candidates = [path if path.is_absolute() else Path.cwd() / path]
    candidates.extend(
        [
            repo_root / "ml" / "artifacts" / "fake_review" / filename,
            repo_root / "ml" / "artifacts" / "fake_full" / filename,
            repo_root / "ml" / "artifacts" / filename,
        ]
    )
    return candidates


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_score(value: float) -> int:
    return max(0, min(100, round(value)))


fake_review_service = FakeReviewService(
    settings.fake_review_model_path,
    settings.fake_review_vectorizer_path,
)
