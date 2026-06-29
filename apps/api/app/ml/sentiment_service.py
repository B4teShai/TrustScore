"""Sentiment inference with a safe deterministic fallback."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import logging
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.ml.preprocessing import (
    NEGATIVE_KEYWORDS,
    POSITIVE_KEYWORDS,
    ReviewFeatures,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SentimentResult:
    """Product-level sentiment score and confidence metadata."""

    score: int
    confidence: float
    mode: str


class SentimentService:
    """Wrap optional Transformers inference behind a cheap fallback."""

    def __init__(self, model_name: str, artifact_path: str | None) -> None:
        self.model_name = model_name
        self.artifact_path = artifact_path

    @cached_property
    def _artifact_pipeline(self) -> Any | None:
        for path in _artifact_candidates(self.artifact_path, "sentiment_tfidf_logreg.joblib"):
            if not path.exists():
                continue
            try:
                import joblib

                return joblib.load(path)
            except Exception as exc:
                logger.warning(
                    "sentiment_artifact_load_failed",
                    extra={"error_type": type(exc).__name__},
                )
                return None
        return None

    @property
    def artifact_status(self) -> str:
        return "loaded" if self._artifact_pipeline is not None else "missing_or_unavailable"

    @cached_property
    def _pipeline(self) -> Any | None:
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
            return pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
            )
        except Exception as exc:
            logger.info(
                "sentiment_transformers_pipeline_unavailable",
                extra={"error_type": type(exc).__name__},
            )
            return None

    def score_product(
        self,
        features: ReviewFeatures,
        product_rating: float | None,
    ) -> SentimentResult:
        if not features.cleaned_reviews:
            return self._fallback_from_rating(product_rating)

        if self._artifact_pipeline is not None:
            try:
                return self._score_with_artifact(features)
            except Exception as exc:
                logger.warning(
                    "sentiment_artifact_scoring_failed",
                    extra={"error_type": type(exc).__name__},
                )

        if self._pipeline is not None:
            try:
                return self._score_with_transformers(features)
            except Exception as exc:
                logger.warning(
                    "sentiment_transformers_scoring_failed",
                    extra={"error_type": type(exc).__name__},
                )

        return self._score_with_keywords(features, product_rating)

    def _score_with_artifact(self, features: ReviewFeatures) -> SentimentResult:
        pipeline = self._artifact_pipeline
        review_texts = [review.text for review in features.cleaned_reviews[:30]]
        if pipeline is None or not review_texts:
            return SentimentResult(score=50, confidence=0.1, mode="artifact_empty")

        scores: list[float] = []
        confidences: list[float] = []
        if hasattr(pipeline, "predict_proba"):
            labels = [str(label) for label in pipeline.classes_]
            probabilities = pipeline.predict_proba(review_texts)
            label_scores = {"negative": 20, "neutral": 55, "positive": 90}
            for row in probabilities:
                scores.append(
                    sum(label_scores.get(label, 50) * float(prob) for label, prob in zip(labels, row, strict=True))
                )
                confidences.append(max(float(value) for value in row))
        else:
            predictions = pipeline.predict(review_texts)
            label_scores = {"negative": 20, "neutral": 55, "positive": 90}
            scores.extend(label_scores.get(str(label), 50) for label in predictions)
            confidences.extend([0.55] * len(scores))

        return SentimentResult(
            score=_clamp_score(sum(scores) / len(scores)),
            confidence=round(sum(confidences) / len(confidences), 2),
            mode="tfidf_logreg_artifact",
        )

    def _score_with_transformers(self, features: ReviewFeatures) -> SentimentResult:
        review_texts = [review.text for review in features.cleaned_reviews[:20]]
        predictions = self._pipeline(review_texts, truncation=True)

        scores: list[float] = []
        confidences: list[float] = []
        for prediction in predictions:
            label = str(prediction.get("label", "")).upper()
            probability = float(prediction.get("score", 0.5))
            if "NEG" in label:
                scores.append((1 - probability) * 100)
            else:
                scores.append(probability * 100)
            confidences.append(probability)

        if not scores:
            return SentimentResult(score=50, confidence=0.25, mode="fallback_empty")

        return SentimentResult(
            score=_clamp_score(sum(scores) / len(scores)),
            confidence=round(sum(confidences) / len(confidences), 2),
            mode="transformers",
        )

    def _score_with_keywords(
        self,
        features: ReviewFeatures,
        product_rating: float | None,
    ) -> SentimentResult:
        scores: list[float] = []
        for review in features.cleaned_reviews:
            positive_hits = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in review.text)
            negative_hits = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in review.text)
            if review.rating is not None:
                base = review.rating / 5 * 100
            else:
                base = 50
            scores.append(base + positive_hits * 8 - negative_hits * 14)

        if not scores:
            return self._fallback_from_rating(product_rating)

        review_count_factor = min(1.0, features.total_reviews / 10)
        return SentimentResult(
            score=_clamp_score(sum(scores) / len(scores)),
            confidence=round(0.35 + review_count_factor * 0.35, 2),
            mode="keyword_fallback",
        )

    def _fallback_from_rating(self, product_rating: float | None) -> SentimentResult:
        if product_rating is None:
            return SentimentResult(score=50, confidence=0.1, mode="fallback_missing")

        return SentimentResult(
            score=_clamp_score(product_rating / 5 * 100),
            confidence=0.35,
            mode="rating_fallback",
        )


def _clamp_score(value: float) -> int:
    return max(0, min(100, round(value)))


def _artifact_candidates(raw_path: str | None, filename: str) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[4]
    candidates: list[Path] = []
    if raw_path:
        path = Path(raw_path)
        candidates.append(path if path.is_absolute() else Path.cwd() / path)
        candidates.append(repo_root / "ml" / "artifacts" / path.name)
    candidates.extend(
        [
            repo_root / "ml" / "artifacts" / "sentiment" / filename,
            repo_root / "ml" / "artifacts" / filename,
        ]
    )
    return candidates


sentiment_service = SentimentService(
    settings.sentiment_model_name,
    settings.sentiment_artifact_path,
)
