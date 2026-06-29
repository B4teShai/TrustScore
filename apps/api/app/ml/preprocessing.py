"""Review cleaning and lightweight feature extraction for TrustScore inference."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import html
import re

from app.schemas.product_analysis import ReviewInput


HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
NEGATIVE_KEYWORDS = (
    "fake",
    "broken",
    "scam",
    "refund",
    "late",
    "poor quality",
    "never arrived",
    "bad",
    "terrible",
    "defective",
    "damaged",
)
POSITIVE_KEYWORDS = (
    "good",
    "great",
    "excellent",
    "perfect",
    "fast",
    "love",
    "works",
    "quality",
    "recommend",
)


@dataclass(frozen=True)
class CleanReview:
    """Normalized review text with metadata preserved for scoring."""

    text: str
    rating: float | None
    verified_purchase: bool | None


@dataclass(frozen=True)
class ReviewFeatures:
    """Aggregated review-pattern features used by runtime scoring."""

    cleaned_reviews: list[CleanReview]
    total_reviews: int
    unique_review_count: int
    duplicate_review_rate: float
    short_five_star_rate: float
    avg_review_length: float
    negative_keyword_rate: float
    extreme_rating_ratio: float
    verified_purchase_ratio: float
    rating_sentiment_mismatch_rate: float


def clean_review_text(value: str) -> str:
    """Normalize review text without changing its underlying meaning."""
    without_html = HTML_TAG_RE.sub(" ", html.unescape(value))
    return WHITESPACE_RE.sub(" ", without_html).strip().lower()


def _keyword_signal(text: str) -> int:
    negative_hits = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in text)
    positive_hits = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in text)
    if negative_hits > positive_hits:
        return -1
    if positive_hits > negative_hits:
        return 1
    return 0


def _has_rating_sentiment_mismatch(review: CleanReview) -> bool:
    if review.rating is None:
        return False

    signal = _keyword_signal(review.text)
    if signal < 0 and review.rating >= 4:
        return True
    if signal > 0 and review.rating <= 2:
        return True
    return False


def extract_review_features(reviews: list[ReviewInput]) -> ReviewFeatures:
    """Clean reviews and calculate product-level review features."""
    normalized_reviews = [
        CleanReview(
            text=cleaned,
            rating=review.rating,
            verified_purchase=review.verified_purchase,
        )
        for review in reviews
        if (cleaned := clean_review_text(review.text))
    ]
    total_reviews = len(normalized_reviews)

    if total_reviews == 0:
        return ReviewFeatures(
            cleaned_reviews=[],
            total_reviews=0,
            unique_review_count=0,
            duplicate_review_rate=0.0,
            short_five_star_rate=0.0,
            avg_review_length=0.0,
            negative_keyword_rate=0.0,
            extreme_rating_ratio=0.0,
            verified_purchase_ratio=0.0,
            rating_sentiment_mismatch_rate=0.0,
        )

    text_counts = Counter(review.text for review in normalized_reviews)
    duplicate_count = sum(count - 1 for count in text_counts.values() if count > 1)
    unique_reviews = [
        review
        for index, review in enumerate(normalized_reviews)
        if review.text not in {item.text for item in normalized_reviews[:index]}
    ]

    token_lengths = [len(review.text.split()) for review in normalized_reviews]
    short_five_star_count = sum(
        1
        for review, token_length in zip(normalized_reviews, token_lengths, strict=True)
        if review.rating is not None and review.rating >= 4.5 and token_length <= 6
    )
    negative_keyword_count = sum(
        1
        for review in normalized_reviews
        if any(keyword in review.text for keyword in NEGATIVE_KEYWORDS)
    )
    extreme_rating_count = sum(
        1
        for review in normalized_reviews
        if review.rating is not None and (review.rating <= 1.5 or review.rating >= 4.5)
    )
    verified_count = sum(1 for review in normalized_reviews if review.verified_purchase)
    mismatch_count = sum(
        1 for review in normalized_reviews if _has_rating_sentiment_mismatch(review)
    )

    return ReviewFeatures(
        cleaned_reviews=unique_reviews,
        total_reviews=total_reviews,
        unique_review_count=len(unique_reviews),
        duplicate_review_rate=duplicate_count / total_reviews,
        short_five_star_rate=short_five_star_count / total_reviews,
        avg_review_length=sum(token_lengths) / total_reviews,
        negative_keyword_rate=negative_keyword_count / total_reviews,
        extreme_rating_ratio=extreme_rating_count / total_reviews,
        verified_purchase_ratio=verified_count / total_reviews,
        rating_sentiment_mismatch_rate=mismatch_count / total_reviews,
    )
