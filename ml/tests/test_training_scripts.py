from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import pandas as pd

from ml.training.calibrate_trustscore import build_parser as build_calibrator_parser
from ml.training.calibrate_trustscore import calibrate_trustscore
from ml.training.download_datasets import build_parser as build_download_parser
from ml.training.download_datasets import download_datasets
from ml.training.train_all import build_parser as build_train_all_parser
from ml.training.train_all import train_all
from ml.training.train_fake_review_model import build_parser as build_fake_parser
from ml.training.train_fake_review_model import train_fake_review_model
from ml.training.train_risk_models import build_parser as build_risk_parser
from ml.training.train_risk_models import load_amazon_meta
from ml.training.train_risk_models import train_risk_models
from ml.training.train_sentiment_model import build_parser as build_sentiment_parser
from ml.training.train_sentiment_model import load_sentiment_dataset
from ml.training.train_sentiment_model import train_sentiment_model


class _DatasetStub:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def shuffle(self, *, seed: int, buffer_size: int) -> "_DatasetStub":
        self.shuffle_seed = seed
        self.shuffle_buffer_size = buffer_size
        return self

    def take(self, count: int):
        return iter(self.rows[:count])

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_fake_review_training_writes_backend_compatible_artifacts(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path / "fake_reviews.csv",
        [
            {"category": "Home", "rating": 5, "text": "Excellent quality and fast delivery", "label": 0},
            {"category": "Home", "rating": 4, "text": "Works well after a month of use", "label": 0},
            {"category": "Tech", "rating": 5, "text": "Real purchase and solid build quality", "label": 0},
            {"category": "Tech", "rating": 3, "text": "Average but usable product for price", "label": 0},
            {"category": "Home", "rating": 5, "text": "Great product love it very pretty", "label": 1},
            {"category": "Home", "rating": 5, "text": "Amazing item perfect perfect perfect", "label": 1},
            {"category": "Tech", "rating": 1, "text": "Generated fake review with odd wording", "label": 1},
            {"category": "Tech", "rating": 5, "text": "Best product ever highly recommended wow", "label": 1},
        ],
    )
    output_dir = tmp_path / "artifacts"
    args = build_fake_parser().parse_args(
        [
            "--local-csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--test-size",
            "0.25",
            "--min-df",
            "1",
            "--n-estimators",
            "10",
        ]
    )

    metadata = train_fake_review_model(args)

    model_path = output_dir / "fake_review_rf.joblib"
    vectorizer_path = output_dir / "fake_review_vectorizer.joblib"
    assert model_path.exists()
    assert vectorizer_path.exists()
    assert (output_dir / "fake_review_feature_config.json").exists()
    assert (output_dir / "model_metadata.json").exists()
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    probabilities = model.predict_proba(vectorizer.transform(["excellent real purchase"]))
    assert probabilities.shape == (1, 2)
    assert metadata["dataset"]["row_count"] == 8


def test_sentiment_training_supports_no_save_local_csv(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path / "sentiment.csv",
        [
            {"rating": 1, "title": "Bad", "text": "Broke quickly"},
            {"rating": 2, "title": "Poor", "text": "Not worth the price"},
            {"rating": 3, "title": "Okay", "text": "Average and usable"},
            {"rating": 3, "title": "Fine", "text": "Nothing special"},
            {"rating": 5, "title": "Great", "text": "Excellent quality"},
            {"rating": 4, "title": "Good", "text": "Works well"},
        ],
    )
    output_dir = tmp_path / "artifacts"
    args = build_sentiment_parser().parse_args(
        [
            "--local-csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--test-size",
            "0.5",
            "--min-df",
            "1",
            "--no-save",
        ]
    )

    metadata = train_sentiment_model(args)

    assert "f1_macro" in metadata["metrics"]
    assert not (output_dir / "sentiment_tfidf_logreg.joblib").exists()


def test_sentiment_remote_loader_uses_json_files_without_dataset_script(monkeypatch) -> None:
    calls = []
    rows = [
        {"rating": 5, "title": "Great", "text": "Excellent quality"},
        {"rating": 1, "title": "Bad", "text": "Broke quickly"},
    ]

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return _DatasetStub(rows)

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    args = build_sentiment_parser().parse_args(
        ["--categories", "All_Beauty", "--sample-size", "2"]
    )

    frame, dataset_info = load_sentiment_dataset(args)

    assert calls[0][0] == ("json",)
    assert "trust_remote_code" not in calls[0][1]
    assert calls[0][1]["streaming"] is True
    assert calls[0][1]["data_files"]["full"] == [
        "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/All_Beauty.jsonl"
    ]
    assert dataset_info.row_count == 2
    assert set(frame["label"]) == {"negative", "positive"}


def test_risk_model_training_handles_realistic_local_sources(tmp_path: Path) -> None:
    amazon_csv = _write_csv(
        tmp_path / "amazon_meta.csv",
        [
            {"title": "Premium Headphones", "main_category": "Electronics", "store": "Known Store", "average_rating": 4.7, "rating_number": 500, "price": 100},
            {"title": "Budget Cable", "main_category": "Electronics", "store": "Known Store", "average_rating": 4.4, "rating_number": 80, "price": 105},
            {"title": "Basic Adapter", "main_category": "Electronics", "store": "Mid Store", "average_rating": 3.9, "rating_number": 20, "price": 110},
            {"title": "Unknown Charger", "main_category": "Electronics", "store": "New Store", "average_rating": 2.8, "rating_number": 2, "price": 10},
            {"title": "Luxury Speaker", "main_category": "Electronics", "store": "Known Store", "average_rating": 4.9, "rating_number": 800, "price": 300},
            {"title": "Overpriced Hub", "main_category": "Electronics", "store": "Mid Store", "average_rating": 3.6, "rating_number": 30, "price": 310},
        ],
    )
    policy_csv = _write_csv(
        tmp_path / "target.csv",
        [
            {"shipping_returns_policy": "30-day return refund and warranty available"},
            {"shipping_returns_policy": "14 day return or exchange"},
            {"shipping_returns_policy": "No return information visible"},
            {"shipping_returns_policy": "Refund accepted within 7 days"},
            {"shipping_returns_policy": "Warranty replacement and 30 days return"},
            {"shipping_returns_policy": "Shipping only"},
        ],
    )
    output_dir = tmp_path / "artifacts"
    args = build_risk_parser().parse_args(
        [
            "--amazon-meta-csv",
            str(amazon_csv),
            "--policy-csv",
            str(policy_csv),
            "--output-dir",
            str(output_dir),
            "--test-size",
            "0.34",
            "--n-estimators",
            "10",
            "--min-df",
            "1",
        ]
    )

    metadata = train_risk_models(args)

    assert (output_dir / "seller_reliability_tfidf_rf.joblib").exists()
    assert (output_dir / "price_safety_tfidf_rf.joblib").exists()
    assert (output_dir / "policy_clarity_tfidf_rf.joblib").exists()
    assert "seller_reliability" in metadata["metrics"]
    assert "price_safety" in metadata["metrics"]
    assert "policy_clarity" in metadata["metrics"]


def test_risk_remote_loader_uses_parquet_files_without_dataset_script(monkeypatch) -> None:
    calls = []
    rows = [
        {
            "title": "Premium Headphones",
            "main_category": "Electronics",
            "store": "Known Store",
            "average_rating": 4.7,
            "rating_number": 500,
            "price": 100,
        },
        {
            "title": "Budget Cable",
            "main_category": "Electronics",
            "store": "Known Store",
            "average_rating": 4.4,
            "rating_number": 80,
            "price": 105,
        },
    ]

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return _DatasetStub(rows)

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    args = build_risk_parser().parse_args(
        ["--categories", "All_Beauty", "--sample-size", "2"]
    )

    frame, dataset_info = load_amazon_meta(args)

    assert calls[0][0] == ("parquet",)
    assert "trust_remote_code" not in calls[0][1]
    assert calls[0][1]["streaming"] is True
    assert calls[0][1]["data_files"]["full"] == [
        "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw_meta_All_Beauty/full-*.parquet"
    ]
    assert dataset_info.row_count == 2
    assert set(frame["source_category"]) == {"All_Beauty"}


def test_trustscore_calibrator_skips_without_human_labels() -> None:
    args = build_calibrator_parser().parse_args(["--no-save"])

    result = calibrate_trustscore(args)

    assert result["status"] == "skipped"
    assert "trust_score" in result["required_columns"]


def test_download_datasets_writes_sample_manifest(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "theArijitDas/Fake-Reviews-Dataset":
            return _DatasetStub(
                [
                    {"category": "Home", "rating": 5, "text": "Real review", "label": 0},
                    {"category": "Home", "rating": 5, "text": "Fake review", "label": 1},
                ]
            )
        if args[0] == "json":
            return _DatasetStub(
                [
                    {"rating": 5, "title": "Great", "text": "Excellent quality"},
                    {"rating": 1, "title": "Bad", "text": "Broke quickly"},
                ]
            )
        return _DatasetStub(
            [
                {
                    "title": "Premium Headphones",
                    "main_category": "Electronics",
                    "store": "Known Store",
                    "average_rating": 4.7,
                    "rating_number": 500,
                    "price": 100,
                },
                {
                    "title": "Budget Cable",
                    "main_category": "Electronics",
                    "store": "Known Store",
                    "average_rating": 4.4,
                    "rating_number": 80,
                    "price": 105,
                },
            ]
        )

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda _url: pd.DataFrame(
            [
                {"shipping_returns_policy": "30-day return refund and warranty"},
                {"shipping_returns_policy": "No returns visible"},
            ]
        ),
    )
    args = build_download_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--categories",
            "All_Beauty",
            "--fake-sample-size",
            "2",
            "--amazon-review-sample-size",
            "2",
            "--amazon-meta-sample-size",
            "2",
            "--target-policy-sample-size",
            "2",
        ]
    )

    manifest = download_datasets(args)

    assert (tmp_path / "fake_reviews.csv").exists()
    assert (tmp_path / "amazon_reviews_sample.csv").exists()
    assert (tmp_path / "amazon_meta_sample.csv").exists()
    assert (tmp_path / "target_policy_sample.csv").exists()
    assert (tmp_path / "manifest.json").exists()
    assert len(manifest["datasets"]) == 4
    assert calls


def test_train_all_uses_local_ml_data_samples(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_csv(
        data_dir / "fake_reviews.csv",
        [
            {"category": "Home", "rating": 5, "text": "Excellent quality and fast delivery", "label": 0},
            {"category": "Home", "rating": 4, "text": "Works well after a month of use", "label": 0},
            {"category": "Tech", "rating": 5, "text": "Real purchase and solid build quality", "label": 0},
            {"category": "Tech", "rating": 3, "text": "Average but usable product for price", "label": 0},
            {"category": "Home", "rating": 5, "text": "Great product love it very pretty", "label": 1},
            {"category": "Home", "rating": 5, "text": "Amazing item perfect perfect perfect", "label": 1},
            {"category": "Tech", "rating": 1, "text": "Generated fake review with odd wording", "label": 1},
            {"category": "Tech", "rating": 5, "text": "Best product ever highly recommended wow", "label": 1},
        ],
    )
    _write_csv(
        data_dir / "amazon_reviews_sample.csv",
        [
            {"rating": 1, "title": "Bad", "text": "Broke quickly"},
            {"rating": 2, "title": "Poor", "text": "Not worth the price"},
            {"rating": 3, "title": "Okay", "text": "Average and usable"},
            {"rating": 3, "title": "Fine", "text": "Nothing special"},
            {"rating": 5, "title": "Great", "text": "Excellent quality"},
            {"rating": 4, "title": "Good", "text": "Works well"},
        ],
    )
    _write_csv(
        data_dir / "amazon_meta_sample.csv",
        [
            {"title": "Premium Headphones", "main_category": "Electronics", "store": "Known Store", "average_rating": 4.7, "rating_number": 500, "price": 100},
            {"title": "Budget Cable", "main_category": "Electronics", "store": "Known Store", "average_rating": 4.4, "rating_number": 80, "price": 105},
            {"title": "Basic Adapter", "main_category": "Electronics", "store": "Mid Store", "average_rating": 3.9, "rating_number": 20, "price": 110},
            {"title": "Unknown Charger", "main_category": "Electronics", "store": "New Store", "average_rating": 2.8, "rating_number": 2, "price": 10},
            {"title": "Luxury Speaker", "main_category": "Electronics", "store": "Known Store", "average_rating": 4.9, "rating_number": 800, "price": 300},
            {"title": "Overpriced Hub", "main_category": "Electronics", "store": "Mid Store", "average_rating": 3.6, "rating_number": 30, "price": 310},
        ],
    )
    _write_csv(
        data_dir / "target_policy_sample.csv",
        [
            {"shipping_returns_policy": "30-day return refund and warranty available"},
            {"shipping_returns_policy": "14 day return or exchange"},
            {"shipping_returns_policy": "No return information visible"},
            {"shipping_returns_policy": "Refund accepted within 7 days"},
            {"shipping_returns_policy": "Warranty replacement and 30 days return"},
            {"shipping_returns_policy": "Shipping only"},
        ],
    )
    output_dir = tmp_path / "artifacts"
    args = build_train_all_parser().parse_args(
        [
            "--skip-download",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--test-size",
            "0.5",
            "--fake-sample-size",
            "8",
            "--sentiment-sample-size",
            "6",
            "--risk-sample-size",
            "6",
            "--fake-estimators",
            "10",
            "--risk-estimators",
            "10",
        ]
    )

    summary = train_all(args)

    assert (output_dir / "fake_review" / "fake_review_rf.joblib").exists()
    assert (output_dir / "sentiment" / "sentiment_tfidf_logreg.joblib").exists()
    assert (output_dir / "risk" / "seller_reliability_tfidf_rf.joblib").exists()
    assert (output_dir / "training_summary.json").exists()
    assert summary["trustscore_calibration"]["status"] == "skipped"
