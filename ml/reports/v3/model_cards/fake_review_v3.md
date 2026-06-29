# Model Card — fake_review_v3

- **Task:** fake review detection (original vs generated)
- **Version:** v3 (0.3.0)
- **Generated:** 2026-06-18T07:44:10Z

## Intended use
Estimate the probability a product review is machine-generated; one of five TrustScore signals.

## Training data
theArijitDas/Fake-Reviews-Dataset (balanced ~40k). Independent labels (not weak labels).

## Metrics
```json
{
  "accuracy": 0.9492,
  "artifact_size_kb": 704.8,
  "f1": 0.949,
  "feature_set": "tfidf",
  "infer_latency_ms_per_1k": 1.4,
  "model": "calibrated_linear_svc",
  "roc_auc": 0.9889,
  "train_time_s": 1.03
}
```

## Limitations
- Distinguishes generated vs original text, not all fraud (e.g. paid genuine-looking reviews).
- Does not detect duplicate review spam; pair with the heuristic.
- Cross-category generalisation reported separately (see evaluation_report.md).

## Ethical considerations
- False positives can unfairly flag honest sellers; surface as a signal, not a verdict.
- English-only; non-English reviews are out of scope.

