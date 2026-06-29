# Model Card — risk_v3_price_anomaly

- **Task:** price-safety anomaly detection
- **Version:** v3 (0.3.0)
- **Generated:** 2026-06-18T08:00:24Z

## Intended use
Flag products whose price is anomalous within its category (possible scam-low or overpriced).

## Training data
v2 Amazon metadata sample (price + category only); unsupervised, no label.

## Metrics
```json
{
  "injected_anomaly_recall": 1.0,
  "normal_false_flag_rate": 0.124
}
```

## Limitations
- Flags statistical outliers, not confirmed fraud.
- Category medians depend on catalogue coverage.
- 66% of metadata rows lack a price; those default to neutral.

## Ethical considerations
- A low score is a caution signal, not proof of wrongdoing.
- Avoids the v2 weak-label leakage by not training on a constructed label.

