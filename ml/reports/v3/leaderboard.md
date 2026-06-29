# v3 Algorithm Leaderboard — fake review detection

Task: original vs computer-generated review. Train 32420 / test 8106.
Selection metric: ROC-AUC, with deployment cost (size, latency) as tie-breaker.

| Rank | Model | Features | Acc | F1 | ROC-AUC | Train s | Latency µs/row | Size KB |
|---|---|---|---|---|---|---|---|---|
| 1 | calibrated_linear_svc | tfidf | 0.9492 | 0.949 | 0.9889 | 1.03 | 1.4 | 704.8 |
| 2 | lightgbm | tfidf | 0.9419 | 0.9416 | 0.9872 | 36.54 | 7.7 | 2071.1 |
| 3 | logreg | tfidf | 0.9425 | 0.9424 | 0.9854 | 0.56 | 0.5 | 235.1 |
| 4 | xgboost | tfidf | 0.9218 | 0.9212 | 0.98 | 72.53 | 2.5 | 812.7 |
| 5 | random_forest | tfidf | 0.8979 | 0.8985 | 0.9606 | 9.71 | 16.3 | 146632.2 |
| 6 | stacking | embeddings | 0.8373 | 0.8384 | 0.9158 | 43.27 | 7.0 | 150836.7 |
| 7 | voting_soft | embeddings | 0.8346 | 0.8346 | 0.913 | 13.98 | 6.7 | 150832.7 |
| 8 | xgboost | embeddings | 0.8215 | 0.8243 | 0.908 | 4.52 | 0.5 | 1354.3 |
| 9 | lightgbm | embeddings | 0.8253 | 0.828 | 0.9077 | 4.73 | 2.2 | 1072.8 |
| 10 | catboost | embeddings | 0.8154 | 0.819 | 0.8971 | 3.09 | 0.6 | 354.6 |
| 11 | calibrated_linear_svc | embeddings | 0.8187 | 0.8191 | 0.8927 | 3.89 | 0.9 | 10.6 |
| 12 | logreg | embeddings | 0.8067 | 0.8075 | 0.8842 | 0.06 | 0.2 | 2.2 |
| 13 | random_forest | embeddings | 0.7543 | 0.7435 | 0.8332 | 9.89 | 6.2 | 149474.8 |

**Winner:** `calibrated_linear_svc` on `tfidf` — ROC-AUC 0.9889, acc 0.9492, 704.8 KB. Chosen for top AUC at acceptable deployment cost.

