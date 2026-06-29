# Quick Prompt to Give Codex

```text
Implement my project: AI TrustScore Browser Extension for Online Shopping.

Use the docs in this repository as requirements. Build a monorepo with:
- apps/extension: Chrome Extension Manifest V3, React, TypeScript, Vite.
- apps/api: Python FastAPI backend.
- ml/training: fake review model training script.
- db/schema.sql: Supabase PostgreSQL schema.

Main workflow:
Product Page Data -> Preprocessing -> Feature Extraction -> AI/ML Models -> TrustScore Engine -> Risk Level + Explanation -> User Feedback -> Feedback Storage.

Models:
1. Review Sentiment: BERT/DistilBERT, output sentiment score.
2. Fake Review Detection: Random Forest, output fake review probability and review authenticity score.
3. Seller/Price/Policy Risk: rule-based scoring + optional ML, output seller reliability, price safety, and policy clarity.

TrustScore formula:
30% Review Authenticity + 20% Seller Reliability + 20% Sentiment Score + 15% Return Policy Clarity + 10% Price Safety + 5% User Feedback History.

Risk classification:
>=80 Low Risk, >=50 Medium Risk, else High Risk.

Implement backend first, then extension UI, then database logging, then ML training script. Add tests for scoring logic. Keep privacy safe: do not collect passwords, payment data, cookies, or private user data.
```
