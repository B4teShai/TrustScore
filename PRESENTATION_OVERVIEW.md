# AI TrustScore — Project Overview (for presentation)

This document explains the project for someone presenting it, not running it. It covers what
the product does, why it's built the way it is, what data trains it, and how it works when a
real user uses it.

## 1. What TrustScore is

Online shoppers can't easily tell whether a product listing is trustworthy: reviews can be
faked, sellers can be unreliable, and prices can be inflated relative to the real market.
**AI TrustScore** automates that judgment call.

It's a **Chrome extension + FastAPI backend**, focused on **Amazon product pages only** so it
can scrape reliably and score correctly rather than half-working on every site. A shopper opens
an Amazon product page, clicks the extension, and gets back:

- A **trust score (0–100)**
- A **risk level** — Low / Medium / High
- Plain-language **reasons** explaining the score

The score is a weighted combination of **five independent signals**, each produced by its own
model or rule, so the result is explainable rather than a black box.

## 2. High-level pipeline

The DOM scraping happens **client-side**, inside the extension, reading the already-rendered
Amazon page in the shopper's own logged-in tab. That is deliberate: Amazon bot-blocks anonymous
server-side fetches and returns partial/empty HTML, so scraping in the browser is what makes the
data — and therefore the score — reliable. Only the extracted fields are sent to the backend.

```
 Shopper lands on an Amazon product page
              │
              ▼
      Chrome Extension — React popup
              │
              ▼
      In-tab DOM extractor (chrome.scripting)
      reads: title, price, seller, reviews, return policy, rating, review count
      + fetches Amazon review pages (/product-reviews/<ASIN>) for many reviews
              │  sends extracted fields (no raw HTML, no URL-only fetch)
              ▼
      FastAPI backend — Inference API   (POST /v1/scan-extracted, Amazon-only)
              │
      ┌───────┴────────────────────┐
      ▼                            ▼
  ML Predictors                Weighted Engine
  (fake-review, sentiment,          │
   price-anomaly models)            ▼
      │                     TrustScore (0–100) + risk level
      │                            │
      ▼                            ▼
  PostgreSQL — scan history   Response → UI popup
  (persisted)                 (score, risk, reasons, evidence, "what we read")
```

Details the diagram doesn't show, but are real in the code, worth knowing for Q&A:
- **Amazon-only, in-browser scraping.** The extension runs only on `amazon.*` pages; other
  sites show an "Amazon only" message. The backend also rejects non-Amazon URLs
  (`422 unsupported_site`). The old server-side URL-fetch path (`POST /v1/scan`) is no longer
  used by the extension because Amazon bot-blocks it.
- **Many reviews.** Amazon renders only ~8 reviews on the product page, so the extension also
  fetches the dedicated review pages (`sortBy=helpful`, up to ~50 reviews) via a same-origin
  request from the logged-in tab, then merges and de-duplicates. A larger, fixed review sample
  is also why **rescanning the same product returns the same score** instead of drifting.
- **"What we read" panel.** The popup shows the actual data it collected — the real review
  texts, seller, price, rating, review count, return policy — so the score is transparent.
- **Price safety** is enriched with a live market-price lookup (Serper shopping search +
  currency conversion) so the scorer compares the listing against a real reference price, not
  just the listing itself.
- If an Anthropic API key is configured, **Claude** rewrites the final explanation in the
  shopper's own page language before it's returned — the score itself never changes, only the
  wording of the explanation.

## 3. Algorithm — how the TrustScore is produced

The important algorithm idea is: **TrustScore is not one end-to-end model**. It is an
explainable ensemble-style scoring algorithm. The backend converts one product page into
several 0-100 component scores, keeps only the components that have enough evidence, then
normalizes a weighted average into the final public score.

### 3a. Inputs

For one scan, the backend works with structured product-page fields:

- Product identity: URL, site, title, image, description.
- Price context: listed price, listed currency, target market, average market reference price,
  number of comparable listings, and optional exchange-rate conversion evidence.
- Seller context: seller name, official-store/platform-seller flags, marketplace fulfillment,
  seller rating, seller review count, and seller tenure when visible.
- Review context: up to 50 review samples (from the product page plus the Amazon review pages)
  with review text, rating, date, and verified-purchase flag.
- Policy context: visible return/refund/warranty wording.

The page text is cleaned before scoring: HTML and marketplace UI boilerplate are removed,
whitespace is normalized, and review text is lowercased. From the cleaned reviews, the backend
derives product-level pattern features such as duplicate review rate, unique review count,
short five-star review rate, negative keyword rate, extreme-rating ratio, verified-purchase
ratio, and rating/sentiment mismatch rate.

### 3b. Component scoring

Every component returns a **score from 0 to 100**, where higher means safer or more trustworthy.

1. **Review authenticity**

   Preferred path: each unique cleaned review is transformed with the trained word/character
   TF-IDF vectorizer, the calibrated LinearSVC model estimates fake-review probability, and the
   product-level fake probability is the average across visible reviews.

   ```
   fake_probability = average(model_probability_fake(review_i))
   review_authenticity = round(100 * (1 - fake_probability))
   ```

   Fallback path if the artifact is unavailable:

   ```
   suspicious =
       duplicate_review_rate * 0.32
     + short_five_star_rate * 0.24
     + rating_sentiment_mismatch_rate * 0.18
     + negative_keyword_rate * 0.14
     + extreme_rating_ratio * 0.08
     + (1 - verified_purchase_ratio) * 0.04

   review_authenticity = round(100 * (1 - suspicious))
   ```

   If no reviews are visible, this component returns a neutral 50 with low confidence.

2. **Sentiment**

   Preferred path: the TF-IDF + Logistic Regression sentiment artifact predicts
   negative/neutral/positive probabilities for the first visible reviews. The classes are
   converted into numeric trust values and averaged:

   ```
   negative = 20, neutral = 55, positive = 90
   sentiment = round(sum(class_score * class_probability))
   ```

   Runtime fallbacks are, in order: local DistilBERT sentiment pipeline if available, then a
   keyword/rating rule, then product rating only. The score still stays on the same 0-100 scale.

3. **Seller reliability**

   v3 deliberately uses a transparent rule instead of the rejected leaky supervised classifier.
   It combines direct seller identity signals and marketplace popularity signals:

   - Official brand store or platform seller: strong positive seller identity signal.
   - Platform-fulfilled listing: moderate positive signal.
   - Seller rating: mapped from 0-5 stars to 0-100.
   - Seller review count: log-scaled so reputation volume helps but eventually saturates.
   - Seller active years: capped around five years.
   - If direct seller reputation is missing, marketplace popularity (rating volume, recent
     purchases, and average rating) is used as a tempered prior, not as a full replacement for
     seller identity.

4. **Return-policy clarity**

   This is also a transparent rule. A missing policy is not scored. If policy text is visible,
   the score starts low and receives points for concrete buyer-protection language:

   ```
   score = 30
   +25 if return/refund wording is present
   +25 if a time window is present, such as 30 days
   +10 if warranty/exchange/replacement wording is present
   +10 if the policy has enough wording to be meaningful
   ```

5. **Price safety**

   Price safety is active only when both the listed price and a verified market reference price
   are available. The v3 production set includes an unsupervised price-anomaly artifact, and the
   served score path uses a leakage-safe market-ratio rule:

   ```
   price_ratio = listed_price / average_market_price

   if price_ratio < 0.35: price_safety = 25   # extremely low, suspicious
   elif price_ratio < 0.60: price_safety = 45 # unusually low
   elif price_ratio <= 1.40: price_safety = 90 # normal market range
   elif price_ratio <= 2.00: price_safety = 70 # high but not extreme
   else: price_safety = 50                    # too high to trust automatically
   ```

   If the market reference cannot be verified, the score trace says price was not scored instead
   of pretending the listing is safe or unsafe.

### 3c. Active components and final formula

The backend does not blindly average missing evidence. It first decides which components are
active:

- Always active when a product page is accepted: review authenticity, seller reliability, and
  sentiment. If reviews or seller data are sparse, those components expose low confidence and
  missing inputs.
- Price safety becomes active only when both `price` and `average_market_price` exist.
- Return-policy clarity becomes active only when return-policy text is visible.
- User feedback history becomes active only when the shopper has previously voted 👍/👎 on that
  product. The vote is stored per product and replayed on later scans as a 0–100 feedback value,
  applied with a small weight (**0.05**). With no prior vote it stays inert and does not affect
  the score. Because the stored vote does not change on a plain rescan, feedback moves the score
  a little but rescanning does not.

Then the final score is:

```
TrustScore = round(
  sum(component_score_i * weight_i for active components)
  / sum(weight_i for active components)
)
```

Default weights:

| Component | Weight |
|---|---:|
| Review authenticity | 0.30 |
| Seller reliability | 0.20 |
| Sentiment | 0.20 |
| Return-policy clarity | 0.15 |
| Price safety | 0.10 |
| User feedback history | 0.05 (active only when the shopper has voted on that product) |

Because the formula divides by the sum of **active** weights, missing price or policy evidence
does not automatically punish or reward a product. Instead, the response separately lists
missing inputs and confidence.

Risk level is a simple public band over the final score:

```
80-100  -> Low Risk
50-79   -> Medium Risk
0-49    -> High Risk
```

### 3d. Confidence, reasons, and audit trace

TrustScore returns more than a number. The response includes:

- **Confidence**, calculated from review volume, fake-review model certainty, sentiment certainty,
  seller completeness, price completeness, and policy completeness. Low evidence or weak product
  identity marks the scan as `low_evidence_triage`.
- **Top reasons**, selected from the three weakest active component scores.
- **Evidence**, such as duplicate review rate, seller name, seller rating, recent demand, listed
  price, market reference price, and return-policy snippet.
- **Model mode**, showing whether each component used a trained artifact, transparent v3 rule,
  fallback heuristic, or was not scored because evidence was missing.

This makes the algorithm explainable in a presentation: the system can say not only "this
listing scored 68," but also which signals created that number and which inputs were missing.

## 4. The production signals — why each exists and how it's computed

| Signal | Weight | How it's computed | Reported metric |
|---|---|---|---|
| Review authenticity | 30% | Calibrated **LinearSVC** classifier over word + character **TF-IDF** features of review text | **0.9889 ROC-AUC** (baseline was 0.961) |
| Sentiment | 20% | **TF-IDF + Logistic Regression**, with a transformer (DistilBERT) fallback, then a keyword-rule fallback | **86.7% accuracy / 0.9647 ROC-AUC** (baseline was 86.4% accuracy) |
| Seller reliability | 20% | **Transparent rule** on rating, review count, and active years (tenure) | Deterministic — no accuracy metric, by design (see leakage story below) |
| Return-policy clarity | 15% | **Transparent rule** matching return/warranty/time-window wording | Deterministic — same reasoning as seller reliability |
| Price safety | 10% | v3 leakage-safe market-price ratio rule, backed by an unsupervised **IsolationForest + ratio-rule** production artifact and live market-price lookup | **1.0 recall** on injected price anomalies |
| User feedback history | 5% (only when the shopper voted on that product) | Shopper's stored 👍/👎 for the product, replayed as a 0–100 value and applied with a small weight | Deterministic — a small, bounded nudge, not a trained metric |

The winning review-authenticity model (TF-IDF + LinearSVC) was chosen partly for being
**lightweight**: ~705KB on disk and low inference latency, versus a much heavier transformer
model for a similar or worse result.

## 5. Why the pipeline looks like this — scientific rigor over raw accuracy

This is the most useful part of the project to present, because it's three real examples of
checking a model's results before trusting them, instead of just reporting the best number.

### 5a. The leakage story (v1 → v2 → v3)

- **v1** — baseline models using text-only features for every signal. Seller/price risk
  accuracy was weak: **0.58 / 0.60**.
- **v2** — added numeric metadata (seller rating, review count, price ratio) to the seller and
  price risk models. Accuracy jumped to near **1.0**. On the surface, this looked like a huge
  win.
- **v3** — the team didn't just trust the improved accuracy; they tested it, and found
  **label leakage**: the numeric features could reconstruct the training labels almost
  perfectly, even when tested on product categories the model had never seen. That means the
  model had learned the *rule used to create the labels*, not anything about real trust — a
  classic ML trap that inflates offline metrics while teaching the model nothing useful.

  The team **rejected the leaky models** rather than ship a fake accuracy number, and replaced
  seller/price risk scoring with a transparent rule plus an unsupervised anomaly detector
  instead — the "Robust Ruleset" shown in the results.

### 5b. Out-of-domain generalization check (sentiment model)

To make sure the sentiment model wasn't just memorizing Amazon-specific language, it was
tested on datasets it never trained on:

| Test set | In/out of domain | Accuracy | ROC-AUC |
|---|---|---|---|
| Amazon (held-out) | in-domain | 0.90 | 0.96 |
| Yelp Polarity | out-of-domain | 0.89 | 0.96 |
| IMDb | out-of-domain | 0.86 | 0.94 |
| SST-2 | out-of-domain | 0.72 | 0.80 |

Performance drops on SST-2 (short, terse movie-review snippets) — an honestly reported
weak spot, not hidden from the results.

### 5c. Calibration check

A model can be accurate but still overconfident or underconfident in its predicted
probabilities. The fake-review classifier's probabilities were checked against a reliability
diagram and calibrated, bringing **Expected Calibration Error (ECE) down to 0.007** — close to
perfectly calibrated, meaning a "90% fake" prediction is actually right about 90% of the time.

### A second finding worth mentioning

For fake-review detection specifically, a simple **TF-IDF** model outperformed frozen
transformer embeddings — a counter-intuitive result that was reported honestly rather than
discarded in favor of a "fancier" model.

**Takeaway for the presentation:** good AI development isn't just "add more features, get a
better number" — it's stress-testing *why* the number improved, checking it generalizes beyond
the training distribution, and checking the model's confidence is trustworthy, before shipping it.

## 6. Data — what trains this, and what it uses live

| Dataset | Rows | Used for |
|---|---|---|
| Fake-Reviews-Dataset (`theArijitDas/Fake-Reviews-Dataset`, Hugging Face) | ~40k | Review-authenticity model |
| Amazon Reviews 2023 (`McAuley-Lab/Amazon-Reviews-2023`) — review text + star rating | ~450k | Sentiment model, using star rating as a weak label (1–2★ = negative, 3★ = neutral, 4–5★ = positive) |
| Amazon Reviews 2023 — product metadata (parquet: store, rating, price, category — only 9 categories ship metadata parquet) | ~180k | Price-anomaly / risk signal |
| Yelp Polarity / IMDb / SST-2 | ~192k combined | Out-of-domain testing of the sentiment model (section 5b) |
| Target return-policy sample | ~1k | Return-policy-clarity model (trained, currently served by a rule instead) |

At runtime, two more data sources feed a single scan:

- The **actual product page** being scored, scraped live (title, price, seller info, reviews,
  return policy).
- **Serper** shopping search for comparable market prices, plus a currency-conversion lookup
  so prices in different currencies compare correctly.

**Important nuance:** there is no public dataset with a real "trust score" label. The final
0–100 score is **not** learned end-to-end — it's a hand-set weighted formula applied to the
active submodel and rule outputs described above.

## 7. Training, briefly

- Training code lives in `ml/training/`. Each iteration (v1, v2, v3) is fully isolated —
  nothing in v3 touches v1/v2 code or artifacts.
- Trained model files are versioned under `ml/artifacts/{v1,v2,v3}/`, plus one curated
  `ml/artifacts/best/` set that picks the single winning model per signal.
- `best_manifest.json` records exactly which model won each signal, its accuracy metric, which
  version it came from, and a `leakage_safe` flag — an explicit, auditable record of the
  decision described in section 5.
- Training is **optional** for a demo: the repository ships with the pre-trained artifacts
  already committed, so the backend runs immediately without retraining anything.

## 8. How it's used at runtime

1. Shopper clicks the extension on an **Amazon** product page. The extension reads the rendered
   DOM in the current tab and also fetches the Amazon review pages for more reviews, then sends
   the extracted fields to the backend (`POST /v1/scan-extracted`). Non-Amazon pages are blocked
   client-side and by the backend (`422 unsupported_site`).
2. Extracted structured data — title, price, seller, up to 50 reviews, return policy, rating,
   review count — is sanitized on the backend.
3. All component scorers run — the ML models score what they can; if a model artifact
   is missing, that signal falls back to a heuristic, and the response says which happened.
4. Price safety is enriched with a live market-price lookup and currency conversion.
5. The weighted formula combines the active component scores into the final 0–100 trust score
   and Low/Medium/High risk level. If the shopper has a stored 👍/👎 for this product, it is
   applied as a small (0.05-weight) nudge; a plain rescan with no new vote returns the same score.
6. If an Anthropic API key is configured, **Claude** writes a short, plain-language
   explanation of the result, in the shopper's own page language (currently English and
   Japanese are the primary targets) — otherwise a rule-based English explanation is used. The
   score itself is identical either way; only the wording of the explanation changes.
7. The full result — score, risk level, component scores, reasons, evidence, and exactly which
   model/fallback produced each number — is returned to the extension and saved (to Postgres if
   configured, otherwise to a local file), so every scan is auditable after the fact. The popup
   also shows a **"what we read"** panel with the actual reviews, seller, price, and policy it
   collected, so the shopper can see what the score is based on.

## 9. Deployment

The whole stack (Postgres, API, extension dev server, and a trainer step that assembles the
best model set) runs from a single Docker Compose file. The hosted API is deployed to
**DigitalOcean**, and the production extension build points at that hosted API.

## 10. Conclusion & future work

| | |
|---|---|
| **What?** | Explainable shopping trust score |
| **Implemented?** | 5-signal ML pipeline + rigorous validation stages |
| **Result?** | 0.9889 ROC-AUC on review authenticity; leakage found & fixed; calibrated probabilities (ECE 0.007) |
| **Develop further?** | Real human-labeled trust-score data, transformer fine-tuning, public release |

Future work called out for this project:

- **Collect human-labeled ground truth** for the seller/price risk signals — the current rules
  are transparent and leakage-safe, but a real labeled dataset would let a proper model replace
  them.
- **Deploy DistilBERT** for more nuanced sentiment analysis (the current best model deliberately
  chose TF-IDF over transformers for the fake-review signal, but sentiment may benefit
  differently — this is future work, not yet done).

## 11. Where to look for more

- `ml/README.md` — full training instructions and the v1/v2/v3 write-up
- `ml/reports/v3/FINDINGS.md` — the leakage investigation in detail
- `docs/SYSTEM_ARCHITECTURE.md` — full technical architecture
- `presentation/presentation-2/` — existing slides
