# AI TrustScore — 5-Slide Talk Script (≈5 minutes)

Deck: `presentation-2-final.html` (open in browser; press **P** for Marp presenter mode to see
these notes beside each slide). Target ~5:00 total. Keep one slide ≈ one minute.

---

## Slide 1 — Title (≈30s)
"Hi, I'm Yesui. My project is **AI TrustScore** — a browser extension that looks at any online
product page and gives shoppers a single, explainable **trust score from 0 to 100**, plus a risk
level and short reasons. Online stores are full of fake reviews, shady sellers and misleading
prices, and shoppers don't have time to check all of it. TrustScore does that automatically. In
five minutes I'll cover how it's built, the results, and how I stress-tested my own results to
keep them honest."

## Slide 2 — What it is & how it works (≈60s)
"Here's the system. A **Chrome extension** reads the product page — title, price, seller, reviews,
return policy. It sends that to a **FastAPI backend**, which runs **five ML signals**: is the
review text genuine or generated, what's the sentiment, is the seller reliable, is the price safe,
is the return policy clear. Each is a 0–100 score; we combine them with a weighted formula into the
final **TrustScore** and a Low/Medium/High risk level, with short reasons."

## Slide 3 — Model Performance & Convergence (≈45s)
"Before looking at datasets, let's see how the model behaves. On the left, our training convergence
shows binary cross-entropy loss decreasing steadily, indicating a stable learning process without
overfitting. On the right, our ROC curve demonstrates exceptional discriminatory power with a
**0.9889 AUC**. This means the model is highly effective at distinguishing between genuine customer
feedback and synthetic, machine-generated reviews."

## Slide 4 — Datasets & results (≈60s)
"I trained on public data: a 40-thousand **fake-review** set, 450-thousand **Amazon reviews** for
sentiment, 180-thousand **product-metadata** rows for risk. In our evaluation, the winner was a
**calibrated LinearSVC on TF-IDF**: it achieved top accuracy while remaining extremely lightweight
at only **705 KB**, making it very efficient for production deployment."

## Slide 5 — The research insight: honesty over hype (≈75s)
"The core of our scientific rigor was catching a major bug. In the experimental phase, our risk
models showed near 1.00 accuracy. We investigated and found **Data Leakage**: the labels were
accidentally derived from the input features. Instead of reporting a 'perfect' but fake number,
we rejected that model. We moved to an **unsupervised anomaly detector** for price and transparent
heuristics for sellers. As you can see in the chart, we also tested **domain shift** — how the model
performs on entirely different websites like Yelp and IMDb — which gives us a much more honest
reflection of real-world performance."

## Slide 6 — Conclusions & next steps (≈60s)
"To wrap up: we've built a fully functional, reproducible end-to-end system. The headline result is a
fake-review detector with **0.989 ROC-AUC** that holds up under rigorous stress-testing. Key lesson:
scientific honesty beats hype every time. Future work will focus on human-labeled ground truth and
more advanced transformer models like DistilBERT. **Thank you — happy to take questions.**"

---

## Timing cheat-sheet
| Slide | Topic | Time |
|---|---|---|
| 1 | Title / problem | 0:30 |
| 2 | System & implementation | 1:00 |
| 3 | Performance (Loss/AUC) | 0:45 |
| 4 | Datasets & results | 1:00 |
| 5 | Research honesty (leakage, OOD) | 1:15 |
| 6 | Conclusions & 4 questions | 1:00 |
| | **Total** | **≈5:30** |

## Likely Q&A
- *Why did TF-IDF beat transformers?* Generated-vs-original detection relies on surface lexical
  artifacts (word/char n-grams); a frozen sentence-embedding normalises those away. A fine-tuned
  transformer might win but needs a GPU.
- *How did you catch the leakage?* Trained a model on the numeric features alone and on a held-out
  category — 1.00 both times means the label is a deterministic function of those features.
- *Is the risk score useless now?* No — we keep transparent rules plus an **unsupervised price-anomaly**
  signal that doesn't depend on the leaked label.
