# Q&A Prep — AI TrustScore, Part 2

Likely questions from the critique of the slide deck, with ready answers. Read this once before
presenting; you don't need to memorize it word for word.

---

### Q: "Slide 3 shows a loss curve over 20 epochs with binary cross-entropy — but you said the
### model is a LinearSVC. SVMs don't train via epochs of cross-entropy loss. What is this chart
### actually showing?"

**Honest answer, don't bluff this one:**

"The production fake-review model is a LinearSVC — trained via a convex hinge-loss optimizer
(scikit-learn's liblinear solver) to convergence, not via epoch-by-epoch gradient descent on a
cross-entropy loss the way a neural network trains. The chart on that slide illustrates general
training-convergence behavior for the presentation rather than a literal per-epoch log from the
LinearSVC itself, since the SVM doesn't produce that kind of curve.

What the LinearSVC *does* go through that this metric-set actually reflects is **probability
calibration**: after fitting the SVM, we wrap it in `CalibratedClassifierCV` (Platt/sigmoid
scaling) so its raw decision-function scores become usable probabilities — and that calibration
step is what the ECE 0.007 number on the results slide is actually measuring."

If asked to produce the literal training log: be upfront that it isn't a plotted log of that
exact run — offer to show the ROC curve and calibration (reliability) diagram instead, since
those *are* real, measured outputs.

---

### Q: "What are the actual weights in the weighted scoring formula, and why those numbers?"

**Answer:**

| Signal | Weight |
|---|---|
| Review authenticity | 30% |
| Sentiment | 20% |
| Seller reliability | 20% |
| Return-policy clarity | 15% |
| Price safety | 10% |
| User feedback history | 0% (allocated 5%, not active — no real feedback data collected yet) |

Rationale, if pressed:
- **Review authenticity gets the highest weight (30%)** because fake reviews are the most
  direct way a listing can be manipulated, and it's also the most confidently validated signal
  (0.9889 ROC-AUC, tested cross-category).
- **Price safety gets the lowest active weight (10%)**, despite intuitively mattering a lot,
  because it's the least certain signal: it's an unsupervised anomaly heuristic (not a
  calibrated probability like the ML models), and it depends on a live market-price lookup
  that isn't always available for every listing.
- **User feedback history is weighted 0%** on purpose — it's a placeholder for a future
  feedback loop; there's no real user feedback data yet, so it isn't allowed to silently
  influence the score.
- These weights are a **hand-set design choice**, not learned from data — there's no public
  dataset with a real "trust score" label to fit them against. Say this plainly if asked; it's
  consistent with the "no fabricated ground truth" theme of the whole project.

---

### Q: "The price-safety model gets 1.0 recall — that sounds too good. Is that real-world
### validated?"

**Answer, be upfront:**

"That recall is measured on **injected synthetic price anomalies** — we don't have a
real-world, human-labeled dataset of 'this price was actually a scam' to test against, so we
inject artificial outliers into real price data and check the detector catches them. It's a
sanity check that the anomaly detector works as intended, not a real-world audited benchmark.
Collecting real labeled ground truth for this is explicitly called out as future work."

---

### Q: "Is it 0.989 or 0.9889 ROC-AUC? Your slides use both."

**Answer:** "0.9889 — some places round it to three decimals for readability, same number."
(Worth fixing in the deck itself before presenting if there's time — pick one and use it
everywhere.)

---

### Q: "705KB and 'low latency' compared to what, exactly?"

**Answer, frame it as a comparison rather than an absolute claim:**

"The comparison is against the alternative we actually tried and rejected: frozen transformer
embeddings for the same fake-review task. A transformer encoder like DistilBERT is on the order
of 250MB+ on disk and needs materially more compute per prediction than a sparse TF-IDF vector
plus a linear classifier, which runs in well under a millisecond on CPU. We didn't do a formal
latency benchmark with exact millisecond figures, so if asked for a precise number, say so
rather than inventing one — the size comparison (705KB vs. hundreds of MB) is the defensible,
verifiable claim."

---

### Q: "Slide 5's headline is about the risk-model leakage, but the chart is about the sentiment
### model. What's the connection?"

**Answer:** "They're two separate rigor checks presented together, not one causal chain: the
risk-model leakage discovery is one example of testing a result before trusting it, and the
sentiment cross-domain test (Yelp/IMDb/SST-2) is a second, independent example — checking that
the sentiment model wasn't just memorizing Amazon-specific language. Both support the same
theme (don't trust a number until you've tried to break it), but they're different models and
different failure modes."

---

### Quick reference — numbers to have on hand

- Review authenticity: **0.9889 ROC-AUC**, 0.91 mean cross-category OOD, baseline was 0.961.
- Sentiment: **86.7% accuracy / 0.9647 ROC-AUC** in-domain; OOD — Yelp 0.89/0.96, IMDb 0.86/0.94,
  SST-2 0.72/0.80.
- Seller/price risk: v1/v2 baseline 0.58–0.60 → looked like ~1.0 with leakage → rejected, now a
  transparent rule + IsolationForest (1.0 recall on injected anomalies).
- Calibration: **ECE 0.007** after isotonic/Platt calibration (was 0.078 before).
- Datasets: Fake-Reviews ~40k rows, Amazon Reviews 2023 ~450k rows, Amazon metadata ~180k rows,
  Yelp/IMDb/SST-2 ~192k rows combined for OOD testing.
