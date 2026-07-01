# What Changed — slide text for "AI presentation 2.pdf"

The PDF is an image-based deck, so it can't be edited from text here. Below is ready-to-paste
copy for a **"What changed / v2 improvements"** slide (or 2 slides). Pick the short version for a
bullet slide, or the detailed version for speaker notes.

---

## Slide title options

- "What changed in v2 — reliable Amazon scoring"
- "Improvements: Amazon-only, more reviews, stable scores"

---

## Short version (bullet slide)

**Focused on doing one thing correctly.**

- **Amazon-only.** The scraper now targets Amazon product pages only, so it scrapes reliably
  instead of half-working everywhere. Other sites show an "Amazon only" message.
- **In-browser scraping (no bot-block).** Reads the page in the shopper's own logged-in tab
  instead of fetching it server-side, which Amazon blocks. Data is now complete and reliable.
- **Many reviews.** Also pulls Amazon's review pages (up to ~50 reviews), not just the ~8 shown
  on the product page — a bigger, truer sample.
- **Stable score on rescan.** A larger, fixed review sample + deterministic scoring means
  rescanning the same product returns the same score (no more drift).
- **Feedback nudges the score — a little.** A shopper's 👍/👎 is stored per product and applied
  with a small 5% weight; it moves the score slightly, but rescanning alone never does.
- **"What we read" panel.** The popup now shows the actual reviews, seller, price, rating, and
  return policy it collected — so the score is transparent, not a black box.

---

## Detailed version (speaker notes)

**1. Amazon-only, by design.**
A trust score is only useful if the data behind it is complete. A generic scraper produced
missing fields on many sites, which made the score unreliable. We scoped the product to Amazon:
the extension activates only on `amazon.*` pages, and the backend rejects non-Amazon URLs with a
clear `unsupported_site` error. The design still generalizes — another marketplace is a new
extraction profile, not a rewrite.

**2. Client-side DOM scraping instead of server fetch.**
Amazon bot-blocks anonymous server-side requests and returns partial/empty HTML — that was the
root cause of unreliable scores. Now the extension reads the already-rendered DOM in the
shopper's own logged-in tab and sends only the extracted fields to the backend
(`POST /v1/scan-extracted`). The old URL-fetch path is no longer used.

**3. Collect as many reviews as possible.**
Amazon renders only ~8 reviews on a product page. The extension now also fetches the dedicated
review pages (`/product-reviews/<ASIN>`, sorted by "helpful") from the same logged-in tab,
merges and de-duplicates them, up to the 50-review cap.

**4. Deterministic score on rescan.**
Scoring was always deterministic; the drift came from the browser lazy-loading a different set of
reviews each scan. The larger, fixed, helpful-sorted sample stabilizes the input, so the same
product returns the same score.

**5. Feedback affects the score — but only a little.**
The shopper's 👍/👎 is stored per product and replayed on later scans as a 0–100 value applied
with a small **0.05** weight. It nudges the score without overriding the evidence-based signals,
and because a plain rescan carries the same stored vote, feedback moves the score while rescanning
does not.

**6. Transparency — "what we read".**
The result popup now shows the real collected data: the actual review texts (with stars and
verified-purchase badges), seller, listed price, overall rating, total review count, and return
policy. The score is explainable down to the inputs it used.

**7. Rules verified.**
Seller reliability, price safety, and return-policy clarity keep their transparent, deterministic
rules (official-store/platform/rating/review-count/tenure with a popularity fallback; price-vs-
market ratio bands; return/time-window/warranty wording). These were confirmed correct and are
stable across rescans.

---

## One-line summary (for a closing bullet)

> v2 makes the score **reliable and honest**: scrape Amazon correctly in-browser, use as many
> reviews as possible, keep the score stable on rescan, let feedback nudge it slightly, and show
> the shopper exactly what the score was based on.
