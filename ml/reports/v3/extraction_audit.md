# v3 Phase 5 — Data Extraction Audit (proposal)

Scope decision: extraction is delivered as an **audited proposal + scaffolding**, not a
backend rewrite. This documents the current pipeline, gaps, and concrete improvements.

## Current pipeline

- `apps/api/app/page_fetching/fetcher.py` — URL validation (SSRF guard via
  `validate_public_web_url`), static fetch, optional rendered fetch (Playwright, gated by
  `ENABLE_RENDERED_FETCH`).
- `apps/api/app/extraction/product_page.py` — `extract_product_page(html, url)`:
  1. Parse with BeautifulSoup; collect **JSON-LD** `Product` objects (`_json_ld_products`).
  2. Title from JSON-LD `name` else heuristic `_find_title`.
  3. Price/currency via JSON-LD then `PRICE_RE`; rating/review-count via JSON-LD then regex
     (`RATING_RE`, `REVIEW_COUNT_RE`); seller, description, image, reviews, return policy.
  4. `_product_signals` + `_is_product_detail` decide `detected`.
- The browser extension provides a DOM-extracted fallback (`/scan-extracted`).

## Strengths
- JSON-LD-first with regex fallback is robust across many e-commerce templates.
- Detection gate avoids scoring non-product pages.
- SSRF validation before fetch.

## Gaps / risks
1. **No structured coverage metric** — there is no measurement of how often each field
   (price, seller, reviews, policy) is successfully extracted.
2. **Currency normalisation** — `PRICE_RE` matches `$ USD MNT ₮ € £` but downstream treats
   price as a bare float; mixed-currency comparison to `average_market_price` is unguarded.
3. **Missing-field handling is implicit** — fields silently become `None`; the response does
   not distinguish "absent on page" from "extraction failed".
4. **No validation of extracted ranges** — e.g. rating must be 0–5, review_count ≥ 0,
   price > 0; malformed JSON-LD values can pass through.
5. **`average_market_price` is always `None`** server-side — the price-safety signal depends
   on it but extraction never sources it (only the extension may pass it).
6. **No retry/error-recovery taxonomy** — a fetch/parse failure returns a generic reason.

## Proposed improvements
1. **Coverage metrics**: emit per-field extraction coverage (see `extraction_coverage.py`)
   and log it; expose counts in `/model-info`.
2. **Validation layer**: a `validate_extracted(product)` that clamps/normalises rating∈[0,5],
   review_count≥0, price>0, currency∈ISO set; demote out-of-range values to `None` with a
   recorded `extraction_warnings` list.
3. **Currency normalisation**: map symbols→ISO codes; only compute price ratios within the
   same currency, else skip price-safety with a clear reason.
4. **Explicit missing-field signals**: add `missing_fields: list[str]` to the response so the
   UI/score can down-weight low-coverage scans (already partly done via `completeness`).
5. **Error recovery**: classify failures (network, blocked, parse, no-product) and return a
   typed reason so the extension can choose the DOM fallback intelligently.
6. **Market price sourcing**: optionally derive `average_market_price` from category medians
   computed offline (reusing the v2 metadata pipeline) so price-safety has a server-side basis.

## Scaffolding delivered
- `ml/training/v3/extraction_coverage.py` — runs `extract_product_page` over HTML fixtures
  and reports field-population coverage (a template for a real coverage harness on a page corpus).
