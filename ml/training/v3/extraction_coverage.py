"""Phase 5 scaffolding — extraction field-coverage harness (demonstration).

Runs the live backend extractor over HTML fixtures and reports how often each field
populates. A template for running against a real product-page corpus. Read-only; does not
modify the backend.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from ml.training.common import write_json
    from ml.training.v3 import REPORTS_DIR
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from training.common import write_json  # type: ignore
    from training.v3 import REPORTS_DIR  # type: ignore


_FIXTURES: list[tuple[str, str]] = [
    (
        "https://shop.example.com/product/123",
        """<html><head>
        <script type="application/ld+json">{"@type":"Product","name":"Wireless Earbuds",
        "offers":{"@type":"Offer","price":"29.99","priceCurrency":"USD"},
        "aggregateRating":{"ratingValue":"4.3","reviewCount":"128"}}</script></head>
        <body><h1>Wireless Earbuds</h1><p>30-day return and refund policy with warranty.</p>
        <div class="review">Great sound quality and battery life.</div></body></html>""",
    ),
    (
        "https://shop.example.com/item/777",
        """<html><body><h1>Mystery Gadget</h1>
        <span>Price: $8.00</span><span>2 ratings</span></body></html>""",
    ),
    (
        "https://example.com/blog/not-a-product",
        "<html><body><h1>How to shop safely</h1><p>Some article text.</p></body></html>",
    ),
]


def run(args: argparse.Namespace) -> dict[str, Any]:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3] / "apps" / "api"))
    from app.extraction.product_page import extract_product_page

    fields = ["product_title", "price", "currency", "seller", "return_policy", "reviews", "rating", "review_count"]
    counts = {f: 0 for f in fields}
    detected = 0
    per_fixture = []
    for url, html in _FIXTURES:
        res = extract_product_page(html, url)
        rec: dict[str, Any] = {"url": url, "detected": res.detected}
        if res.detected and res.product is not None:
            detected += 1
            p = res.product
            present = {
                "product_title": bool(p.product_title), "price": p.price is not None,
                "currency": bool(p.currency), "seller": p.seller is not None,
                "return_policy": bool(p.return_policy), "reviews": len(p.reviews) > 0,
                "rating": p.rating is not None, "review_count": p.review_count is not None,
            }
            for f, ok in present.items():
                counts[f] += int(ok)
            rec["present_fields"] = [f for f, ok in present.items() if ok]
        per_fixture.append(rec)

    coverage = {f: round(c / max(detected, 1), 3) for f, c in counts.items()}
    report = {"n_fixtures": len(_FIXTURES), "n_detected_products": detected,
              "field_coverage_over_detected": coverage, "per_fixture": per_fixture,
              "note": "Fixtures are illustrative; point this at a real product-page corpus for production metrics."}
    write_json(REPORTS_DIR / "extraction_coverage.json", report)
    print(f"[extraction] coverage -> {REPORTS_DIR / 'extraction_coverage.json'}: {coverage}")
    return report


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


if __name__ == "__main__":
    run(build_parser().parse_args())
