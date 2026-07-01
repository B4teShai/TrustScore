import { afterEach, describe, expect, it, vi } from "vitest";

import {
  analyzeProductWithPreviewFallback,
  canonicalProductPageUrl,
  getOrCreateBrowserId,
  getStoredResultForPage,
  normalizeProductPreview,
  productPayloadFromPreview,
  safeImageUrl,
  storeLastResult,
} from "./App";
import type { ProductAnalysisResponse } from "../shared/types";

const result: ProductAnalysisResponse = {
  scan_id: "11111111-1111-4111-8111-111111111111",
  product: {
    url: "https://example.com/product/123",
    site: "example.com",
    product_title: "Wireless Headphones",
  },
  trust_score: 82,
  risk_level: "Low Risk",
  confidence: 0.75,
  component_scores: {
    review_authenticity: 82,
    seller_reliability: 82,
    sentiment: 82,
    return_policy_clarity: 82,
    price_safety: 82,
    user_feedback_history: 50,
  },
  top_reasons: ["Review patterns look mostly natural."],
  evidence: [
    {
      component: "review_authenticity",
      summary: "Review authenticity uses visible review text.",
      evidence: ["2 visible reviews"],
      missing_inputs: [],
      confidence: 0.75,
    },
  ],
  missing_inputs: [],
  score_semantics: "TrustScore is normalized over active non-feedback weights.",
  recommendation: "Signals look healthy.",
  model_version: "0.3.0",
  fetch_mode: "http",
  extraction_signals: ["title", "price"],
  model_modes: { sentiment: "keyword_fallback" },
  model_artifact_status: { sentiment: "missing_or_unavailable" },
  model_versions: { trustscore: "0.1.0" },
  is_mock: false,
};

const apiResult: ProductAnalysisResponse = {
  ...result,
  model_modes: {
    fake_review: "heuristic_fallback",
    sentiment: "keyword_fallback",
    seller_reliability: "rule_fallback",
    price_safety: "not_scored_missing_market_reference",
    return_policy_clarity: "not_scored_missing_policy",
    user_feedback_history: "not_applied",
  },
  model_artifact_status: {
    fake_review: "missing_or_unavailable",
    sentiment: "missing_or_unavailable",
    risk: "missing_or_unavailable",
  },
  model_versions: {
    trustscore: "0.3.0",
    fake_review: "0.3.0",
    sentiment: "0.2.0",
    risk: "0.3.0",
  },
};

describe("popup storage helpers", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("stores and reads the last result for the same page only", async () => {
    installChromeStorage();

    await storeLastResult(result);

    await expect(getStoredResultForPage("https://example.com/product/123")).resolves.toMatchObject({
      scan_id: result.scan_id,
    });
    await expect(getStoredResultForPage("https://example.com/product/other")).resolves.toBeNull();
  });

  it("creates a stable browser id only when requested", async () => {
    installChromeStorage();

    const first = await getOrCreateBrowserId();
    const second = await getOrCreateBrowserId();

    expect(first).toBeTruthy();
    expect(second).toBe(first);
  });

  it("drops local and private product image URLs", () => {
    expect(safeImageUrl("http://127.0.0.1/admin.png")).toBeUndefined();
    expect(safeImageUrl("http://10.0.0.5/product.jpg")).toBeUndefined();
    expect(safeImageUrl("https://shop.localhost/product.jpg")).toBeUndefined();
    expect(safeImageUrl("https://cdn.example.com/product.jpg")).toBe(
      "https://cdn.example.com/product.jpg",
    );
  });

  it("canonicalizes long Amazon product URLs before scanning", () => {
    const url =
      "https://www.amazon.com/Amazon-Essentials-Womens-Standard-Regular/dp/B07RMMDY21/ref=sr_1_1?keywords=jeans&" +
      `tag=${"a".repeat(3000)}`;

    expect(canonicalProductPageUrl(url)).toBe("https://www.amazon.com/dp/B07RMMDY21");
    expect(
      canonicalProductPageUrl(
        "https://www.amazon.co.jp/dp/B0DZC1K3B7?ref=abc&fbclid=very-long",
      ),
    ).toBe("https://www.amazon.co.jp/dp/B0DZC1K3B7");
  });

  it("normalizes active-tab marketplace previews into extracted scan payloads", () => {
    const preview = normalizeProductPreview({
      title: "Soccer Ball Engraved Glasses",
      imageUrl: "https://i.etsystatic.com/123/listing.jpg",
      seller: "GoalGiftShop",
      sellerRating: 4.9,
      sellerReviewCount: 1284,
      price: 24.5,
      currency: "usd",
      rating: 4.8,
      reviewCount: 312,
      returnPolicy: "Returns accepted within 30 days.",
      reviews: [
        {
          text: "Brief content visible, double tap to read full content. Great engraving quality. Read more Read less",
          rating: 5,
        },
      ],
    });

    expect(preview?.currency).toBe("USD");
    expect(preview?.reviews?.[0]?.text).toBe("Great engraving quality.");

    const payload = productPayloadFromPreview({
      host: "www.etsy.com",
      preview: preview ?? undefined,
      targetMarket: "US",
      url: "https://www.etsy.com/listing/1566610967/soccer-ball-engraved-glasses",
    });

    expect(payload).toMatchObject({
      currency: "USD",
      price: 24.5,
      product_title: "Soccer Ball Engraved Glasses",
      review_count: 312,
      seller: { name: "GoalGiftShop", rating: 4.9, review_count: 1284 },
      site: "www.etsy.com",
    });
  });

  it("drops localized marketplace preview prices outside the target market", () => {
    const preview = normalizeProductPreview({
      title: "Natural Burlap Placemats",
      price: 164690.19,
      currency: "MNT",
    });

    const payload = productPayloadFromPreview({
      host: "www.amazon.com",
      preview: preview ?? undefined,
      targetMarket: "US",
      url: "https://www.amazon.com/dp/B0GXVNG3TR",
    });

    expect(payload).toMatchObject({
      product_title: "Natural Burlap Placemats",
      site: "www.amazon.com",
    });
    expect(payload?.price).toBeUndefined();
    expect(payload?.currency).toBeUndefined();
  });

  it("keeps same-market JPY marketplace preview prices", () => {
    const preview = normalizeProductPreview({
      title: "Japanese Snack Box",
      price: 2659,
      currency: "JPY",
    });

    const payload = productPayloadFromPreview({
      host: "www.amazon.co.jp",
      preview: preview ?? undefined,
      targetMarket: "JP",
      url: "https://www.amazon.co.jp/dp/B0TESTJP12",
    });

    expect(payload).toMatchObject({
      currency: "JPY",
      price: 2659,
      product_title: "Japanese Snack Box",
      site: "www.amazon.co.jp",
    });
  });

  it("keeps supported JPY preview prices on Amazon US pages", () => {
    const preview = normalizeProductPreview({
      title: "Straight Leg Jeans for Women",
      seller: "Mars power",
      price: 5682,
      currency: "JPY",
      reviewCount: 1037,
    });

    const payload = productPayloadFromPreview({
      host: "www.amazon.com",
      preview: preview ?? undefined,
      targetMarket: "US",
      url: "https://www.amazon.com/dp/B0TESTUSJP",
    });

    expect(payload).toMatchObject({
      currency: "JPY",
      price: 5682,
      product_title: "Straight Leg Jeans for Women",
      review_count: 1037,
      seller: { name: "Mars power" },
      site: "www.amazon.com",
    });
  });

  it("uses URL scan before active-tab extracted fallback", async () => {
    installChromeStorage();
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, apiResult));
    vi.stubGlobal("fetch", fetchMock);

    await analyzeProductWithPreviewFallback({
      host: "www.amazon.com",
      preview: {
        title: "Wireless Keyboard",
        price: 29.99,
        currency: "USD",
        seller: "Example Store",
      },
      targetMarket: "US",
      url: "https://www.amazon.com/dp/B0TEST1234",
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe("https://walrus-app-38mjb.ondigitalocean.app/api/v1/scan");
  });

  it("falls back to active-tab scan after product detection failure with strong preview", async () => {
    installChromeStorage();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse(422, {
          detail: {
            code: "product_not_detected",
            message: "This page does not have enough product-detail signals.",
          },
        }),
      )
      .mockResolvedValueOnce(jsonResponse(200, { market_reference: { active: true } }))
      .mockResolvedValueOnce(jsonResponse(200, apiResult));
    vi.stubGlobal("fetch", fetchMock);

    await analyzeProductWithPreviewFallback({
      host: "www.amazon.co.jp",
      preview: {
        title: "Magic Trackpad",
        imageUrl: "https://example.com/trackpad.jpg",
        price: 16800,
        currency: "JPY",
        seller: "Apple Store",
      },
      targetMarket: "JP",
      url: "https://www.amazon.co.jp/dp/B0TESTJP14",
    });

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls[2][0]).toBe("https://walrus-app-38mjb.ondigitalocean.app/api/v1/scan-extracted");
  });

  it("does not fall back to extracted scan for weak review-page previews", async () => {
    installChromeStorage();
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse(422, {
        detail: {
          code: "product_not_detected",
          message: "This page does not have enough product-detail signals.",
        },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      analyzeProductWithPreviewFallback({
        host: "www.bestbuy.com",
        preview: {
          title: "Customer Ratings & Reviews",
          reviews: [{ text: "Comfortable mouse after long use." }],
        },
        targetMarket: "US",
        url: "https://www.bestbuy.com/site/reviews/logitech-mouse/6282602",
      }),
    ).rejects.toMatchObject({ code: "product_not_detected" });

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

function installChromeStorage() {
  const store: Record<string, unknown> = {};
  vi.stubGlobal("chrome", {
    runtime: {},
    storage: {
      local: {
        get(key: string, callback: (items: Record<string, unknown>) => void) {
          callback({ [key]: store[key] });
        },
        set(items: Record<string, unknown>, callback?: () => void) {
          Object.assign(store, items);
          callback?.();
        },
      },
    },
  });
}

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
