import { afterEach, describe, expect, it, vi } from "vitest";

import {
  amazonLocalePathPrefix,
  analyzeProductWithPreviewFallback,
  canonicalProductPageUrl,
  getOrCreateBrowserId,
  getStoredResultForPage,
  normalizeProductPreview,
  productPayloadFromPreview,
  safeImageUrl,
  storeLastResult,
} from "./App";
import { resetApiClientCompatibilityCacheForTests } from "../shared/apiClient";
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
    resetApiClientCompatibilityCacheForTests();
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

  it("cleans breadcrumb category paths and maps them into the payload", () => {
    const preview = normalizeProductPreview({
      title: "MIULEE Blackout Linen Textured Curtains",
      categoryPath: [
        "  Home & Kitchen  ",
        "home & kitchen",
        "",
        42,
        "Window Treatments",
        "Curtains & Drapes",
        "Panels",
        "Extra One",
        "Extra Two",
        "Extra Three",
        "Extra Four",
        "Extra Five",
      ],
    });

    expect(preview?.categoryPath).toEqual([
      "Home & Kitchen",
      "Window Treatments",
      "Curtains & Drapes",
      "Panels",
      "Extra One",
      "Extra Two",
      "Extra Three",
      "Extra Four",
    ]);

    const payload = productPayloadFromPreview({
      host: "www.amazon.com",
      preview: preview ?? undefined,
      targetMarket: "US",
      url: "https://www.amazon.com/dp/B08SBYPF14",
    });

    expect(payload?.category_path).toEqual(preview?.categoryPath);

    const withoutCrumbs = normalizeProductPreview({ title: "Plain Product" });
    expect(withoutCrumbs?.categoryPath).toBeUndefined();
  });

  it("keeps the Amazon locale path prefix for review-page fetches", () => {
    expect(
      amazonLocalePathPrefix("/-/en/Naotico-Upholstered-Headboard/dp/B0FC247FC1/"),
    ).toBe("/-/en");
    expect(amazonLocalePathPrefix("/-/en_GB/dp/B0FC247FC1/")).toBe("/-/en_GB");
    expect(amazonLocalePathPrefix("/dp/B0FC247FC1/")).toBe("");
    expect(amazonLocalePathPrefix("/-/enx/dp/B0FC247FC1/")).toBe("");
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

  it("keeps strong Amazon preview evidence while dropping localized page currency", () => {
    const preview = normalizeProductPreview({
      title: "mommore Kids Backpack for Boys Girls 4-8 Kindergarten Elementary School Backpack",
      imageUrl: "https://m.media-amazon.com/images/I/816XvP67HUL._AC_SX679_.jpg",
      seller: "mommore Store",
      price: 107454.17,
      currency: "MNT",
      rating: 4.7,
      reviewCount: 1686,
      returnPolicy: "30-day refund / replacement",
      reviews: [
        {
          text: "Brief content visible, double tap to read full content. Very durable backpack after daily use. Read more Read less",
          rating: 5,
          verified_purchase: true,
        },
      ],
    });

    const payload = productPayloadFromPreview({
      host: "www.amazon.com",
      preview: preview ?? undefined,
      targetMarket: "US",
      url: "https://www.amazon.com/dp/B0B28J2S6N",
    });

    expect(payload).toMatchObject({
      product_title: "mommore Kids Backpack for Boys Girls 4-8 Kindergarten Elementary School Backpack",
      review_count: 1686,
      reviews: [{ text: "Very durable backpack after daily use.", rating: 5 }],
      seller: { name: "mommore Store" },
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

  it("uses active-tab extracted scan before URL scan when preview is strong", async () => {
    installChromeStorage();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(200, { market_reference: { active: true } }))
      .mockResolvedValueOnce(jsonResponse(200, { ...apiResult, fetch_mode: "extension_dom" }));
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

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0][0]).toBe("https://walrus-app-38mjb.ondigitalocean.app/api/v1/model-info");
    expect(fetchMock.mock.calls[1][0]).toBe("https://walrus-app-38mjb.ondigitalocean.app/api/v1/scan-extracted");
    expect(fetchMock.mock.calls.map(([url]) => url)).not.toContain(
      "https://walrus-app-38mjb.ondigitalocean.app/api/v1/scan",
    );
  });

  it("does not fall back to a URL scan when the active-tab scan fails", async () => {
    installChromeStorage();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(200, { market_reference: { active: true } }))
      .mockResolvedValueOnce(
        jsonResponse(503, {
          detail: {
            code: "scan_extracted_unavailable",
            message: "Active-tab scan is temporarily unavailable.",
          },
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      analyzeProductWithPreviewFallback({
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
      }),
    ).rejects.toMatchObject({ code: "scan_extracted_unavailable" });

    expect(fetchMock.mock.calls.map(([url]) => url)).not.toContain(
      "https://walrus-app-38mjb.ondigitalocean.app/api/v1/scan",
    );
  });

  it("rejects weak review-page previews without any network scan", async () => {
    installChromeStorage();
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      analyzeProductWithPreviewFallback({
        host: "www.amazon.com",
        preview: {
          title: "Customer Ratings & Reviews",
          reviews: [{ text: "Comfortable mouse after long use." }],
        },
        targetMarket: "US",
        url: "https://www.amazon.com/reviews/B0TEST1234",
      }),
    ).rejects.toMatchObject({ code: "product_not_detected" });

    expect(fetchMock).not.toHaveBeenCalled();
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
