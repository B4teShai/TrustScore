import { afterEach, describe, expect, it, vi } from "vitest";

import {
  canonicalProductPageUrl,
  getOrCreateBrowserId,
  getStoredResultForPage,
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
