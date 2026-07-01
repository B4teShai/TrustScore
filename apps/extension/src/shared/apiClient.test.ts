import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  analyzeExtractedProduct,
  analyzeProduct,
  resetApiClientCompatibilityCacheForTests,
  toLegacyExtractedPayload,
} from "./apiClient";
import type { ProductAnalysisResponse } from "./types";

const API_BASE_URL = "https://walrus-app-38mjb.ondigitalocean.app";

const productResponse: ProductAnalysisResponse = {
  scan_id: "11111111-1111-4111-8111-111111111111",
  product: {
    url: "https://example.com/product/123",
    site: "example.com",
    product_title: "Wireless Headphones",
  },
  trust_score: 72,
  risk_level: "Medium Risk",
  confidence: 0.62,
  component_scores: {
    review_authenticity: 70,
    seller_reliability: 70,
    sentiment: 70,
    return_policy_clarity: 70,
    price_safety: 70,
    user_feedback_history: 50,
  },
  top_reasons: ["Return policy may need manual checking."],
  evidence: [
    {
      component: "return_policy_clarity",
      summary: "Return-policy clarity checks visible policy wording.",
      evidence: ["Policy snippet: 30-day returns"],
      missing_inputs: [],
      confidence: 0.8,
    },
  ],
  missing_inputs: [],
  score_semantics: "TrustScore is normalized over active non-feedback weights.",
  recommendation: "Check seller details before buying.",
  model_version: "0.3.0",
  fetch_mode: "http",
  extraction_signals: ["title", "price"],
  model_modes: {
    fake_review: "heuristic_fallback",
    sentiment: "keyword_fallback",
    seller_reliability: "rule_fallback",
    price_safety: "rule_fallback",
    return_policy_clarity: "rule_fallback",
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
    sentiment: "0.1.0",
    risk: "0.3.0",
  },
  is_mock: false,
};

describe("apiClient", () => {
  afterEach(() => {
    resetApiClientCompatibilityCacheForTests();
    vi.unstubAllGlobals();
  });

  it("posts scans to the canonical v1 route", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, productResponse));
    vi.stubGlobal("fetch", fetchMock);

    const result = await analyzeProduct({ url: "https://example.com/product/123" });

    expect(result.scan_id).toBe(productResponse.scan_id);
    expect(fetchMock).toHaveBeenCalledWith(
      `${API_BASE_URL}/api/v1/scan`,
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ url: "https://example.com/product/123" }),
      }),
    );
  });

  it("posts active-tab fallback scans to the extracted product route", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, productResponse));
    vi.stubGlobal("fetch", fetchMock);

    const result = await analyzeExtractedProduct({
      product: {
        url: "https://example.com/product/123",
        site: "example.com",
        product_title: "Wireless Headphones",
        seller: { name: "Example Store" },
        reviews: [],
      },
    });

    expect(result.scan_id).toBe(productResponse.scan_id);
    expect(fetchMock).toHaveBeenCalledWith(
      `${API_BASE_URL}/api/v1/scan-extracted`,
      expect.objectContaining({
        method: "POST",
      }),
    );
  });

  it("uses a legacy active-tab payload when model-info identifies an older API", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(200, { model_version: "1.0.0" }))
      .mockResolvedValueOnce(jsonResponse(200, productResponse));
    vi.stubGlobal("fetch", fetchMock);

    const result = await analyzeExtractedProduct({
      target_market: "US",
      product: {
        url: "https://example.com/product/123",
        site: "example.com",
        product_title: "Wireless Headphones",
        reviews: [],
        units_bought_recent: 1000,
      },
    });

    const secondBody = JSON.parse(fetchMock.mock.calls[1][1].body as string) as {
      product: Record<string, unknown>;
      target_market?: string;
    };

    expect(result.scan_id).toBe(productResponse.scan_id);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0][0]).toBe(`${API_BASE_URL}/api/v1/model-info`);
    expect(secondBody.target_market).toBeUndefined();
    expect(secondBody.product.units_bought_recent).toBeUndefined();
  });

  it("retries active-tab scans with a legacy payload when validation still rejects new fields", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(200, { market_reference: { active: true } }))
      .mockResolvedValueOnce(
        jsonResponse(422, {
          detail: [
            {
              type: "extra_forbidden",
              loc: ["body", "target_market"],
              msg: "Extra inputs are not permitted",
              input: "US",
            },
          ],
        }),
      )
      .mockResolvedValueOnce(jsonResponse(200, productResponse));
    vi.stubGlobal("fetch", fetchMock);

    const result = await analyzeExtractedProduct({
      target_market: "US",
      product: {
        url: "https://example.com/product/123",
        site: "example.com",
        product_title: "Wireless Headphones",
        reviews: [],
        units_bought_recent: 1000,
      },
    });

    const firstBody = JSON.parse(fetchMock.mock.calls[1][1].body as string) as Record<string, unknown>;
    const secondBody = JSON.parse(fetchMock.mock.calls[2][1].body as string) as {
      product: Record<string, unknown>;
      target_market?: string;
    };

    expect(result.scan_id).toBe(productResponse.scan_id);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(firstBody.target_market).toBe("US");
    expect(secondBody.target_market).toBeUndefined();
    expect(secondBody.product.units_bought_recent).toBeUndefined();
  });

  it("creates legacy-safe active-tab payloads", () => {
    const payload = toLegacyExtractedPayload({
      target_market: "US",
      locale: "en-US",
      product: {
        url: "https://example.com/product/123",
        product_title: "Wireless Headphones",
        reviews: [],
        units_bought_recent: 1000,
      },
    });

    expect(payload.target_market).toBeUndefined();
    expect(payload.locale).toBe("en-US");
    expect(payload.product.units_bought_recent).toBeUndefined();
  });

  it("accepts nulls for missing optional product metadata", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse(200, {
        ...productResponse,
        product: {
          ...productResponse.product,
          seller_name: null,
          description: undefined,
        },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await analyzeProduct({ url: "https://next.mn/mn/product/gc-q257cafv" });

    expect(result.product.seller_name).toBeNull();
  });

  it("surfaces structured backend errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(422, {
          detail: {
            code: "product_not_detected",
            message: "This page does not have enough product-detail signals.",
          },
        }),
      ),
    );

    await expect(analyzeProduct({ url: "https://example.com/about" })).rejects.toMatchObject({
      code: "product_not_detected",
      message: "This page does not have enough product-detail signals.",
      status: 422,
    } satisfies Partial<ApiError>);
  });

  it("rejects invalid success payloads", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(200, { ok: true })));

    await expect(analyzeProduct({ url: "https://example.com/product/123" })).rejects.toMatchObject({
      code: "invalid_response",
      status: 502,
    } satisfies Partial<ApiError>);
  });

  it("rejects responses with missing component scores", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(200, {
          ...productResponse,
          component_scores: {
            ...productResponse.component_scores,
            price_safety: undefined,
          },
        }),
      ),
    );

    await expect(analyzeProduct({ url: "https://example.com/product/123" })).rejects.toMatchObject({
      code: "invalid_response",
      status: 502,
    } satisfies Partial<ApiError>);
  });

  it("rejects responses with invalid risk levels", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(200, {
          ...productResponse,
          risk_level: "Unknown",
        }),
      ),
    );

    await expect(analyzeProduct({ url: "https://example.com/product/123" })).rejects.toMatchObject({
      code: "invalid_response",
      status: 502,
    } satisfies Partial<ApiError>);
  });

  it("rejects responses without artifact status metadata", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(200, {
          ...productResponse,
          model_artifact_status: undefined,
        }),
      ),
    );

    await expect(analyzeProduct({ url: "https://example.com/product/123" })).rejects.toMatchObject({
      code: "invalid_response",
      status: 502,
    } satisfies Partial<ApiError>);
  });
});

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
