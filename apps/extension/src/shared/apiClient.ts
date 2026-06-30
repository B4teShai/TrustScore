import type {
  ExtractedProductAnalysisPayload,
  FeedbackPayload,
  FeedbackResponse,
  HealthCheckResponse,
  ProductAnalysisPayload,
  ProductAnalysisResponse,
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
let modernExtractedPayloadSupported: boolean | null = null;

export class ApiError extends Error {
  status: number;
  code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function requestJson<TResponse>(
  path: string,
  options?: RequestInit,
  validate?: (value: unknown) => value is TResponse,
): Promise<TResponse> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    throw await parseApiError(response);
  }

  const json = await response.json();
  if (validate && !validate(json)) {
    throw new ApiError("Backend returned an invalid TrustScore response.", 502, "invalid_response");
  }
  return json as TResponse;
}

export function getHealth(): Promise<HealthCheckResponse> {
  return requestJson<HealthCheckResponse>("/health");
}

export function analyzeProduct(
  payload: ProductAnalysisPayload,
): Promise<ProductAnalysisResponse> {
  return requestJson<ProductAnalysisResponse>(
    "/api/v1/scan",
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
    isProductAnalysisResponse,
  );
}

export async function analyzeExtractedProduct(
  payload: ExtractedProductAnalysisPayload,
): Promise<ProductAnalysisResponse> {
  const requestPayload =
    hasModernExtractedOnlyFields(payload) && !(await backendSupportsModernExtractedPayload())
      ? toLegacyExtractedPayload(payload)
      : payload;
  try {
    return await requestJson<ProductAnalysisResponse>(
      "/api/v1/scan-extracted",
      {
        method: "POST",
        body: JSON.stringify(requestPayload),
      },
      isProductAnalysisResponse,
    );
  } catch (error) {
    if (!(error instanceof ApiError) || !shouldRetryLegacyExtractedPayload(error, requestPayload)) {
      throw error;
    }
    modernExtractedPayloadSupported = false;
    return requestJson<ProductAnalysisResponse>(
      "/api/v1/scan-extracted",
      {
        method: "POST",
        body: JSON.stringify(toLegacyExtractedPayload(requestPayload)),
      },
      isProductAnalysisResponse,
    );
  }
}

export function submitFeedback(
  payload: FeedbackPayload,
): Promise<FeedbackResponse> {
  return requestJson<FeedbackResponse>("/api/v1/feedback", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

async function parseApiError(response: Response): Promise<ApiError> {
  let code: string | undefined;
  let message = `API request failed with status ${response.status}`;

  try {
    const body: unknown = await response.json();
    if (isBackendErrorBody(body)) {
      code = body.detail.code;
      message = body.detail.message;
    } else if (isPydanticErrorBody(body)) {
      code = "validation_error";
      message = "The page details changed while the scan was starting.";
    }
  } catch {
    // Keep the generic HTTP status message when the body is not JSON.
  }

  return new ApiError(message, response.status, code);
}

function isBackendErrorBody(
  value: unknown,
): value is { detail: { code: string; message: string } } {
  if (!value || typeof value !== "object" || !("detail" in value)) {
    return false;
  }
  const detail = (value as { detail?: unknown }).detail;
  return Boolean(
    detail &&
      typeof detail === "object" &&
      typeof (detail as { code?: unknown }).code === "string" &&
      typeof (detail as { message?: unknown }).message === "string",
  );
}

function isPydanticErrorBody(value: unknown): value is { detail: unknown[] } {
  return Boolean(
    value &&
      typeof value === "object" &&
      Array.isArray((value as { detail?: unknown }).detail),
  );
}

function shouldRetryLegacyExtractedPayload(
  error: ApiError,
  payload: ExtractedProductAnalysisPayload,
): boolean {
  return (
    error.status === 422 &&
    error.code === "validation_error" &&
    (payload.target_market !== undefined || payload.product.units_bought_recent !== undefined)
  );
}

function hasModernExtractedOnlyFields(payload: ExtractedProductAnalysisPayload): boolean {
  return payload.target_market !== undefined || payload.product.units_bought_recent !== undefined;
}

async function backendSupportsModernExtractedPayload(): Promise<boolean> {
  if (modernExtractedPayloadSupported !== null) {
    return modernExtractedPayloadSupported;
  }
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/model-info`);
    if (!response.ok) {
      return true;
    }
    const body: unknown = await response.json();
    modernExtractedPayloadSupported =
      Boolean(body && typeof body === "object" && "market_reference" in body);
    return modernExtractedPayloadSupported;
  } catch {
    return true;
  }
}

export function toLegacyExtractedPayload(
  payload: ExtractedProductAnalysisPayload,
): ExtractedProductAnalysisPayload {
  const legacyPayload: ExtractedProductAnalysisPayload = {
    ...payload,
    product: { ...payload.product },
  };
  delete legacyPayload.target_market;
  delete legacyPayload.product.units_bought_recent;
  return legacyPayload;
}

export function resetApiClientCompatibilityCacheForTests(): void {
  modernExtractedPayloadSupported = null;
}

function isProductAnalysisResponse(value: unknown): value is ProductAnalysisResponse {
  if (!value || typeof value !== "object") {
    return false;
  }
  const candidate = value as Partial<ProductAnalysisResponse>;
  return Boolean(
    typeof candidate.scan_id === "string" &&
      isUuid(candidate.scan_id) &&
      candidate.product &&
      isProductMetadata(candidate.product) &&
      isScore(candidate.trust_score) &&
      isRiskLevel(candidate.risk_level) &&
      isConfidence(candidate.confidence) &&
      isComponentScores(candidate.component_scores) &&
      Array.isArray(candidate.top_reasons) &&
      candidate.top_reasons.length <= 3 &&
      candidate.top_reasons.every((reason) => typeof reason === "string") &&
      Array.isArray(candidate.evidence) &&
      candidate.evidence.every(isComponentEvidence) &&
      Array.isArray(candidate.missing_inputs) &&
      candidate.missing_inputs.every((input) => typeof input === "string") &&
      typeof candidate.score_semantics === "string" &&
      typeof candidate.recommendation === "string" &&
      typeof candidate.model_version === "string" &&
      typeof candidate.fetch_mode === "string" &&
      Array.isArray(candidate.extraction_signals) &&
      candidate.extraction_signals.every((signal) => typeof signal === "string") &&
      isStringRecord(candidate.model_modes) &&
      hasModelModeKeys(candidate.model_modes) &&
      isPlainRecord(candidate.model_artifact_status) &&
      isStringRecord(candidate.model_versions) &&
      hasModelVersionKeys(candidate.model_versions) &&
      typeof candidate.is_mock === "boolean",
  );
}

function isProductMetadata(value: unknown): value is ProductAnalysisResponse["product"] {
  if (!value || typeof value !== "object") {
    return false;
  }
  const product = value as Partial<ProductAnalysisResponse["product"]>;
  return Boolean(
    typeof product.url === "string" &&
      product.url.length > 0 &&
      typeof product.product_title === "string" &&
      product.product_title.length > 0 &&
      optionalString(product.site) &&
      optionalString(product.product_image_url) &&
      optionalNumber(product.price) &&
      optionalString(product.currency) &&
      optionalString(product.seller_name),
  );
}

function isComponentScores(value: unknown): value is ProductAnalysisResponse["component_scores"] {
  if (!value || typeof value !== "object") {
    return false;
  }
  const scores = value as Record<string, unknown>;
  return COMPONENT_SCORE_KEYS.every((key) => isScore(scores[key]));
}

function isComponentEvidence(value: unknown): value is ProductAnalysisResponse["evidence"][number] {
  if (!value || typeof value !== "object") {
    return false;
  }
  const evidence = value as Partial<ProductAnalysisResponse["evidence"][number]>;
  return Boolean(
    typeof evidence.component === "string" &&
      COMPONENT_SCORE_KEYS.includes(evidence.component) &&
      typeof evidence.summary === "string" &&
      Array.isArray(evidence.evidence) &&
      evidence.evidence.every((item) => typeof item === "string") &&
      Array.isArray(evidence.missing_inputs) &&
      evidence.missing_inputs.every((item) => typeof item === "string") &&
      isConfidence(evidence.confidence),
  );
}

function isScore(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 100;
}

function isConfidence(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

function isRiskLevel(value: unknown): value is ProductAnalysisResponse["risk_level"] {
  return value === "Low Risk" || value === "Medium Risk" || value === "High Risk";
}

function optionalString(value: unknown): boolean {
  return value === undefined || value === null || typeof value === "string";
}

function optionalNumber(value: unknown): boolean {
  return value === undefined || value === null || (typeof value === "number" && Number.isFinite(value));
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function isStringRecord(value: unknown): value is Record<string, string> {
  return (
    isPlainRecord(value) &&
    Object.values(value).every((recordValue) => typeof recordValue === "string")
  );
}

function hasModelModeKeys(value: Record<string, string>): boolean {
  return MODEL_MODE_KEYS.every((key) => typeof value[key] === "string");
}

function hasModelVersionKeys(value: Record<string, string>): boolean {
  return MODEL_VERSION_KEYS.every((key) => typeof value[key] === "string");
}

function isUuid(value: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
    value,
  );
}

const COMPONENT_SCORE_KEYS = [
  "review_authenticity",
  "seller_reliability",
  "sentiment",
  "return_policy_clarity",
  "price_safety",
  "user_feedback_history",
] as const;

const MODEL_MODE_KEYS = [
  "fake_review",
  "sentiment",
  "seller_reliability",
  "price_safety",
  "return_policy_clarity",
  "user_feedback_history",
] as const;

const MODEL_VERSION_KEYS = ["trustscore", "fake_review", "sentiment", "risk"] as const;
