export type RiskLevel = "Low Risk" | "Medium Risk" | "High Risk";
export type TargetMarket = "auto" | "US" | "JP" | "EU" | "UK";

export type ProductAnalysisPayload = {
  url: string;
  browser_id?: string;
  locale?: string;
  target_market?: TargetMarket;
};

export type SellerInfo = {
  name?: string;
  rating?: number;
  review_count?: number;
  years_active?: number;
  sold_by?: string;
  ships_from?: string;
  fulfilled_by?: string;
  brand_store_name?: string;
  is_platform_seller?: boolean;
  is_platform_fulfilled?: boolean;
  is_official_store?: boolean;
  seller_source?: string;
};

export type ReviewInput = {
  text: string;
  rating?: number;
  date?: string;
  verified_purchase?: boolean;
};

export type ExtractedProductPayload = {
  url: string;
  site?: string;
  product_title: string;
  description?: string;
  product_image_url?: string;
  price?: number;
  currency?: string;
  average_market_price?: number;
  seller?: SellerInfo;
  return_policy?: string;
  reviews: ReviewInput[];
  rating?: number;
  review_count?: number;
  units_bought_recent?: number;
  feedback_score?: number;
};

export type ExtractedProductAnalysisPayload = {
  product: ExtractedProductPayload;
  browser_id?: string;
  locale?: string;
  target_market?: TargetMarket;
};

export type ProductMetadata = {
  url: string;
  site?: string | null;
  product_title: string;
  product_image_url?: string | null;
  price?: number | null;
  currency?: string | null;
  seller_name?: string | null;
};

export type ComponentScores = {
  review_authenticity: number;
  seller_reliability: number;
  sentiment: number;
  return_policy_clarity: number;
  price_safety: number;
  user_feedback_history: number;
};

export type ComponentKey = keyof ComponentScores;

export type ComponentEvidence = {
  component: ComponentKey;
  summary: string;
  evidence: string[];
  missing_inputs: string[];
  confidence: number;
};

export type MarketContext = {
  requested_market: TargetMarket;
  resolved_market: string;
  resolved_country?: string | null;
  expected_currency: string;
  listed_currency?: string | null;
};

export type ScoreTraceItem = {
  component: ComponentKey;
  score?: number | null;
  status: "scored" | "not_scored" | "fallback" | "missing_evidence";
  mode: string;
  confidence: number;
  evidence: string[];
  missing_inputs: string[];
};

export type ProductAnalysisResponse = {
  scan_id: string;
  product: ProductMetadata;
  trust_score: number;
  risk_level: RiskLevel;
  confidence: number;
  component_scores: ComponentScores;
  top_reasons: string[];
  evidence: ComponentEvidence[];
  missing_inputs: string[];
  score_semantics: string;
  recommendation: string;
  recommendation_source?: "rule" | "ai";
  language?: string;
  model_version: string;
  fetch_mode: string;
  extraction_signals: string[];
  model_modes: Record<string, string>;
  model_artifact_status: Record<string, unknown>;
  model_versions: Record<string, string>;
  score_status?: "scored" | "low_evidence_triage";
  page_type?: "product" | "review_page" | "search" | "category" | "cart" | "account" | "unknown";
  product_identity_confidence?: number;
  canonical_product_url?: string | null;
  market_context?: MarketContext | null;
  score_trace?: ScoreTraceItem[];
  is_mock: boolean;
};

export type FeedbackPayload = {
  scan_id: string;
  browser_id?: string;
  helpful: boolean;
  issue_category?:
    | "score_too_high"
    | "score_too_low"
    | "wrong_product"
    | "wrong_seller"
    | "wrong_reviews"
    | "wrong_price"
    | "wrong_policy"
    | "wrong_page_type"
    | "wrong_market"
    | "wrong_currency"
    | "wrong_extracted_field"
    | "missing_evidence"
    | "other";
  corrected_component?: ComponentKey;
  expected_risk_level?: RiskLevel;
  comment?: string;
};

export type FeedbackResponse = {
  status: "saved" | "accepted";
};

export type HealthCheckResponse = {
  status: "ok";
  service: "ai-trustscore-api";
};
