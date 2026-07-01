import { useEffect, useMemo, useState, type ReactElement, type ReactNode } from "react";

import "./App.css";
import {
  ApiError,
  analyzeExtractedProduct,
  analyzeProduct,
  submitFeedback,
} from "../shared/apiClient";
import type {
  ComponentEvidence,
  ComponentScores,
  ExtractedProductPayload,
  ProductAnalysisResponse,
  TargetMarket,
} from "../shared/types";

type IconProps = {
  size?: number;
  sw?: number;
};

type RiskKey = "low" | "med" | "high";
type ViewState = "detecting" | "home" | "loading" | "result" | "feedback" | "empty" | "error";
type ProductMeta = {
  host: string;
  imageUrl?: string;
  price?: string;
  title: string;
  seller: string;
  url?: string;
};
type PreviewReview = {
  text: string;
  rating?: number;
  verified_purchase?: boolean;
};

type ProductPreview = {
  title?: string;
  imageUrl?: string;
  seller?: string;
  sellerRating?: number;
  sellerReviewCount?: number;
  price?: number;
  currency?: string;
  rating?: number;
  reviewCount?: number;
  unitsBoughtRecent?: number;
  description?: string;
  returnPolicy?: string;
  reviews?: PreviewReview[];
  lang?: string;
};
type CurrentPage = {
  host: string;
  preview?: ProductPreview;
  tabId?: number;
  targetMarket: Exclude<TargetMarket, "auto">;
  url: string;
};

const LAST_RESULT_KEY = "trustscore:lastResult";
const BROWSER_ID_KEY = "trustscore:browserId";
const REVIEW_UI_COPY_RE =
  /(brief content visible,\s*double tap to read full content\.?|full content visible,\s*double tap to read brief content\.?|read more\s+read less|the media could not be loaded\.?)/gi;

function TrustLogo({ size = 28 }: { size?: number }) {
  const id = `trust-logo-${size}`;
  return (
    <svg
      aria-hidden="true"
      className="logo-mark"
      height={size}
      viewBox="0 0 64 64"
      width={size}
    >
      <defs>
        <linearGradient id={id} x1="10" x2="54" y1="8" y2="58">
          <stop offset="0" stopColor="#ff4fa3" />
          <stop offset="0.55" stopColor="#f43f5e" />
          <stop offset="1" stopColor="#fb7185" />
        </linearGradient>
      </defs>
      <rect fill={`url(#${id})`} height="56" rx="16" width="56" x="4" y="4" />
      <circle cx="32" cy="32" fill="none" r="15" stroke="white" strokeWidth="5" />
      <path
        d="M32 13v8M32 43v8M13 32h8M43 32h8"
        stroke="white"
        strokeLinecap="round"
        strokeWidth="5"
      />
      <path d="M22 42 42 22" stroke="#ffe4f1" strokeLinecap="round" strokeWidth="5" />
      <circle cx="32" cy="32" fill="white" r="4" />
    </svg>
  );
}

const Icon = ({
  children,
  size = 16,
  sw = 1.8,
}: IconProps & { children: ReactNode }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeLinecap="round"
    strokeLinejoin="round"
    strokeWidth={sw}
    aria-hidden="true"
  >
    {children}
  </svg>
);

const I = {
  AlertOctagon: (props: IconProps) => (
    <Icon {...props}>
      <path d="M8.5 3h7L21 8.5v7L15.5 21h-7L3 15.5v-7L8.5 3z" />
      <path d="M12 8v4M12 16v.1" />
    </Icon>
  ),
  AlertTriangle: (props: IconProps) => (
    <Icon {...props}>
      <path d="M12 4l9.5 16h-19L12 4z" />
      <path d="M12 10v4M12 17.5v.1" />
    </Icon>
  ),
  Brain: (props: IconProps) => (
    <Icon {...props}>
      <path d="M9 4a3 3 0 0 0-3 3v.5A3 3 0 0 0 4 10a3 3 0 0 0 1 5.5V17a3 3 0 0 0 4 3" />
      <path d="M15 4a3 3 0 0 1 3 3v.5a3 3 0 0 1 2 2.5a3 3 0 0 1-1 5.5V17a3 3 0 0 1-4 3" />
      <path d="M12 4v16" />
    </Icon>
  ),
  Check: (props: IconProps) => (
    <Icon {...props}>
      <path d="M5 12l5 5L20 7" />
    </Icon>
  ),
  CheckCircle: (props: IconProps) => (
    <Icon {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="M8 12.5l3 3 5-6" />
    </Icon>
  ),
  Lock: (props: IconProps) => (
    <Icon {...props}>
      <rect x="5" y="11" width="14" height="9" rx="2" />
      <path d="M8 11V8a4 4 0 0 1 8 0v3" />
    </Icon>
  ),
  Price: (props: IconProps) => (
    <Icon {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="M14.5 9.5C14 8.5 13 8 12 8c-1.5 0-2.5 1-2.5 2.2c0 2.6 5 1.6 5 4c0 1.4-1.2 2.3-2.5 2.3c-1.2 0-2.4-.7-2.7-2" />
      <path d="M12 6.5V8M12 16v1.5" />
    </Icon>
  ),
  Return: (props: IconProps) => (
    <Icon {...props}>
      <path d="M3 12a9 9 0 1 0 3-6.7" />
      <path d="M3 4v5h5" />
    </Icon>
  ),
  ReviewAuth: (props: IconProps) => (
    <Icon {...props}>
      <path d="M4 5h16v10H8l-4 4V5z" />
      <path d="M8.5 10h7M8.5 7h4" />
    </Icon>
  ),
  Search: (props: IconProps) => (
    <Icon {...props}>
      <circle cx="11" cy="11" r="6" />
      <path d="M16 16l4 4" />
    </Icon>
  ),
  Seller: (props: IconProps) => (
    <Icon {...props}>
      <circle cx="12" cy="8" r="4" />
      <path d="M4 21a8 8 0 0 1 16 0" />
    </Icon>
  ),
  Sentiment: (props: IconProps) => (
    <Icon {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="M9 14s1 2 3 2s3-2 3-2" />
      <path d="M9 9.5v.1M15 9.5v.1" />
    </Icon>
  ),
  Settings: (props: IconProps) => (
    <Icon {...props}>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3a1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.5a1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8a1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1a1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5a1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z" />
    </Icon>
  ),
  Shield: (props: IconProps) => (
    <Icon {...props}>
      <path d="M12 3l8 3v6c0 4.5-3.2 8.4-8 9c-4.8-.6-8-4.5-8-9V6l8-3z" />
    </Icon>
  ),
  ShoppingBag: (props: IconProps) => (
    <Icon {...props}>
      <path d="M5 8h14l-1 12H6L5 8z" />
      <path d="M9 8V5a3 3 0 0 1 6 0v3" />
    </Icon>
  ),
  Sparkles: (props: IconProps) => (
    <Icon {...props}>
      <path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.5 5.5L8 8M16 16l2.5 2.5M5.5 18.5L8 16M16 8l2.5-2.5" />
    </Icon>
  ),
  ThumbsDown: (props: IconProps) => (
    <Icon {...props}>
      <path d="M17 13V4h3a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1h-3z" />
      <path d="M17 13l-4 7a3 3 0 0 1-3-3v-3H5a2 2 0 0 1-2-2.4l1.4-7A2 2 0 0 1 6.4 4H17" />
    </Icon>
  ),
  ThumbsUp: (props: IconProps) => (
    <Icon {...props}>
      <path d="M7 11v9H4a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1h3z" />
      <path d="M7 11l4-7a3 3 0 0 1 3 3v3h5a2 2 0 0 1 2 2.4l-1.4 7A2 2 0 0 1 16.6 20H7" />
    </Icon>
  ),
};

const COMPONENT_CONFIG: Record<
  keyof ComponentScores,
  {
    icon: (props: IconProps) => ReactElement;
    label: string;
  }
> = {
  review_authenticity: {
    icon: I.ReviewAuth,
    label: "Review authenticity",
  },
  seller_reliability: {
    icon: I.Seller,
    label: "Seller reliability",
  },
  sentiment: {
    icon: I.Sentiment,
    label: "Sentiment score",
  },
  return_policy_clarity: {
    icon: I.Return,
    label: "Return policy clarity",
  },
  price_safety: {
    icon: I.Price,
    label: "Price safety",
  },
  user_feedback_history: {
    icon: I.ThumbsUp,
    label: "Feedback history",
  },
};

const LOADING_TASKS = [
  "Reading visible page signals",
  "Scoring review evidence",
  "Checking seller fields",
  "Checking listed price",
  "Checking return-policy text",
];

const FEEDBACK_CHIPS: Array<{
  component?: keyof ComponentScores;
  issue:
    | "score_too_high"
    | "score_too_low"
    | "wrong_product"
    | "wrong_seller"
    | "wrong_reviews"
    | "wrong_price"
    | "wrong_policy"
    | "missing_evidence"
    | "other";
  label: string;
  value: string;
}> = [
  { label: "Wrong score", value: "wrong_score", issue: "other" },
  { label: "Wrong price/currency", value: "wrong_price", issue: "wrong_price", component: "price_safety" },
  { label: "Wrong seller/policy", value: "wrong_seller_policy", issue: "missing_evidence" },
  { label: "Wrong product", value: "wrong_product", issue: "wrong_product" },
  { label: "Missing reviews", value: "missing_reviews", issue: "wrong_reviews", component: "review_authenticity" },
  { label: "Other", value: "other", issue: "other" },
] as const;

function riskFromLevel(riskLevel: ProductAnalysisResponse["risk_level"]): RiskKey {
  if (riskLevel === "Low Risk") {
    return "low";
  }
  if (riskLevel === "High Risk") {
    return "high";
  }
  return "med";
}

function riskFromScore(score: number): RiskKey {
  if (score >= 80) {
    return "low";
  }
  if (score >= 50) {
    return "med";
  }
  return "high";
}

function riskLabel(risk: RiskKey) {
  if (risk === "low") {
    return "Low risk";
  }
  if (risk === "high") {
    return "High risk";
  }
  return "Medium risk";
}

function reasonTone(index: number, risk: RiskKey) {
  if (risk === "low") {
    return "tick";
  }
  if (risk === "high" && index < 2) {
    return "bad";
  }
  return "warn";
}

function componentTone(score: number): RiskKey {
  return riskFromScore(score);
}

function isSupportedPageUrl(url: string | undefined) {
  return Boolean(url && /^https?:\/\//i.test(url));
}

export async function getStoredResultForPage(
  pageUrl: string,
): Promise<ProductAnalysisResponse | null> {
  const stored = await storageGet<ProductAnalysisResponse>(LAST_RESULT_KEY);
  if (stored?.product?.url === pageUrl) {
    return stored;
  }
  return null;
}

export async function storeLastResult(result: ProductAnalysisResponse): Promise<void> {
  await storageSet(LAST_RESULT_KEY, result);
}

export async function getOrCreateBrowserId(): Promise<string | undefined> {
  const existing = await storageGet<string>(BROWSER_ID_KEY);
  if (existing) {
    return existing;
  }

  const generated =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `browser-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  await storageSet(BROWSER_ID_KEY, generated);
  return generated;
}

function storageGet<T>(key: string): Promise<T | undefined> {
  return new Promise((resolve) => {
    if (typeof chrome === "undefined" || !chrome.storage?.local) {
      resolve(undefined);
      return;
    }

    chrome.storage.local.get(key, (items) => {
      if (chrome.runtime.lastError) {
        resolve(undefined);
        return;
      }
      resolve(items[key] as T | undefined);
    });
  });
}

function storageSet(key: string, value: unknown): Promise<void> {
  return new Promise((resolve) => {
    if (typeof chrome === "undefined" || !chrome.storage?.local) {
      resolve();
      return;
    }

    chrome.storage.local.set({ [key]: value }, () => {
      resolve();
    });
  });
}

async function getCurrentPageUrl(): Promise<CurrentPage | null> {
  if (typeof chrome === "undefined" || !chrome.tabs?.query) {
    throw new Error("Chrome extension APIs are not available.");
  }

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!isSupportedPageUrl(tab?.url)) {
    return null;
  }

  try {
    const url = new URL(tab.url ?? "");
    const scanUrl = canonicalProductPageUrl(url);
    const preview = typeof tab.id === "number" ? await readActiveTabProductPreview(tab.id) : undefined;
    return {
      host: url.hostname,
      preview: preview ?? undefined,
      tabId: tab.id,
      targetMarket: resolveTargetMarket(scanUrl, preview?.lang),
      url: scanUrl,
    };
  } catch {
    return null;
  }
}

export function canonicalProductPageUrl(value: string | URL): string {
  const url = typeof value === "string" ? new URL(value) : new URL(value.href);
  url.hash = "";

  const amazonAsin = asinFromAmazonPath(url);
  if (amazonAsin) {
    return `${url.origin}/dp/${amazonAsin}`;
  }

  return url.href;
}

function asinFromAmazonPath(url: URL): string | null {
  if (!/(^|\.)amazon\./i.test(url.hostname)) {
    return null;
  }
  const match = url.pathname.match(/\/(?:dp|gp\/product)\/([A-Z0-9]{10})(?:[/?]|$)/i);
  return match ? match[1].toUpperCase() : null;
}

async function readActiveTabProductPreview(tabId: number): Promise<ProductPreview | null> {
  if (typeof chrome === "undefined" || !chrome.scripting?.executeScript) {
    return null;
  }

  try {
    const [result] = await chrome.scripting.executeScript({
      target: { tabId },
      func: extractProductPreviewFromDocument,
    });
    return normalizeProductPreview(result?.result);
  } catch {
    return null;
  }
}

export function normalizeProductPreview(value: unknown): ProductPreview | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const candidate = value as ProductPreview;
  const title = cleanText(candidate.title);
  const seller = cleanText(candidate.seller);
  const imageUrl = safeImageUrl(candidate.imageUrl);
  const price = validNumber(candidate.price);
  const rating = validNumber(candidate.rating);
  const reviewCount = validInteger(candidate.reviewCount);
  const unitsBoughtRecent = validInteger(candidate.unitsBoughtRecent);
  const currency = cleanCurrency(candidate.currency);
  const lang = cleanText(candidate.lang)?.slice(0, 35);
  const sellerRating = validNumber(candidate.sellerRating);
  const sellerReviewCount = validInteger(candidate.sellerReviewCount);
  const description = cleanText(candidate.description)?.slice(0, 4000);
  const returnPolicy = cleanText(candidate.returnPolicy)?.slice(0, 4000);
  const reviews = normalizeReviews(candidate.reviews);

  if (!title && !imageUrl && !seller) {
    return null;
  }

  return {
    title,
    imageUrl,
    seller,
    sellerRating,
    sellerReviewCount,
    price,
    currency,
    rating,
    reviewCount,
    unitsBoughtRecent,
    description,
    returnPolicy,
    reviews,
    lang,
  };
}

function normalizeReviews(value: unknown): PreviewReview[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const reviews: PreviewReview[] = [];
  for (const item of value) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const candidate = item as PreviewReview;
    const text = cleanReviewText(candidate.text)?.slice(0, 2000);
    if (!text) {
      continue;
    }
    const rating = validNumber(candidate.rating);
    reviews.push({
      text,
      rating: typeof rating === "number" && rating <= 5 ? rating : undefined,
      verified_purchase:
        typeof candidate.verified_purchase === "boolean" ? candidate.verified_purchase : undefined,
    });
    if (reviews.length >= 30) {
      break;
    }
  }
  return reviews;
}

function cleanText(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const compact = value.replace(/\s+/g, " ").trim();
  return compact || undefined;
}

function cleanReviewText(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  return cleanText(value.replace(REVIEW_UI_COPY_RE, " "));
}

function cleanCurrency(value: unknown): string | undefined {
  return normalizeCurrencyCode(cleanText(value));
}

function validNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : undefined;
}

function validInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : undefined;
}

export function productPayloadFromPreview(page: CurrentPage): ExtractedProductPayload | null {
  const preview = page.preview;
  if (!preview?.title) {
    return null;
  }
  const priceInfo = normalizePreviewPrice(
    preview.price,
    preview.currency,
    page.host,
    page.targetMarket,
  );

  const seller =
    preview.seller || preview.sellerRating !== undefined || preview.sellerReviewCount !== undefined
      ? {
          name: limitText(preview.seller, 160),
          rating: preview.sellerRating,
          review_count: preview.sellerReviewCount,
          brand_store_name: looksLikeOfficialStore(preview.seller) ? limitText(preview.seller, 160) : undefined,
          is_official_store: looksLikeOfficialStore(preview.seller) || undefined,
          is_platform_seller: looksLikePlatformSeller(preview.seller) || undefined,
          seller_source: preview.seller ? "active_tab" : undefined,
        }
      : undefined;

  return {
    url: page.url,
    site: limitText(page.host, 255),
    product_title: limitText(preview.title, 240) ?? "Unknown product",
    description: limitText(preview.description, 4000),
    product_image_url: preview.imageUrl,
    price: priceInfo.price,
    currency: priceInfo.currency,
    seller,
    return_policy: limitText(preview.returnPolicy, 4000),
    reviews: (preview.reviews ?? []).slice(0, 30),
    rating: preview.rating,
    review_count: preview.reviewCount,
    units_bought_recent: preview.unitsBoughtRecent,
  };
}

function looksLikeOfficialStore(value?: string): boolean {
  return Boolean(
    value &&
      /\b(Apple|Amazon|Logitech|StarTech|Sony|Samsung|Microsoft|Anker|Belkin|Dell|HP)\b.*\b(Store|Official)\b/i.test(
        value,
      ),
  );
}

function looksLikePlatformSeller(value?: string): boolean {
  return Boolean(value && /\b(Amazon|Best Buy|Walmart|Target|Etsy|eBay)\b/i.test(value));
}

function pageLocale(page: CurrentPage): string | undefined {
  const fromPage = page.preview?.lang;
  if (fromPage) {
    return fromPage;
  }
  if (typeof navigator !== "undefined" && navigator.language) {
    return navigator.language;
  }
  return undefined;
}

function resolveTargetMarket(
  urlValue: string,
  locale?: string,
): Exclude<TargetMarket, "auto"> {
  let host = "";
  try {
    host = new URL(urlValue).hostname.toLowerCase();
  } catch {
    host = urlValue.toLowerCase();
  }
  if (host.endsWith(".co.jp") || host.endsWith(".jp")) {
    return "JP";
  }
  if (host.endsWith(".co.uk") || host.endsWith(".uk")) {
    return "UK";
  }
  if (/\.(de|fr|it|es|nl|be|ie|at|pt|pl|se)$/i.test(host)) {
    return "EU";
  }
  if (
    host === "amazon.com" ||
    host.endsWith(".amazon.com") ||
    host === "ebay.com" ||
    host.endsWith(".ebay.com") ||
    host === "etsy.com" ||
    host.endsWith(".etsy.com")
  ) {
    return "US";
  }
  const lang = locale?.toLowerCase().replace("_", "-") ?? "";
  const primary = lang.split("-", 1)[0];
  if (primary === "ja" || lang.endsWith("-jp")) {
    return "JP";
  }
  if (lang === "en-gb" || lang === "en-uk" || lang.endsWith("-gb") || lang.endsWith("-uk")) {
    return "UK";
  }
  if (["de", "fr", "it", "es", "nl", "pt", "pl", "sv"].includes(primary)) {
    return "EU";
  }
  return "US";
}

function expectedCurrencyForMarket(market: Exclude<TargetMarket, "auto">): string {
  if (market === "JP") {
    return "JPY";
  }
  if (market === "EU") {
    return "EUR";
  }
  if (market === "UK") {
    return "GBP";
  }
  return "USD";
}

function normalizeCurrencyCode(value?: string): string | undefined {
  const raw = value?.trim().toUpperCase();
  const token = raw?.replace(/\s+/g, "");
  if (!token) {
    return undefined;
  }
  if (token === "$" || token === "US$" || token === "USD") {
    return "USD";
  }
  if (token === "¥" || token === "￥" || token === "円" || token === "JPY" || token === "JP¥") {
    return "JPY";
  }
  if (token === "€" || token === "EUR") {
    return "EUR";
  }
  if (token === "£" || token === "GBP") {
    return "GBP";
  }
  if (token === "MNT" || token === "₮") {
    return "MNT";
  }
  const tokens = raw?.replace(/[_-]/g, " ").split(/\s+/) ?? [];
  for (const code of ["USD", "JPY", "EUR", "GBP", "MNT"]) {
    if (tokens.includes(code)) {
      return code;
    }
  }
  return /^[A-Z]{3}$/.test(token) ? token : token.slice(0, 16);
}

function normalizePreviewPrice(
  price: number | undefined,
  currency: string | undefined,
  _host: string,
  market: Exclude<TargetMarket, "auto">,
): { price?: number; currency?: string; ignoredCurrency?: string } {
  if (price === undefined) {
    return {};
  }
  const expected = expectedCurrencyForMarket(market);
  const normalizedCurrency = normalizeCurrencyCode(currency);
  if (normalizedCurrency && !["USD", "JPY", "EUR", "GBP"].includes(normalizedCurrency)) {
    return { ignoredCurrency: normalizedCurrency };
  }
  return { price, currency: normalizedCurrency ?? expected };
}

const RTL_LANGS = new Set(["ar", "he", "fa", "ur", "ps", "sd"]);

function isRtlLanguage(language?: string): boolean {
  if (!language) {
    return false;
  }
  return RTL_LANGS.has(language.toLowerCase().split("-")[0]);
}

function formatMoney(price: number, currency?: string, locale?: string): string {
  if (currency) {
    try {
      return new Intl.NumberFormat(locale, { style: "currency", currency }).format(price);
    } catch {
      // Unknown currency code: fall through to a plain format.
    }
  }
  return `${currency ? `${currency} ` : ""}${price.toLocaleString(locale)}`;
}

function limitText(value: string | undefined, maxLength: number): string | undefined {
  const compact = value?.replace(/\s+/g, " ").trim();
  return compact ? compact.slice(0, maxLength) : undefined;
}

async function extractProductPreviewFromDocument(): Promise<ProductPreview> {
  const host = window.location.hostname.toLowerCase();
  const reviewUiCopyRe =
    /(brief content visible,\s*double tap to read full content\.?|full content visible,\s*double tap to read brief content\.?|read more\s+read less|the media could not be loaded\.?)/gi;

  function text(selector: string): string | undefined {
    const element = document.querySelector(selector);
    const value =
      element instanceof HTMLMetaElement
        ? element.content
        : element?.textContent ?? undefined;
    return compact(value);
  }

  function attr(selector: string, attribute: string): string | undefined {
    const value = document.querySelector(selector)?.getAttribute(attribute) ?? undefined;
    return compact(value);
  }

  function firstText(selectors: string[]): string | undefined {
    return firstTexts(selectors, 1)[0];
  }

  function firstTexts(selectors: string[], limit: number): string[] {
    const out: string[] = [];
    const seen = new Set<string>();
    for (const selector of selectors) {
      for (const element of Array.from(document.querySelectorAll(selector))) {
        const value = compact(
          element instanceof HTMLMetaElement
            ? element.getAttribute("content")
            : element.textContent,
        );
        if (!value || seen.has(value)) {
          continue;
        }
        out.push(value);
        seen.add(value);
        if (out.length >= limit) {
          return out;
        }
      }
    }
    return out;
  }

  function firstAttr(selectors: string[], attribute: string): string | undefined {
    for (const selector of selectors) {
      const value = attr(selector, attribute);
      if (value) {
        return value;
      }
    }
    return undefined;
  }

  function compact(value: string | null | undefined): string | undefined {
    const cleaned = value?.replace(/\s+/g, " ").trim();
    return cleaned || undefined;
  }

  function cleanReviewBody(value: string | null | undefined): string | undefined {
    return compact(value?.replace(reviewUiCopyRe, " "));
  }

  function isAmazon(): boolean {
    return /(^|\.)amazon\./i.test(host);
  }

  function isEbay(): boolean {
    return /(^|\.)ebay\./i.test(host);
  }

  function isEtsy(): boolean {
    return /(^|\.)etsy\./i.test(host);
  }

  function absoluteUrl(value: string | undefined): string | undefined {
    if (!value) {
      return undefined;
    }
    try {
      return new URL(value, window.location.href).href;
    } catch {
      return undefined;
    }
  }

  function dynamicImage(value: string | undefined): string | undefined {
    if (!value) {
      return undefined;
    }
    try {
      const parsed = JSON.parse(value) as Record<string, [number, number]>;
      return Object.entries(parsed)
        .map(([url, size]) => ({
          area: Array.isArray(size) ? Number(size[0]) * Number(size[1]) : 0,
          url,
        }))
        .sort((a, b) => b.area - a.area)[0]?.url;
    } catch {
      return undefined;
    }
  }

  function srcsetImage(value: string | undefined): string | undefined {
    if (!value) {
      return undefined;
    }
    return value
      .split(",")
      .map((candidate) => {
        const [url, descriptor] = candidate.trim().split(/\s+/);
        return {
          score: Number(descriptor?.replace(/[wx]$/i, "")) || 1,
          url,
        };
      })
      .sort((a, b) => b.score - a.score)[0]?.url;
  }

  function imageFromSelectors(selectors: string[]): string | undefined {
    for (const selector of selectors) {
      const element = document.querySelector(selector);
      if (!element) {
        continue;
      }
      const dynamic = dynamicImage(element.getAttribute("data-a-dynamic-image") ?? undefined);
      const fromSrcset = srcsetImage(
        element.getAttribute("srcset") ?? element.getAttribute("data-srcset") ?? undefined,
      );
      const raw =
        dynamic ||
        element.getAttribute("data-old-hires") ||
        element.getAttribute("data-src") ||
        element.getAttribute("src") ||
        fromSrcset;
      const imageUrl = absoluteUrl(compact(raw ?? undefined));
      if (imageUrl) {
        return imageUrl;
      }
    }
    return undefined;
  }

  function detectCurrency(value: string): string | undefined {
    if (/₮|MNT/i.test(value)) {
      return "MNT";
    }
    if (/¥|円|JPY/i.test(value)) {
      return "JPY";
    }
    if (/\$|USD/i.test(value)) {
      return "USD";
    }
    if (/€|EUR/i.test(value)) {
      return "EUR";
    }
    if (/£|GBP/i.test(value)) {
      return "GBP";
    }
    return undefined;
  }

  function parseAmount(raw: string, currency: string | undefined): number | undefined {
    const s = raw.replace(/[^\d.,]/g, "");
    if (!s) {
      return undefined;
    }
    // Yen has no minor unit: strip every separator and read an integer.
    if (currency === "JPY") {
      return Number(s.replace(/[.,]/g, "")) || undefined;
    }
    // Otherwise treat a trailing .NN or ,NN (1-2 digits) as the decimal part and
    // every other separator as a thousands separator.
    const dec = Math.max(s.lastIndexOf("."), s.lastIndexOf(","));
    const trailing = dec === -1 ? 0 : s.length - dec - 1;
    if (dec !== -1 && trailing >= 1 && trailing <= 2) {
      const intPart = s.slice(0, dec).replace(/[.,]/g, "");
      return Number(`${intPart}.${s.slice(dec + 1)}`) || undefined;
    }
    return Number(s.replace(/[.,]/g, "")) || undefined;
  }

  function priceFromText(value: string | undefined): { price?: number; currency?: string } {
    if (!value) {
      return {};
    }
    const currency = detectCurrency(value);
    const prefixMatch = value.match(
      /(US\s*\$|C\$|A\$|\$|USD|MNT|₮|€|EUR|£|GBP|¥|￥|円|JPY)\s*([0-9][0-9.,]*[0-9]|[0-9])/i,
    );
    if (prefixMatch) {
      const parsedCurrency = detectCurrency(prefixMatch[1]) || currency;
      return {
        currency: parsedCurrency,
        price: parseAmount(prefixMatch[2], parsedCurrency),
      };
    }
    const suffixMatch = value.match(
      /([0-9][0-9.,]*[0-9]|[0-9])\s*(USD|MNT|EUR|GBP|JPY|円|CAD|AUD)/i,
    );
    if (suffixMatch) {
      const parsedCurrency = detectCurrency(suffixMatch[2]) || currency;
      return {
        currency: parsedCurrency,
        price: parseAmount(suffixMatch[1], parsedCurrency),
      };
    }
    const numMatch = value.match(/[0-9][0-9.,]*[0-9]|[0-9]/);
    if (!numMatch) {
      return {};
    }
    return {
      currency,
      price: parseAmount(numMatch[0], currency),
    };
  }

  function numberFromText(value: string | undefined): number | undefined {
    const match = value?.match(/([0-9][0-9,.]*)/);
    return match ? Number(match[1].replace(/,/g, "")) : undefined;
  }

  function sellerRatingFromPercent(value: string | undefined): number | undefined {
    const match = value?.match(/([0-9]{1,3}(?:[.,][0-9]+)?)\s*%\s*(?:positive|feedback)?/i);
    if (!match) {
      return undefined;
    }
    const percent = Number(match[1].replace(",", "."));
    if (!Number.isFinite(percent) || percent < 0 || percent > 100) {
      return undefined;
    }
    return Math.round((percent / 20) * 10) / 10;
  }

  function sellerReviewCountFromText(value: string | undefined): number | undefined {
    const match = value?.match(
      /([0-9][0-9,.]*)\s*(?:feedback|seller reviews?|shop reviews?|reviews?)/i,
    );
    return match ? Number(match[1].replace(/[,.](?=\d{3}\b)/g, "")) : undefined;
  }

  function cleanSeller(value: string | undefined): string | undefined {
    return compact(
      value
        ?.replace(/^Visit the\s+/i, "")
        .replace(/^Brand:\s*/i, "")
        .replace(/^Sold by\s*/i, "")
        .replace(/^Seller:\s*/i, "")
        .replace(/^Shop\s+/i, "")
        .replace(/\s+on\s+Etsy$/i, "")
        .replace(/\s+\|\s*eBay$/i, "")
        .replace(/\s+Storefront$/i, " Store"),
    );
  }

  function texts(selector: string): string[] {
    return Array.from(document.querySelectorAll(selector))
      .map((element) => compact(element.textContent ?? undefined))
      .filter((value): value is string => Boolean(value));
  }

  function reviewsFromDom(): Array<{ text: string; rating?: number; verified_purchase?: boolean }> {
    const nodes = Array.from(
      document.querySelectorAll(
        [
          '[data-hook="review"]',
          '[data-hook="cmps-review"]',
          'div[id^="customer_review-"]',
          'div[id^="customer_review_foreign-"]',
          '[data-review-region]',
          '[data-review-text]',
          '[data-testid*="review"]',
          '.review-item-content',
          '.review',
        ].join(", "),
      ),
    ).slice(0, 40);
    const out: Array<{ text: string; rating?: number; verified_purchase?: boolean }> = [];
    const seen = new Set<string>();
    for (const node of nodes) {
      const body = cleanReviewBody(
        // Current Amazon layout uses the reviewText / reviewTextContainer hooks;
        // the legacy review-body / review-collapsed hooks are kept as fallbacks.
        node.querySelector('[data-hook="reviewText"] span')?.textContent ||
          node.querySelector('[data-hook="reviewText"]')?.textContent ||
          node.querySelector('[data-hook="reviewTextContainer"] span')?.textContent ||
          node.querySelector('[data-hook="reviewTextContainer"]')?.textContent ||
          node.querySelector('[data-hook="review-body"] span')?.textContent ||
          node.querySelector('[data-hook="review-body"]')?.textContent ||
          node.querySelector('[data-hook="review-collapsed"] span')?.textContent ||
          node.querySelector(".review-text-content span")?.textContent ||
          node.querySelector(".review-text-content, .cr-original-review-text")?.textContent ||
          node.querySelector('[data-review-text]')?.textContent ||
          node.querySelector(".review-item-content")?.textContent ||
          node.querySelector("p")?.textContent ||
          node.textContent ||
          undefined,
      );
      if (!body || body.length < 20 || seen.has(body)) {
        continue;
      }
      seen.add(body);
      const ratingAlt =
        node.querySelector(
          '[data-hook*="review-star-rating"] .a-icon-alt, .review-rating .a-icon-alt, i[class*="a-star"] .a-icon-alt',
        )?.textContent ?? "";
      const ratingMatch = ratingAlt.match(/([0-5](?:[.,][0-9])?)\s*(?:out of|\/|つ星|個の)?\s*5?/i);
      out.push({
        text: body.slice(0, 2000),
        rating: ratingMatch ? Number(ratingMatch[1].replace(",", ".")) : undefined,
        verified_purchase: node.querySelector('[data-hook="avp-badge"]') ? true : undefined,
      });
    }
    return out;
  }

  function descriptionFromDom(): string | undefined {
    return compact(
      texts("#feature-bullets li:not(.aok-hidden) .a-list-item").join(". ") ||
        firstText([
          "[data-testid='ux-layout-section__item']",
          "[data-region='listing-page-description']",
          "[data-id='description-text']",
          "#viTabs_0_is",
        ]) ||
        text("#productDescription") ||
        text("#bookDescription_feature_div") ||
        attr("meta[name='description']", "content"),
    );
  }

  function returnPolicyFromDom(): string | undefined {
    return compact(
      firstText([
        "[data-testid='x-returns-minview']",
        "[data-testid='ux-labels-values-Returns']",
        "[data-testid='returns-policy']",
        "[data-policies-return-policy]",
        "[data-region='listing-page-policies']",
        "[data-region='return-policy']",
        "[id*='return-policy']",
        "[class*='return-policy']",
      ]) ||
        text("#productSupportAndReturnPolicy-secondary-content") ||
        text("#productSupportAndReturnPolicy_feature_div") ||
        text("#RETURNS_POLICY_feature_div") ||
        text("#dp-returns-policy_feature_div") ||
        text('[data-hook="returns-policy"]') ||
        text("#returns-policy-anchor-text") ||
        text('a[data-csa-c-content-id*="return"]') ||
        text('[id*="RETURNS_POLICY"]'),
    );
  }

  function titleFromDom(): string | undefined {
    if (isEbay()) {
      return (
        firstText([
          "h1.x-item-title__mainTitle span",
          "[data-testid='x-item-title'] h1",
          "[data-testid='x-item-title']",
          "h1 span.ux-textspans",
          "h1",
        ]) || attr("meta[property='og:title']", "content")
      );
    }
    if (isEtsy()) {
      return (
        firstText([
          "h1[data-buy-box-listing-title]",
          "[data-buy-box-region='title'] h1",
          "h1.wt-text-body-01",
          "h1",
        ]) || attr("meta[property='og:title']", "content")
      );
    }
    return (
      text("#productTitle") ||
      text("[data-feature-name='title'] h1") ||
      text("h1") ||
      attr("meta[property='og:title']", "content") ||
      document.title
    );
  }

  function imageFromDom(): string | undefined {
    if (isEbay()) {
      return imageFromSelectors([
        "img#icImg",
        "[data-testid='ux-image-carousel-item'] img",
        ".ux-image-carousel-item img",
        ".ux-image-carousel img",
        "img[srcset]",
      ]);
    }
    if (isEtsy()) {
      return imageFromSelectors([
        "[data-carousel] img",
        ".listing-page-image-carousel-component img",
        "[data-listing-page-image] img",
        "img[srcset]",
      ]);
    }
    return imageFromSelectors(["#landingImage", "img[srcset]", "img"]);
  }

  function sellerFromDom(): string | undefined {
    if (isEbay()) {
      return cleanSeller(
        firstText([
          "[data-testid='x-sellercard-atf__info'] a",
          "[data-testid='x-sellercard-atf__info']",
          ".x-sellercard-atf__info__about-seller a",
          ".x-sellercard-atf__info__about-seller",
          "[data-testid='ux-seller-section__item'] a",
          "[class*='seller'] a",
        ]),
      );
    }
    if (isEtsy()) {
      return cleanSeller(
        firstText([
          "a[data-buy-box-region='shop-name']",
          "[data-buy-box-region='shop-name'] a",
          "a[data-region='shop-name']",
          "[data-region='shop-name'] a",
          "a[href*='/shop/'][data-region]",
          ".shop-name a",
          "a[href*='/shop/']",
        ]),
      );
    }
    return cleanSeller(
      text("#bylineInfo") ||
        text("#sellerProfileTriggerId") ||
        text("[tabular-attribute-name='Sold by'] .tabular-buybox-text") ||
        text("#merchant-info") ||
        attr("meta[property='product:brand']", "content") ||
        attr("meta[name='brand']", "content"),
    );
  }

  function priceTextFromDom(): string | undefined {
    if (isEbay()) {
      return firstText([
        "[data-testid='x-price-primary'] span",
        ".x-price-primary span",
        ".x-price-approx span",
        "[itemprop='price']",
      ]);
    }
    if (isEtsy()) {
      return (
        firstText([
          "[data-buy-box-region='price']",
          "p[data-buy-box-region='price']",
          ".wt-text-title-03",
          "[data-selector='price-only']",
        ]) || attr("meta[property='product:price:amount']", "content")
      );
    }
    if (isAmazon()) {
      return amazonPriceTextFromDom();
    }
    return (
      text("#corePrice_feature_div .a-price .a-offscreen") ||
      text("#corePrice_feature_div .a-price") ||
      splitAmazonPriceText() ||
      text(".a-price .a-offscreen") ||
      text(".a-price") ||
      text("#priceblock_ourprice") ||
      text("#priceblock_dealprice") ||
      text("#price_inside_buybox") ||
      attr("meta[property='product:price:amount']", "content")
    );
  }

  function amazonPriceTextFromDom(): string | undefined {
    return (
      firstText([
        "#corePriceDisplay_desktop_feature_div .a-price .a-offscreen",
        "#apex_desktop .a-price .a-offscreen",
        "#corePrice_feature_div .a-price .a-offscreen",
        "#desktop_buybox .a-price .a-offscreen",
        "#buybox .a-price .a-offscreen",
        "#newBuyBoxPrice",
        "#priceblock_ourprice",
        "#priceblock_dealprice",
        "#price_inside_buybox",
      ]) ||
      splitAmazonPriceText() ||
      amazonOfferPriceText() ||
      attr("meta[property='product:price:amount']", "content")
    );
  }

  function amazonOfferPriceText(): string | undefined {
    for (const selector of [
      "#desktop_buybox",
      "#buybox",
      "#centerCol",
      "#twister",
      "#variation_color_name",
      "[data-feature-name='twister']",
    ]) {
      const value = text(selector);
      const price = pricePhraseFromText(value);
      if (price) {
        return price;
      }
    }
    return undefined;
  }

  function pricePhraseFromText(value: string | undefined): string | undefined {
    if (!value) {
      return undefined;
    }
    const currencyAmount = String.raw`(?:US\s*\$|C\$|A\$|\$|USD|MNT|₮|€|EUR|£|GBP|¥|￥|円|JPY)\s*[0-9][0-9.,]*`;
    const fromMatch = value.match(
      new RegExp(String.raw`(?:options?|offers?)\s+from\s*(${currencyAmount})`, "i"),
    );
    if (fromMatch) {
      return fromMatch[1];
    }
    const directMatch = value.match(new RegExp(`(${currencyAmount})`, "i"));
    return directMatch?.[1];
  }

  function splitAmazonPriceText(): string | undefined {
    const containers = Array.from(
      document.querySelectorAll(
        "#corePriceDisplay_desktop_feature_div, #apex_desktop, #corePrice_feature_div, #desktop_buybox, #buybox",
      ),
    );
    for (const container of containers) {
      const symbol = compact(container.querySelector(".a-price-symbol")?.textContent);
      const whole = compact(container.querySelector(".a-price-whole")?.textContent)?.replace(/[.,]\s*$/, "");
      const fraction = compact(container.querySelector(".a-price-fraction")?.textContent);
      if (!whole) {
        continue;
      }
      if (symbol && fraction) {
        return `${symbol}${whole}.${fraction}`;
      }
      if (symbol) {
        return `${symbol}${whole}`;
      }
      if (fraction) {
        return `${whole}.${fraction}`;
      }
      return whole;
    }
    return undefined;
  }

  function ratingTextFromDom(): string | undefined {
    if (isEtsy()) {
      return firstText([
        "[data-review-star-rating]",
        "[aria-label*='out of 5 stars']",
        "[aria-label*='stars']",
      ]);
    }
    if (isEbay()) {
      return firstText(["[data-testid*='rating']", "[aria-label*='out of 5 stars']"]);
    }
    return text("#acrPopover") || attr("#acrPopover", "title");
  }

  function reviewCountTextFromDom(): string | undefined {
    if (isEtsy()) {
      return (
        firstAttr(["[data-reviews-total-count]"], "data-reviews-total-count") ||
        firstText(["[data-reviews-total-count]", "a[href*='reviews']", "button[aria-label*='review']"])
      );
    }
    if (isEbay()) {
      return firstText(["[data-testid*='review-count']", "[aria-label*='review']", ".reviews"]);
    }
    return text("#acrCustomerReviewText");
  }

  function sellerFeedbackTextFromDom(): string | undefined {
    if (isEbay()) {
      return compact(
        firstTexts(
          [
            "[data-testid='x-sellercard-atf__data-item']",
            ".x-sellercard-atf__data-item",
            "[data-testid='ux-seller-section__item']",
            "[class*='seller']",
          ],
          8,
        ).join(" "),
      );
    }
    if (isEtsy()) {
      return compact(
        firstTexts(
          ["[data-region='shop-rating']", "[aria-label*='shop reviews']", "a[href*='reviews']"],
          8,
        ).join(" "),
      );
    }
    return undefined;
  }

  // "8K+ bought in past month" style social-proof badge: recent sales volume
  // is a strong marketplace-legitimacy signal, so capture it as an integer.
  function recentPurchasesFromDom(): number | undefined {
    const haystack = compact(
      text("#social-proofing-faceout-title-tk_bought") ||
        text('[id*="social-proofing"]') ||
        document.body.innerText,
    );
    const match = haystack?.match(/([0-9][0-9.,]*)\s*([KkMm])?\+?\s+bought\s+in\s+past\s+month/i);
    if (!match) {
      return undefined;
    }
    const base = Number(match[1].replace(/,/g, ""));
    if (!Number.isFinite(base)) {
      return undefined;
    }
    const unit = match[2]?.toLowerCase();
    const scaled = unit === "k" ? base * 1000 : unit === "m" ? base * 1000000 : base;
    return Math.round(scaled);
  }

  // Amazon hydrates the on-page reviews block after load; if reviews are
  // indicated but no cards are in the DOM yet, wait briefly for them.
  async function waitForReviews(): Promise<void> {
    const present = () =>
      document.querySelector('[data-hook="review"], div[id^="customer_review-"]');
    if (present()) {
      return;
    }
    const indicated = document.querySelector(
      "#acrCustomerReviewText, [data-hook='total-review-count']",
    );
    if (!indicated) {
      return;
    }
    for (let attempt = 0; attempt < 6 && !present(); attempt += 1) {
      await new Promise((resolve) => {
        setTimeout(resolve, 250);
      });
    }
  }

  await waitForReviews();

  const title = titleFromDom();
  const imageUrl = imageFromDom() || absoluteUrl(attr("meta[property='og:image']", "content"));
  const seller = sellerFromDom();
  const priceInfo = priceFromText(priceTextFromDom());
  const sellerFeedback = sellerFeedbackTextFromDom();

  const lang = compact(
    document.documentElement.getAttribute("lang") ||
      attr("meta[http-equiv='content-language']", "content") ||
      attr("meta[property='og:locale']", "content"),
  );

  return {
    title: compact(title),
    imageUrl,
    seller,
    sellerRating: sellerRatingFromPercent(sellerFeedback),
    sellerReviewCount: sellerReviewCountFromText(sellerFeedback),
    price: priceInfo.price,
    currency: priceInfo.currency || attr("meta[property='product:price:currency']", "content"),
    rating: numberFromText(ratingTextFromDom()),
    reviewCount: numberFromText(reviewCountTextFromDom()),
    unitsBoughtRecent: recentPurchasesFromDom(),
    description: descriptionFromDom(),
    returnPolicy: returnPolicyFromDom(),
    reviews: reviewsFromDom(),
    lang,
  };
}

function productMetaFromResult(result: ProductAnalysisResponse): ProductMeta {
  return {
    host: result.product.site || "current page",
    imageUrl: safeImageUrl(result.product.product_image_url ?? undefined),
    title: result.product.product_title,
    seller: result.product.seller_name ?? "Seller not visible",
    url: result.product.url,
  };
}

export function safeImageUrl(value?: string): string | undefined {
  if (!value) {
    return undefined;
  }

  try {
    const url = new URL(value);
    if (!/^https?:$/i.test(url.protocol) || isLocalOrPrivateHost(url.hostname)) {
      return undefined;
    }
    return url.href;
  } catch {
    return undefined;
  }
}

function isLocalOrPrivateHost(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, "").replace(/\.$/, "");
  if (
    host === "localhost" ||
    host === "localhost.localdomain" ||
    host.endsWith(".localhost") ||
    host.endsWith(".local")
  ) {
    return true;
  }

  if (host.includes(":")) {
    return isPrivateIpv6(host);
  }

  return isPrivateIpv4(host);
}

function isPrivateIpv4(host: string): boolean {
  const parts = host.split(".");
  if (parts.length !== 4) {
    return false;
  }

  const octets = parts.map((part) => Number(part));
  if (octets.some((part) => !Number.isInteger(part) || part < 0 || part > 255)) {
    return false;
  }

  const [first, second] = octets;
  return (
    first === 0 ||
    first === 10 ||
    first === 127 ||
    first >= 224 ||
    (first === 100 && second >= 64 && second <= 127) ||
    (first === 169 && second === 254) ||
    (first === 172 && second >= 16 && second <= 31) ||
    (first === 192 && second === 168)
  );
}

function isPrivateIpv6(host: string): boolean {
  return (
    host === "::" ||
    host === "::1" ||
    host.startsWith("fc") ||
    host.startsWith("fd") ||
    host.startsWith("fe80:")
  );
}

function ProductThumbnail({ imageUrl, title }: { imageUrl?: string; title: string }) {
  const [src, setSrc] = useState(imageUrl);

  useEffect(() => {
    setSrc(imageUrl);
  }, [imageUrl]);

  function handleImageError() {
    setSrc(undefined);
  }

  return (
    <div className="product-thumb">
      {src ? (
        <img
          alt={title ? `${title} product image` : "Product image"}
          onError={() => {
            handleImageError();
          }}
          referrerPolicy="no-referrer"
          src={src}
        />
      ) : (
        <I.ShoppingBag size={22} />
      )}
    </div>
  );
}

function ScoreRing({
  score,
  risk,
  size = "normal",
}: {
  score: number;
  risk: RiskKey;
  size?: "normal" | "compact";
}) {
  const radius = 76;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - score / 100);

  return (
    <div
      aria-label={`TrustScore ${score} out of 100, ${riskLabel(risk)}`}
      className={`score-ring score-ring-${size} risk-${risk}`}
      role="img"
    >
      <svg width="168" height="168" viewBox="0 0 168 168">
        <circle cx="84" cy="84" r={radius} className="ring-bg" />
        <circle
          cx="84"
          cy="84"
          r={radius}
          className="ring-fg"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="score-num">
        <div className="num">{score}</div>
        <div className="out-of">TrustScore</div>
      </div>
    </div>
  );
}

type ScanReliability = {
  evidenceLabel: "High evidence" | "Partial evidence" | "Limited evidence" | "Insufficient evidence";
  isWeak: boolean;
  message: string;
};

function scanReliability(result: ProductAnalysisResponse): ScanReliability {
  const confidence = result.confidence;
  const identity = result.product_identity_confidence ?? 1;
  const pageType = result.page_type ?? "product";
  const scoreStatus = result.score_status ?? "scored";
  const missingCore = result.missing_inputs.filter((item) =>
    ["visible review text", "seller profile", "product price"].includes(item),
  ).length;
  const isWeak =
    scoreStatus === "low_evidence_triage" ||
    pageType !== "product" ||
    confidence < 0.55 ||
    identity < 0.75 ||
    missingCore >= 2;

  if (pageType !== "product" || identity < 0.45) {
    return {
      evidenceLabel: "Insufficient evidence",
      isWeak: true,
      message: "Product details may be incomplete or this page may not be a product detail page.",
    };
  }
  if (isWeak) {
    return {
      evidenceLabel: "Limited evidence",
      isWeak: true,
      message: "Treat this as a triage signal because important page evidence was missing or unscored.",
    };
  }
  if (confidence < 0.75) {
    return {
      evidenceLabel: "Partial evidence",
      isWeak: false,
      message: "Some useful product evidence was read, but manual checks are still needed.",
    };
  }
  return {
    evidenceLabel: "High evidence",
    isWeak: false,
    message: "TrustScore read enough visible page evidence for a normal scoring view.",
  };
}

type CoverageStatus = "read" | "partial" | "missing" | "not-scored";
type CoverageItem = {
  detail: string;
  icon: (props: IconProps) => ReactElement;
  label: string;
  status: CoverageStatus;
};

function signalCoverage(result: ProductAnalysisResponse): CoverageItem[] {
  const evidence = new Map(result.evidence.map((item) => [item.component, item] as const));
  const reviews = evidence.get("review_authenticity");
  const seller = evidence.get("seller_reliability");
  const price = evidence.get("price_safety");
  const policy = evidence.get("return_policy_clarity");
  const priceEvidence = price?.evidence ?? [];
  const listedPriceRead = priceEvidence.some((item) => item.toLowerCase().includes("listed price"));
  const priceScored = !result.model_modes.price_safety?.startsWith("not_scored");
  const policyScored = !result.model_modes.return_policy_clarity?.startsWith("not_scored");
  const marketReference = priceEvidence.find((item) =>
    item.toLowerCase().startsWith("market reference found"),
  );
  const marketReferenceFailure = priceEvidence.find((item) =>
    item.toLowerCase().startsWith("no verified market reference"),
  );

  return [
    {
      detail: reviews?.evidence[0] ?? "No visible review text",
      icon: I.ReviewAuth,
      label: "Reviews",
      status: reviews?.evidence.length ? "read" : "missing",
    },
    {
      detail: seller?.evidence[0] ?? "No seller profile",
      icon: I.Seller,
      label: "Seller",
      status: seller?.evidence.length
        ? seller.missing_inputs.length
          ? "partial"
          : "read"
        : "missing",
    },
    {
      detail: price?.evidence[0] ?? "No visible price",
      icon: I.Price,
      label: "Price",
      status: priceScored
        ? listedPriceRead
          ? "read"
          : "missing"
        : price?.evidence.length
          ? "not-scored"
          : "missing",
    },
    {
      detail: policy?.evidence[0] ?? "No return-policy text",
      icon: I.Return,
      label: "Returns",
      status: policyScored ? (policy?.evidence.length ? "read" : "missing") : "not-scored",
    },
    {
      detail: marketReference ?? marketReferenceFailure ?? "No verified market reference found",
      icon: I.Search,
      label: "Market",
      status: marketReference ? "read" : "not-scored",
    },
  ];
}

function statusText(status: CoverageStatus): string {
  if (status === "read") {
    return "Read";
  }
  if (status === "partial") {
    return "Partial";
  }
  if (status === "not-scored") {
    return "Not scored";
  }
  return "Missing";
}

function SignalCoverageSection({ result }: { result: ProductAnalysisResponse }) {
  return (
    <>
      <div className="sec-h">
        <h3>What was read</h3>
      </div>
      <div className="signal-coverage">
        {signalCoverage(result).map((item) => {
          const CoverageIcon = item.icon;
          return (
            <div className={`coverage-item is-${item.status}`} key={item.label}>
              <div className="coverage-icon">
                <CoverageIcon size={13} />
              </div>
              <div className="coverage-copy">
                <div className="lbl">{item.label}</div>
                <div className="sub">{item.detail}</div>
              </div>
              <div className="coverage-status">{statusText(item.status)}</div>
            </div>
          );
        })}
      </div>
    </>
  );
}

function MarketContextSection({ result }: { result: ProductAnalysisResponse }) {
  const context = result.market_context;
  const priceTrace = result.score_trace?.find((item) => item.component === "price_safety");
  if (!context && !priceTrace) {
    return null;
  }
  const priceStatus =
    priceTrace?.status === "not_scored"
      ? priceTrace.evidence[0] || priceTrace.missing_inputs[0] || "Price safety was not scored."
      : priceTrace?.status
        ? `Price safety ${priceTrace.status.replace("_", " ")}.`
        : "Price status unavailable.";

  return (
    <>
      <div className="sec-h">
        <h3>Market context</h3>
      </div>
      <div className="market-context">
        <div>
          <span>Market</span>
          <strong>
            {context?.resolved_country || context?.resolved_market || "Unknown"}
          </strong>
        </div>
        <div>
          <span>Currency</span>
          <strong>
            {context?.listed_currency || "Not visible"} / {context?.expected_currency || "expected"}
          </strong>
        </div>
        <p>{priceStatus}</p>
      </div>
    </>
  );
}

function Header({ state }: { state: ViewState }) {
  return (
      <div className="popup-head">
        <div className="logo">
          <TrustLogo size={28} />
        </div>
      <div>
        <div className="brand-name">AI TrustScore</div>
        <div className="brand-sub">
          {state === "loading" ? "Analyzing..." : "Shopping safety AI"}
        </div>
      </div>
      <div className="head-spacer" />
    </div>
  );
}

function HomeState({
  onAnalyze,
  onRefresh,
  page,
}: {
  onAnalyze: () => void;
  onRefresh: () => void;
  page: CurrentPage;
}) {
  const preview = page.preview;
  const title = preview?.title || page.host;
  const seller = preview?.seller || "Seller not visible";
  const previewPrice = preview
    ? normalizePreviewPrice(preview.price, preview.currency, page.host, page.targetMarket)
    : {};
  const price =
    typeof previewPrice.price === "number"
      ? formatMoney(previewPrice.price, previewPrice.currency, preview?.lang)
      : previewPrice.ignoredCurrency
        ? "Localized price ignored"
        : "Price not visible";

  return (
    <div className="popup-body fade-swap">
      <div className="card product-card">
        <ProductThumbnail imageUrl={preview?.imageUrl} title={title} />
        <div className="product-copy">
          <div className="detected-line">
            <span />
            {preview?.title ? "Product preview ready" : "Ready to analyze URL"}
          </div>
          <div className="product-title">{title}</div>
          <div className="product-meta">{page.host}</div>
        </div>
      </div>

      <div className="preview-facts">
        <div>
          <span>Seller</span>
          <strong>{seller}</strong>
        </div>
        <div>
          <span>Price</span>
          <strong>{price}</strong>
        </div>
        <div>
          <span>Reviews</span>
          <strong>{preview?.reviewCount ? preview.reviewCount.toLocaleString() : "Not visible"}</strong>
        </div>
      </div>

      <div className="section-kicker">Run analysis on</div>
      <div className="analysis-grid">
        {[
          { icon: I.ReviewAuth, label: "Reviews" },
          { icon: I.Seller, label: "Seller" },
          { icon: I.Price, label: "Price" },
          { icon: I.Return, label: "Returns" },
        ].map((item) => {
          const AnalysisIcon = item.icon;
          return (
            <div className="analysis-tile" key={item.label}>
              <AnalysisIcon size={14} />
              {item.label}
            </div>
          );
        })}
      </div>

      <button className="btn btn-primary btn-block" onClick={onAnalyze} type="button">
        <I.Sparkles size={15} />
        Analyze product
      </button>
      <button className="btn btn-secondary btn-block btn-sm refresh-page" onClick={onRefresh} type="button">
        <I.Search size={13} />
        Re-check active tab
      </button>
    </div>
  );
}

function LoadingState({ progress }: { progress: number }) {
  const taskIdx = Math.min(4, Math.floor((progress / 100) * LOADING_TASKS.length));

  return (
    <div className="popup-body fade-swap" role="status" aria-label="Analyzing product evidence">
      <div className="scanner">
        <div className="ring-bg" />
        <div className="ring-fg" />
        <div className="core">
          <TrustLogo size={38} />
        </div>
      </div>

      <div className="loading-title">Analyzing this product</div>
      <div className="loading-copy">
        Reading visible product evidence and scoring only verified signals.
      </div>

      <div className="progress-block">
        <div className="bar">
          <span style={{ width: `${progress}%` }} />
        </div>
        <div className="progress-meta">
          <span>{LOADING_TASKS[taskIdx]}</span>
          <span>{Math.round(progress)}%</span>
        </div>
      </div>

      <div className="tasks">
        {LOADING_TASKS.map((task, index) => (
          <div
            className={`task ${index < taskIdx ? "is-done" : ""} ${index === taskIdx ? "is-active" : ""
              }`}
            key={task}
          >
            <div className="t-dot">
              {index < taskIdx ? <I.Check size={10} sw={3} /> : null}
            </div>
            <div>{task}</div>
            <div className="t-time">{index < taskIdx ? "✓" : index === taskIdx ? "..." : ""}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function DetectingState() {
  return (
    <div className="popup-body fade-swap">
      <div className="empty">
        <div className="empty-ic">
          <I.Search size={26} />
        </div>
        <h3>Checking current page</h3>
        <p>Checking whether this page has visible product evidence.</p>
      </div>
    </div>
  );
}

function SummaryReasons({ reasons, risk }: { reasons: string[]; risk: RiskKey }) {
  const visibleReasons = reasons.slice(0, 2);
  if (!visibleReasons.length) {
    return null;
  }

  return (
    <div className="summary-reasons" aria-label="Main score reasons">
      <div className="summary-reasons-title">Main reasons</div>
      {visibleReasons.map((reason, index) => {
        const tone = reasonTone(index, risk);
        const ReasonIcon =
          tone === "tick" ? I.CheckCircle : tone === "bad" ? I.AlertOctagon : I.AlertTriangle;
        return (
          <div className={`summary-reason is-${tone}`} key={reason}>
            <ReasonIcon size={13} />
            <span>{reason}</span>
          </div>
        );
      })}
    </div>
  );
}

function ResultState({
  onAnalyze,
  onFeedback,
  result,
}: {
  onAnalyze: () => void;
  onFeedback: () => void;
  result: ProductAnalysisResponse;
}) {
  const risk = riskFromLevel(result.risk_level);
  const components = Object.entries(result.component_scores) as Array<
    [keyof ComponentScores, number]
  >;
  const evidenceByComponent = new Map(
    result.evidence.map((item) => [item.component, item] as const),
  );
  const sourceLabel = result.fetch_mode === "extension_dom" ? "Active-tab scan" : "Page scan";
  const isLowConfidence = result.confidence < 0.55;
  const reliability = scanReliability(result);
  const confidencePct = Math.round(result.confidence * 100);

  function isScored(name: keyof ComponentScores): boolean {
    return name !== "user_feedback_history" && !result.model_modes[name]?.startsWith("not_scored");
  }

  function isGrounded(name: keyof ComponentScores): boolean {
    if (!isScored(name)) {
      return false;
    }
    const item = evidenceByComponent.get(name);
    return item ? item.evidence.length > 0 : false;
  }

  return (
    <div className="popup-body fade-swap">
      <div className="card product-card">
        <ProductThumbnail
          imageUrl={productMetaFromResult(result).imageUrl}
          title={productMetaFromResult(result).title}
        />
        <div className="product-copy">
          <div className="detected-line">
            <span />
            {sourceLabel}
          </div>
          <div className="product-title">{productMetaFromResult(result).title}</div>
          <div className="product-meta">
            {productMetaFromResult(result).host} · {productMetaFromResult(result).seller}
          </div>
        </div>
      </div>

      {reliability.isWeak ? (
        <div className="score-summary is-limited">
          <ScoreRing risk={risk} score={result.trust_score} size="compact" />
          <div className="score-summary-copy">
            <div className="score-badge">
              <I.AlertTriangle size={12} />
              {reliability.evidenceLabel}
            </div>
            <h3>Score needs evidence check</h3>
            <p>{reliability.message}</p>
            <div className="score-summary-meta">
              <span>{riskLabel(risk)}</span>
              <span>Confidence {confidencePct}%</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="score-wrap">
          <ScoreRing risk={risk} score={result.trust_score} />
          <div className={`pill pill-${risk}`}>
            <span className="dot" />
            {riskLabel(risk)}
          </div>
          <div className="confidence-line">
            Confidence {confidencePct}% · {reliability.evidenceLabel}
          </div>
        </div>
      )}

      <div className="confidence-note">
        Confidence reflects readable page evidence, not a guarantee of product safety.
      </div>

      <div className={`recco ${isLowConfidence ? "is-caution" : ""}`}>
        <div className="ic">
          {isLowConfidence ? <I.AlertTriangle size={16} /> : <I.Sparkles size={16} />}
        </div>
        <div>
          <h4>{isLowConfidence ? "Evidence Limit" : "Shopping Guidance"}</h4>
          <p>{result.recommendation}</p>
        </div>
      </div>

      {result.missing_inputs.length ? (
        <div className="missing-strip" role="status">
          <I.AlertTriangle size={13} />
          <span>Missing: {result.missing_inputs.slice(0, 4).join(", ")}</span>
        </div>
      ) : null}

      <SummaryReasons reasons={result.top_reasons} risk={risk} />

      <details className="scan-details">
        <summary className="details-summary">
          <span className="details-label">
            <I.Search size={13} />
            <span className="show-details-label">Show full scan details</span>
            <span className="hide-details-label">Hide full scan details</span>
          </span>
          <span className="mono muted">
            {components.filter(([name]) => isGrounded(name)).length} of {components.length} with data
          </span>
        </summary>
        <div className="details-body">
          <SignalCoverageSection result={result} />
          <MarketContextSection result={result} />

          <div className="sec-h">
            <h3>All reasons</h3>
          </div>
          <div className="card reasons-card">
            {result.top_reasons.map((reason, index) => {
              const tone = reasonTone(index, risk);
              const ReasonIcon =
                tone === "tick"
                  ? I.CheckCircle
                  : tone === "bad"
                    ? I.AlertOctagon
                    : I.AlertTriangle;
              return (
                <div className="reason" key={reason}>
                  <div className={tone}>
                    <ReasonIcon size={14} />
                  </div>
                  <div>{reason}</div>
                </div>
              );
            })}
          </div>

          <EvidenceSection evidence={result.evidence} />

          <div className="sec-h">
            <h3>Model breakdown</h3>
          </div>
          <div className="metrics-list">
            {[...components]
              .sort((a, b) => Number(isGrounded(b[0])) - Number(isGrounded(a[0])))
              .map(([name, score]) => {
            const config = COMPONENT_CONFIG[name];
            const MetricIcon = config.icon;
            const grounded = isGrounded(name);
            const tone = grounded ? componentTone(score) : "med";
            const subtitle =
              name === "user_feedback_history"
                ? "Not applied to score"
                : grounded
                  ? metricSubtitle(name, score)
                  : result.model_modes[name]?.startsWith("not_scored")
                    ? "Not scored for this page"
                    : "No visible data on this page";
            return (
              <div className={`metric is-${tone} ${grounded ? "" : "is-nodata"}`} key={name}>
                <div className="ic">
                  <MetricIcon size={14} />
                </div>
                <div className="metric-main">
                  <div className="lbl">{config.label}</div>
                  <div className="sub">{subtitle}</div>
                  <div className={`bar is-${tone}`}>
                    <span style={{ width: `${grounded ? score : 0}%` }} />
                  </div>
                </div>
                <div className="val">{grounded ? score : "—"}</div>
              </div>
            );
              })}
          </div>
        </div>
      </details>

      <div className="result-actions">
        <button className="btn btn-secondary btn-sm" onClick={onAnalyze} type="button">
          <I.Search size={13} />
          Re-scan
        </button>
        <button className="btn btn-secondary btn-sm" onClick={onFeedback} type="button">
          <I.ThumbsUp size={13} />
          Feedback
        </button>
      </div>
    </div>
  );
}

function metricSubtitle(name: keyof ComponentScores, score: number) {
  if (name === "review_authenticity") {
    return score >= 80 ? "Mostly natural patterns" : score >= 50 ? "Some patterns to check" : "Suspicious patterns";
  }
  if (name === "seller_reliability") {
    return score >= 80 ? "Strong seller profile" : score >= 50 ? "Mixed seller signals" : "Weak seller signal";
  }
  if (name === "sentiment") {
    return score >= 80 ? "Mostly positive reviews" : score >= 50 ? "Mixed customer sentiment" : "Negative review signals";
  }
  if (name === "return_policy_clarity") {
    return score >= 80 ? "Policy looks clear" : score >= 50 ? "Manual check useful" : "Policy unclear";
  }
  if (name === "price_safety") {
    return score >= 80 ? "Within safe range" : score >= 50 ? "Compare with market" : "Unusual price risk";
  }
  return "Not applied to score";
}

function EvidenceSection({ evidence }: { evidence: ComponentEvidence[] }) {
  const visible = evidence.filter((item) => item.component !== "user_feedback_history").slice(0, 5);
  if (!visible.length) {
    return null;
  }

  return (
    <>
      <div className="sec-h">
        <h3>Evidence used</h3>
      </div>
      <div className="evidence-list">
        {visible.map((item) => {
          const config = COMPONENT_CONFIG[item.component];
          const EvidenceIcon = config.icon;
          return (
            <div className="evidence-item" key={item.component}>
              <div className="ic">
                <EvidenceIcon size={13} />
              </div>
              <div className="evidence-copy">
                <div className="lbl">{config.label}</div>
                <div className="sub">{item.evidence[0] || item.summary}</div>
                {item.evidence.slice(1, 4).map((detail) => (
                  <div className="sub" key={detail}>{detail}</div>
                ))}
                {item.missing_inputs.length ? (
                  <div className="miss">Missing {item.missing_inputs.slice(0, 2).join(", ")}</div>
                ) : null}
              </div>
              <div className="confidence-chip">{Math.round(item.confidence * 100)}%</div>
            </div>
          );
        })}
      </div>
    </>
  );
}

function FeedbackBlock({ scanId }: { scanId: string }) {
  const [vote, setVote] = useState<"yes" | "no" | null>(null);
  const [comment, setComment] = useState("");
  const [selectedChip, setSelectedChip] = useState<(typeof FEEDBACK_CHIPS)[number] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState<"saved" | "accepted" | null>(null);

  async function handleSubmit(
    nextVote = vote,
    chip: typeof selectedChip = selectedChip,
    note = comment,
  ) {
    if (!nextVote) {
      return;
    }

    setError(null);
    setIsSubmitting(true);
    try {
      const browserId = await getOrCreateBrowserId();
      const response = await submitFeedback({
        scan_id: scanId,
        browser_id: browserId,
        helpful: nextVote === "yes",
        issue_category: chip?.issue || (nextVote === "no" ? "other" : undefined),
        corrected_component: chip?.component,
        comment: note || undefined,
      });
      setFeedbackStatus(response.status);
      setSubmitted(true);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Could not submit feedback.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  if (submitted) {
    return (
      <div className="feedback fade-swap">
        <div className="feedback-thanks">
          <I.CheckCircle size={18} />
          <div>
            <b>Thanks.</b>{" "}
            {feedbackStatus === "saved"
              ? "Your feedback was saved for evaluation."
              : "Your feedback was accepted for this local demo."}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="feedback fade-swap">
      <h4>Was this result helpful?</h4>
      <div className="fb-row">
        <button
          aria-pressed={vote === "yes"}
          className={`fb-btn is-yes ${vote === "yes" ? "is-active" : ""}`}
          disabled={isSubmitting}
          onClick={() => {
            setVote("yes");
            setSelectedChip(null);
            void handleSubmit("yes", null, "");
          }}
          type="button"
        >
          <I.ThumbsUp size={13} />
          Yes
        </button>
        <button
          aria-pressed={vote === "no"}
          className={`fb-btn is-no ${vote === "no" ? "is-active" : ""}`}
          onClick={() => {
            setVote("no");
            setSelectedChip(null);
          }}
          type="button"
        >
          <I.ThumbsDown size={13} />
          No
        </button>
        {vote === "no" ? (
          <button
            className="btn btn-primary btn-sm"
            disabled={isSubmitting}
            onClick={() => {
              void handleSubmit();
            }}
            type="button"
          >
            {isSubmitting ? "Submitting..." : "Submit"}
          </button>
        ) : null}
      </div>
      {vote === "no" ? (
        <div className="fb-chip-grid" aria-label="What was off">
          {FEEDBACK_CHIPS.map((chip) => (
            <button
              aria-pressed={selectedChip?.value === chip.value}
              className={`fb-chip ${selectedChip?.value === chip.value ? "is-active" : ""}`}
              key={chip.value}
              onClick={() => setSelectedChip(chip)}
              type="button"
            >
              {chip.label}
            </button>
          ))}
        </div>
      ) : null}
      {vote === "no" ? (
        <textarea
          className="fb-input"
          onChange={(event) => setComment(event.target.value)}
          placeholder="Add a short note for review (optional)"
          rows={2}
          value={comment}
        />
      ) : null}
      {error ? <div className="helper-text">{error}</div> : null}
    </div>
  );
}

function ErrorState({
  code,
  error,
  onAnalyze,
}: {
  code?: string | null;
  error: string;
  onAnalyze: () => void;
}) {
  return (
    <div className="popup-body fade-swap">
      <div className="empty">
        <div className="empty-ic">
          <I.AlertTriangle size={26} />
        </div>
        <h3>{errorTitle(code)}</h3>
        <p>{errorMessage(code, error)}</p>
        <p>{errorHint(code)}</p>
        <button className="btn btn-secondary btn-block btn-sm" onClick={onAnalyze} type="button">
          <I.Search size={13} />
          Scan again
        </button>
      </div>
    </div>
  );
}

function errorTitle(code?: string | null): string {
  if (code === "product_not_detected") {
    return "Product page not detected";
  }
  if (code === "invalid_product_url") {
    return "Unsupported URL";
  }
  if (code === "product_page_unavailable") {
    return "Page scan blocked";
  }
  if (code === "validation_error") {
    return "Scan details need a refresh";
  }
  return "Could not scan this page";
}

function errorMessage(code: string | null | undefined, fallback: string): string {
  if (code === "product_not_detected") {
    return "TrustScore could not find enough product details on this page.";
  }
  if (code === "invalid_product_url") {
    return "This page type is not supported for product safety scanning.";
  }
  if (code === "product_page_unavailable") {
    return "This shop blocked the page scan before enough product details were readable.";
  }
  if (code === "validation_error") {
    return "The page details changed while the scan was starting.";
  }
  return fallback && !fallback.toLowerCase().includes("backend")
    ? fallback
    : "TrustScore could not complete the scan right now.";
}

function errorHint(code?: string | null): string {
  if (code === "product_not_detected") {
    return "Open a shopping product detail page with visible product, price, seller, or review signals.";
  }
  if (code === "invalid_product_url") {
    return "Use a public HTTP or HTTPS product page. Local, private-network, credentialed, or non-default-port URLs are blocked.";
  }
  if (code === "product_page_unavailable") {
    return "Re-check the active tab so TrustScore can use the product details visible in your browser.";
  }
  if (code === "validation_error") {
    return "Re-check the page, then scan again.";
  }
  return "Refresh the product page and scan again.";
}

function EmptyState({
  reason,
  onRefresh,
}: {
  reason: string;
  onRefresh: () => void;
}) {
  return (
    <div className="popup-body fade-swap">
      <div className="empty">
        <div className="empty-ic">
          <I.Search size={26} />
        </div>
        <h3>No analyzable URL</h3>
        <p>{reason}</p>
        <p>Open an HTTP or HTTPS shopping product page and try again.</p>
        <button className="btn btn-secondary btn-block btn-sm" onClick={onRefresh} type="button">
          <I.Search size={13} />
          Check active tab
        </button>
      </div>
    </div>
  );
}

export async function analyzeProductWithPreviewFallback(
  page: CurrentPage,
): Promise<ProductAnalysisResponse> {
  const browserId = await getOrCreateBrowserId();
  const locale = pageLocale(page);
  const previewProduct = productPayloadFromPreview(page);

  try {
    return await analyzeProduct({
      url: page.url,
      browser_id: browserId,
      locale,
      target_market: page.targetMarket,
    });
  } catch (error) {
    if (
      !(error instanceof ApiError) ||
      !canFallbackToExtractedScan(error) ||
      !previewProduct ||
      !hasStrongProductPreview(page)
    ) {
      throw error;
    }
  }

  return analyzeExtractedProduct({
    product: previewProduct,
    browser_id: browserId,
    locale,
    target_market: page.targetMarket,
  });
}

function canFallbackToExtractedScan(error: ApiError): boolean {
  return error.code === "product_page_unavailable" || error.code === "product_not_detected";
}

function hasStrongProductPreview(page: CurrentPage): boolean {
  const preview = page.preview;
  if (!preview?.title || isUnsupportedProductPageType(page.url)) {
    return false;
  }
  const signals = [
    preview.price !== undefined,
    Boolean(preview.seller),
    preview.rating !== undefined || preview.reviewCount !== undefined,
    Boolean(preview.imageUrl),
    Boolean(preview.returnPolicy),
    Boolean(preview.reviews?.length),
  ].filter(Boolean).length;
  return signals >= 2;
}

function isUnsupportedProductPageType(urlValue: string): boolean {
  try {
    const path = new URL(urlValue).pathname.toLowerCase();
    return (
      path.includes("/review") ||
      path.includes("/reviews") ||
      path.includes("/search") ||
      path.includes("/category") ||
      path.includes("/browse") ||
      path.includes("/cart") ||
      path.includes("/checkout") ||
      path.includes("/account") ||
      path.includes("/signin") ||
      path.includes("/login")
    );
  } catch {
    return true;
  }
}

function App() {
  const [state, setState] = useState<ViewState>("detecting");
  const [currentPage, setCurrentPage] = useState<CurrentPage | null>(null);
  const [result, setResult] = useState<ProductAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorCode, setErrorCode] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const activeRisk = useMemo(() => {
    if (!result) {
      return "med";
    }
    return riskFromLevel(result.risk_level);
  }, [result]);

  useEffect(() => {
    if (state !== "loading") {
      return;
    }

    setProgress(0);
    const id = window.setInterval(() => {
      setProgress((current) => Math.min(95, current + 5 + Math.random() * 8));
    }, 170);

    return () => window.clearInterval(id);
  }, [state]);

  useEffect(() => {
    void handlePageDetection();
  }, []);

  async function handlePageDetection() {
    setState("detecting");
    setError(null);
    setErrorCode(null);
    setResult(null);

    try {
      const page = await getCurrentPageUrl();
      setCurrentPage(page);
      const storedResult = page ? await getStoredResultForPage(page.url) : null;
      if (storedResult) {
        setResult(storedResult);
        setState("result");
        return;
      }
      setState(page ? "home" : "empty");
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : "Could not inspect the current page.";
      setErrorCode(requestError instanceof ApiError ? requestError.code ?? null : null);
      setError(message);
      setState("error");
    }
  }

  async function handleAnalyzeClick() {
    if (!currentPage) {
      setState("empty");
      return;
    }

    setState("loading");
    setError(null);
    setErrorCode(null);

    try {
      const response = await analyzeProductWithPreviewFallback(currentPage);
      await storeLastResult(response);
      setProgress(100);
      window.setTimeout(() => {
        setResult(response);
        setState("result");
      }, 350);
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : "TrustScore could not complete the scan right now.";
      setErrorCode(requestError instanceof ApiError ? requestError.code ?? null : null);
      setError(message);
      setState("error");
    }
  }

  return (
    <main
      aria-live={state === "loading" ? "polite" : "off"}
      className={`popup risk-${activeRisk}`}
      dir={isRtlLanguage(result?.language) ? "rtl" : "ltr"}
    >
      <Header state={state} />
      {state === "detecting" ? <DetectingState /> : null}
      {state === "home" && currentPage ? (
        <HomeState
          onAnalyze={handleAnalyzeClick}
          onRefresh={handlePageDetection}
          page={currentPage}
        />
      ) : null}
      {state === "loading" ? <LoadingState progress={progress} /> : null}
      {state === "result" && result ? (
        <ResultState
          onAnalyze={handleAnalyzeClick}
          onFeedback={() => setState("feedback")}
          result={result}
        />
      ) : null}
      {state === "feedback" && result ? (
        <>
          <ResultState
            onAnalyze={handleAnalyzeClick}
            onFeedback={() => undefined}
            result={result}
          />
          <FeedbackBlock scanId={result.scan_id} />
        </>
      ) : null}
      {state === "error" && error ? (
        <ErrorState
          code={errorCode}
          error={error}
          onAnalyze={currentPage ? handleAnalyzeClick : handlePageDetection}
        />
      ) : null}
      {state === "empty" ? (
        <EmptyState
          onRefresh={handlePageDetection}
          reason="Open an HTTP or HTTPS page first."
        />
      ) : null}
    </main>
  );
}

export default App;
