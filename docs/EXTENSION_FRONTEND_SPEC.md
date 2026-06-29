# Browser Extension and Frontend Specification

## Purpose
Build a Chrome Extension with a React popup UI that previews the active-tab product title, image, and seller, then analyzes the product through FastAPI.

## Frameworks
- Browser Extension: Chrome Extension Manifest V3.
- Frontend: React + TypeScript + Vite.
- Styling: simple CSS, Tailwind optional.

## Extension files

```text
apps/extension/
  public/
    icons/
  src/
    background/
      background.ts
    popup/
      App.tsx
      components/
        ScoreBadge.tsx
        RiskLabel.tsx
        ReasonsList.tsx
        FeedbackButtons.tsx
      main.tsx
    shared/
      types.ts
      apiClient.ts
  manifest.json
  package.json
  vite.config.ts
```

## Manifest V3 example

```json
{
  "manifest_version": 3,
  "name": "AI TrustScore",
  "version": "0.1.0",
  "description": "AI TrustScore Browser Extension for online shopping.",
  "permissions": ["activeTab", "scripting", "storage"],
  "host_permissions": ["http://localhost:8000/*", "http://127.0.0.1:8000/*"],
  "background": {
    "service_worker": "background.js",
    "type": "module"
  },
  "action": {
    "default_popup": "popup.html",
    "default_title": "AI TrustScore"
  }
}
```

The extension does not need product-page content-script host permissions. It uses `activeTab` plus `scripting` only after the user opens the popup, reads a small product preview from the active tab, and keeps required host permissions limited to the backend API origin. Product images are rendered directly in the popup after local/private URL filtering; the extension does not proxy product images through the background worker.

## Product analysis request shape

```ts
export type ProductScanPayload = {
  url: string;
};
```

## Backend extraction strategy

FastAPI first fetches the URL, parses product metadata and HTML, optionally
renders JavaScript-heavy pages, then returns product metadata with the
TrustScore response. If the retailer blocks backend fetching, the popup falls
back to `POST /api/v1/scan-extracted` with only the active-tab preview fields
needed for a demo scan: URL, title, image URL, seller, price, rating, and review
count when visible.

Missing extracted fields should reduce confidence, not break prediction.

## Popup UI requirements

### Loading state
Show:

```text
Analyzing product page...
```

### Result state
Show:
- TrustScore number.
- Risk level badge.
- Confidence.
- Top 3 reasons.
- Recommendation.
- Component scores in a small list or bar.
- Helpful / Not Helpful buttons.

### Error state
Show:

```text
Could not analyze this page. Open a product page with visible reviews and try again.
```

## Popup result example

```text
TrustScore: 68 / 100
Risk Level: Medium Risk
Confidence: 76%

Top reasons:
1. Some reviews look repeated or suspicious.
2. Return policy is unclear.
3. Several reviews mention product quality problems.

Recommendation:
Check return policy and seller details before buying.

Was this helpful? Yes / No
```

## API client behavior
- Call `POST /api/v1/scan` with `{ url }`; create or reuse a `browser_id` only when submitting feedback.
- Store last result in `chrome.storage.local`.
- Submit feedback to `POST /api/v1/feedback`.
- Create and send an anonymous browser ID only when feedback is submitted.
- Parse backend `detail.code` and `detail.message` for user-facing errors.
- Handle backend unavailable state.

## UI color recommendation
- Low Risk: green.
- Medium Risk: amber/yellow.
- High Risk: red.
- Unknown / low confidence: gray.

## Privacy behavior
The extension should not read or send:
- Passwords.
- Payment information.
- Cookies.
- Private user account data.
- Personal browsing history.

Only current product-page data should be sent for analysis.
