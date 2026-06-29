// DesignSystem.jsx — colors / type / components reference board.
const I = window.I;

function Swatch({ bg, name, hex, dark, sub }) {
  return (
    <div>
      <div
        className={`ds-swatch${dark ? " dark" : ""}`}
        style={{ background: bg }}
      >
        {hex}
      </div>
      <div className="ds-name">{name}</div>
      {sub && <div className="ds-hex">{sub}</div>}
    </div>
  );
}

function DesignSystemBoard() {
  return (
    <div className="ds">
      {/* Brand */}
      <div className="ds-cell" style={{ gridColumn: "span 12" }}>
        <p className="ds-h">Brand · AI tech</p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(6, 1fr)",
            gap: 12,
          }}
        >
          <Swatch
            bg="linear-gradient(135deg,#4f46e5,#7c3aed,#a855f7)"
            name="AI Gradient"
            hex="indigo→violet"
            sub="Used for primary actions, score ring (high-trust), AI moments"
          />
          <Swatch bg="#4f46e5" name="Brand" hex="#4F46E5" />
          <Swatch bg="#7c3aed" name="Brand 2" hex="#7C3AED" />
          <Swatch bg="#eef2ff" name="Brand 50" hex="#EEF2FF" dark />
          <Swatch bg="#0b1220" name="Ink" hex="#0B1220" />
          <Swatch bg="#64748b" name="Muted" hex="#64748B" />
        </div>
      </div>

      {/* Risk system */}
      <div className="ds-cell" style={{ gridColumn: "span 12" }}>
        <p className="ds-h">Risk system · the heart of the product</p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: 16,
          }}
        >
          {[
            {
              tone: "Low",
              range: "75–100",
              color: "#059669",
              bg: "#ecfdf5",
              dark: false,
              ic: <I.CheckCircle size={18} />,
              copy: "Reviews and seller checks pass. Safe to buy.",
            },
            {
              tone: "Medium",
              range: "45–74",
              color: "#d97706",
              bg: "#fffbeb",
              dark: false,
              ic: <I.AlertTriangle size={18} />,
              copy: "Some signals worth checking before buying.",
            },
            {
              tone: "High",
              range: "0–44",
              color: "#dc2626",
              bg: "#fef2f2",
              dark: false,
              ic: <I.AlertOctagon size={18} />,
              copy: "Multiple red flags — recommend avoiding.",
            },
          ].map((r, i) => (
            <div
              key={i}
              style={{
                border: `1px solid ${r.color}33`,
                background: r.bg,
                borderRadius: 12,
                padding: 14,
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  color: r.color,
                }}
              >
                {r.ic}
                <div style={{ fontWeight: 600, fontSize: 14 }}>
                  {r.tone} risk
                </div>
                <div
                  className="mono"
                  style={{ marginLeft: "auto", fontSize: 12 }}
                >
                  {r.range}
                </div>
              </div>
              <div
                style={{
                  marginTop: 8,
                  fontSize: 12.5,
                  color: "var(--ink-2)",
                  lineHeight: 1.45,
                }}
              >
                {r.copy}
              </div>
              <div style={{ marginTop: 10, display: "flex", gap: 6 }}>
                <span
                  style={{
                    background: r.color,
                    color: "white",
                    padding: "3px 9px",
                    borderRadius: 999,
                    fontSize: 11,
                    fontWeight: 600,
                  }}
                >
                  {r.tone}
                </span>
                <span
                  style={{
                    background: "white",
                    color: r.color,
                    padding: "3px 9px",
                    borderRadius: 999,
                    fontSize: 11,
                    fontWeight: 600,
                    border: `1px solid ${r.color}33`,
                  }}
                >
                  Pill
                </span>
                <span
                  className="mono"
                  style={{ color: r.color, fontSize: 11, alignSelf: "center" }}
                >
                  {r.color}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Typography */}
      <div className="ds-cell" style={{ gridColumn: "span 7" }}>
        <p className="ds-h">Typography · Geist + Geist Mono</p>
        <div className="ds-type-row">
          <span className="meta">Display · 28/600</span>
          <span
            style={{ fontSize: 28, fontWeight: 600, letterSpacing: "-0.02em" }}
          >
            Trust at a glance.
          </span>
        </div>
        <div className="ds-type-row">
          <span className="meta">Title · 16/600</span>
          <span
            style={{ fontSize: 16, fontWeight: 600, letterSpacing: "-0.01em" }}
          >
            Model breakdown
          </span>
        </div>
        <div className="ds-type-row">
          <span className="meta">Body · 13/400</span>
          <span style={{ fontSize: 13 }}>
            94% of reviews appear authentic and human-written.
          </span>
        </div>
        <div className="ds-type-row">
          <span className="meta">Eyebrow · 11/600 · UPPER</span>
          <span
            style={{
              fontSize: 11,
              fontWeight: 600,
              color: "var(--muted)",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
            }}
          >
            Top reasons
          </span>
        </div>
        <div className="ds-type-row">
          <span className="meta">Numeric · Mono</span>
          <span
            className="mono"
            style={{ fontSize: 24, letterSpacing: "-0.03em", fontWeight: 600 }}
          >
            74
            <span style={{ fontSize: 12, color: "var(--muted)" }}> / 100</span>
          </span>
        </div>
      </div>

      {/* Spacing & Radii */}
      <div className="ds-cell" style={{ gridColumn: "span 5" }}>
        <p className="ds-h">Spacing · 4-pt grid</p>
        <div
          style={{
            display: "flex",
            alignItems: "flex-end",
            gap: 10,
            marginBottom: 18,
          }}
        >
          {[4, 8, 12, 16, 24, 32].map((n) => (
            <div
              key={n}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 6,
              }}
            >
              <div
                style={{
                  width: n,
                  height: n,
                  background: "var(--brand)",
                  borderRadius: 2,
                }}
              />
              <div
                className="mono"
                style={{ fontSize: 10, color: "var(--muted)" }}
              >
                {n}
              </div>
            </div>
          ))}
        </div>
        <p className="ds-h" style={{ marginTop: 10 }}>
          Radii
        </p>
        <div style={{ display: "flex", gap: 10 }}>
          {[
            { r: 6, n: "r-1" },
            { r: 10, n: "r-2" },
            { r: 14, n: "r-3" },
            { r: 18, n: "r-4" },
            { r: 999, n: "pill" },
          ].map((x) => (
            <div
              key={x.n}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 4,
              }}
            >
              <div
                style={{
                  width: 36,
                  height: 36,
                  background: "var(--brand-50)",
                  border: "1px solid var(--brand-100)",
                  borderRadius: x.r,
                }}
              />
              <div
                className="mono"
                style={{ fontSize: 10, color: "var(--muted)" }}
              >
                {x.n}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Buttons */}
      <div className="ds-cell" style={{ gridColumn: "span 6" }}>
        <p className="ds-h">Buttons</p>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button className="btn btn-primary">
              <I.Sparkles size={14} /> Primary
            </button>
            <button className="btn btn-secondary">Secondary</button>
            <button className="btn btn-ghost">Ghost</button>
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button className="btn btn-primary btn-sm">Small primary</button>
            <button className="btn btn-secondary btn-sm">
              Small secondary
            </button>
            <button className="btn btn-secondary btn-sm">
              <I.Refresh size={12} /> Re-scan
            </button>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <span className="pill pill-low">
              <span className="dot" />
              Low risk
            </span>
            <span className="pill pill-med">
              <span className="dot" />
              Medium risk
            </span>
            <span className="pill pill-high">
              <span className="dot" />
              High risk
            </span>
            <span className="pill pill-brand">
              <I.Sparkles size={10} />
              AI
            </span>
          </div>
        </div>
      </div>

      {/* Iconography */}
      <div className="ds-cell" style={{ gridColumn: "span 6" }}>
        <p className="ds-h">Iconography · 1.8px stroke, 24px grid</p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(6, 1fr)",
            gap: 12,
          }}
        >
          {[
            { ic: <I.Shield />, n: "Shield" },
            { ic: <I.Sparkles />, n: "AI" },
            { ic: <I.ReviewAuth />, n: "Reviews" },
            { ic: <I.Seller />, n: "Seller" },
            { ic: <I.Sentiment />, n: "Sentiment" },
            { ic: <I.Price />, n: "Price" },
            { ic: <I.Return />, n: "Returns" },
            { ic: <I.Brain />, n: "Model" },
            { ic: <I.CheckCircle />, n: "Pass" },
            { ic: <I.AlertTriangle />, n: "Warn" },
            { ic: <I.AlertOctagon />, n: "Block" },
            { ic: <I.Lock />, n: "Privacy" },
          ].map((x, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 6,
              }}
            >
              <div
                style={{
                  width: 36,
                  height: 36,
                  borderRadius: 8,
                  background: "var(--surface-2)",
                  border: "1px solid var(--border-2)",
                  color: "var(--ink-2)",
                  display: "grid",
                  placeItems: "center",
                }}
              >
                {x.ic}
              </div>
              <div
                style={{
                  fontSize: 10,
                  color: "var(--muted)",
                  fontFamily: "var(--font-mono)",
                }}
              >
                {x.n}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Why this design rationale */}
      <div
        className="ds-cell"
        style={{
          gridColumn: "span 12",
          background: "var(--brand-soft-grad)",
          border: "1px solid var(--brand-100)",
        }}
      >
        <p className="ds-h" style={{ color: "var(--brand)" }}>
          Why this design improves trust & usability
        </p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: 16,
          }}
        >
          {[
            {
              h: "One number, one decision",
              p: "The 0–100 score is the largest element on the screen. Users get an answer in under a second; the breakdown is for those who want to dig deeper.",
            },
            {
              h: "Color = action, not decoration",
              p: "Green / amber / red map to a single mental model — buy, check, avoid. The same colors appear on the floating badge so users can decide before opening the popup.",
            },
            {
              h: "Explainable AI",
              p: "Three short reasons + five model signals make the AI verdict auditable. Users see what was checked, not just a magic score.",
            },
            {
              h: "Calm, neutral chrome",
              p: "Soft shadows, restrained palette, generous spacing. The interface stays out of the way so the verdict reads first; the AI gradient appears only on brand and primary actions.",
            },
          ].map((b, i) => (
            <div key={i}>
              <div
                style={{
                  fontSize: 13.5,
                  fontWeight: 600,
                  color: "var(--ink)",
                  marginBottom: 4,
                  letterSpacing: "-0.005em",
                }}
              >
                {b.h}
              </div>
              <div
                style={{
                  fontSize: 12.5,
                  color: "var(--ink-2)",
                  lineHeight: 1.5,
                  textWrap: "pretty",
                }}
              >
                {b.p}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

window.DesignSystemBoard = DesignSystemBoard;
