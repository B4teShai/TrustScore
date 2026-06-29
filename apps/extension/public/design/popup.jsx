// Popup.jsx — TrustScore browser-extension popup, all 6 states.
// Renders a full popup chrome (header / body / footer) sized at 360px wide.
//
// Exposes:
//   <Popup state="home" | "loading" | "result" | "feedback" | "empty"
//          score={number} risk="low|med|high"
//          interactive  onStateChange />
//
//   <FloatingBadge score risk />
//
// Designed for use both inside design-canvas artboards (static) and as an
// interactive flow (interactive=true cycles through states).

const { useState, useEffect, useRef } = React;
const I = window.I;

const RISK_META = {
  low:  { label: 'Low risk',    color: 'var(--low)',  pillCls: 'pill-low',  ring: '#10b981',
          gradient: ['#10b981', '#059669'],
          recommendation: 'Looks safe to buy. Reviews and seller checks all pass.' },
  med:  { label: 'Medium risk', color: 'var(--med)',  pillCls: 'pill-med',  ring: '#f59e0b',
          gradient: ['#f59e0b', '#d97706'],
          recommendation: 'Check seller details and recent reviews before buying.' },
  high: { label: 'High risk',   color: 'var(--high)', pillCls: 'pill-high', ring: '#ef4444',
          gradient: ['#ef4444', '#dc2626'],
          recommendation: 'We recommend avoiding this listing — multiple red flags detected.' },
};

const riskFromScore = (s) => s >= 75 ? 'low' : s >= 45 ? 'med' : 'high';

// ───────── Score ring ─────────
function ScoreRing({ score, risk, size = 168, animate = true }) {
  const meta = RISK_META[risk];
  const r = (size - 16) / 2;
  const C = 2 * Math.PI * r;
  const targetOff = C * (1 - score / 100);
  const [off, setOff] = useState(animate ? C : targetOff);
  useEffect(() => {
    if (!animate) { setOff(targetOff); return; }
    const id = requestAnimationFrame(() => setOff(targetOff));
    return () => cancelAnimationFrame(id);
  }, [targetOff, animate]);

  const id = `grad-${risk}-${size}`;
  return (
    <div className="score-ring" style={{ width: size, height: size }}>
      <svg width={size} height={size}>
        <defs>
          <linearGradient id={id} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor={meta.gradient[0]} />
            <stop offset="100%" stopColor={meta.gradient[1]} />
          </linearGradient>
        </defs>
        <circle cx={size/2} cy={size/2} r={r}
          fill="none" stroke="#eef0f4" strokeWidth="10" />
        <circle cx={size/2} cy={size/2} r={r}
          fill="none" stroke={`url(#${id})`} strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={C} strokeDashoffset={off}
          style={{ transition: 'stroke-dashoffset 1.1s cubic-bezier(.2,.7,.3,1)' }}/>
      </svg>
      <div className="score-num">
        <div className="num">{score}</div>
        <div className="out-of">/ 100 · TrustScore</div>
      </div>
    </div>
  );
}

// ───────── Header ─────────
function Header({ onClose, state }) {
  return (
    <div className="popup-head">
      <div className="logo"><I.Shield size={16} sw={2.2}/></div>
      <div>
        <div className="brand-name">AI TrustScore</div>
        <div className="brand-sub">{state === 'loading' ? 'Analyzing…' : 'Shopping safety AI'}</div>
      </div>
      <div className="head-spacer" />
      <button className="head-icon" title="Settings"><I.Settings size={16}/></button>
    </div>
  );
}

// ───────── States ─────────
function HomeState({ onAnalyze, productMeta }) {
  return (
    <div className="popup-body fade-swap">
      <div>
        <div className="card" style={{ padding: 14, display: 'grid', gridTemplateColumns: '52px 1fr', gap: 12, alignItems: 'center' }}>
          <div style={{
            width: 52, height: 52, borderRadius: 10,
            background: '#f1f5f9', display: 'grid', placeItems: 'center',
            color: 'var(--muted)', fontSize: 22,
          }}>
            <I.ShoppingBag size={22}/>
          </div>
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: 11, color: 'var(--low)', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: 5 }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--low)' }} />
              Product page detected
            </div>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--ink)', marginTop: 3, lineHeight: 1.3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {productMeta.title}
            </div>
            <div style={{ fontSize: 11.5, color: 'var(--muted)', marginTop: 2 }}>
              {productMeta.host} · {productMeta.seller}
            </div>
          </div>
        </div>

        <div style={{ margin: '14px 2px 6px', fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 600 }}>
          Run analysis on
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
          {[
            { ic: <I.ReviewAuth size={14}/>, lbl: 'Reviews' },
            { ic: <I.Seller size={14}/>, lbl: 'Seller' },
            { ic: <I.Price size={14}/>, lbl: 'Price' },
            { ic: <I.Return size={14}/>, lbl: 'Returns' },
          ].map((x, i) => (
            <div key={i} style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '8px 10px', borderRadius: 8,
              background: 'var(--surface-2)',
              border: '1px solid var(--border-2)',
              fontSize: 12, color: 'var(--ink-2)', fontWeight: 500,
            }}>
              <span style={{ color: 'var(--brand)' }}>{x.ic}</span>{x.lbl}
            </div>
          ))}
        </div>

        <button className="btn btn-primary btn-block" style={{ marginTop: 14 }} onClick={onAnalyze}>
          <I.Sparkles size={15}/> Analyze product
        </button>
        <div style={{ textAlign: 'center', fontSize: 11, color: 'var(--muted)', marginTop: 10 }}>
          Takes ~3 seconds · Free for the first 25 scans
        </div>
      </div>
    </div>
  );
}

function LoadingState({ progress, taskIdx }) {
  const tasks = [
    'Scraping product metadata',
    'Checking reviews for authenticity',
    'Verifying seller history',
    'Comparing price across vendors',
    'Reading return policy',
  ];
  return (
    <div className="popup-body fade-swap">
      <div>
        <div className="scanner">
          <div className="ring-bg"/>
          <div className="ring-fg"/>
          <div className="core"><I.Brain size={26} sw={1.6}/></div>
        </div>
        <div style={{ textAlign: 'center', fontSize: 14, fontWeight: 600, letterSpacing: '-0.01em', color: 'var(--ink)' }}>
          Analyzing this product
        </div>
        <div style={{ textAlign: 'center', fontSize: 12.5, color: 'var(--muted)', marginTop: 4, lineHeight: 1.4 }}>
          BERT + Random Forest models running on reviews, seller, price, and policy.
        </div>

        <div style={{ marginTop: 14 }}>
          <div className="bar"><span style={{ width: `${progress}%`, transition: 'width 0.4s ease' }}/></div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--muted)', marginTop: 6, fontFamily: 'var(--font-mono)' }}>
            <span>{tasks[taskIdx]}</span>
            <span>{Math.round(progress)}%</span>
          </div>
        </div>

        <div className="tasks">
          {tasks.map((t, i) => (
            <div key={i} className={`task ${i < taskIdx ? 'is-done' : i === taskIdx ? 'is-active' : ''}`}>
              <div className="t-dot">{i < taskIdx ? <I.Check size={10} sw={3}/> : null}</div>
              <div>{t}</div>
              <div className="t-time">{i < taskIdx ? '✓' : i === taskIdx ? '…' : ''}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

const REASONS_BY_RISK = {
  low: [
    { tone: 'tick', text: '94% of recent reviews appear authentic and human-written.' },
    { tone: 'tick', text: 'Seller has 4 years of history with consistent fulfillment.' },
    { tone: 'tick', text: 'Price is within 6% of the median across 12 retailers.' },
  ],
  med: [
    { tone: 'tick', text: 'Seller has been active for 2 years with mostly positive ratings.' },
    { tone: 'warn', text: '18% of reviews flagged as likely AI-generated or templated.' },
    { tone: 'warn', text: 'Return policy lists a 14-day window with restocking fees.' },
  ],
  high: [
    { tone: 'bad', text: '47% of reviews show duplicate or paid-promotion patterns.' },
    { tone: 'bad', text: 'Seller registered 3 weeks ago — no verified track record.' },
    { tone: 'warn', text: 'Price is 62% below market median — common scam indicator.' },
  ],
};

const METRICS_BY_RISK = {
  low: [
    { ic: <I.ReviewAuth size={14}/>, lbl: 'Review authenticity', sub: 'BERT classifier · 1,284 reviews', val: 92, tone: 'low' },
    { ic: <I.Seller size={14}/>, lbl: 'Seller reliability', sub: 'Verified · 4y history', val: 88, tone: 'low' },
    { ic: <I.Sentiment size={14}/>, lbl: 'Sentiment score', sub: 'Mostly positive · stable', val: 81, tone: 'low' },
    { ic: <I.Price size={14}/>, lbl: 'Price safety', sub: 'Within market range', val: 76, tone: 'low' },
    { ic: <I.Return size={14}/>, lbl: 'Return policy clarity', sub: '30-day full refund', val: 84, tone: 'low' },
  ],
  med: [
    { ic: <I.ReviewAuth size={14}/>, lbl: 'Review authenticity', sub: '18% likely templated', val: 64, tone: 'med' },
    { ic: <I.Seller size={14}/>, lbl: 'Seller reliability', sub: '2y · mostly positive', val: 71, tone: 'med' },
    { ic: <I.Sentiment size={14}/>, lbl: 'Sentiment score', sub: 'Mixed · 22% negative', val: 68, tone: 'med' },
    { ic: <I.Price size={14}/>, lbl: 'Price safety', sub: '12% below median', val: 79, tone: 'low' },
    { ic: <I.Return size={14}/>, lbl: 'Return policy clarity', sub: '14d · restocking fee', val: 58, tone: 'med' },
  ],
  high: [
    { ic: <I.ReviewAuth size={14}/>, lbl: 'Review authenticity', sub: '47% duplicates detected', val: 22, tone: 'high' },
    { ic: <I.Seller size={14}/>, lbl: 'Seller reliability', sub: 'New seller · 3 weeks', val: 18, tone: 'high' },
    { ic: <I.Sentiment size={14}/>, lbl: 'Sentiment score', sub: 'Polarized · suspicious', val: 41, tone: 'high' },
    { ic: <I.Price size={14}/>, lbl: 'Price safety', sub: '62% below median', val: 27, tone: 'high' },
    { ic: <I.Return size={14}/>, lbl: 'Return policy clarity', sub: 'Not clearly stated', val: 35, tone: 'high' },
  ],
};

function ResultState({ score, risk, onFeedback }) {
  const meta = RISK_META[risk];
  const reasons = REASONS_BY_RISK[risk];
  const metrics = METRICS_BY_RISK[risk];
  const ToneIcon = ({ tone }) => tone === 'tick' ? <I.CheckCircle size={14} sw={2}/> : tone === 'warn' ? <I.AlertTriangle size={14} sw={2}/> : <I.AlertOctagon size={14} sw={2}/>;
  return (
    <div className="popup-body fade-swap">
      <div>
        <div className="score-wrap">
          <ScoreRing score={score} risk={risk} />
          <div className={`pill ${meta.pillCls}`} style={{ fontSize: 12, padding: '5px 11px' }}>
            <span className="dot"/> {meta.label}
          </div>
        </div>

        <div className="recco" style={{ marginTop: 14 }}>
          <div className="ic"><I.Sparkles size={16}/></div>
          <div>
            <h4>AI Recommendation</h4>
            <p>{meta.recommendation}</p>
          </div>
        </div>

        <div className="sec-h"><h3>Top reasons</h3></div>
        <div className="card" style={{ padding: '4px 12px' }}>
          {reasons.map((r, i) => (
            <div key={i} className="reason">
              <div className={r.tone}><ToneIcon tone={r.tone}/></div>
              <div>{r.text}</div>
            </div>
          ))}
        </div>

        <div className="sec-h"><h3>Model breakdown</h3><span style={{ fontSize: 11, color: 'var(--muted)' }} className="mono">5 signals</span></div>
        <div style={{ display: 'grid', gap: 7 }}>
          {metrics.map((m, i) => (
            <div key={i} className={`metric is-${m.tone}`}>
              <div className="ic">{m.ic}</div>
              <div style={{ minWidth: 0 }}>
                <div className="lbl">{m.lbl}</div>
                <div className="sub">{m.sub}</div>
                <div className="bar" style={{ marginTop: 6 }}>
                  <span className={`is-${m.tone}`} style={{ width: `${m.val}%`, position: 'absolute', inset: 0, background: m.tone === 'low' ? 'linear-gradient(90deg,#10b981,#059669)' : m.tone === 'med' ? 'linear-gradient(90deg,#fbbf24,#f59e0b)' : 'linear-gradient(90deg,#f87171,#dc2626)' }}/>
                </div>
              </div>
              <div className="val">{m.val}</div>
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', gap: 8, marginTop: 14 }}>
          <button className="btn btn-secondary btn-sm" style={{ flex: 1 }}><I.Refresh size={13}/> Re-scan</button>
          <button className="btn btn-secondary btn-sm" style={{ flex: 1 }} onClick={onFeedback}><I.ThumbsUp size={13}/> Feedback</button>
          <button className="btn btn-secondary btn-sm"><I.ExternalLink size={13}/></button>
        </div>
      </div>
    </div>
  );
}

function FeedbackBlock({ onClose }) {
  const [vote, setVote] = useState(null);
  const [comment, setComment] = useState('');
  const [submitted, setSubmitted] = useState(false);
  if (submitted) {
    return (
      <div className="feedback fade-swap">
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', color: 'var(--low)' }}>
          <I.CheckCircle size={18}/>
          <div style={{ fontSize: 12.5, color: 'var(--ink-2)' }}>
            <b style={{ color: 'var(--ink)' }}>Thanks!</b> Your feedback trains the model.
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
          className={`fb-btn is-yes ${vote === 'yes' ? 'is-active' : ''}`}
          onClick={() => setVote('yes')}>
          <I.ThumbsUp size={13}/> Yes
        </button>
        <button
          className={`fb-btn is-no ${vote === 'no' ? 'is-active' : ''}`}
          onClick={() => setVote('no')}>
          <I.ThumbsDown size={13}/> No
        </button>
        {vote && <button
          className="btn btn-primary btn-sm"
          style={{ marginLeft: 'auto' }}
          onClick={() => setSubmitted(true)}>
          Submit
        </button>}
      </div>
      {vote && (
        <textarea
          className="fb-input"
          rows={2}
          placeholder={vote === 'yes' ? 'What worked well? (optional)' : 'What was off about this result? (optional)'}
          value={comment}
          onChange={(e) => setComment(e.target.value)}
        />
      )}
    </div>
  );
}

function EmptyState({ onTry }) {
  return (
    <div className="popup-body fade-swap">
      <div className="empty">
        <div className="empty-ic"><I.Search size={26}/></div>
        <h3>No product found on this page</h3>
        <p>Open a shopping product page on Amazon, Shopee, Lazada, eBay, AliExpress, or other supported stores to analyze its TrustScore.</p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6, margin: '0 0 14px' }}>
          {['Amazon', 'Shopee', 'Lazada', 'eBay', 'AliExpress', 'Etsy'].map(s => (
            <div key={s} style={{
              fontSize: 11.5, color: 'var(--muted)', padding: '6px 8px',
              background: 'var(--surface-2)', borderRadius: 8,
              border: '1px solid var(--border-2)',
            }}>{s}</div>
          ))}
        </div>

        <button className="btn btn-secondary btn-block btn-sm" onClick={onTry}>
          <I.Refresh size={13}/> Try detection again
        </button>
      </div>
    </div>
  );
}

// ───────── Footer ─────────
function Footer({ state }) {
  return (
    <div className="popup-foot">
      <I.Lock size={11}/>
      <span>Private — analyzed locally on the page</span>
      <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', color: 'var(--muted-2)' }}>v1.0.2</span>
    </div>
  );
}

// ───────── Top-level Popup ─────────
function Popup({
  state: stateProp = 'home',
  score = 74,
  risk: riskProp,
  interactive = false,
  productMeta = { title: 'Sony WH-1000XM5 Wireless Headphones', host: 'amazon.com', seller: 'AudioTech Direct' },
  shadow = true,
  onStateChange,
}) {
  const [state, setState] = useState(stateProp);
  const [progress, setProgress] = useState(0);
  const [taskIdx, setTaskIdx] = useState(0);
  const [showFeedback, setShowFeedback] = useState(false);
  const risk = riskProp || riskFromScore(score);

  // sync with prop when not interactive
  useEffect(() => { if (!interactive) setState(stateProp); }, [stateProp, interactive]);

  // loading progression
  useEffect(() => {
    if (state !== 'loading') return;
    setProgress(0); setTaskIdx(0);
    let p = 0;
    const id = setInterval(() => {
      p += 4 + Math.random() * 5;
      if (p >= 100) {
        p = 100;
        clearInterval(id);
        setProgress(100);
        setTaskIdx(5);
        setTimeout(() => transition('result'), 500);
      } else {
        setProgress(p);
        setTaskIdx(Math.min(4, Math.floor((p / 100) * 5)));
      }
    }, 280);
    return () => clearInterval(id);
  }, [state]);

  const transition = (s) => { setState(s); onStateChange && onStateChange(s); };

  const cls = 'popup' + (shadow ? ' popup-bare' : '');
  return (
    <div className={cls}>
      <Header state={state} />
      {state === 'home' && <HomeState
        productMeta={productMeta}
        onAnalyze={() => interactive ? transition('loading') : null} />}
      {state === 'loading' && <LoadingState progress={progress} taskIdx={taskIdx} />}
      {state === 'result' && <ResultState score={score} risk={risk}
        onFeedback={() => setShowFeedback(true)} />}
      {state === 'feedback' && <ResultState score={score} risk={risk}
        onFeedback={() => null} />}
      {state === 'empty' && <EmptyState onTry={() => interactive ? transition('home') : null} />}
      {((state === 'result' && showFeedback) || state === 'feedback') && <FeedbackBlock />}
      {state !== 'loading' && <Footer state={state}/>}
    </div>
  );
}

// ───────── Floating badge ─────────
function FloatingBadge({ score = 74, risk: riskProp, style }) {
  const risk = riskProp || riskFromScore(score);
  const meta = RISK_META[risk];
  const color = risk === 'low' ? 'var(--low)' : risk === 'med' ? 'var(--med)' : 'var(--high)';
  const bg = risk === 'low' ? 'var(--low-bg)' : risk === 'med' ? 'var(--med-bg)' : 'var(--high-bg)';
  const ringColor = risk === 'low' ? '#10b981' : risk === 'med' ? '#f59e0b' : '#ef4444';
  // small ring
  const r = 13, C = 2 * Math.PI * r;
  const off = C * (1 - score / 100);
  return (
    <div className="fab" style={style}>
      <div className="fab-ring" style={{ background: bg }}>
        <svg width="32" height="32" style={{ position: 'absolute', inset: 0, transform: 'rotate(-90deg)' }}>
          <circle cx="16" cy="16" r={r} fill="none" stroke="rgba(0,0,0,0.06)" strokeWidth="3"/>
          <circle cx="16" cy="16" r={r} fill="none" stroke={ringColor} strokeWidth="3" strokeLinecap="round" strokeDasharray={C} strokeDashoffset={off}/>
        </svg>
        <div className="fab-num" style={{ color }}>{score}</div>
      </div>
      <div className="fab-meta">
        <div className="fab-label">TrustScore</div>
        <div className="fab-status" style={{ color }}>{meta.label}</div>
      </div>
    </div>
  );
}

Object.assign(window, { Popup, FloatingBadge, ScoreRing, RISK_META, riskFromScore });
