// Browser.jsx — fake retailer product page with the TrustScore floating badge
// and the popup expanded above the toolbar. Used as a hero artboard.

const I = window.I;
const { FloatingBadge, Popup } = window;

function BrowserMock({ score = 74, risk = 'med', popupOpen = true, popupState = 'result' }) {
  return (
    <div className="browser">
      <div className="browser-bar">
        <div className="dots"><span/><span/><span/></div>
        <div style={{ display: 'flex', gap: 4, color: 'var(--muted)' }}>
          <I.ChevronRight size={14} style={{ transform: 'rotate(180deg)' }}/>
          <I.ChevronRight size={14} />
          <I.Refresh size={14} />
        </div>
        <div className="url">
          <span className="lock"><I.Lock size={11}/></span>
          <span style={{ color: 'var(--ink-2)', fontWeight: 500 }}>amazon.com</span>
          <span style={{ color: 'var(--muted-2)' }}>/dp/B09XS7JWHH/sony-wh-1000xm5-wireless-headphones</span>
        </div>
        <div className="ext-tray">
          <div className="ext"><I.Star size={14}/></div>
          <div className="ext active" title="AI TrustScore"><I.Shield size={14}/></div>
        </div>
      </div>
      <div className="browser-body">
        <div className="product-page">
          <div>
            <div className="pp-hero">
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, color: 'var(--muted-2)', fontSize: 11 }}>
                <I.ShoppingBag size={36} sw={1.4}/>
                <span style={{ fontFamily: 'var(--font-mono)' }}>product image</span>
              </div>
            </div>
            <div className="pp-thumbs">
              <span/><span/><span/><span/><span/>
            </div>
          </div>
          <div className="pp-info">
            <div style={{ fontSize: 12, color: 'var(--brand)', fontWeight: 500, marginBottom: 6 }}>Sony Official Store</div>
            <h1>Sony WH-1000XM5 Wireless Noise Cancelling Headphones</h1>
            <div className="seller">Sold by <b>AudioTech Direct</b> · Ships from Singapore</div>
            <div className="pp-stars">
              <span className="stars"><I.Star size={13}/><I.Star size={13}/><I.Star size={13}/><I.Star size={13}/><I.Star size={13}/></span>
              <span style={{ marginLeft: 4 }}>4.6</span>
              <span>·</span>
              <span>1,284 ratings</span>
            </div>
            <div className="pp-price">
              <span className="now">$248.00</span>
              <span className="was">$399.00</span>
              <span className="save">Save 38%</span>
            </div>
            <div className="pp-cta">
              <button className="add">Add to cart</button>
              <button className="buy">Buy now</button>
            </div>
            <ul className="pp-bullets">
              <li>Industry-leading noise cancellation with 8 microphones</li>
              <li>Up to 30 hours battery life · 3 min quick charge</li>
              <li>Multipoint connection · Adaptive sound control</li>
            </ul>
          </div>
        </div>

        {/* Floating injected badge */}
        <FloatingBadge score={score} risk={risk} style={{ right: 28, bottom: 28 }}/>

        {/* Popup expanded above the extension toolbar icon */}
        {popupOpen && (
          <div style={{ position: 'absolute', top: 12, right: 16, transformOrigin: 'top right' }}>
            <Popup state={popupState} score={score} risk={risk} />
          </div>
        )}
      </div>
    </div>
  );
}

window.BrowserMock = BrowserMock;
