"""Build the final 6-slide presentation PDF.

Clean light teal/coral brand, 16:9, generated with matplotlib.
Includes all 7 sketchy figures: architecture, data_flow, performance_metrics, 
calibration_reliability, sentiment_cross_domain, ml_workflow, and signals_summary.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = HERE / "presentation-2-final.pdf"

TEAL = "#1aa6a6"
CORAL = "#ff6f61"
NAVY = "#3d5a80"
INK = "#1d2d35"
GREY = "#5b6b73"
LIGHT = "#e8f6f6"
AMBER = "#ee9b00"

W, H = 13.333, 7.5  # 16:9 inches
plt.rcParams["font.family"] = "DejaVu Sans"


def _page(pdf: PdfPages):
    fig = plt.figure(figsize=(W, H), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def _close(pdf: PdfPages, fig):
    pdf.savefig(fig)
    plt.close(fig)


def _header(ax, num: int, title: str, kicker: str = ""):
    ax.add_patch(Rectangle((0, 90), 100, 10, color=TEAL, zorder=1))
    ax.add_patch(Rectangle((0, 90), 1.6, 10, color=CORAL, zorder=2))
    ax.text(3.2, 95, title, fontsize=25, fontweight="bold", color="white", va="center", zorder=3)
    if kicker:
        ax.text(97, 95, kicker, fontsize=13, color=LIGHT, va="center", ha="right", zorder=3)
    # footer
    ax.add_patch(Rectangle((0, 0), 100, 4.5, color="#f4f7f7", zorder=1))
    ax.text(3.2, 2.25, "AI TrustScore  ·  Yesui Erkhembayar (s24229521)", fontsize=10.5,
            color=GREY, va="center")
    ax.text(97, 2.25, f"{num} / 6", fontsize=10.5, color=GREY, va="center", ha="right")


def _bullets(ax, x, y, items, dy=6.2, size=15.5, width=58):
    import textwrap

    cy = y
    for it in items:
        lead = it.get("lead", "")
        body = it.get("body", "")
        ax.text(x, cy, "▸", fontsize=14, color=CORAL, va="top")
        wrapped = textwrap.fill(body, width=width)
        lines = wrapped.split("\n")
        ax.text(x + 2.2, cy, (lead + "  " if lead else ""), fontsize=size, color=TEAL,
                fontweight="bold", va="top")
        first_indent = (len(lead) * 0.62 + 2) if lead else 0
        ax.text(x + 2.2 + first_indent, cy, lines[0], fontsize=size, color=INK, va="top")
        for ln in lines[1:]:
            cy -= 3.5
            ax.text(x + 2.2, cy, ln, fontsize=size, color=INK, va="top")
        cy -= dy


def _image(ax, path: Path, x, y, w, h):
    if not path.exists():
        ax.text(x + w / 2, y + h / 2, f"[Missing: {path.name}]", ha="center", color=GREY)
        return
    img = mpimg.imread(path)
    ih, iw = img.shape[0], img.shape[1]
    ar = iw / ih
    box_ar = (w / 100 * W) / (h / 100 * H)
    if ar > box_ar:  # fit width
        nw = w
        nh = w * (box_ar / ar)
    else:
        nh = h
        nw = h * (ar / box_ar)
    iax = ax.inset_axes([(x + (w - nw) / 2) / 100, (y + (h - nh) / 2) / 100, nw / 100, nh / 100])
    iax.imshow(img)
    iax.axis("off")


def _table(ax, x, y, col_w, headers, rows, row_h=5.2, size=12.5, header_color=TEAL):
    tw = sum(col_w)
    ax.add_patch(Rectangle((x, y - row_h), tw, row_h, color=header_color, zorder=2))
    cx = x
    for h, w in zip(headers, col_w):
        ax.text(cx + 0.8, y - row_h / 2, h, fontsize=size, color="white", fontweight="bold",
                va="center", zorder=3)
        cx += w
    cy = y - row_h
    for i, row in enumerate(rows):
        bg = "#f1f8f8" if i % 2 == 0 else "white"
        ax.add_patch(Rectangle((x, cy - row_h), tw, row_h, color=bg, zorder=1))
        cx = x
        for j, (cell, w) in enumerate(zip(row, col_w)):
            color = INK
            weight = "normal"
            if isinstance(cell, tuple):
                cell, color, weight = cell[0], cell[1], cell[2]
            ax.text(cx + 0.8, cy - row_h / 2, cell, fontsize=size, color=color, fontweight=weight,
                    va="center", zorder=3)
            cx += w
        cy -= row_h
    # outer border
    ax.add_patch(Rectangle((x, cy), tw, y - cy, fill=False, edgecolor="#cfe3e3", lw=1, zorder=4))


def _chip(ax, x, y, text, color):
    ax.add_patch(FancyBboxPatch((x, y), len(text) * 0.62 + 2, 3.4,
                 boxstyle="round,pad=0.2,rounding_size=0.8", color=color, zorder=3))
    ax.text(x + (len(text) * 0.62 + 2) / 2, y + 1.7, text, ha="center", va="center",
            fontsize=11, color="white", fontweight="bold", zorder=4)


# --------------------------------------------------------------------------- #
def slide1(ax):
    ax.add_patch(Rectangle((0, 0), 100, 100, color="white"))
    ax.add_patch(Rectangle((0, 60), 100, 2.0, color=TEAL))
    ax.add_patch(Rectangle((0, 60), 22, 2.0, color=CORAL))
    ax.text(8, 76, "AI TrustScore", fontsize=52, fontweight="bold", color=TEAL)
    ax.text(8, 67, "Explainable trust scoring for online shopping", fontsize=22, color=INK)
    ax.text(8, 50, "Problem", fontsize=16, fontweight="bold", color=CORAL)
    ax.text(8, 45.5, "Fake reviews, unreliable sellers and misleading prices — shoppers can't",
            fontsize=15, color=INK)
    ax.text(8, 42, "check it all before buying.", fontsize=15, color=INK)
    ax.text(8, 34, "What it does", fontsize=16, fontweight="bold", color=CORAL)
    ax.text(8, 29.5, "A browser extension that gives any product page a 0–100 trust score,",
            fontsize=15, color=INK)
    ax.text(8, 26, "a risk level (Low / Medium / High) and short, plain-language reasons.",
            fontsize=15, color=INK)
    for i, (t, c) in enumerate([("Browser extension", NAVY), ("FastAPI backend", TEAL),
                                ("5 ML signals", CORAL), ("Iterative Stages", AMBER)]):
        _chip(ax, 8 + i * 21, 15, t, c)
    ax.text(92, 9, "Yesui Erkhembayar · s24229521 · 18 Jun 2026", fontsize=12, color=GREY, ha="right")


def slide2(ax):
    _header(ax, 2, "System Design & Implementation")
    _image(ax, FIG / "architecture.png", 2, 45, 46, 40)
    _bullets(ax, 5, 40, [
        {"lead": "Signals:", "body": "Authenticity · Sentiment · Seller · Price · Policy."},
        {"lead": "Architecture:", "body": "React Popup Extension + FastAPI (Python 3.13) backend."},
        {"lead": "Flow:", "body": "Real-time DOM scraping → ML inference → Explainable TrustScore."},
    ], dy=6.5, size=14, width=85)


def slide3(ax):
    _header(ax, 3, "Multi-Signal Performance Summary")
    _image(ax, FIG / "signals_summary.png", 5, 38, 90, 48)
    _bullets(ax, 5, 35, [
        {"lead": "Holistic Validation:", "body": "Each signal is measured by the most relevant metric."},
        {"lead": "High Power:", "body": "Both text-based models (Fake detection and Sentiment) exceed 0.96 ROC-AUC."},
        {"lead": "Robustness:", "body": "Non-ML signals use deterministic rules to avoid bias and leakage."},
    ], dy=6.5, size=14, width=85)


def slide4(ax):
    _header(ax, 4, "Dataset Overview & Results")
    ax.text(5, 84, "Primary Training Datasets", fontsize=14, fontweight="bold", color=CORAL)
    _table(ax, 5, 82, [25, 12, 12], ["Dataset", "Rows", "Use"], [
        ["Fake-Reviews (ArijitDas)", "40,526", "Fake detection"],
        ["Amazon Reviews 2023", "450,000", "Sentiment"],
        ["Amazon Metadata", "180,000", "Risk/Price"],
        ["Yelp/IMDB/SST-2", "192,349", "OOD Testing"],
    ], row_h=5.0, size=11.5)
    
    ax.text(55, 84, "Model Components", fontsize=14, fontweight="bold", color=CORAL)
    _table(ax, 55, 82, [25, 15], ["Component", "Final Metric"], [
        ["Fake Review Authenticity", ("0.9889 AUC", TEAL, "bold")],
        ["Product Sentiment", ("0.9647 AUC", TEAL, "bold")],
        ["Seller/Price Risk", ("Robust Ruleset", AMBER, "bold")],
        ["System Calibration", ("ECE 0.007", TEAL, "bold")],
    ], row_h=5.0, size=11.5)

    _image(ax, FIG / "performance_metrics.png", 15, 8, 70, 30)


def slide5(ax):
    _header(ax, 5, "Scientific Rigor & Honest Validation")
    _image(ax, FIG / "sentiment_cross_domain.png", 52, 45, 45, 40)
    ax.text(5, 84, "Finding & Fixing Leakage", fontsize=14, fontweight="bold", color=CORAL)
    _bullets(ax, 5, 80, [
        {"lead": "The Discovery:", "body": "Experimental risk models hit 1.0 accuracy. We found labels were mathematically derived from input features (leakage)."},
        {"lead": "The Fix:", "body": "Rejected the biased model. Switched to unsupervised IsolationForest for price anomalies and transparent heuristics."},
        {"lead": "Validation:", "body": "Confirmed cross-domain generalization (e.g. IMDb/Yelp) to reflect real-world performance."},
    ], dy=10, size=13.5, width=45)
    ax.text(75, 42, "Honest domain shift validation", fontsize=10, color=GREY, ha="center")


def slide6(ax):
    _header(ax, 6, "Conclusions & Future Work")
    _image(ax, FIG / "ml_workflow.png", 52, 45, 45, 40)
    _bullets(ax, 5, 82, [
        {"lead": "Success:", "body": "Fully functional, reproducible end-to-end trust scoring system."},
        {"lead": "Lesson:", "body": "Scientific honesty beats hype—leakage detection led to robustness."},
        {"lead": "Future:", "body": "Human-labeled ground truth and DistilBERT deployment."},
    ], dy=8, size=14.5, width=45)
    
    # 4 questions box
    ax.add_patch(FancyBboxPatch((5, 10), 90, 25, boxstyle="round,pad=0.4,rounding_size=1.2",
                 facecolor=LIGHT, edgecolor=TEAL, lw=1.4))
    qa = [("What?", "Explainable shopping trust score"),
          ("Implemented?", "5-signal ML pipeline + rigorous validation stages"),
          ("Result?", "0.9889 AUC; leakage found & fixed; calibrated probabilities"),
          ("Develop further?", "Real labels, transformer fine-tuning, public release")]
    cy = 28
    for q, a in qa:
        ax.text(8, cy, q, fontsize=12.5, fontweight="bold", color=CORAL, va="center")
        ax.text(28, cy, a, fontsize=12.5, color=INK, va="center")
        cy -= 4.5


def main() -> None:
    with PdfPages(OUT) as pdf:
        for builder in (slide1, slide2, slide3, slide4, slide5, slide6):
            fig, ax = _page(pdf)
            builder(ax)
            _close(pdf, fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
