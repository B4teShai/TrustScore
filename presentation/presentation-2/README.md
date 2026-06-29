# Presentation 2 — AI TrustScore (final)

Final 5-slide / ~5-minute deck for Project Part 2 (author: Yesui Erkhembayar, s24229521).

## Contents
- **`presentation-2-final.md`** — the deck (Marp). 5 slides; speaker notes embedded as
  presenter-mode comments.
- **`presentation-2-final.html`** — rendered deck. Open in a browser; press **P** for presenter
  view (slides + notes).
- **`SPEAKER_NOTES.md`** — the talking script per slide, with timing and Q&A prep.
- **`make_diagrams.py`** — regenerates the design diagrams in `figures/`.
- **`figures/`** — `architecture.png`, `data_flow.png`, `ml_workflow.png` (design diagrams),
  `comparison_accuracy.png`, `sentiment_cross_domain.png` (results used in the deck).

## View / export
```bash
open presentation-2-final.html

# re-render after editing the markdown
npx -y @marp-team/marp-cli presentation-2-final.md -o presentation-2-final.html --html

# export to PDF (requires a local Chrome/Chromium)
npx -y @marp-team/marp-cli presentation-2-final.md -o presentation-2-final.pdf --pdf

# regenerate diagrams
python make_diagrams.py
```

## Detailed reports (authoritative)
The numbers in the deck come from the reproducible ML pipeline. Full detail lives in:
- `../../ml/reports/v3/RESULTS.md` — one-glance results
- `../../ml/reports/v3/FINDINGS.md` — full research findings (leakage, fix, OOD, calibration)
- `../../ml/reports/v3/PRODUCTION_MODELS.md` — final model chosen per signal
