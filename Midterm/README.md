# Midterm — Retail Hype & Sentiment Network on SET50

> Course **DADS7201 — Social Network Analysis**, NIDA.
> Deliverable: 2-page A4 report — **submitted** (PDF kept locally, not
> pushed to the public repo).

End-to-end SNA pipeline that turns **real Pantip Sinthorn chatter** into a
weighted co-mention network over the **SET50 H1-2026** constituents, then
cross-validates against a return-correlation graph built from `yfinance`
daily data. Adds a **permutation + bootstrap significance layer** and a
**human-annotated sentiment gold-set** to defend the numbers.

---

## 1. What we built

| Layer | Tooling | Output |
|---|---|---|
| Stock universe | SET50 H1 2026 (official PDF) | `data/stocks.csv` (50 tickers, sectored) |
| Market data | `yfinance` (30-day event window + 6-month corr window) | `data/prices.csv`, `data/returns.csv` |
| Real chatter | Playwright + Pantip XHR interception (resumable) | `data/pantip_topics_index.csv`, `data/posts_raw.csv` |
| Mention extraction | strict-boundary SET50 regex + phase split | `data/posts.csv`, `data/mentions.csv` |
| Sentiment | WangchanBERTa 3-class + 100-post gold-set validation | `data/sentiment.csv`, `data/sentiment_gold.csv`, `output/sentiment_validation.*` |
| Network analysis | NetworkX + python-louvain | `output/centrality.csv`, `communities.csv`, `evolution_comparison.csv`, `hype_hubs.csv` |
| Cross-validation | Pearson corr graph vs co-mention graph | `output/overlap_sensitivity.csv`, `attention_without_correlation.csv`, `correlation_without_attention.csv`, `validated_attention_edges.csv` |
| Significance | 1,000× permutation on phase edges + 500× bootstrap Jaccard | `output/significance.json` |
| Report + slides | python-docx / LaTeX / python-pptx | 2-page & 4-page `.docx` + `.pdf` + `.tex` + 12-slide `.pptx` (built locally under `report/`, not published) |

---

## 2. Headline results (final run)

**Corpus.** 1,383 SET50-tagged Pantip Sinthorn topics scraped 2026-01-01
→ 2026-06-18; **97 kept** (≥ 2 SET50 mentions); **312 mention records**;
3.2 stocks/post average. Phase split (10-day windows around the event
anchor): Before = 27 posts, During = 42, After = 27.

**Network evolution.**

| Phase  | Edges | Density | Avg Degree | Communities | Modularity |
|--------|------:|--------:|-----------:|------------:|-----------:|
| Before | 73    | 0.060   | 2.92       | 25          | 0.415      |
| During | 104   | 0.085   | 4.16       | 24          | 0.404      |
| After  | 252   | 0.206   | 10.08      | 20          | 0.078      |

→ Density rises ~3.5× from Before to After and modularity collapses
(0.415 → 0.078) — attention **expands and de-fragments** after the event.

**Statistical significance** (`output/significance.json`):

- Density gap **After − Before = +0.146** (raw)
- Permutation test (1,000 shuffles of phase labels): **p = 0.032**
- Bootstrap 95% CI on Jaccard(co-mention ∩ |r|≥0.3 correlation):
  **[0.093, 0.140]** (observed = 0.144, boot mean = 0.121)

→ The Before → After densification is significant at α = 0.05, but the
attention/correlation overlap is *low* and its CI does not cross 0.5.

**Top 5 Hype Hubs** (composite of Degree × MentionCount × AvgSentiment × |Return|):

| Rank | Symbol | HypeScore | Sector |
|---:|---|---:|---|
| 1 | KBANK  | 0.849 | Banking |
| 2 | ADVANC | 0.834 | Information & Communication Technology |
| 3 | GULF   | 0.825 | Energy & Utilities |
| 4 | BBL    | 0.802 | Banking |
| 5 | SCB    | 0.799 | Banking |

Big-cap banks + energy dominate — consistent with the retail-attention
prior. Full ranking in `output/hype_hubs.csv`.

**Louvain communities on the full graph.** 14 communities in total, but
five of them are non-trivial (≥ 4 members):

| Community | Size | Members (short) |
|---|---:|---|
| C2 — PTT / bluechip | 14 | PTT, PTTEP, PTTGC, SCC, SCGP, IVL, BDMS, BH, COM7, CPF, GPSC, TOP, AOT, BANPU |
| C5 — Consumer / real | 9 | CPALL, HMPRO, CRC, BJC, BTS, BEM, CBG, OSP, SAWAD |
| C4 — Banks (pure) | 8 | BBL, KBANK, KKP, KTB, SCB, TCAP, TISCO, TTB |
| C1 — Comms + growth | 6 | ADVANC, GULF, KTC, MINT, TRUE, WHA |
| C7 — Electronics / speculative | 4 | CCET, CPN, DELTA, OR |

The pure 8-bank cluster is the strongest signal — no bank strays into
any other community.

**Attention vs Fundamentals overlay** (co-mention vs 6-month return
Pearson correlation, `|r| ≥ 0.3`):

- 274 attention edges vs 276 correlation edges — **62 in both**
- **Jaccard = 0.127** at `|r| ≥ 0.3` (drops fast: 0.078 at 0.4, 0.024
  at 0.5, 0 at 0.7)
- **212 "hype-only"** edges — top: CCET–DELTA (w=10), DELTA–OR (w=9),
  KBANK–TRUE (w=9), ADVANC–KBANK (w=8), ADVANC–TRUE (w=7)
- **214 "overlooked co-movers"** — top: CPF–HMPRO (r=0.61), EGCO–RATCH
  (r=0.60), BJC–LH (r=0.56), MTC–SAWAD (r=0.55), LH–RATCH (r=0.55)

→ Retail chatter is **not** a proxy for fundamentals; the two networks
share only ~13% of edges at the standard threshold.

**Sentiment validation.** WangchanBERTa vs 100-post hand-annotated
gold-set — see `output/sentiment_validation.json`.

---

## 3. How to reproduce

```powershell
pip install -r requirements.txt
python -m playwright install chromium

# 0. SET50 universe + yfinance prices (~10 s)
python scripts/01_collect.py

# 1. Scrape Pantip — resumable, two phases
python scripts/01b_scrape_pantip.py listing      # ~12 min
python scripts/01b_scrape_pantip.py bodies       # ~50 min

# 2. SET50 mention extraction + phase split
python scripts/01c_extract_mentions.py

# 3. Real Thai sentiment via WangchanBERTa + gold-set validation
python scripts/02_sentiment_real.py
python scripts/02b_validate_sentiment.py

# 4. Aggregate sentiment + network + figures + overlay + significance
python scripts/02_sentiment.py
python scripts/03_network.py
python scripts/04_visualization.py
python scripts/06_correlation_overlay.py
python scripts/08_significance.py

# 5. Report + slides
python scripts/05_report.py                # 4-page .docx
python scripts/05_report_2page.py          # 2-page .docx (submission)
python scripts/05_report_latex.py          # LaTeX version
python scripts/07_make_slides.py           # .pptx
```

Detailed walkthrough + debugging tips are kept locally in
`REAL_DATA_PIPELINE.md`.  LaTeX build instructions are in
`report/COMPILE_LATEX.md`.

---

## 4. Repo layout

```
Midterm/
├── requirements.txt
├── REAL_DATA_PIPELINE.md          ← pipeline-runner guide
├── implementation_plan.md         ← initial planning (historical)
├── implementation.md              ← initial methodology notes (historical)
├── skill.md                       ← initial spec (historical)
├── โจทย์_Midterm.txt              ← assignment brief
│
├── scripts/
│   ├── 01_collect.py              SET50 universe + yfinance
│   ├── 01b_scrape_pantip.py       Playwright Phase A + requests Phase B
│   ├── 01c_extract_mentions.py    regex + phase split
│   ├── 02_sentiment_real.py       WangchanBERTa classifier
│   ├── 02b_validate_sentiment.py  100-post gold-set F1
│   ├── 02_sentiment.py            per-stock aggregation
│   ├── 03_network.py              centrality + Louvain
│   ├── 04_visualization.py        all figures (PNG + SVG)
│   ├── 05_report.py               4-page .docx
│   ├── 05_report_2page.py         2-page .docx (submitted layout)
│   ├── 05_report_latex.py         LaTeX / .tex
│   ├── 06_correlation_overlay.py  attention vs fundamentals
│   ├── 07_make_slides.py          .pptx deck
│   └── 08_significance.py         permutation + bootstrap
│
├── data/
│   ├── stocks.csv                 SET50 H1 2026 universe
│   ├── prices.csv / returns.csv   yfinance event window
│   ├── event_anchor.json          event window centre (from returns)
│   ├── pantip_topics_index.csv    Phase A output (1,383 rows)
│   ├── posts_raw.csv              Phase B output (1,383 rows)
│   ├── posts.csv                  filtered (97 rows, ≥ 2 SET50 mentions)
│   ├── mentions.csv               long format (312 rows)
│   ├── sentiment.csv              WangchanBERTa labels
│   └── sentiment_gold.csv         100-post hand-annotated gold-set
│
├── output/
    ├── centrality.csv             50 stocks × 3 centralities
    ├── communities.csv            Louvain partition
    ├── evolution_comparison.csv   3-phase metrics
    ├── hype_hubs.csv              composite ranking
    ├── stock_sentiment.csv        per-stock sentiment
    ├── overlap_sensitivity.csv    Jaccard at |r| in {0.3, 0.4, 0.5, 0.6, 0.7}
    ├── attention_without_correlation.csv     "hype only" pairs (212)
    ├── correlation_without_attention.csv     "overlooked" pairs (214)
    ├── validated_attention_edges.csv         pairs in both networks (62)
    ├── significance.json          permutation p + bootstrap Jaccard CI
    ├── sentiment_validation.{csv,json}       gold-set F1 breakdown
    └── *.png / *.svg              8 figures (network_full, evolution,
                                    hype, communities, centrality_top10,
                                    overlay, attention_vs_fundamentals,
                                    overlap_sensitivity)

```

---

## 5. Caveats & what's not done

- **Sentiment-stock pairing is post-level, not span-level.** A long post
  mentioning DELTA negatively + GULF positively assigns the same label
  to both. Future work: aspect-based sentiment.
- **Yfinance event window (30 d) is short for stable correlations.** The
  attention/fundamentals overlay uses a **separate 6-month return fetch**
  to compensate.
- **Ticker collision with common words.** `OR`, `BH`, `TU`, `LH` are
  common English words / particles. The strict boundary regex catches
  most Thai false positives but a final manual sanity check is
  recommended before publication.
- **Sample size after filtering (97 posts, 27/42/27 by phase)** is small
  → the permutation p = 0.032 is significant but the modest sample
  makes effect-size claims tentative.
