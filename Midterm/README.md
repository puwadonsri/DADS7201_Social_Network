# Midterm — Retail Hype & Sentiment Network on SET50

> Course **DADS7201 — Social Network Analysis**, NIDA.
>
>
> Predictive extension is developed in [`../Final/`](../Final/) — see
> ** Follow-up work** at the bottom.

End-to-end SNA pipeline that turns **real Pantip Sinthorn chatter** into a
weighted co-mention network over the **SET50 H1-2026** constituents, then
cross-validates against a return-correlation graph built from `yfinance`
daily data. Adds a **permutation + bootstrap significance layer** and a
**human-annotated sentiment gold-set** to defend the numbers.

---

## Summary

โปรเจกต์ Midterm สร้าง **ไปป์ไลน์ SNA เชิงพรรณนา (descriptive)** ที่แปลง
ข้อความจริงจากห้องสินธร Pantip ให้เป็น **เครือข่ายการถูกพูดถึงร่วมกัน
(co-mention network)** ของหุ้น SET50 แล้วนำไป cross-validate กับกราฟ
สหสัมพันธ์ผลตอบแทน (return correlation) จาก `yfinance`

**ข้อมูล:** เก็บกระทู้สินธร 1,374 กระทู้ (1 ม.ค. – 18 มิ.ย. 2026) →
คัดเหลือ **96 กระทู้** ที่กล่าวถึงหุ้น SET50 ≥ 2 ตัว (311 เรคคอร์ด),
จุดอ้างอิงเหตุการณ์ **T = 28 พ.ค. 2026** (วันที่โพสต์หนาแน่นสุด)

**ผลลัพธ์หลัก (สมมติฐาน H1–H4):**
- **H1 (degree → ความสนใจ):** ✅ สนับสนุน
- **H2 (คอมมูนิตี้ → เคลื่อนไหวราคาไปด้วยกัน):** ✅ สนับสนุน (วิเคราะห์
  เพิ่มหลังส่ง, +5.4σ, p = 0.001) — แต่มี *sector confound*
- **H3 (sentiment → hub):** ⚠️ อ่อน เพราะตัวจำแนก sentiment แม่นระดับสุ่ม
  (accuracy 0.34, κ = 0.14) → เป็น *ข้อจำกัดของโมเดล* ไม่ใช่ของข้อมูล
  (แก้ต่อใน Final)
- **H4 (เหตุการณ์ → ความหนาแน่นกราฟ):** ✅ ความหนาแน่นเพิ่ม 0.060 → 0.206
  ช่วง Before → After, permutation test p = 0.032

**ผลเสริม — ความสนใจ ≠ ปัจจัยพื้นฐาน:** กราฟ co-mention กับกราฟสหสัมพันธ์
ทับกันแค่ ~13% (Jaccard = 0.13 ที่ \|r\|≥0.3, CI 95% [0.09, 0.14]) →
กระแสรายย่อยไม่ใช่ตัวแทนของปัจจัยพื้นฐาน

**Top 5 Hype Hub:** KBANK, ADVANC, GULF, BBL, SCB (แบงก์ใหญ่ + พลังงาน)

> ⚠️ **ข้อควรระวัง:** กลุ่มตัวอย่างหลังกรองเล็ก (96 กระทู้) → ค่า p
> มีนัยสำคัญแต่การอ้าง effect-size ยังต้องระวัง ส่วนการต่อยอดเชิงทำนาย
> อยู่ใน [`../Final/`](../Final/)

---

## 1. What we built

| Layer | Tooling | Output |
|---|---|---|
| Stock universe | SET50 H1 2026 (official PDF) | `data/stocks.csv` (50 tickers, sectored) |
| Market data | `yfinance` (30-day event window + 6-month corr window) | `data/prices.csv`, `data/returns.csv` |
| Real chatter | Playwright + Pantip XHR interception (resumable) | `data/pantip_topics_index.csv`, `data/posts_raw.csv` |
| Mention extraction | strict-boundary SET50 regex + phase split | `data/posts.csv`, `data/mentions.csv` |
| Sentiment | WangchanBERTa 3-class + 50-post gold-set validation | `data/sentiment.csv`, `data/sentiment_gold.csv`, `output/sentiment_validation.*` |
| Network analysis | NetworkX + python-louvain | `output/centrality.csv`, `communities.csv`, `evolution_comparison.csv`, `hype_hubs.csv` |
| Cross-validation | Pearson corr graph vs co-mention graph | `output/overlap_sensitivity.csv`, `attention_without_correlation.csv`, `correlation_without_attention.csv`, `validated_attention_edges.csv` |
| Significance | 1,000× permutation on phase edges + 500× bootstrap Jaccard | `output/significance.json` |
| Report (submitted) | python-docx | 2-page `.docx` → `report/DADS7201_Midterm_Report_2page.docx` |
| **Post-submission additions** | | |
| H2 community coherence | 1,000× permutation on within vs between community pairwise return `r` | `output/h2_test.json`, `h2_null_distribution.{png,svg}` |
| Extended report + slides | python-docx / LaTeX / python-pptx | extended `.docx` + `.tex` / `.pdf` (LNCS) + 12-slide `.pptx` (built locally under `report/`, not published) |

---

## 2. Headline results (final run)

**Corpus.** 1,374 SET50-tagged Pantip Sinthorn topics scraped 2026-01-01
→ 2026-06-18; **96 kept** (≥ 2 SET50 mentions); **311 mention records**;
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

> ⚠️ **N caveat.** Only 27 / 42 / 27 posts per phase after the
> event-anchor filter. The permutation test (below) controls for
> chance in phase-label assignment, but *effect-size* claims remain
> sample-size-limited — the 3.5× density ratio in particular should
> not be reported as a stable point estimate. Multi-event replication
> is the natural robustness check; see §5 Follow-up work.

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

The pure 8-bank cluster is the *cleanest partition* — but partly
**tautological**: retail Sinthorn posts routinely name major banks
together ("SCB/KBANK/KTB บวกยกแผง", "แบงก์ใหญ่ลงยกกลุ่ม"), so a Louvain
algorithm grouping them is partly a property of retail linguistic
habit rather than a purely attention-driven finding. §3.1 (H2 test) is
a partial statistical control for this; a stronger test would be a
**sector-stratified permutation null**.

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

**Sentiment validation.** WangchanBERTa vs 50-post hand-annotated
gold-set — **classifier is chance-level**: accuracy 0.34, Cohen κ = 0.14,
Positive recall = 0.00 (all 23 gold-Positive posts predicted as Neutral).
The H3 hypothesis (sentiment → hub score) is therefore weak largely as a
**classifier artefact**, not a data one. Full breakdown in
`output/sentiment_validation.json`. A domain-adapted classifier is the
natural next step — and is done in the Final project (see §5): a
5-fold CV fine-tune of WangchanBERTa on the same 50 gold posts lifts
accuracy 0.34 → **0.50** and Positive recall 0.00 → **0.78**, confirming
that the Midterm H3 weakness is largely a **classifier artefact**.

---

## 3. Post-submission analysis (not in the graded report)

Developed after the 2-page submission for follow-up work. Written up
here for completeness; do not read this as part of the submitted
findings.

### 3.1 H2 — community coherence in return co-movement

Test whether Louvain communities from co-mention edges actually co-move
at the price level. For each stock pair we take Pearson `r` on the same
121-trading-day yfinance window as the correlation graph. Statistic

S = mean(within-community `r`) − mean(between-community `r`)

evaluated over the 5 non-trivial Louvain communities (40 stocks; 171
within + 609 between pairs; GULF and TIDLOR dropped due to missing
yfinance history).

- Observed **S = +0.075** (mean within `r` = 0.255 vs mean between
  `r` = 0.180)
- Permutation null (1,000 shuffles of community labels): mean ≈ 0.000,
  SD = 0.014 → observed sits **+5.4σ** above null → **p = 0.001**
- Per-community mean internal `r`, ranked:

| Community | n | mean r | Character |
|---|:---:|---:|---|
| C4 | 8 | **+0.324** | Pure Banks (BBL, KBANK, KKP, KTB, SCB, TCAP, TISCO, TTB) |
| C5 | 9 | +0.280 | Consumer / Real (BEM, BJC, BTS, CBG, CPALL, ...) |
| C1 | 5 | +0.246 | Growth / Comms (ADVANC, KTC, MINT, TRUE, WHA) |
| C2 | 14 | +0.233 | Bluechip (PTT, PTTEP, IVL, AOT, ...) |
| C7 | 4 | +0.144 | Speculative (CCET, CPN, DELTA, OR) |

**H2 is supported statistically** — co-mention communities do co-move
at the returns level. **Caveat:** Louvain communities largely align
with SET industry sectors (C4 = 8/8 banks; C5 mostly Consumer + Real
Estate), so part of this effect is a **sector confound** — the Pearson
`r` between two banks is elevated *because they are both banks*, not
purely because retail talk about them together. A properly attention-
driven test would need a **sector-stratified permutation null** (shuffle
labels within-sector, not across the full universe) — deferred to
future work.

Full breakdown in `output/h2_test.json`; null-distribution histogram at
`output/h2_null_distribution.png`.

---

## 4. How to reproduce

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
python scripts/09_test_h2.py               # post-submission H2 test

# 5. Report + slides
python scripts/05_report_2page.py          # ★ 2-page .docx (submitted)
python scripts/05_report.py                # extended .docx (post-submission)
python scripts/05_report_latex.py          # LNCS .tex → tectonic → .pdf
python scripts/07_make_slides.py           # 12-slide .pptx
```

Detailed walkthrough + debugging tips are kept locally in
`REAL_DATA_PIPELINE.md`.  LaTeX build instructions are in
`report/COMPILE_LATEX.md`.

---

## 5. Follow-up work — Final project

The three main Midterm limitations (chance-level sentiment classifier;
descriptive-only findings; small event window) are addressed in the
Final project [`../Final/`](../Final/) via three predictive layers
spanning course Weeks 3–7:

- **Path A — Temporal link prediction** on the same co-mention graph,
  comparing 5 methods (Jaccard / Adamic–Adar / Preferential Attachment
  / Node2Vec / GraphSAGE). Best test AUC 0.73 (Preferential Attachment).
- **Path B — Return prediction** with graph + attention features.
  Directional accuracy stays near chance (~0.55) but the long–short
  portfolio Sharpe proxy climbs **3.07 → 4.30** as graph features are
  added — attention moves the money, not the average.
- **Path C — WangchanBERTa fine-tune** (5-fold CV on the same 50-post
  gold set): accuracy **0.34 → 0.50**, Positive recall **0.00 → 0.78**.
  Confirms H3 weakness is largely a classifier artefact.

Report + slides: [`../Final/report/DADS7201_Final_Report.pdf`](../Final/report/DADS7201_Final_Report.pdf)
(4-page LNCS), [`../Final/report/DADS7201_Final_Slides.pptx`](../Final/report/DADS7201_Final_Slides.pptx)
(14 slides with Thai speaker notes).

---

## 6. Repo layout

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
│   ├── 02b_validate_sentiment.py  50-post gold-set F1
│   ├── 02_sentiment.py            per-stock aggregation
│   ├── 03_network.py              centrality + Louvain
│   ├── 04_visualization.py        all figures (PNG + SVG)
│   ├── 05_report_2page.py         ★ 2-page .docx (submitted layout)
│   ├── 05_report.py               extended .docx (post-submission)
│   ├── 05_report_latex.py         LNCS .tex
│   ├── 06_correlation_overlay.py  attention vs fundamentals
│   ├── 07_make_slides.py          .pptx deck
│   ├── 08_significance.py         permutation + bootstrap
│   └── 09_test_h2.py              post-submission H2 community-coherence test
│
├── data/
│   ├── stocks.csv                 SET50 H1 2026 universe
│   ├── prices.csv / returns.csv   yfinance event window
│   ├── event_anchor.json          event window centre (from returns)
│   ├── pantip_topics_index.csv    Phase A output (1,374 topics)
│   ├── posts_raw.csv              Phase B output (1,374 posts)
│   ├── posts.csv                  filtered (96 posts, ≥ 2 SET50 mentions)
│   ├── mentions.csv               long format (311 records)
│   ├── sentiment.csv              WangchanBERTa labels
│   └── sentiment_gold.csv         50-post hand-annotated gold-set
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
    ├── h2_test.json               H2 permutation p + per-community mean r
    └── *.png / *.svg              9 figures (network_full, evolution,
                                    hype, communities, centrality_top10,
                                    overlay, attention_vs_fundamentals,
                                    overlap_sensitivity, h2_null_distribution)

```

---

## 7. Caveats & what's not done

- **Sentiment classifier is chance-level on this corpus.** WangchanBERTa
  hits accuracy 0.34 / κ = 0.14 on the 50-post gold set — Positive posts
  are systematically misclassified as Neutral (recall = 0.00). Any claim
  built on `AvgSentiment` (including the sentiment term in HypeScore) is
  therefore **not defensible**; the H3 hypothesis is treated as a
  classifier artefact, not a data one. **Fix implemented in the Final
  project** (see §5): 5-fold CV fine-tune on the same 50 gold posts
  raises accuracy to **0.50** and Positive recall to **0.78** (though
  Cohen κ only moves 0.14 → 0.17). A larger
  labelling budget (200–500 posts) is the natural next step.
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
- **Sample size after filtering (96 posts, 27/42/27 by phase)** is small
  → the permutation p = 0.032 is significant but the modest sample
  makes effect-size claims tentative.
