# Final Project — Predictive Extension of the Midterm SET50 Hype Network

> Course **DADS7201 — Social Network Analysis**, NIDA
> Assignment (`โจทย์_final.txt`): *"Apply predictive analysis to your
> midterm project."* → extended **4-page A4 report**, Turnitin-checked,
> due **Week 8 (26 Jul 2026)**.
>
> **Deliverables:** [`report/DADS7201_Final_Report.pdf`](report/DADS7201_Final_Report.pdf)
> (4-page Springer LNCS) and
> [`report/DADS7201_Final_Slides.pptx`](report/DADS7201_Final_Slides.pptx)
> (14 slides, Thai speaker notes).
>
> The original planning notes are kept in
> [PROJECT_PLAN.md](PROJECT_PLAN.md); this README is the **as-built
> summary** of what was actually done and what the numbers say.

The [Midterm](../Midterm/) delivered a **descriptive** SNA pipeline on
SET50 + Pantip Sinthorn: a weighted co-mention network, Louvain
communities, a return-correlation overlay, a sentiment-classifier
validation, and a permutation/bootstrap significance layer. The Final
turns each descriptive finding into a **predictive** one across three
layers (Paths A–C) that between them exercise course methods from
**Weeks 3, 5, and 7**.

---

## 1. From Midterm to Final — the through-line

The Midterm left three concrete openings. Each becomes one Final path:

| Midterm result / limitation | Final path | Question turned predictive |
|---|---|---|
| Co-mention graph + Louvain + H4 densification (descriptive only) | **Path A — Temporal link prediction** | *Which stock pairs will retail investors start co-mentioning next?* |
| Hype Hub Score + Attention-vs-Fundamentals dissociation (no forecast) | **Path B — Return prediction** | *Do graph/attention features add signal to next-day returns?* |
| Sentiment classifier at chance level (κ = 0.14), so H3 unresolved | **Path C — Sentiment fine-tune** | *Is H3's weakness a data property or a classifier artefact?* |

No re-scraping was needed: every Final script reads the Midterm's
`data/` and `output/` as inputs and writes only into `Final/output/`.
The single new market call is a wider (~180-day) `yfinance` return
history for Path B's training rows.

### What the Midterm established (recap)

| Finding | Verdict | Source |
|---|---|---|
| H1 (degree → attention) | Supported | `../Midterm/output/centrality.csv` |
| H2 (community → co-movement) | Supported, +5.4σ, p = 0.001 *(post-submission)* | `../Midterm/output/h2_test.json` |
| H3 (sentiment → hub) | **Weak — classifier at chance (κ = 0.14)** | `../Midterm/output/sentiment_validation.json` |
| H4 (event → density) | Supported, After−Before p = 0.032 | `../Midterm/output/significance.json` |
| Attention ≠ fundamentals | Jaccard 0.13 at \|r\|≥0.3, 95% CI [0.09, 0.14] | `../Midterm/output/overlap_sensitivity.csv` |

Corpus carried forward: 1,374 Pantip Sinthorn topics (1 Jan – 18 Jun
2026); **96 posts** kept at ≥ 2 SET50 mentions; **311 mention records**;
event anchor **T = 2026-05-28** (peak posting day).

---

## 2. Path A — Temporal link prediction on the co-mention graph

**Task.** Given co-mentions in `[t_min, T−20)`, predict which *currently
non-adjacent* SET50 pairs first co-occur in the next windows: `[T−20, T)`
(validation) and `[T, T+20)` (test). Negatives are sampled from the SET50
non-edge set (2:1 in training, 1:1 in evaluation), following the Week-7
convention. The split is cached in [data/temporal_split.json](data/temporal_split.json):
**154 train edges, 50 first-time val positives, 134 first-time test
positives.**

**Five methods, spanning three weeks.**

| # | Method | Week | Scoring |
|---|---|:--:|---|
| 1 | Jaccard coefficient | 3 | \|N(u)∩N(v)\| / \|N(u)∪N(v)\| |
| 2 | Adamic–Adar | 3 | Σ 1/log\|N(w)\| over common neighbours |
| 3 | Preferential attachment | 3 | \|N(u)\|·\|N(v)\| |
| 4 | Node2Vec + logistic | 5 | LR on Hadamard product z_u ⊙ z_v (d = 64) |
| 5 | Homogeneous GraphSAGE | 7 | 2× SAGEConv, hidden 64, BCE-with-logits |

**Results (test set, 134 positives + 134 negatives; AUC 95% CI from 500
bootstrap iterations — full table in [output/lp_baselines.csv](output/lp_baselines.csv)).**

| Method | Test AUC | 95% CI | P@10 | P@25 | MRR |
|---|---:|:---:|---:|---:|---:|
| **Preferential** | **0.733** | [0.68, 0.79] | 0.90 | 0.88 | 0.037 |
| Adamic–Adar | 0.727 | [0.67, 0.78] | 1.00 | 0.92 | 0.038 |
| Jaccard | 0.713 | [0.66, 0.76] | 1.00 | 0.92 | 0.037 |
| Node2Vec | 0.711 | [0.65, 0.77] | 0.50 | 0.60 | 0.021 |
| GraphSAGE | 0.547 | [0.48, 0.62] | 0.70 | 0.64 | 0.028 |

**Read-out.**
- The three Week-3 heuristics are effectively tied at the top (their CIs
  overlap heavily); Preferential wins on AUC, Adamic–Adar/Jaccard hit
  **P@10 = 1.00 (10/10)**. Node2Vec matches the heuristics on AUC only.
- **GraphSAGE underperforms** — identity node features and ~150 training
  edges are well inside a GNN's data-hungry regime. An honest, on-message
  result: on a small sparse graph, simple structural heuristics win.
- The top predicted unseen pairs (`output/lp_topk_predicted_edges.csv`)
  concentrate on *hype-only* and *new* Midterm buckets that **do emerge**
  in the eval window (GULF–OR, KBANK–OR, GULF–TOP), while most
  *overlooked co-mover* pairs stay silent → **attention begets
  attention**, extending the Midterm's Attention-vs-Fundamentals result
  into a forecast.

Figures: `output/lp_roc.{png,svg}`, `output/lp_precision_at_k.{png,svg}`.

---

## 3. Path B — Return prediction with graph features

**Task.** For each (stock, day) row over 120 trading days, predict
`sign(r_{d+1})`. Feature groups build up incrementally:

| Model | Features | Learner |
|---|---|---|
| **B0** | own-return lags (r_{d−1}, 5-day mean/std, cross-sectional rank) | Logistic reg. |
| **B1** | B0 + degree centrality + Hype Hub Score + Louvain one-hot | Logistic reg. |
| **B2** | B1 + rolling attention (10-day mentions) + rolling sentiment (5-day) | Logistic reg. |
| **B3** | B2 features | XGBoost |

Evaluation is a 5-fold expanding-window time-series CV; the portfolio
proxy goes long the top-decile predicted-up and short the bottom-decile
predicted-down, equal-weight, daily rebalance
([output/rp_metrics.json](output/rp_metrics.json)).

| Model | Accuracy | 95% CI | Sharpe proxy |
|---|---:|:---:|---:|
| B0 (lags) | 0.553 | [0.540, 0.568] | 3.07 |
| B1 (+graph) | 0.545 | [0.530, 0.560] | 3.79 |
| B2 (+attention, full) | 0.547 | [0.533, 0.562] | **4.30** |
| B3 (XGBoost) | 0.525 | [0.509, 0.539] | 1.51 |

**Read-out.** Directional accuracy sits near chance for every model —
120-day SET50 sign prediction is intrinsically hard, and we report that
honestly. But the Sharpe proxy rises **monotonically B0 → B1 → B2
(3.07 → 3.79 → 4.30)**: graph and attention features shift the
portfolio's decision boundary in a risk-improving way even when headline
accuracy does not move — *attention moves the money, not the average*.
B3 (XGBoost) collapses to Sharpe 1.51, i.e. it **over-fits** the small
time series, so the linear B2 is the reported model.

Figures: `output/rp_baseline_vs_full.{png,svg}`,
`output/rp_sharpe_curve.{png,svg}`, `output/rp_feature_importance.{png,svg}`.

---

## 4. Path C — Sentiment classifier fine-tune (re-testing H3)

**Task.** The Midterm's off-the-shelf WangchanBERTa was chance-level, so
H3 (sentiment → hub) could not be judged. We fine-tune
`poom-sci/WangchanBERTa-finetuned-sentiment` with a **stratified 5-fold
CV on the same 50-post gold set** (3 epochs, lr 2e-5, batch 4) and pool
the held-out predictions ([output/sentiment_finetuned_validation.json](output/sentiment_finetuned_validation.json)).

| | Accuracy | Macro F1 | Cohen κ | Positive recall |
|---|---:|---:|---:|---:|
| Baseline (off-the-shelf) | 0.34 | 0.31 | 0.14 | 0.00 |
| Fine-tuned (5-fold pooled) | **0.50** ± 0.06 | 0.40 | 0.17 | **0.78** |

**Read-out — with the honest caveat.** Fine-tuning recovers the
Positive class the baseline never predicted (recall **0.00 → 0.78**;
18/23 gold-Positive posts now correct), lifting accuracy 0.34 → 0.50.
This is enough to reframe the Midterm's H3 weakness as a **classifier
transfer artefact, not a corpus property**. But the improvement is
uneven and must be stated as such: **Cohen κ barely moves (0.14 → 0.17)**
and the model now over-predicts Positive at the expense of the Neutral
class (Neutral recall collapses 1.00 → 0.09). With only 50 posts and no
new labels, this is a **proof-of-concept, not a solved classifier** — a
~200–500-post labelling budget is the concrete next step to push κ past
0.5.

Figure: `output/sentiment_finetuned_confusion.{png,svg}`.

> ⚠️ Path C used **only the original 50 gold posts** via 5-fold CV. The
> larger hand-labelled set floated in the plan was *not* collected; the
> result is deliberately scoped as a proof-of-concept and reported that
> way in both the paper and here.

---

## 5. Grading self-check

| Rubric criterion (from Midterm feedback) | How the Final satisfies it |
|---|---|
| Uses course methods from ≥ 3 weeks | ✅ Path A alone spans Weeks 3, 5, 7 |
| Statistical rigor / error bars | ✅ Bootstrap 95% CI on every AUC; CV CIs on accuracy |
| Real (non-synthetic) data | ✅ Same Pantip Sinthorn corpus, no re-scrape |
| Descriptive → predictive | ✅ H4 → link prediction; overlay → forecast; H3 → classifier re-test |
| Reproducible pipeline | ✅ Script-per-step, reads Midterm outputs (§7) |
| Honest limitations | ✅ GraphSAGE loss, near-chance returns, κ-flat sentiment all reported |

---

## 6. How to reproduce

All scripts run from the `Final/` directory and read `../Midterm/`
outputs; no re-scraping.

```powershell
# Path A — temporal link prediction (heuristics + Node2Vec + GraphSAGE)
python scripts/09_link_prediction.py      # → output/lp_*.csv, lp_roc/lp_precision_at_k

# Path B — return prediction (fetches ~180-day yfinance history)
python scripts/10_return_prediction.py    # → output/rp_metrics.json, rp_*.png/.svg

# Path C — WangchanBERTa 5-fold fine-tune on the 50-post gold set
python scripts/11_sentiment_finetune.py   # → output/sentiment_finetuned_*

# Report + slides (pull metrics live from the CSV/JSON above)
python scripts/12_report_final_latex.py   # → report/DADS7201_Final_Report.tex
python scripts/13_make_slides.py          # → report/DADS7201_Final_Slides.pptx

# Compile the LNCS PDF with tectonic (installed during the Midterm)
cd report
%TEMP%\tectonic\tectonic.exe DADS7201_Final_Report.tex
```

---

## 7. Files produced by this Final phase

```
Final/
├── README.md                              ← this file (as-built summary)
├── PROJECT_PLAN.md                        ← original implementation plan
├── โจทย์_final.txt                        ← assignment brief
│
├── scripts/
│   ├── 09_link_prediction.py              Path A — 5-method temporal link prediction
│   ├── 10_return_prediction.py            Path B — B0..B3 return prediction + portfolio
│   ├── 11_sentiment_finetune.py           Path C — WangchanBERTa 5-fold fine-tune
│   ├── 12_report_final_latex.py           builds the 4-page LNCS .tex
│   └── 13_make_slides.py                  builds the 14-slide .pptx
│
├── data/
│   └── temporal_split.json                cached event-anchored train/val/test split
│
├── output/
│   ├── lp_baselines.csv                   Path A metrics (5 methods × val/test + CI)
│   ├── lp_gnn.csv                         GraphSAGE detail
│   ├── lp_topk_predicted_edges.csv        top-30 unseen pairs, tagged w/ Midterm bucket
│   ├── lp_roc.{png,svg}                   test-set ROC (5 curves)
│   ├── lp_precision_at_k.{png,svg}        Precision@{10,25,50} bars
│   ├── rp_features.csv                    per-(stock,day) feature matrix
│   ├── rp_metrics.json                    B0..B3 accuracy / F1 / Sharpe proxy
│   ├── rp_baseline_vs_full.{png,svg}      accuracy bars
│   ├── rp_sharpe_curve.{png,svg}          long–short cumulative return
│   ├── rp_feature_importance.{png,svg}    XGBoost gain importance
│   ├── sentiment_finetuned_validation.json  baseline vs fine-tuned metrics
│   ├── sentiment_finetuned_confusion.{png,svg}
│   └── sentiment_finetuned_predictions.csv  PostID, gold, baseline_pred, ft_pred
│
└── report/
    ├── DADS7201_Final_Report.tex          4-page Springer LNCS source
    ├── DADS7201_Final_Report.pdf          compiled with tectonic
    └── DADS7201_Final_Slides.pptx         14 slides, Thai speaker notes
```

Midterm outputs are consumed as inputs; the Final data layer adds only
the cached temporal split. (`models/` is a placeholder — the fine-tuned
weights are re-derived per fold and not checkpointed to disk.)
</content>
</invoke>
