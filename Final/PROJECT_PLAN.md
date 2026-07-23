# Project Plan — Predictive Extension

Complements [README.md](README.md). This file is the *implementation-level* plan the report is written against. Every table, figure, and metric mentioned here has a designated CSV / JSON path so the report generator can pull them live (same pattern the Midterm used).

---

## Path A — Temporal link prediction on the co-mention graph

### Task definition

Given the co-mention graph built from Pantip posts in `[t_min, T]`, predict which stock pairs `(u, v)` currently *without* a co-mention edge will *acquire* one in the next K days `[T+1, T+K]`.

Formally: for each candidate pair (u, v) with no edge at time T, output a probability `p_uv` that the edge appears within the horizon.

### Temporal split

- **Anchor.** Same event date as Midterm (`data/event_anchor.json` → 2026-05-28).
- **Train edges.** All co-mentions in posts dated `[t_min, T=event_date−1]`.
- **Val edges.** Positive: co-mentions that first appear in `[event_date, event_date+9]`.
- **Test edges.** Positive: co-mentions that first appear in `[event_date+10, event_date+19]`.
- **Negatives.** Randomly sampled non-edges from the same node universe (SET50, 50 nodes) with `neg_ratio = 2:1` for train, 1:1 for val/test (Week-7 convention).
- Cache the split at `data/temporal_split.json` for reproducibility.

### Methods (5 to compare)

| # | Method | Origin | Score for pair (u,v) |
|---|---|---|---|
| 1 | Jaccard coefficient | Week 3 | `|N(u) ∩ N(v)| / |N(u) ∪ N(v)|` |
| 2 | Adamic–Adar | Week 3 | `Σ_{w ∈ N(u) ∩ N(v)} 1 / log|N(w)|` |
| 3 | Preferential attachment | Week 3 | `|N(u)| · |N(v)|` |
| 4 | Node2Vec + logistic on Hadamard product | Week 5 | `LR(z_u ⊙ z_v)` with 64-dim Node2Vec (`p=1, q=1`, walks=10, len=40) |
| 5 | GraphSAGE (2 layers, 64 hidden) | Week 7 | dot(z_u, z_v) from SAGEConv trained with `BCEWithLogitsLoss` |

Methods 1–3 are Weeks-3 heuristics (no training). Methods 4–5 use `RandomLinkSplit`, `LinkNeighborLoader`, `BCEWithLogitsLoss`, and `Adam(lr=1e-3)` exactly as the Week-7 tutorial.

### Metrics

- **ROC-AUC** on val + test
- **Precision@k** at k ∈ {10, 25, 50}
- **Mean Reciprocal Rank (MRR)** for held-out positives
- **95% bootstrap CI** on AUC (500 iterations) so each score has an error bar — this satisfies the "statistical rigor" bullet from the Midterm rubric.

### Output artefacts

```
output/
├── lp_baselines.csv                # (method, split, AUC, Prec@10, Prec@25, MRR, AUC_ci_lo, AUC_ci_hi)
├── lp_gnn.csv                      # same shape, GraphSAGE only
├── lp_topk_predicted_edges.csv     # 30 highest-probability unseen pairs w/ midterm bucket (hype-only / overlooked / new)
├── lp_roc.png/.svg                 # 5-curve ROC on test
└── lp_precision_at_k.png/.svg      # bar chart at k = 10, 25, 50
```

### Story hook for the report

Cross-reference each predicted edge with the Midterm's Attention-vs-Fundamentals bucket. If the model predicts that many currently *overlooked co-mover* pairs will acquire co-mention edges, that's a strong claim: retail eventually catches up with fundamentals. If it predicts new *hype-only* pairs instead, retail hype is self-reinforcing.

---

## Path B — Return prediction with graph features

### Task definition

For each SET50 stock `s` and each trading day `d` in the window `[event_date−30, event_date+30]`, predict either:
- **Classification:** sign of the next-day return `sign(r_{d+1})` ∈ {up, down}
- **Regression:** magnitude of `r_{d+1}`

We report classification headline (interpretable) plus regression as secondary.

### Feature matrix

Per (stock, day) row:

| Group | Features | Source |
|---|---|---|
| Own returns lag | `r_{d}`, `r_{d-1}`, `r_{d-5:d-1}` mean / std | `data/returns.csv` |
| Centrality (rolling) | degree, betweenness, eigenvector on 10-day co-mention window ending at d | recomputed daily |
| Community | one-hot(Louvain community_id from window ending at d) | recomputed daily |
| Attention volume | daily mention count for stock s | `data/mentions.csv` |
| Sentiment | mean post sentiment for stock s over `[d-5, d]` | `data/sentiment.csv` (or Path C output) |
| Hype Hub Score | composite on window ending at d | recomputed |
| Cross-sectional | market rank of centrality / sentiment on day d | derived |

### Models

- **Baseline B0:** past-return-only logistic regression (lag features).
- **B1:** B0 + centrality + community features.
- **B2:** B1 + sentiment + attention features (full model).
- **B3 (optional):** XGBoost on B2 features, 5-fold time-series CV.

### Evaluation

- Time-series CV (expanding window: train on `[t_min, T-k]`, test on `[T-k+1, T]` for k in {5, 10, 15, 20, 25} days).
- **Directional accuracy** = `mean(pred_sign == true_sign)`.
- **Precision on high-confidence** predictions (top-decile probability).
- **Sharpe-ratio proxy:** naïve portfolio going long top-decile predicted-up stocks, short bottom-decile predicted-down — daily rebalance. Compare to equal-weight SET50 benchmark.

### Output artefacts

```
output/
├── rp_features.csv                 # full feature matrix per (stock, day)
├── rp_metrics.json                 # {model → {accuracy, precision@decile, F1, sharpe_proxy}}
├── rp_feature_importance.png/.svg  # XGBoost gain importance top-15
├── rp_baseline_vs_full.png/.svg    # bar chart of accuracy, B0..B3
└── rp_sharpe_curve.png/.svg        # cumulative return of graph-feature strategy vs benchmark
```

### Story hook

"Do graph features add signal beyond own-return lags?" If B2 > B0 by ≥3 pp, we report the delta with a bootstrap CI to defend it.

---

## Path C — Sentiment classifier fine-tuning

### Task definition

Fine-tune `poom-sci/WangchanBERTa-finetuned-sentiment` on 200–300 hand-labelled Pantip investment posts (extending the 50-post gold set from Midterm), and re-evaluate H3.

### Data

- **Reuse:** `data/sentiment_gold.csv` (50 posts, Claude-labelled in Midterm).
- **Add:** stratified sample 200 more posts from `data/posts.csv` (aim for ~70 Positive, ~70 Negative, ~60 Neutral by manual class balancing). Hand-labeller: Claude (documented as second annotator; if human-annotator time permits, double-label 30 posts for inter-annotator κ).
- **Split:** 80% train, 20% held-out test (stratified).

### Training

- **Base:** `poom-sci/WangchanBERTa-finetuned-sentiment` as starting point (transfer learning from Thai sentiment weights).
- **Framework:** `transformers.Trainer` on CPU (or Colab GPU if runtime becomes an issue).
- **Hyperparameters:** batch 8, lr 2e-5, epochs 5, weight decay 0.01, early stopping on val macro F1.
- **Time-box:** finish in 1 day. If accuracy > 0.55, use in Paths A/B. If not, report the failure honestly and note that domain-adapted training on labelled Thai financial data is still limited.

### Output artefacts

```
output/
├── sentiment_finetuned_validation.json   # accuracy / macro F1 / Cohen κ / confusion matrix (on 20% held-out)
├── sentiment_finetuned_confusion.png/.svg
└── sentiment_finetuned_predictions.csv   # PostID, gold, model_pred, model_pred_ft, delta
```

If the fine-tuned model works, re-run `02_sentiment.py` with `SENTIMENT_MODEL = 'finetuned'` and see whether H3 flips.

---

## Report structure (4-page LNCS)

Same class file and typography as the Midterm (`llncs.cls`, TNR 10 pt, SVG figures with PNG fallback).

**Section budget:**

| Section | Content | Lines (LNCS 10 pt) |
|---|---|---:|
| Abstract + keywords | Recap + new predictive result headlines | 10 |
| 1 Overview | 3-sentence recap linking to `Midterm/README.md` | 5 |
| 2 Prior findings (H1–H4 table + Jaccard result) | Compressed to one page total incl. small figure | 25 |
| 3 Predictive framing | Why link-prediction + return-prediction | 10 |
| 4 Method A — Temporal link prediction | Split protocol + 5 methods + metrics | 20 |
| 5 Method A results | Table (AUC + CI for 5 methods) + ROC figure | 15 |
| 6 Method B — Return prediction | Feature list + models + evaluation | 15 |
| 7 Method B results | Table + Sharpe-proxy curve | 15 |
| 8 Method C — Sentiment fine-tune | Before/after metrics | 8 |
| 9 Discussion + limitations | Honest — what works, what doesn't, what next | 15 |
| References | 8–10 items (Kipf & Welling, Hamilton, Grover & Leskovec, etc.) | 12 |

**Figures (3 total):**
1. Fig. 1 (side-by-side): Midterm evolution_metrics.svg + overlap sensitivity.svg (recap)
2. Fig. 2: ROC curves + Precision@k bar chart (Path A)
3. Fig. 3: Feature importance (XGBoost) + cumulative return curve (Path B)

**References to add** (beyond Midterm's 6):
- **Grover & Leskovec (2016)** — Node2Vec
- **Hamilton, Ying, Leskovec (2017)** — GraphSAGE
- **Kipf & Welling (2017)** — GCN
- **Perozzi, Al-Rfou, Skiena (2014)** — DeepWalk
- (optional) **Lü & Zhou (2011)** — Link prediction in complex networks survey

---

## Scripts to write (skeletons in `scripts/`)

| Script | Owns | Depends on |
|---|---|---|
| `09_link_prediction.py` | Path A end-to-end | Midterm `data/mentions.csv`, `data/posts.csv`, `data/event_anchor.json` |
| `10_return_prediction.py` | Path B end-to-end | Midterm `data/returns.csv` + all `output/*.csv` |
| `11_sentiment_finetune.py` | Path C training + eval | `data/sentiment_gold.csv`, transformers |
| `12_report_final_latex.py` | Compose 4-page LNCS `.tex` from CSVs/JSONs | everything above + Midterm outputs |

All scripts follow the Midterm convention: read from `Midterm/data/` and `Midterm/output/`, write into `Final/output/`, no re-scraping.

---

## Answering "what would get full marks"

Three levers the professor watches for:

1. **Depth of course-methods coverage.** Path A alone spans Weeks 3–7. Include the 5-method comparison table — one glance shows we've used every technique taught in the semester.
2. **Statistical honesty.** Every headline AUC and every accuracy delta comes with a bootstrap CI. Report ties this to the Midterm feedback that we've already internalised.
3. **Turning descriptive findings into testable predictions.** The Midterm's Attention-vs-Fundamentals dissociation becomes a *falsifiable claim*: does the link-prediction model successfully forecast the emergence of new *overlooked co-mover* edges? Either answer is publishable.

If we deliver all three, plus Path B's finance-relevant portfolio proxy, the report is easily A-territory even without Path C. Path C is the "sixth degree of polish" — nice to have but not the deciding factor.
