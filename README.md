# DADS7201 — Social Network Analysis

Coursework for **DADS7201 — Social Network Analysis** at NIDA, organised
by week. Each week is a self-contained Streamlit application with its own
data, dependencies, and deploy configuration.

## Weeks

| Week | Topic | Folder |
|---|---|---|
| 1 | Bipartite network: SET50 companies ↔ top-5 major shareholders | [`Week1/`](Week1/) |
| 2 | Centrality on a conflict-news graph in Neo4j Aura (Degree / Closeness / Betweenness / Eigenvector / Katz / PageRank) | [`Week2/`](Week2/) |
| 3 | Neo4j Graph Data Science — Bridges + 5 centralities + Louvain vs K-Means community detection (edges_rows + Karate + IMDB) + HW2 Streamlit dashboard on Stanford MemeTracker | [`Week3/`](Week3/) |
| 4 | Similarity & Node Embeddings on GDS — similarity functions, Node Similarity, KNN, FastRP, end-to-end product recommendation | [`Week4/`](Week4/) |
| Midterm | Retail Hype & Sentiment Network on SET50 (Pantip Sinthorn + WangchanBERTa + correlation overlay) | [`Midterm/`](Midterm/) |
| 5 | Node Embeddings — random walk + softmax NLL from scratch, DeepWalk / Node2Vec (BFS vs DFS), GCN preview on Karate Club (PyG) | [`Week5/`](Week5/) |
| 6 | Graph Neural Networks — GCN vs MLP on Cora (node classification), 3-layer GCN + `global_mean_pool` on MUTAG (graph classification), GAT with attention heads | [`Week6/`](Week6/) |
| 7 | Link Prediction on MovieLens — heterogeneous GraphSAGE with `to_hetero` + `LinkNeighborLoader` + dot-product classifier (Val AUC 0.93) | [`Week7/`](Week7/) |
| Final | Predictive extension of the Midterm SET50 hype network — temporal link prediction (Preferential AUC 0.73), return prediction (Sharpe 3.07→4.30), WangchanBERTa sentiment fine-tune (H3 re-test) | [`Final/`](Final/) |

## How this repo is organised

Each `WeekN/` folder is **independent**:

- has its own `streamlit_app.py`, `requirements.txt`, `packages.txt`
- can be deployed as a separate Streamlit Cloud app pointing at
  `WeekN/streamlit_app.py`
- shares no code with other weeks — copy/paste any patterns you want
  to reuse

## Adding a new week

```powershell
# from the repo root
mkdir Week2
# copy the Week1 skeleton as a starting point if useful
Copy-Item -Recurse Week1\.streamlit, Week1\requirements.txt, Week1\packages.txt Week2\
# build your app in Week2/streamlit_app.py
git add Week2
git commit -m "Week2: initial app"
git push
```

Then on https://share.streamlit.io: **New app** → repo
`puwadonsri/DADS7201_Social_Network`, main file `Week2/streamlit_app.py`.

## Author

Puwadon Sri ([@puwadonsri](https://github.com/puwadonsri)) ·
DADS, NIDA · 2026
