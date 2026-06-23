"""
Louvain on Zachary's Karate Club + comparison with K-Means and ground truth.

Pipeline:
  G = gds.graph.load_karate_club()
  Louvain                -> communities
  FastRP -> K-Means k=2  -> communities (matched to ground truth labels)
  Compare both against the 2-faction ground truth.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neo4j_utils import get_gds, output_path

# Zachary's Karate Club ground truth (Mr.Hi=0, Officer=1).
# GDS load_karate_club uses 1-indexed node IDs (1..34).
# Mr.Hi faction: 1,2,3,4,5,6,7,8,9,11,12,13,14,17,18,20,22
# Officer faction: 10,15,16,19,21,23..34
_MR_HI = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17, 18, 20, 22}
GT = {nid: (0 if nid in _MR_HI else 1) for nid in range(1, 35)}

gds = get_gds()

# --- 1. load karate ---
print("==> Loading Karate Club...")
try:
    gds.run_cypher("CALL gds.graph.drop('karate', false)")
except Exception:
    pass
G = gds.graph.load_karate_club("karate", undirected=False)
print(f"    Nodes: {G.node_count()}, Edges: {G.relationship_count()}")

# --- 2. Louvain ---
print("==> Louvain...")
louvain_df = gds.louvain.stream(G)
louvain_stats = gds.louvain.stats(G)
modularity = float(louvain_stats["modularity"])
n_comm = int(louvain_stats["communityCount"])
print(f"    modularity = {modularity:.4f}, communities = {n_comm}")
louvain = dict(zip(louvain_df["nodeId"].astype(int), louvain_df["communityId"].astype(int)))


# --- 3. K-Means (k=2 — match the ground truth granularity) ---
print("==> FastRP + K-Means (k=2)...")
gds.fastRP.mutate(G, embeddingDimension=128, randomSeed=42, mutateProperty="embedding")
kmeans_df = gds.kmeans.stream(
    G, nodeProperty="embedding", k=2, randomSeed=42,
    computeSilhouette=True, concurrency=1,
)
kmeans = dict(zip(kmeans_df["nodeId"].astype(int), kmeans_df["communityId"].astype(int)))
sil = kmeans_df["silhouette"].dropna()
avg_sil = float(sil.mean()) if len(sil) > 0 else float("nan")
print(f"    K-Means silhouette = {avg_sil:.4f}")


# --- 4. accuracy vs ground truth (majority vote alignment) ---
def accuracy(pred_dict):
    by_cluster = {}
    for nid, cid in pred_dict.items():
        by_cluster.setdefault(cid, []).append(GT[nid])
    cluster_to_gt = {cid: max(set(v), key=v.count) for cid, v in by_cluster.items()}
    correct = sum(1 for nid, cid in pred_dict.items() if cluster_to_gt[cid] == GT[nid])
    return correct, len(pred_dict), cluster_to_gt


lc, lt, l_map = accuracy(louvain)
kc, kt, k_map = accuracy(kmeans)
print(f"\n==> vs ground truth (2 factions):")
print(f"    Louvain   accuracy = {lc}/{lt} = {lc / lt:.2%}  ({n_comm} clusters)")
print(f"    K-Means   accuracy = {kc}/{kt} = {kc / kt:.2%}  (k=2)")


# --- 5. 2D embedding for visualization ---
print("\n==> Generating 2D layout (FastRP)...")
try:
    gds.run_cypher("CALL gds.graph.drop('karate2d', false)")
except Exception:
    pass
G2 = gds.graph.load_karate_club("karate2d", undirected=False)
gds.fastRP.mutate(G2, embeddingDimension=2, randomSeed=42, mutateProperty="emb2d")
emb2d_df = gds.graph.streamNodeProperties(G2, ["emb2d"])
prop_col = [c for c in emb2d_df.columns if c != "nodeId"][0]
emb2d_df["x"] = emb2d_df[prop_col].apply(lambda v: v[0])
emb2d_df["y"] = emb2d_df[prop_col].apply(lambda v: v[1])

# Stream relationships from the in-memory karate graph (does NOT touch Neo4j DB).
edges = gds.run_cypher(
    "CALL gds.graph.relationships.stream('karate2d') YIELD sourceNodeId, targetNodeId "
    "RETURN sourceNodeId AS src, targetNodeId AS dst"
)

# --- 6. plot ---
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
PALETTE = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]
gt_palette = {0: "#66c2a5", 1: "#fc8d62"}

# index for layout
pos = {int(r["nodeId"]): (r["x"], r["y"]) for _, r in emb2d_df.iterrows()}


def draw(ax, label_map, title, color_for):
    for _, r in edges.iterrows():
        s, t = int(r["src"]), int(r["dst"])
        if s in pos and t in pos:
            x1, y1 = pos[s]
            x2, y2 = pos[t]
            ax.plot([x1, x2], [y1, y2], color="#cccccc", linewidth=0.5, alpha=0.4, zorder=2)
    for nid, (x, y) in pos.items():
        c = color_for[label_map[nid]]
        ax.scatter(x, y, c=c, s=350, edgecolors="black", linewidths=0.4, zorder=5)
        ax.text(x, y, str(nid), fontsize=8, ha="center", va="center", fontweight="bold", zorder=6)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("FastRP dim-1")
    ax.set_ylabel("FastRP dim-2")


louvain_color = {cid: PALETTE[i % len(PALETTE)] for i, cid in enumerate(sorted(set(louvain.values())))}
kmeans_color = {cid: PALETTE[i % len(PALETTE)] for i, cid in enumerate(sorted(set(kmeans.values())))}

draw(axes[0], louvain,
     f"Louvain ({n_comm} communities, mod={modularity:.3f})\nacc vs GT = {lc / lt:.0%}",
     louvain_color)
draw(axes[1], kmeans,
     f"K-Means k=2 (sil={avg_sil:.3f})\nacc vs GT = {kc / kt:.0%}",
     kmeans_color)
draw(axes[2], GT, "Ground Truth (Mr.Hi vs Officer)", gt_palette)

plt.suptitle("Karate Club — Louvain vs K-Means vs Ground Truth", fontsize=15, fontweight="bold")
plt.tight_layout()

img_path = output_path("images", "karate_louvain_vs_kmeans.png")
os.makedirs(os.path.dirname(img_path), exist_ok=True)
plt.savefig(img_path, dpi=180, bbox_inches="tight")
print(f"\n==> Saved: {img_path}")

gds.close()
print("Done.")
