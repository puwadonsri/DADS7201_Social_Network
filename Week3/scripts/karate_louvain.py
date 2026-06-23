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
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neo4j_utils import get_gds, output_path
from _viz_utils import PALETTE, draw_hull, draw_nodes, fr_layout, style_axes

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


# --- 5. Edges (for drawing) — stream from in-memory graph ---
print("\n==> Streaming edges + computing FR layout...")
edges_df = gds.run_cypher(
    "CALL gds.graph.relationships.stream('karate') YIELD sourceNodeId, targetNodeId "
    "RETURN sourceNodeId AS src, targetNodeId AS dst"
)
all_nodes = sorted(set(edges_df["src"]).union(set(edges_df["dst"])))
all_edges = [(int(r["src"]), int(r["dst"])) for _, r in edges_df.iterrows()]
pos = fr_layout(all_nodes, all_edges, iterations=200, seed=11, k_scale=1.5)

# --- 6. plot ---
gt_palette = {0: "#66c2a5", 1: "#fc8d62"}
louvain_color = {cid: PALETTE[i % len(PALETTE)] for i, cid in enumerate(sorted(set(louvain.values())))}
kmeans_color = {cid: PALETTE[i % len(PALETTE)] for i, cid in enumerate(sorted(set(kmeans.values())))}


def draw_panel(ax, label_map, color_for, title, subtitle, *, mistakes=None):
    # convex hull background per community
    groups = defaultdict(list)
    for nid, cid in label_map.items():
        if nid in pos:
            groups[cid].append(pos[nid])
    for cid, pts in groups.items():
        draw_hull(ax, pts, color=color_for[cid], alpha=0.16, lw=1.6)

    # edges — coloured if within community
    for s, t in all_edges:
        if s not in pos or t not in pos:
            continue
        x1, y1 = pos[s]
        x2, y2 = pos[t]
        same = label_map.get(s) == label_map.get(t)
        ax.plot(
            [x1, x2], [y1, y2],
            color=color_for[label_map[s]] if same else "#bbbbbb",
            linewidth=1.6 if same else 0.5,
            alpha=0.5 if same else 0.3,
            zorder=2,
        )

    # nodes — flag mistakes (vs ground truth) with red ring
    mistakes = mistakes or set()
    for nid in all_nodes:
        x, y = pos[nid]
        c = color_for[label_map[nid]]
        edge = "#d62728" if nid in mistakes else "#222"
        lw = 2.2 if nid in mistakes else 0.6
        ax.scatter(x, y, c=c, s=520, edgecolors=edge, linewidths=lw, alpha=0.95, zorder=5)
        ax.text(x, y, str(nid), fontsize=9, ha="center", va="center",
                fontweight="bold", color="#111", zorder=6)

    style_axes(ax, title, subtitle)


# Mark mistakes (where predicted faction differs from ground truth)
louvain_mistakes = {nid for nid, cid in louvain.items() if l_map[cid] != GT[nid]}
kmeans_mistakes = {nid for nid, cid in kmeans.items() if k_map[cid] != GT[nid]}

fig, axes = plt.subplots(1, 3, figsize=(24, 9), facecolor="#fafbfc")

draw_panel(
    axes[0], louvain, louvain_color,
    "Louvain", f"{n_comm} communities · modularity = {modularity:.3f} · acc = {lc/lt:.0%}",
    mistakes=louvain_mistakes,
)
draw_panel(
    axes[1], kmeans, kmeans_color,
    "K-Means (k=2)", f"silhouette = {avg_sil:.3f} · acc = {kc/kt:.0%}",
    mistakes=kmeans_mistakes,
)
draw_panel(
    axes[2], GT, gt_palette,
    "Ground Truth", "Mr.Hi (teal) vs Officer (orange) — 2-faction split",
)

plt.suptitle(
    "Zachary's Karate Club — Louvain vs K-Means vs Ground Truth",
    fontsize=18, fontweight="bold", y=0.99, color="#222",
)
fig.text(
    0.5, 0.02,
    "Layout: Fruchterman-Reingold from KNOWS edges.  "
    "Red-ringed nodes = mis-classified vs ground-truth faction (majority-vote alignment).",
    ha="center", fontsize=10, color="#666", style="italic",
)
plt.tight_layout(rect=(0, 0.03, 1, 0.97))

img_path = output_path("images", "karate_louvain_vs_kmeans.png")
os.makedirs(os.path.dirname(img_path), exist_ok=True)
plt.savefig(img_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n==> Saved: {img_path}")

gds.close()
print("Done.")
