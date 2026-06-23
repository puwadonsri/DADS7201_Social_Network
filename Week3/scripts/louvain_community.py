"""
Louvain community detection on edges_rows.csv (Cypher / GDS).

Pipeline:
  1. Project undirected graph.
  2. Run Louvain (stream + stats) to get communities + modularity.
  3. Run K-Means (k=4, FastRP 128-dim) for side-by-side comparison.
  4. Side-by-side PNG (matplotlib circular layout) — Louvain vs K-Means.
  5. Print agreement metrics (NMI/ARI) between Louvain and K-Means.
"""
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neo4j_utils import get_driver, output_path
from _viz_utils import (
    PALETTE, draw_edges, draw_hull, draw_nodes, fr_layout, style_axes,
)

GRAPH_NAME = "louvainGraph"
driver = get_driver()


def run(q, params=None):
    with driver.session(database="neo4j") as s:
        return list(s.run(q, params or {}))


# --- 1. (Re)project undirected graph ---
print("==> Projecting graph...")
run("CALL gds.graph.drop($n, false)", {"n": GRAPH_NAME})
run(
    """
    MATCH (s:Person)-[r:KNOWS]->(t:Person)
    RETURN gds.graph.project($n, s, t, {}, { undirectedRelationshipTypes: ['*'] })
    """,
    {"n": GRAPH_NAME},
)

# --- 2. Louvain ---
print("==> Running Louvain...")
stats = run(
    f"""
    CALL gds.louvain.stats('{GRAPH_NAME}')
    YIELD modularity, communityCount, communityDistribution
    RETURN modularity, communityCount, communityDistribution
    """
)
modularity = stats[0]["modularity"]
n_comm = stats[0]["communityCount"]
print(f"    modularity = {modularity:.4f},  communities = {n_comm}")

louvain_rows = run(
    f"""
    CALL gds.louvain.stream('{GRAPH_NAME}')
    YIELD nodeId, communityId
    RETURN gds.util.asNode(nodeId).name AS name, communityId
    ORDER BY communityId, name
    """
)
louvain = {r["name"]: r["communityId"] for r in louvain_rows}

louvain_groups = defaultdict(list)
for name, cid in louvain.items():
    louvain_groups[cid].append(name)
print("    Louvain communities:")
for cid in sorted(louvain_groups):
    members = louvain_groups[cid]
    print(f"      cid={cid} ({len(members)}): {', '.join(members)}")

# --- 3. K-Means k=4 for comparison (FastRP embedding) ---
print("==> Running K-Means (k=4) for comparison...")
run(
    f"""
    CALL gds.fastRP.mutate('{GRAPH_NAME}', {{
        embeddingDimension: 128, randomSeed: 42, mutateProperty: 'embedding'
    }})
    """
)
kmeans_rows = run(
    f"""
    CALL gds.kmeans.stream('{GRAPH_NAME}', {{
        nodeProperty: 'embedding', k: 4, randomSeed: 42,
        computeSilhouette: true, concurrency: 1
    }})
    YIELD nodeId, communityId, silhouette
    RETURN gds.util.asNode(nodeId).name AS name, communityId, silhouette
    ORDER BY communityId, name
    """
)
kmeans = {r["name"]: r["communityId"] for r in kmeans_rows}
sil_scores = [r["silhouette"] for r in kmeans_rows if r["silhouette"] is not None]
avg_sil = float(np.mean(sil_scores)) if sil_scores else float("nan")
print(f"    K-Means silhouette = {avg_sil:.4f}")

kmeans_groups = defaultdict(list)
for name, cid in kmeans.items():
    kmeans_groups[cid].append(name)
print("    K-Means communities:")
for cid in sorted(kmeans_groups):
    members = kmeans_groups[cid]
    print(f"      cid={cid} ({len(members)}): {', '.join(members)}")

# --- 4. Agreement metrics (NMI + ARI, no sklearn) ---
def contingency(a_labels, b_labels):
    a_vals, b_vals = sorted(set(a_labels)), sorted(set(b_labels))
    a_idx = {v: i for i, v in enumerate(a_vals)}
    b_idx = {v: i for i, v in enumerate(b_vals)}
    m = np.zeros((len(a_vals), len(b_vals)), dtype=int)
    for x, y in zip(a_labels, b_labels):
        m[a_idx[x], b_idx[y]] += 1
    return m


def nmi(a, b):
    c = contingency(a, b).astype(float)
    n = c.sum()
    pi_ = c.sum(axis=1) / n
    pj = c.sum(axis=0) / n
    h_a = -np.sum(pi_[pi_ > 0] * np.log(pi_[pi_ > 0]))
    h_b = -np.sum(pj[pj > 0] * np.log(pj[pj > 0]))
    mi = 0.0
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c[i, j] > 0:
                mi += (c[i, j] / n) * np.log((c[i, j] * n) / (c[i, :].sum() * c[:, j].sum()))
    return mi / np.sqrt(h_a * h_b) if h_a > 0 and h_b > 0 else 0.0


def ari(a, b):
    from math import comb
    c = contingency(a, b)
    sum_comb_c = sum(comb(int(v), 2) for v in c.flatten())
    sum_comb_a = sum(comb(int(v), 2) for v in c.sum(axis=1))
    sum_comb_b = sum(comb(int(v), 2) for v in c.sum(axis=0))
    n = c.sum()
    total = comb(int(n), 2)
    expected = sum_comb_a * sum_comb_b / total
    max_index = (sum_comb_a + sum_comb_b) / 2
    return (sum_comb_c - expected) / (max_index - expected) if max_index != expected else 1.0


all_names = sorted(louvain.keys())
l_labels = [louvain[n] for n in all_names]
k_labels = [kmeans[n] for n in all_names]
print(f"\n==> Agreement (Louvain vs K-Means): NMI={nmi(l_labels, k_labels):.4f}, ARI={ari(l_labels, k_labels):.4f}")

# --- 5. Get edges for visualization ---
edges_rows = run("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name AS a, b.name AS b")
all_edges = [(r["a"], r["b"]) for r in edges_rows]
driver.close()

# --- 6. Shared FR layout so both panels are directly comparable ---
print("==> Computing FR layout...")
pos = fr_layout(all_names, all_edges, iterations=200, seed=7, k_scale=1.7)


def draw_panel(ax, label_map, title, subtitle):
    cids = sorted(set(label_map.values()))
    color_for = {cid: PALETTE[i % len(PALETTE)] for i, cid in enumerate(cids)}

    # convex hull behind each community
    groups: dict = defaultdict(list)
    for name, cid in label_map.items():
        groups[cid].append(pos[name])
    for cid, pts in groups.items():
        draw_hull(ax, pts, color=color_for[cid], alpha=0.16, lw=1.6)

    # edges — coloured if within community, grey if across
    for a, b in all_edges:
        if a not in pos or b not in pos:
            continue
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        same = label_map.get(a) == label_map.get(b)
        ax.plot(
            [x1, x2], [y1, y2],
            color=color_for[label_map[a]] if same else "#bbbbbb",
            linewidth=2.0 if same else 0.6,
            alpha=0.55 if same else 0.35,
            zorder=2,
        )

    # nodes — bigger, with name annotated below
    node_colors = [color_for[label_map[n]] for n in all_names]
    draw_nodes(ax, pos, all_names, sizes=[520] * len(all_names), colors=node_colors,
               edge_lw=0.7, alpha=0.95)
    for name, (x, y) in pos.items():
        ax.text(x, y - 0.07, name, ha="center", va="top",
                fontsize=8, color="#222", zorder=6)

    style_axes(ax, title, subtitle)


fig, axes = plt.subplots(1, 2, figsize=(22, 11), facecolor="#fafbfc")

draw_panel(
    axes[0], louvain,
    "Louvain", f"{n_comm} communities · modularity = {modularity:.3f}",
)
draw_panel(
    axes[1], kmeans,
    "K-Means (k=4)", f"silhouette = {avg_sil:.3f}",
)

plt.suptitle(
    "Community Detection — Louvain vs K-Means on edges_rows.csv",
    fontsize=18, fontweight="bold", y=0.99, color="#222",
)
fig.text(
    0.5, 0.02,
    f"Agreement: NMI = {nmi(l_labels, k_labels):.3f}  ·  ARI = {ari(l_labels, k_labels):.3f}  "
    "·  Layout: Fruchterman-Reingold, edges coloured when within-community.",
    ha="center", fontsize=11, color="#666", style="italic",
)
plt.tight_layout(rect=(0, 0.03, 1, 0.97))

img_path = output_path("images", "community_louvain_vs_kmeans.png")
os.makedirs(os.path.dirname(img_path), exist_ok=True)
plt.savefig(img_path, dpi=180, bbox_inches="tight")
print(f"\n==> Saved: {img_path}")
print("Done.")
