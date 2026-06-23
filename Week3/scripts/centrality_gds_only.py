"""
Bridges + 5 centralities (Betweenness / Closeness / Degree / Eigenvector /
PageRank) on edges_rows.csv, all computed via Neo4j GDS Cypher.

Visual upgrade over the v1 (circular layout) version:
  - Shared Fruchterman-Reingold layout so community structure is visible.
  - Convex-hull halos colored by Louvain community (cosmetic background only).
  - Per-metric perceptually uniform colormap (viridis/plasma/cividis/...).
  - Top-3 nodes get bold labels with a soft white pill; the rest stay subtle.
"""
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neo4j_utils import get_driver, output_path
from _viz_utils import (
    PALETTE, annotate_top, colormap_for, draw_edges, draw_hull,
    draw_nodes, fr_layout, highlight_edges, style_axes,
)

GRAPH_NAME = "bridgesGraph"
driver = get_driver()


def run(q, p=None):
    with driver.session(database="neo4j") as s:
        return list(s.run(q, p or {}))


# --- 1. Project undirected graph ---
print("==> Projecting graph...")
run("CALL gds.graph.drop($n, false)", {"n": GRAPH_NAME})
run(
    """
    MATCH (s:Person)-[r:KNOWS]->(t:Person)
    RETURN gds.graph.project($n, s, t, {}, { undirectedRelationshipTypes: ['*'] })
    """,
    {"n": GRAPH_NAME},
)

# --- 2. Bridges ---
print("==> Bridges...")
bridge_rows = run(
    f"""
    CALL gds.bridges.stream('{GRAPH_NAME}')
    YIELD from, to, remainingSizes
    RETURN gds.util.asNode(from).name AS f,
           gds.util.asNode(to).name   AS t,
           remainingSizes
    """
)
bridge_set = set()
for r in bridge_rows:
    bridge_set.add((r["f"], r["t"]))
    bridge_set.add((r["t"], r["f"]))
print(f"    found {len(bridge_rows)} bridge edge(s)")

# --- 3. Centralities ---
metrics = [
    ("Betweenness", "gds.betweenness.stream"),
    ("Closeness", "gds.closeness.stream"),
    ("Degree", "gds.degree.stream"),
    ("Eigenvector", "gds.eigenvector.stream"),
    ("PageRank", "gds.pageRank.stream"),
]
scores = {}
for label, algo in metrics:
    print(f"==> {label}...")
    rows = run(
        f"""
        CALL {algo}('{GRAPH_NAME}')
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).name AS name, score
        """
    )
    scores[label] = {r["name"]: r["score"] for r in rows}

# --- 4. Louvain (only used to colour the cosmetic background hulls) ---
print("==> Louvain (for background hulls)...")
louvain = {
    r["name"]: r["communityId"]
    for r in run(
        f"""
        CALL gds.louvain.stream('{GRAPH_NAME}')
        YIELD nodeId, communityId
        RETURN gds.util.asNode(nodeId).name AS name, communityId
        """
    )
}

# --- 5. Nodes + edges for plotting ---
all_names = sorted(scores["Degree"].keys())
edge_rows = run("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name AS a, b.name AS b")
all_edges = [(r["a"], r["b"]) for r in edge_rows]
driver.close()

# --- 6. Shared force-directed layout ---
print("==> Computing FR layout...")
pos = fr_layout(all_names, all_edges, iterations=200, seed=7, k_scale=1.7)

# --- 7. Plot 2x3 grid ---
fig, axes = plt.subplots(2, 3, figsize=(22, 14), facecolor="#fafbfc")
axes = axes.flatten()

# community → colour map (for hull background only)
comm_ids = sorted(set(louvain.values()))
comm_color = {cid: PALETTE[i % len(PALETTE)] for i, cid in enumerate(comm_ids)}


def draw_background_hulls(ax):
    groups = defaultdict(list)
    for name, cid in louvain.items():
        groups[cid].append(pos[name])
    for cid, pts in groups.items():
        draw_hull(ax, pts, color=comm_color[cid], alpha=0.10)


# ---- panel 0: Bridges ----
ax = axes[0]
draw_background_hulls(ax)
draw_edges(ax, pos, all_edges, alpha=0.18, color="#888", lw=0.6)
if bridge_set:
    highlight_edges(ax, pos, [(a, b) for a, b in all_edges if (a, b) in bridge_set],
                    color="#d62728", lw=3.0)
deg = scores["Degree"]
node_sizes = [220 + deg[n] * 90 for n in all_names]
node_colors = [comm_color[louvain[n]] for n in all_names]
draw_nodes(ax, pos, all_names, sizes=node_sizes, colors=node_colors)
top3_by_deg = sorted(all_names, key=lambda n: deg[n], reverse=True)[:3]
annotate_top(ax, pos, all_names, [deg[n] for n in all_names], top_k=3, fontsize=11)
subtitle = "no bridges found" if not bridge_set else f"{len(bridge_set)//2} bridge edges (red)"
style_axes(ax, "Bridges", subtitle)

# ---- panels 1..5: centralities ----
for i, (label, _algo) in enumerate(metrics, start=1):
    ax = axes[i]
    draw_background_hulls(ax)
    draw_edges(ax, pos, all_edges, alpha=0.18, color="#888", lw=0.6)

    values = np.array([scores[label].get(n, 0.0) for n in all_names], dtype=float)
    vmin, vmax = values.min(), values.max()
    norm = np.zeros_like(values) if vmax == vmin else (values - vmin) / (vmax - vmin)

    cmap = colormap_for(label)
    node_colors = [cmap(0.15 + 0.8 * v) for v in norm]    # avoid extreme dark/light
    node_sizes = [240 + v * 1600 for v in norm]
    draw_nodes(ax, pos, all_names, sizes=node_sizes, colors=node_colors, edge_lw=0.5)

    annotate_top(ax, pos, all_names, values, top_k=3, fontsize=11)

    top1_name = all_names[int(np.argmax(values))]
    top1_value = float(values.max())
    style_axes(ax, label, f"top: {top1_name}  ({top1_value:.3f})")

plt.suptitle("Neo4j GDS — Bridges + 5 Centralities on edges_rows.csv",
             fontsize=18, fontweight="bold", y=0.995, color="#222")
fig.text(0.5, 0.012,
         "Layout: Fruchterman-Reingold (24 Person nodes, 84 KNOWS edges).  "
         "Background hulls = Louvain communities.  Computed via gds.* in Cypher.",
         ha="center", fontsize=10, color="#666", style="italic")
plt.tight_layout(rect=(0, 0.025, 1, 0.985))

img_path = output_path("images", "centrality_plot_gds.png")
os.makedirs(os.path.dirname(img_path), exist_ok=True)
plt.savefig(img_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n==> Saved: {img_path}")
print("Done.")
