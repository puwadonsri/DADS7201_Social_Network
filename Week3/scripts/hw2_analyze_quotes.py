"""
HW2 — Centralities + Bridges + Louvain on the MemeTracker domain graph.

Pipeline (all via Cypher / GDS):
  1. LOAD CSV → Domain nodes + LINKS edges (with weight) into Neo4j.
  2. Project an undirected weighted graph.
  3. Run Bridges, Betweenness, Closeness, Degree, Eigenvector, PageRank, Louvain.
  4. Render a 7-panel matplotlib figure.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from neo4j_utils import get_driver, output_path  # noqa: E402
from _viz_utils import (  # noqa: E402
    PALETTE, colormap_for, draw_edges, draw_hull, draw_nodes,
    fr_layout, highlight_edges, style_axes,
)

GRAPH = "quotesGraph"
driver = get_driver()


def run(q, p=None):
    with driver.session(database="neo4j") as s:
        return list(s.run(q, p or {}))


# --- 1. (Re)load CSV ---
print("==> Loading CSV into Neo4j...")
run("MATCH (d:Domain) DETACH DELETE d")
run("""
LOAD CSV WITH HEADERS FROM 'file:///quotes_domains.csv' AS row
CREATE (d:Domain {
    name: row.domain,
    post_count: toInteger(row.post_count)
})
""")
run("""
LOAD CSV WITH HEADERS FROM 'file:///quotes_domain_edges.csv' AS row
MATCH (a:Domain {name: row.src}), (b:Domain {name: row.dst})
CREATE (a)-[:LINKS {weight: toInteger(row.weight)}]->(b)
""")
n_nodes = run("MATCH (d:Domain) RETURN count(d) AS c")[0]["c"]
n_rels = run("MATCH ()-[r:LINKS]->() RETURN count(r) AS c")[0]["c"]
print(f"  loaded {n_nodes} Domain nodes, {n_rels} LINKS edges")

# --- 2. Project undirected weighted graph ---
print("==> Projecting undirected graph...")
run("CALL gds.graph.drop($n, false)", {"n": GRAPH})
run(
    """
    MATCH (s:Domain)-[r:LINKS]->(t:Domain)
    RETURN gds.graph.project(
        $n, s, t,
        { relationshipProperties: r { .weight } },
        { undirectedRelationshipTypes: ['*'] }
    )
    """,
    {"n": GRAPH},
)

# --- 3. Bridges ---
print("==> Bridges...")
bridges = run(
    f"""
    CALL gds.bridges.stream('{GRAPH}')
    YIELD from, to, remainingSizes
    RETURN gds.util.asNode(from).name AS f,
           gds.util.asNode(to).name   AS t,
           remainingSizes
    """
)
bridge_set = set()
for r in bridges:
    bridge_set.add((r["f"], r["t"]))
    bridge_set.add((r["t"], r["f"]))
print(f"  {len(bridges)} bridge edge(s)")

# --- 4. Centralities ---
metrics = [
    ("Betweenness", "gds.betweenness.stream"),
    ("Closeness", "gds.closeness.stream"),
    ("Degree", "gds.degree.stream"),
    ("Eigenvector", "gds.eigenvector.stream"),
    ("PageRank", "gds.pageRank.stream"),
]
scores: dict[str, dict[str, float]] = {}
for label, algo in metrics:
    print(f"==> {label}...")
    rows = run(
        f"""
        CALL {algo}('{GRAPH}')
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).name AS name, score
        """
    )
    scores[label] = {r["name"]: r["score"] for r in rows}
    top5 = sorted(scores[label].items(), key=lambda x: -x[1])[:5]
    for n, s in top5:
        print(f"    {n:<35s} {s:.4f}")

# --- 5. Louvain ---
print("==> Louvain...")
lou_stats = run(
    f"CALL gds.louvain.stats('{GRAPH}') YIELD modularity, communityCount RETURN modularity, communityCount"
)[0]
modularity = lou_stats["modularity"]
n_comm = lou_stats["communityCount"]
print(f"  modularity = {modularity:.4f}, communities = {n_comm}")
louvain = {
    r["name"]: r["communityId"]
    for r in run(
        f"""
        CALL gds.louvain.stream('{GRAPH}')
        YIELD nodeId, communityId
        RETURN gds.util.asNode(nodeId).name AS name, communityId
        """
    )
}

# --- 6. Nodes + edges + largest-component restriction (for legibility) ---
wcc_rows = run(
    f"""
    CALL gds.wcc.stream('{GRAPH}')
    YIELD nodeId, componentId
    RETURN gds.util.asNode(nodeId).name AS name, componentId
    """
)
comp_of = {r["name"]: r["componentId"] for r in wcc_rows}
sizes = defaultdict(int)
for cid in comp_of.values():
    sizes[cid] += 1
big_cid = max(sizes, key=lambda c: sizes[c])
core_names = sorted(n for n, c in comp_of.items() if c == big_cid)
print(f"\n==> WCC: {len(sizes)} components, largest = {sizes[big_cid]} nodes (using only this one for layout)")

core_set = set(core_names)
edges_rows = run(
    "MATCH (a:Domain)-[:LINKS]->(b:Domain) RETURN a.name AS a, b.name AS b"
)
all_edges = [(r["a"], r["b"]) for r in edges_rows
             if r["a"] in core_set and r["b"] in core_set]
all_names = core_names

# trim metric dicts to the largest component
for label in scores:
    scores[label] = {n: s for n, s in scores[label].items() if n in core_set}
louvain = {n: c for n, c in louvain.items() if n in core_set}
bridge_set = {(a, b) for a, b in bridge_set if a in core_set and b in core_set}
driver.close()
print(f"==> Nodes for plot: {len(all_names)}, Edges: {len(all_edges)}")

# --- 7. Layout (largest component only) ---
print("==> Computing FR layout...")
pos = fr_layout(all_names, all_edges, iterations=400, seed=11, k_scale=2.0)

# --- 8. Plot 7-panel grid (Bridges + 5 centralities + Louvain) ---
fig, axes = plt.subplots(2, 4, figsize=(28, 14), facecolor="#fafbfc")
axes = axes.flatten()

# Louvain colour per community (used for hull backgrounds + Louvain panel)
comm_ids = sorted(set(louvain.values()))
comm_color = {cid: PALETTE[i % len(PALETTE)] for i, cid in enumerate(comm_ids)}


def draw_backdrop(ax):
    groups = defaultdict(list)
    for name, cid in louvain.items():
        groups[cid].append(pos[name])
    # only hulls for sizeable communities
    for cid, pts in groups.items():
        if len(pts) >= 4:
            draw_hull(ax, pts, color=comm_color[cid], alpha=0.08, lw=0.8)


def label_top(ax, names_in_panel, values, top_k=8, color="#111"):
    paired = sorted(zip(names_in_panel, values), key=lambda x: -x[1])[:top_k]
    for name, _ in paired:
        x, y = pos[name]
        ax.annotate(
            name, (x, y), xytext=(0, 9), textcoords="offset points",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#666", lw=0.4, alpha=0.92),
            zorder=10,
        )


# panel 0: Bridges
ax = axes[0]
draw_backdrop(ax)
draw_edges(ax, pos, all_edges, alpha=0.10, color="#888", lw=0.4)
if bridge_set:
    highlight_edges(
        ax, pos,
        [(a, b) for a, b in all_edges if (a, b) in bridge_set],
        color="#d62728", lw=2.2, alpha=0.85,
    )
deg = scores["Degree"]
deg_arr = np.array([deg[n] for n in all_names], dtype=float)
node_sizes = 60 + 240 * deg_arr / max(deg_arr.max(), 1)
node_colors = [comm_color[louvain[n]] for n in all_names]
draw_nodes(ax, pos, all_names, sizes=node_sizes, colors=node_colors, edge_lw=0.4, alpha=0.92)
label_top(ax, all_names, [deg[n] for n in all_names], top_k=8)
subtitle = f"{len(bridge_set)//2} bridge edges (red)" if bridge_set else "no bridges"
style_axes(ax, "Bridges", subtitle)

# panels 1..5: centralities
for i, (label, _algo) in enumerate(metrics, start=1):
    ax = axes[i]
    draw_backdrop(ax)
    draw_edges(ax, pos, all_edges, alpha=0.08, color="#888", lw=0.4)

    values = np.array([scores[label].get(n, 0.0) for n in all_names], dtype=float)
    vmin, vmax = values.min(), values.max()
    norm = np.zeros_like(values) if vmax == vmin else (values - vmin) / (vmax - vmin)
    cmap = colormap_for(label)
    node_colors = [cmap(0.15 + 0.8 * v) for v in norm]
    node_sizes = 50 + 320 * norm
    draw_nodes(ax, pos, all_names, sizes=node_sizes, colors=node_colors, edge_lw=0.3, alpha=0.92)
    label_top(ax, all_names, values, top_k=8)
    top1 = all_names[int(np.argmax(values))]
    top1_score = float(values.max())
    style_axes(ax, label, f"top: {top1}  ({top1_score:.3f})")

# panel 6: Louvain
ax = axes[6]
groups = defaultdict(list)
for name, cid in louvain.items():
    groups[cid].append(pos[name])
for cid, pts in groups.items():
    if len(pts) >= 3:
        draw_hull(ax, pts, color=comm_color[cid], alpha=0.20, lw=1.6)

for a, b in all_edges:
    if a not in pos or b not in pos:
        continue
    x1, y1 = pos[a]
    x2, y2 = pos[b]
    same = louvain.get(a) == louvain.get(b)
    ax.plot(
        [x1, x2], [y1, y2],
        color=comm_color[louvain[a]] if same else "#bbbbbb",
        linewidth=1.0 if same else 0.3,
        alpha=0.5 if same else 0.25,
        zorder=2,
    )
node_colors = [comm_color[louvain[n]] for n in all_names]
draw_nodes(ax, pos, all_names, sizes=node_sizes, colors=node_colors, edge_lw=0.3, alpha=0.95)
# label top 3 of each non-tiny community
for cid, members in groups.items():
    if len(members) < 4:
        continue
    in_comm = [n for n in all_names if louvain[n] == cid]
    in_comm.sort(key=lambda n: -deg[n])
    for name in in_comm[:3]:
        x, y = pos[name]
        ax.annotate(
            name, (x, y), xytext=(0, 9), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#666", lw=0.4, alpha=0.92),
            zorder=10,
        )
style_axes(ax, "Louvain", f"{n_comm} communities · modularity = {modularity:.3f}")

# panel 7: clear (we use 2x4 grid, last cell is summary text)
ax = axes[7]
ax.axis("off")
summary_lines = [
    "Dataset",
    f"Stanford MemeTracker  quotes_2009-04.txt  (10.9 GB)",
    f"15.3M posts · 26.7M outbound links streamed",
    "",
    "Subset for analysis",
    f"top {n_nodes} domains by post count  ·  {n_rels} LINKS edges (weight ≥ 5)",
    "",
    "Top-1 by metric",
]
for label, _ in metrics:
    top1 = max(scores[label].items(), key=lambda x: x[1])
    summary_lines.append(f"  · {label:<13s}  {top1[0]}  ({top1[1]:.3f})")
summary_lines += [
    "",
    f"Bridges: {len(bridge_set)//2}",
    f"Louvain: {n_comm} communities, modularity = {modularity:.3f}",
]
ax.text(
    0.02, 0.98, "\n".join(summary_lines),
    transform=ax.transAxes, ha="left", va="top",
    fontsize=11, family="monospace", color="#222",
    bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="#999", lw=0.6),
)

plt.suptitle(
    "HW2 — MemeTracker Domain Graph: Bridges + 5 Centralities + Louvain",
    fontsize=20, fontweight="bold", y=0.995, color="#222",
)
fig.text(
    0.5, 0.012,
    f"All metrics computed via gds.* in Cypher.  Layout: Fruchterman-Reingold.  "
    f"Top-200 domains, {n_rels} LINKS edges.",
    ha="center", fontsize=11, color="#666", style="italic",
)
plt.tight_layout(rect=(0, 0.025, 1, 0.98))

img_path = output_path("images", "hw2_quotes_centrality.png")
os.makedirs(os.path.dirname(img_path), exist_ok=True)
plt.savefig(img_path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n==> Saved: {img_path}")

# --- 9. Export JSON snapshot for the Streamlit app ---
import json

snapshot_path = output_path("snapshots", "hw2_quotes.json")
os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)

# Reload post_count from the saved CSV (we lost it after driver.close)
post_count_map = {}
nodes_csv = os.path.join(os.path.dirname(__file__), "..", "homework", "quotes_domains.csv")
if os.path.exists(nodes_csv):
    with open(nodes_csv, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                post_count_map[parts[0]] = int(parts[1])

# Edge weights — re-read from CSV (already in homework/)
edge_weights = {}
edges_csv = os.path.join(os.path.dirname(__file__), "..", "homework", "quotes_domain_edges.csv")
if os.path.exists(edges_csv):
    with open(edges_csv, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                edge_weights[(parts[0], parts[1])] = int(parts[2])

snapshot = {
    "meta": {
        "dataset": "Stanford MemeTracker quotes_2009-04.txt",
        "raw_size_gb": 10.9,
        "n_posts_streamed": 15_312_654,
        "n_links_streamed": 26_704_274,
        "n_domains_unique": 351_260,
        "top_k": 200,
        "min_weight": 5,
        "n_nodes_plotted": len(all_names),
        "n_edges_plotted": len(all_edges),
        "n_bridges": len(bridge_set) // 2,
        "louvain_modularity": float(modularity),
        "louvain_communities": int(n_comm),
    },
    "nodes": [
        {
            "name": name,
            "post_count": post_count_map.get(name, 0),
            "community": int(louvain[name]),
            "x": float(pos[name][0]),
            "y": float(pos[name][1]),
            **{label.lower(): float(scores[label].get(name, 0.0)) for label, _ in metrics},
        }
        for name in all_names
    ],
    "edges": [
        {
            "src": a,
            "dst": b,
            "weight": int(edge_weights.get((a, b), 1)),
            "is_bridge": (a, b) in bridge_set,
        }
        for a, b in all_edges
    ],
    "cypher": {
        "load_nodes": (
            "LOAD CSV WITH HEADERS FROM 'file:///quotes_domains.csv' AS row\n"
            "CREATE (d:Domain {\n"
            "    name: row.domain,\n"
            "    post_count: toInteger(row.post_count)\n"
            "});"
        ),
        "load_edges": (
            "LOAD CSV WITH HEADERS FROM 'file:///quotes_domain_edges.csv' AS row\n"
            "MATCH (a:Domain {name: row.src}), (b:Domain {name: row.dst})\n"
            "CREATE (a)-[:LINKS {weight: toInteger(row.weight)}]->(b);"
        ),
        "project": (
            "MATCH (s:Domain)-[r:LINKS]->(t:Domain)\n"
            "RETURN gds.graph.project(\n"
            "    'quotesGraph', s, t,\n"
            "    { relationshipProperties: r { .weight } },\n"
            "    { undirectedRelationshipTypes: ['*'] }\n"
            ");"
        ),
        "bridges": (
            "CALL gds.bridges.stream('quotesGraph')\n"
            "YIELD from, to, remainingSizes\n"
            "RETURN gds.util.asNode(from).name AS f,\n"
            "       gds.util.asNode(to).name   AS t,\n"
            "       remainingSizes\n"
            "ORDER BY f, t;"
        ),
        "betweenness": (
            "CALL gds.betweenness.stream('quotesGraph')\n"
            "YIELD nodeId, score\n"
            "RETURN gds.util.asNode(nodeId).name AS name, score\n"
            "ORDER BY score DESC LIMIT 10;"
        ),
        "closeness": (
            "CALL gds.closeness.stream('quotesGraph')\n"
            "YIELD nodeId, score\n"
            "RETURN gds.util.asNode(nodeId).name AS name, score\n"
            "ORDER BY score DESC LIMIT 10;"
        ),
        "degree": (
            "CALL gds.degree.stream('quotesGraph')\n"
            "YIELD nodeId, score\n"
            "RETURN gds.util.asNode(nodeId).name AS name, score\n"
            "ORDER BY score DESC LIMIT 10;"
        ),
        "eigenvector": (
            "CALL gds.eigenvector.stream('quotesGraph')\n"
            "YIELD nodeId, score\n"
            "RETURN gds.util.asNode(nodeId).name AS name, score\n"
            "ORDER BY score DESC LIMIT 10;"
        ),
        "pagerank": (
            "CALL gds.pageRank.stream('quotesGraph')\n"
            "YIELD nodeId, score\n"
            "RETURN gds.util.asNode(nodeId).name AS name, score\n"
            "ORDER BY score DESC LIMIT 10;"
        ),
        "louvain": (
            "// stats\n"
            "CALL gds.louvain.stats('quotesGraph')\n"
            "YIELD modularity, communityCount, communityDistribution;\n\n"
            "// per-node assignment\n"
            "CALL gds.louvain.stream('quotesGraph')\n"
            "YIELD nodeId, communityId\n"
            "RETURN communityId, collect(gds.util.asNode(nodeId).name) AS members\n"
            "ORDER BY size(members) DESC;"
        ),
    },
}

with open(snapshot_path, "w", encoding="utf-8") as f:
    json.dump(snapshot, f, indent=2, ensure_ascii=False)
print(f"==> Saved snapshot: {snapshot_path}  ({os.path.getsize(snapshot_path) / 1024:.1f} KB)")

print("Done.")
