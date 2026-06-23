from neo4j import GraphDatabase
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin, pi

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "s3cureP@ssword"
GRAPH_NAME = "bridgesGraph"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def run_query(query, params=None):
    with driver.session(database="neo4j") as session:
        return list(session.run(query, params or {}))

# --- 1. Project ---
print("==> Projecting graph...")
run_query("CALL gds.graph.drop($name, false)", {"name": GRAPH_NAME})
run_query("""
    MATCH (source:Person)-[r:KNOWS]->(target:Person)
    RETURN gds.graph.project($name, source, target, {}, { undirectedRelationshipTypes: ['*'] })
""", {"name": GRAPH_NAME})

# --- 2. Bridges ---
print("==> Running Bridges...")
bridge_rows = run_query(f"""
    CALL gds.bridges.stream('{GRAPH_NAME}')
    YIELD from, to, remainingSizes
    RETURN gds.util.asNode(from).name AS fromName,
           gds.util.asNode(to).name AS toName,
           remainingSizes
    ORDER BY fromName ASC, toName ASC
""")
bridge_set = set()
for r in bridge_rows:
    bridge_set.add((r["fromName"], r["toName"]))
    print(f"    Bridge: {r['fromName']} -- {r['toName']}  |  {r['remainingSizes']}")

# --- 3. Node centralities from GDS ---
centrality_algos = [
    ("Betweenness", "gds.betweenness.stream"),
    ("Closeness", "gds.closeness.stream"),
    ("Degree", "gds.degree.stream"),
    ("Eigenvector", "gds.eigenvector.stream"),
    ("PageRank", "gds.pageRank.stream"),
]

centrality_data = {}
for label, algo in centrality_algos:
    print(f"==> Running {label}...")
    rows = run_query(f"""
        CALL {algo}('{GRAPH_NAME}')
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).name AS name, score
        ORDER BY score DESC
    """)
    centrality_data[label] = {r["name"]: r["score"] for r in rows}
    print(f"    Top 5 — {label}:")
    for r in rows[:5]:
        print(f"      {r['name']}: {r['score']:.4f}")

# --- 4. Get nodes & edges from Neo4j ---
nodes_rows = run_query("""
    MATCH (n:Person)
    RETURN n.name AS name
    ORDER BY n.name
""")
all_nodes = [r["name"] for r in nodes_rows]

edges_rows = run_query("""
    MATCH (a:Person)-[:KNOWS]->(b:Person)
    RETURN a.name AS a, b.name AS b
""")
all_edges = [(r["a"], r["b"]) for r in edges_rows]

driver.close()

# --- 5. Circular layout (no NetworkX) ---
n = len(all_nodes)
pos = {}
for i, name in enumerate(all_nodes):
    angle = 2 * pi * i / n - pi / 2
    pos[name] = (cos(angle), sin(angle))

# --- 6. Edge colors for Bridges ---
edge_colors_bridge = []
for u, v in all_edges:
    if (u, v) in bridge_set or (v, u) in bridge_set:
        edge_colors_bridge.append("red")
    else:
        edge_colors_bridge.append("lightgray")

# --- 7. Plot each metric ---
all_modes = ["Bridges"] + [c[0] for c in centrality_algos]
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

for idx, mode in enumerate(all_modes):
    ax = axes[idx]

    if mode == "Bridges":
        for (u, v), ec in zip(all_edges, edge_colors_bridge):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color=ec, linewidth=2 if ec == "red" else 0.5, alpha=0.7)
        for name, (x, y) in pos.items():
            ax.scatter(x, y, c="skyblue", s=600, edgecolors="black", linewidths=0.5, zorder=5)
            ax.text(x, y, name, ha="center", va="center", fontsize=7, zorder=6)
        ax.set_title("Bridges (red edges)", fontsize=14, fontweight="bold")
    else:
        scores = centrality_data[mode]
        values = np.array([scores.get(n, 0) for n in all_nodes])
        vmin, vmax = values.min(), values.max()
        norm_values = np.zeros_like(values) if vmin == vmax else (values - vmin) / (vmax - vmin)

        for (u, v) in all_edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="lightgray", linewidth=0.5, alpha=0.4)

        for i, name in enumerate(all_nodes):
            x, y = pos[name]
            size = 400 + norm_values[i] * 3000
            color = plt.cm.OrRd(norm_values[i])
            ax.scatter(x, y, c=[color], s=size, edgecolors="black", linewidths=0.5, zorder=5)
            ax.text(x, y, name, ha="center", va="center", fontsize=7, zorder=6)

        ax.set_title(f"{mode} (darker/larger = higher)", fontsize=14, fontweight="bold")

    ax.axis("off")

plt.suptitle("Neo4j GDS Centrality Analysis", fontsize=18, fontweight="bold")
plt.tight_layout()
import os
out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "images")
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "centrality_plot_gds.png"), dpi=200, bbox_inches="tight")
print(f"\n==> Saved: {os.path.join(out_dir, 'centrality_plot_gds.png')}")
print("Done.")
