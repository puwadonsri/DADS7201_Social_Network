from neo4j import GraphDatabase
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin, pi

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "s3cureP@ssword"
GRAPH_NAME = "kmeansGraph"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def run_query(query, params=None):
    with driver.session(database="neo4j") as session:
        return list(session.run(query, params or {}))

# --- 1. Drop old graph, re-project ---
print("==> Projecting graph...")
run_query("CALL gds.graph.drop($name, false)", {"name": GRAPH_NAME})
run_query("""
    MATCH (source:Person)-[r:KNOWS]->(target:Person)
    RETURN gds.graph.project($name, source, target, {}, { undirectedRelationshipTypes: ['*'] })
""", {"name": GRAPH_NAME})

# --- 2. Generate FastRP node embeddings (128-dim) ---
print("==> Generating FastRP embeddings...")
run_query(f"""
    CALL gds.fastRP.mutate('{GRAPH_NAME}', {{
        embeddingDimension: 128,
        randomSeed: 42,
        mutateProperty: 'embedding'
    }})
""")

# --- 3. Try multiple k values to find optimal ---
print("==> Evaluating K-Means for k=2..8...")
k_results = []
for k in range(2, 9):
    rows = run_query(f"""
        CALL gds.kmeans.stream('{GRAPH_NAME}', {{
            nodeProperty: 'embedding',
            k: {k},
            randomSeed: 42,
            computeSilhouette: true,
            concurrency: 1
        }})
        YIELD nodeId, communityId, distanceFromCentroid, silhouette
        RETURN gds.util.asNode(nodeId).name AS name,
               communityId,
               distanceFromCentroid,
               silhouette
        ORDER BY communityId, name
    """)
    communities = {}
    for r in rows:
        cid = r["communityId"]
        communities.setdefault(cid, []).append(r["name"])
    sil_scores = [r["silhouette"] for r in rows if r["silhouette"] is not None]
    avg_sil = np.mean(sil_scores) if sil_scores else 0

    sizes = [len(members) for members in communities.values()]
    print(f"  k={k}: avg_silhouette={avg_sil:.4f}, cluster_sizes={sizes}")
    k_results.append((k, avg_sil, rows, communities))

# optimal k by highest silhouette
best_k, best_sil, best_rows, best_communities = max(k_results, key=lambda x: x[1])
print(f"\n==> Optimal k = {best_k} (silhouette={best_sil:.4f})")

# --- 4. Write best clustering to Neo4j ---
print(f"==> Writing k={best_k} communities to Neo4j...")
run_query(f"""
    CALL gds.kmeans.write('{GRAPH_NAME}', {{
        nodeProperty: 'embedding',
        k: {best_k},
        randomSeed: 42,
        concurrency: 1,
        writeProperty: 'kmeans'
    }})
""")

# --- 5. Print clusters ---
print(f"\n==> Communities (k={best_k}):")
for cid in sorted(best_communities.keys()):
    members = best_communities[cid]
    print(f"  Community {cid} ({len(members)} members): {', '.join(members)}")

# --- 6. Generate 2D embedding and write to Neo4j ---
print("\n==> Generating 2D embedding for visualization...")
run_query("CALL gds.graph.drop('vizGraph', false)")
run_query("""
    MATCH (source:Person)-[r:KNOWS]->(target:Person)
    RETURN gds.graph.project('vizGraph', source, target, {}, { undirectedRelationshipTypes: ['*'] })
""")
run_query("""
    CALL gds.fastRP.write('vizGraph', {
        embeddingDimension: 2,
        randomSeed: 42,
        writeProperty: 'embedding_2d'
    })
""")
viz_rows = run_query(f"""
    MATCH (n:Person)
    RETURN n.name AS name,
           n.embedding_2d[0] AS x,
           n.embedding_2d[1] AS y,
           n.kmeans AS community
""")

# --- 7. Scatter plot ---
colors = plt.cm.Set2(np.linspace(0, 1, best_k))
fig, ax = plt.subplots(figsize=(14, 10))

for r in viz_rows:
    cid = r["community"]
    ax.scatter(r["x"], r["y"], c=[colors[cid]], s=500, edgecolors="black", linewidths=0.5, zorder=5)
    ax.text(r["x"], r["y"], r["name"], fontsize=8, ha="center", va="center", zorder=6)

ax.set_title(f"K-Means Clustering (k={best_k}, silhouette={best_sil:.4f})", fontsize=16, fontweight="bold")
ax.set_xlabel("FastRP dim-1")
ax.set_ylabel("FastRP dim-2")
import os
out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "images")
os.makedirs(out_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "kmeans_clusters.png"), dpi=200, bbox_inches="tight")
print(f"\n==> Saved: {os.path.join(out_dir, 'kmeans_clusters.png')}")

# --- 8. Silhouette elbow plot ---
fig2, ax2 = plt.subplots(figsize=(10, 6))
ks = [x[0] for x in k_results]
sils = [x[1] for x in k_results]
ax2.plot(ks, sils, "bo-", linewidth=2, markersize=8)
ax2.axvline(best_k, color="red", linestyle="--", alpha=0.5, label=f"optimal k={best_k}")
ax2.set_xlabel("k (number of clusters)")
ax2.set_ylabel("Average Silhouette Score")
ax2.set_title("K-Means: Silhouette Score vs k")
ax2.grid(True, alpha=0.3)
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "kmeans_elbow.png"), dpi=200, bbox_inches="tight")
print(f"==> Saved: {os.path.join(out_dir, 'kmeans_elbow.png')}")

driver.close()
print("Done.")
