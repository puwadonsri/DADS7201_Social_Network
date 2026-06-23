from neo4j import GraphDatabase
from pyvis.network import Network

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "s3cureP@ssword"
GRAPH_NAME = "kmeansGraph"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def run_query(query, params=None):
    with driver.session(database="neo4j") as session:
        return list(session.run(query, params or {}))

# --- 1. Project + FastRP + K-Means write ---
print("==> Setting up graph...")
run_query("CALL gds.graph.drop($name, false)", {"name": GRAPH_NAME})
run_query("""
    MATCH (source:Person)-[r:KNOWS]->(target:Person)
    RETURN gds.graph.project($name, source, target, {}, { undirectedRelationshipTypes: ['*'] })
""", {"name": GRAPH_NAME})

print("==> Generating FastRP embeddings...")
run_query(f"""
    CALL gds.fastRP.mutate('{GRAPH_NAME}', {{
        embeddingDimension: 128,
        randomSeed: 42,
        mutateProperty: 'embedding'
    }})
""")

# Evaluate k and pick best
print("==> Finding optimal k...")
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
               silhouette
        ORDER BY communityId, name
    """)
    sil_scores = [r["silhouette"] for r in rows if r["silhouette"] is not None]
    avg_sil = sum(sil_scores) / len(sil_scores) if sil_scores else 0
    k_results.append((k, avg_sil, rows))

best_k, best_sil, best_rows = max(k_results, key=lambda x: x[1])
print(f"  Optimal k = {best_k} (silhouette={best_sil:.4f})")

# Write to Neo4j
print(f"==> Writing k={best_k} to Neo4j...")
run_query(f"""
    CALL gds.kmeans.write('{GRAPH_NAME}', {{
        nodeProperty: 'embedding',
        k: {best_k},
        randomSeed: 42,
        concurrency: 1,
        writeProperty: 'kmeans'
    }})
""")

# --- 2. Get data from Neo4j ---
nodes_rows = run_query("""
    MATCH (n:Person)
    RETURN n.name AS name, n.kmeans AS community
    ORDER BY n.community, n.name
""")

edges_rows = run_query("""
    MATCH (a:Person)-[:KNOWS]->(b:Person)
    RETURN a.name AS a, b.name AS b
""")

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

driver.close()

# Build color map
community_colors = {
    0: "#66c2a5", 1: "#fc8d62", 2: "#8da0cb",
    3: "#e78ac3", 4: "#a6d854", 5: "#ffd92f",
    6: "#e5c494", 7: "#b3b3b3"
}
node_community = {r["name"]: r["community"] for r in nodes_rows}

# --- 3. PyVis HTML ---
net = Network(height="800px", width="100%", directed=False, notebook=False)
net.set_options("""
{
  "nodes": {
    "font": {"size": 14, "face": "Tahoma"},
    "borderWidth": 1,
    "borderWidthSelected": 2
  },
  "edges": {
    "color": {"color": "#aaaaaa", "highlight": "#333333"},
    "width": 1,
    "smooth": {"enabled": true, "type": "continuous"}
  },
  "physics": {
    "enabled": true,
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {"gravitationalConstant": -40, "springLength": 150}
  },
  "interaction": {"hover": true, "tooltipDelay": 0}
}
""")

for r in nodes_rows:
    name = r["name"]
    cid = r["community"]
    color = community_colors.get(cid, "#999999")
    title = f"<b>{name}</b><br>Community: {cid}"
    net.add_node(name, label=name, title=title, color=color, size=20)

for a, b in edges_rows:
    # Edge color = same if same community, gray if different
    ca = node_community.get(a)
    cb = node_community.get(b)
    c = community_colors.get(ca, "#aaaaaa") if ca == cb else "#cccccc"
    w = 2 if ca == cb else 1
    dash = False if ca == cb else True
    net.add_edge(a, b, color=c, width=w, dashes=dash)

import os
out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "html")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "kmeans_communities.html")
net.save_graph(out_path)
print(f"==> Saved: {out_path} (open in browser)")
print("Done.")
