from neo4j import GraphDatabase
from pyvis.network import Network

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

# --- 2. Bridges from GDS ---
print("==> Running Bridges...")
bridge_rows = run_query(f"""
    CALL gds.bridges.stream('{GRAPH_NAME}')
    YIELD from, to, remainingSizes
    RETURN gds.util.asNode(from).name AS fromName,
           gds.util.asNode(to).name AS toName
""")
bridge_set = set()
for r in bridge_rows:
    bridge_set.add((r["fromName"], r["toName"]))
    bridge_set.add((r["toName"], r["fromName"]))
print(f"    Found {len(bridge_rows)} bridge(s)")

# --- 3. Node centralities from GDS ---
centrality_algos = [
    ("Betweenness", "gds.betweenness.stream"),
    ("Closeness", "gds.closeness.stream"),
    ("Degree", "gds.degree.stream"),
    ("Eigenvector", "gds.eigenvector.stream"),
    ("PageRank", "gds.pageRank.stream"),
]
for label, algo in centrality_algos:
    print(f"==> Running {label}...")
    rows = run_query(f"""
        CALL {algo}('{GRAPH_NAME}')
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS n, score
        SET n.{label.lower()} = score
        RETURN n.name AS name, score
        ORDER BY score DESC
    """)
    print(f"    Top: {rows[0]['name']} = {rows[0]['score']:.4f}")

# --- 4. Get all nodes and edges ---
nodes_rows = run_query("""
    MATCH (n:Person)
    RETURN n.name AS name,
           n.betweenness AS betweenness,
           n.closeness AS closeness,
           n.degree AS degree,
           n.eigenvector AS eigenvector,
           n.pagerank AS pagerank
    ORDER BY n.name
""")
all_nodes = [r["name"] for r in nodes_rows]

edges_rows = run_query("""
    MATCH (a:Person)-[:KNOWS]->(b:Person)
    RETURN a.name AS a, b.name AS b
""")

driver.close()

# --- 5. Build pyvis HTML ---
net = Network(height="800px", width="100%", directed=False, notebook=False)
net.set_options("""
{
  "nodes": {
    "font": {"size": 14, "face": "Tahoma"},
    "borderWidth": 1,
    "borderWidthSelected": 2
  },
  "edges": {
    "color": {"color": "#cccccc", "highlight": "#ff0000"},
    "width": 1,
    "smooth": {"enabled": true, "type": "continuous"}
  },
  "physics": {
    "enabled": true,
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {"gravitationalConstant": -40, "springLength": 150}
  },
  "interaction": { "hover": true, "tooltipDelay": 0 }
}
""")

for r in nodes_rows:
    name = r["name"]
    btw = r["betweenness"] or 0
    title = f"<b>{name}</b><br>"
    for algo_name in ["betweenness", "closeness", "degree", "eigenvector", "pagerank"]:
        val = r[algo_name] or 0
        title += f"{algo_name.capitalize()}: {val:.4f}<br>"
    size = 15 + (btw / 40) * 40
    net.add_node(name, label=name, title=title, size=size, group=int(btw > 0))

for a, b in edges_rows:
    color = "#ff4444" if (a, b) in bridge_set else "#aaaaaa"
    width = 4 if (a, b) in bridge_set else 1
    dashes = False if (a, b) in bridge_set else True
    title = "BRIDGE EDGE" if (a, b) in bridge_set else ""
    net.add_edge(a, b, color=color, width=width, dashes=dashes, title=title)

import os
out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "html")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "centrality_pyvis.html")
net.save_graph(out_path)
print(f"\n==> Saved: {out_path} (open in browser)")
print("Done.")
