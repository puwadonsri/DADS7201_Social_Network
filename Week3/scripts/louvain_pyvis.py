"""
Louvain vs K-Means interactive PyVis HTML (edges_rows.csv).

Generates a single HTML file with two side-by-side networks coloured by
Louvain (left) and K-Means k=4 (right). Tooltip shows the community id from
each method so you can hand-trace the differences.
"""
import os
import sys
from collections import defaultdict

from pyvis.network import Network

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neo4j_utils import get_driver, output_path

GRAPH_NAME = "louvainPyvisGraph"
driver = get_driver()


def run(q, params=None):
    with driver.session(database="neo4j") as s:
        return list(s.run(q, params or {}))


print("==> Projecting...")
run("CALL gds.graph.drop($n, false)", {"n": GRAPH_NAME})
run(
    """
    MATCH (s:Person)-[r:KNOWS]->(t:Person)
    RETURN gds.graph.project($n, s, t, {}, { undirectedRelationshipTypes: ['*'] })
    """,
    {"n": GRAPH_NAME},
)

print("==> Louvain...")
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

print("==> K-Means (k=4)...")
run(
    f"""
    CALL gds.fastRP.mutate('{GRAPH_NAME}', {{
        embeddingDimension: 128, randomSeed: 42, mutateProperty: 'embedding'
    }})
    """
)
kmeans = {
    r["name"]: r["communityId"]
    for r in run(
        f"""
        CALL gds.kmeans.stream('{GRAPH_NAME}', {{
            nodeProperty: 'embedding', k: 4, randomSeed: 42, concurrency: 1
        }})
        YIELD nodeId, communityId
        RETURN gds.util.asNode(nodeId).name AS name, communityId
        """
    )
}

edges = [
    (r["a"], r["b"])
    for r in run("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name AS a, b.name AS b")
]
driver.close()


PALETTE = [
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
]


def color_map(label_map):
    cids = sorted(set(label_map.values()))
    return {cid: PALETTE[i % len(PALETTE)] for i, cid in enumerate(cids)}


def build_html(label_map, edges, title, color_for, out_path):
    net = Network(height="700px", width="100%", directed=False, notebook=False)
    net.set_options(
        """
{
  "nodes": {"font": {"size": 14, "face": "Tahoma"}, "borderWidth": 1},
  "edges": {"color": {"color": "#cccccc"}, "width": 1, "smooth": {"type": "continuous"}},
  "physics": {"solver": "forceAtlas2Based",
              "forceAtlas2Based": {"gravitationalConstant": -40, "springLength": 150}},
  "interaction": {"hover": true, "tooltipDelay": 0}
}
"""
    )
    for name, cid in label_map.items():
        tooltip = f"<b>{name}</b><br>Louvain: {louvain.get(name)}<br>K-Means: {kmeans.get(name)}"
        net.add_node(name, label=name, title=tooltip, color=color_for[cid], size=24)
    for a, b in edges:
        if label_map.get(a) == label_map.get(b):
            net.add_edge(a, b, color=color_for[label_map[a]], width=3)
        else:
            net.add_edge(a, b, color="#cccccc", width=1, dashes=True)
    net.save_graph(out_path)
    print(f"    saved: {out_path}")


out_dir = output_path("html")
os.makedirs(out_dir, exist_ok=True)
louvain_html = os.path.join(out_dir, "louvain_network.html")
kmeans_html = os.path.join(out_dir, "kmeans_network.html")

build_html(louvain, edges, "Louvain", color_map(louvain), louvain_html)
build_html(kmeans, edges, "K-Means k=4", color_map(kmeans), kmeans_html)

# combined side-by-side wrapper
combined = output_path("html", "community_louvain_vs_kmeans.html")
with open(combined, "w", encoding="utf-8") as f:
    f.write(
        f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Louvain vs K-Means</title>
<style>
  body {{ font-family: Tahoma, sans-serif; margin: 0; padding: 12px; background: #f7f7f7; }}
  h1 {{ text-align: center; margin: 6px 0 14px 0; }}
  .row {{ display: flex; gap: 12px; }}
  .col {{ flex: 1; background: white; border-radius: 8px; padding: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .col h2 {{ text-align: center; margin: 4px 0 8px 0; font-size: 18px; }}
  iframe {{ width: 100%; height: 720px; border: 1px solid #ddd; border-radius: 6px; }}
  .legend {{ font-size: 12px; color: #555; text-align: center; padding-top: 4px; }}
</style></head>
<body>
<h1>Community Detection — Louvain vs K-Means (edges_rows.csv)</h1>
<div class="row">
  <div class="col"><h2>Louvain</h2><iframe src="louvain_network.html"></iframe>
    <div class="legend">modularity-optimized • {len(set(louvain.values()))} communities</div></div>
  <div class="col"><h2>K-Means (k=4)</h2><iframe src="kmeans_network.html"></iframe>
    <div class="legend">FastRP 128-dim → K-Means • k=4</div></div>
</div>
</body></html>
"""
    )
print(f"==> Combined view: {combined}")
print("Done.")
