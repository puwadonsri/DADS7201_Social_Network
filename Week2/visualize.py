"""Build an interactive HTML visualization of the Aura graph using PyVis.

- Node size = betweenness centrality (scaled)
- Node color = type (Country/Person/Organization)
- Edge label = relationship type
- Hover tooltip = name, role, all centrality scores
"""

from __future__ import annotations

from pathlib import Path

from neo4j import GraphDatabase
from pyvis.network import Network

HERE = Path(__file__).parent
CREDS_FILE = next(HERE.glob("Neo4j-*.txt"))
OUT_HTML = HERE / "graph.html"

TYPE_COLORS = {
    "Country": "#4C9AFF",
    "Person": "#FF8B00",
    "Organization": "#36B37E",
}


def load_creds(path: Path) -> dict[str, str]:
    creds: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            creds[k.strip()] = v.strip()
    return creds


def main() -> None:
    creds = load_creds(CREDS_FILE)
    driver = GraphDatabase.driver(
        creds["NEO4J_URI"], auth=(creds["NEO4J_USERNAME"], creds["NEO4J_PASSWORD"])
    )
    db = creds.get("NEO4J_DATABASE", "neo4j")

    with driver.session(database=db) as session:
        nodes = list(
            session.run(
                "MATCH (n:Entity) "
                "RETURN n.id AS id, n.name AS name, labels(n) AS labels, "
                "n.role AS role, n.country AS country, "
                "n.degree AS degree, n.closeness AS closeness, "
                "n.betweenness AS betweenness, n.eigenvector AS eigenvector, "
                "n.katz AS katz"
            )
        )
        edges = list(
            session.run(
                "MATCH (a:Entity)-[r]->(b:Entity) "
                "RETURN a.id AS src, b.id AS dst, type(r) AS type, "
                "r.description AS description"
            )
        )
    driver.close()

    max_btw = max((r["betweenness"] or 0) for r in nodes) or 1.0

    net = Network(
        height="780px",
        width="100%",
        bgcolor="#1e1e1e",
        font_color="#ffffff",
        directed=True,
        cdn_resources="in_line",
    )

    for r in nodes:
        node_type = next((l for l in r["labels"] if l != "Entity"), "Entity")
        size = 15 + 45 * ((r["betweenness"] or 0) / max_btw)
        title = (
            f"<b>{r['name']}</b><br>"
            f"type: {node_type}<br>"
            f"role: {r['role']}<br>"
            f"country: {r['country']}<br>"
            f"<br><b>Centrality</b><br>"
            f"degree: {r['degree']:.4f}<br>"
            f"closeness: {r['closeness']:.4f}<br>"
            f"betweenness: {r['betweenness']:.4f}<br>"
            f"eigenvector: {r['eigenvector']:.4f}<br>"
            f"katz: {r['katz']:.4f}"
        )
        net.add_node(
            r["id"],
            label=r["name"],
            title=title,
            color=TYPE_COLORS.get(node_type, "#999999"),
            size=size,
        )

    for r in edges:
        net.add_edge(
            r["src"],
            r["dst"],
            label=r["type"],
            title=r["description"],
            color="#888888",
            arrows="to",
        )

    net.set_options(
        """
        {
          "nodes": {"font": {"size": 14, "face": "Sarabun, Tahoma, Arial"}},
          "edges": {
            "font": {"size": 10, "align": "middle", "color": "#cccccc"},
            "smooth": {"type": "dynamic"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}}
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 180,
              "springConstant": 0.04
            },
            "minVelocity": 0.75
          },
          "interaction": {"hover": true, "tooltipDelay": 100}
        }
        """
    )

    html = net.generate_html(notebook=False)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Saved: {OUT_HTML}")
    print(f"  nodes: {len(nodes)}")
    print(f"  edges: {len(edges)}")


if __name__ == "__main__":
    main()
