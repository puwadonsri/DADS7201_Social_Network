"""Compute 5 centrality measures on the Aura graph using NetworkX,
print a ranked table, and write values back as node properties.

Centralities:
- Degree (in + out + total)
- Closeness
- Betweenness
- Eigenvector
- Katz
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
from neo4j import GraphDatabase

HERE = Path(__file__).parent
CREDS_FILE = next(HERE.glob("Neo4j-*.txt"))


def load_creds(path: Path) -> dict[str, str]:
    creds: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            creds[k.strip()] = v.strip()
    return creds


def fetch_graph(session) -> nx.DiGraph:
    G = nx.DiGraph()
    nodes = session.run(
        "MATCH (n:Entity) RETURN n.id AS id, n.name AS name, labels(n) AS labels"
    )
    for r in nodes:
        extra_label = next((l for l in r["labels"] if l != "Entity"), "Entity")
        G.add_node(r["id"], name=r["name"], type=extra_label)

    edges = session.run(
        "MATCH (a:Entity)-[r]->(b:Entity) "
        "RETURN a.id AS src, b.id AS dst, type(r) AS type"
    )
    for r in edges:
        G.add_edge(r["src"], r["dst"], type=r["type"])
    return G


def compute_centralities(G: nx.DiGraph) -> pd.DataFrame:
    n = G.number_of_nodes()
    in_deg = {k: v / (n - 1) for k, v in G.in_degree()}
    out_deg = {k: v / (n - 1) for k, v in G.out_degree()}
    total_deg = {k: in_deg[k] + out_deg[k] for k in G.nodes()}

    closeness = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G)

    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        eigenvector = nx.eigenvector_centrality_numpy(G)

    try:
        katz = nx.katz_centrality(G, alpha=0.1, beta=1.0, max_iter=2000, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        katz = nx.katz_centrality_numpy(G, alpha=0.1, beta=1.0)

    df = pd.DataFrame(
        {
            "id": list(G.nodes()),
            "name": [G.nodes[n]["name"] for n in G.nodes()],
            "type": [G.nodes[n]["type"] for n in G.nodes()],
            "degree_in": [in_deg[n] for n in G.nodes()],
            "degree_out": [out_deg[n] for n in G.nodes()],
            "degree": [total_deg[n] for n in G.nodes()],
            "closeness": [closeness[n] for n in G.nodes()],
            "betweenness": [betweenness[n] for n in G.nodes()],
            "eigenvector": [eigenvector[n] for n in G.nodes()],
            "katz": [katz[n] for n in G.nodes()],
        }
    )
    return df


def write_back(session, df: pd.DataFrame) -> None:
    rows = df[
        [
            "id",
            "degree_in",
            "degree_out",
            "degree",
            "closeness",
            "betweenness",
            "eigenvector",
            "katz",
        ]
    ].to_dict(orient="records")
    session.run(
        "UNWIND $rows AS row "
        "MATCH (n:Entity {id: row.id}) "
        "SET n.degree_in = row.degree_in, "
        "    n.degree_out = row.degree_out, "
        "    n.degree = row.degree, "
        "    n.closeness = row.closeness, "
        "    n.betweenness = row.betweenness, "
        "    n.eigenvector = row.eigenvector, "
        "    n.katz = row.katz",
        rows=rows,
    )


def main() -> None:
    creds = load_creds(CREDS_FILE)
    driver = GraphDatabase.driver(
        creds["NEO4J_URI"], auth=(creds["NEO4J_USERNAME"], creds["NEO4J_PASSWORD"])
    )
    db = creds.get("NEO4J_DATABASE", "neo4j")

    with driver.session(database=db) as session:
        G = fetch_graph(session)
        print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

        df = compute_centralities(G)
        write_back(session, df)

    driver.close()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("=== Top 10 by Degree ===")
    print(df.sort_values("degree", ascending=False).head(10).to_string(index=False))
    print("\n=== Top 10 by Closeness ===")
    print(df.sort_values("closeness", ascending=False).head(10).to_string(index=False))
    print("\n=== Top 10 by Betweenness ===")
    print(df.sort_values("betweenness", ascending=False).head(10).to_string(index=False))
    print("\n=== Top 10 by Eigenvector ===")
    print(df.sort_values("eigenvector", ascending=False).head(10).to_string(index=False))
    print("\n=== Top 10 by Katz ===")
    print(df.sort_values("katz", ascending=False).head(10).to_string(index=False))

    out_csv = HERE / "centrality.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
