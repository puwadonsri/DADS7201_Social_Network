"""Push nodes_aura.csv and relationships_aura.csv into Neo4j Aura.

Reads credentials from the Aura .txt file that lives beside this script,
then uses UNWIND batches via the official driver — no LOAD CSV needed
(Aura cannot read local files).
"""

from __future__ import annotations

import csv
from pathlib import Path

from neo4j import GraphDatabase

HERE = Path(__file__).parent
CREDS_FILE = next(HERE.glob("Neo4j-*.txt"))
NODES_CSV = HERE / "nodes_aura.csv"
RELS_CSV = HERE / "relationships_aura.csv"


def load_creds(path: Path) -> dict[str, str]:
    creds: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            creds[k.strip()] = v.strip()
    return creds


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    creds = load_creds(CREDS_FILE)
    uri = creds["NEO4J_URI"]
    user = creds["NEO4J_USERNAME"]
    pwd = creds["NEO4J_PASSWORD"]
    db = creds.get("NEO4J_DATABASE", "neo4j")

    nodes = read_csv(NODES_CSV)
    rels = read_csv(RELS_CSV)

    nodes_by_label: dict[str, list[dict]] = {}
    for r in nodes:
        label = r["label"]
        nodes_by_label.setdefault(label, []).append(
            {
                "id": r["id"],
                "name": r["name"],
                "role": r["role"],
                "country": r["country"],
            }
        )

    rels_by_type: dict[str, list[dict]] = {}
    for r in rels:
        rels_by_type.setdefault(r["type"], []).append(
            {
                "start": r["start_id"],
                "end": r["end_id"],
                "description": r["description"],
            }
        )

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session(database=db) as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run(
            "CREATE CONSTRAINT entity_id IF NOT EXISTS "
            "FOR (n:Entity) REQUIRE n.id IS UNIQUE"
        )

        for label, rows in nodes_by_label.items():
            cypher = (
                f"UNWIND $rows AS row "
                f"CREATE (n:Entity:{label}) "
                f"SET n.id = row.id, n.name = row.name, "
                f"n.role = row.role, n.country = row.country"
            )
            session.run(cypher, rows=rows)
            print(f"nodes  :{label:<13} {len(rows):>3}")

        for rtype, rows in rels_by_type.items():
            cypher = (
                "UNWIND $rows AS row "
                "MATCH (a:Entity {id: row.start}), (b:Entity {id: row.end}) "
                f"CREATE (a)-[r:{rtype}]->(b) "
                "SET r.description = row.description"
            )
            session.run(cypher, rows=rows)
            print(f"rels   :{rtype:<13} {len(rows):>3}")

        node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"\ntotal nodes: {node_count}")
        print(f"total rels : {rel_count}")

    driver.close()


if __name__ == "__main__":
    main()
