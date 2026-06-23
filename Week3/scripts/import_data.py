"""
Import edges_rows.csv into Neo4j.

Pipeline:
  1. ดูแลให้ neo4j-import/edges_rows.csv มีอยู่ (copy จาก data/ ถ้ายังไม่มี)
  2. LOAD CSV เข้า Neo4j ผ่าน Cypher (ถ้ายังไม่ได้ import)
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neo4j_utils import get_driver

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "edges_rows.csv")
DST_DIR = os.path.join(ROOT, "neo4j-import")
DST = os.path.join(DST_DIR, "edges_rows.csv")

os.makedirs(DST_DIR, exist_ok=True)
if not os.path.exists(DST) or os.path.getmtime(SRC) > os.path.getmtime(DST):
    shutil.copy2(SRC, DST)
    print(f"Copied {SRC} -> {DST}")

driver = get_driver()
with driver.session(database="neo4j") as s:
    r = list(s.run("MATCH (n:Person) RETURN count(n) AS c"))
    print(f"Existing Person nodes: {r[0]['c']}")
    if r[0]["c"] == 0:
        s.run("""
            LOAD CSV WITH HEADERS FROM 'file:///edges_rows.csv' AS row
            MERGE (a:Person {name: row.person_a})
            MERGE (b:Person {name: row.person_b})
            CREATE (a)-[:KNOWS {id: toInteger(row.id)}]->(b)
        """)
        r = list(s.run("MATCH (n:Person) RETURN count(n) AS c"))
        rels = list(s.run("MATCH ()-[r:KNOWS]->() RETURN count(r) AS c"))
        print(f"Imported Person nodes: {r[0]['c']}, KNOWS rels: {rels[0]['c']}")
driver.close()
