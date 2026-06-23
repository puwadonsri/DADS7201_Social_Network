from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "s3cureP@ssword"))
with driver.session(database="neo4j") as s:
    # Drop all leftover graphs
    for g in ["bridgesGraph", "kmeansGraph", "vizGraph", "testGraph", "myGraph"]:
        s.run(f"CALL gds.graph.drop('{g}', false)")
    print("Cleaned up old graphs")

    s.run("""
        MATCH (source:Person)-[r:KNOWS]->(target:Person)
        RETURN gds.graph.project('bridgesGraph', source, target, {}, { undirectedRelationshipTypes: ['*'] })
    """)
    print("Projected graph")

    rows = list(s.run("""
        CALL gds.bridges.stream('bridgesGraph')
        YIELD from, to, remainingSizes
        RETURN gds.util.asNode(from).name AS f, gds.util.asNode(to).name AS t, remainingSizes
    """))
    print(f"Bridges found: {len(rows)}")
    for r in rows:
        print(f"  {r['f']} -- {r['t']}  => components: {r['remainingSizes']}")

driver.close()
