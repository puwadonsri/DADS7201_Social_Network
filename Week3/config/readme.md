# Lab — Week 3 (Neo4j + Graph Data Science)

> โฟลเดอร์โครงสร้าง:
> - `Week3/data/` — ข้อมูลต้นทาง (`edges_rows.csv`)
> - `Week3/neo4j-import/` — โฟลเดอร์ที่ container จะอ่านด้วย `LOAD CSV FROM 'file:///...'`
> - `Week3/scripts/` — Python scripts (centrality, K-Means, Louvain, …)
> - `Week3/outputs/` — รูป/HTML/รายงานผลลัพธ์

## 1) Run Neo4j + GDS

**ทางเลือก A — Docker run (Windows CMD)**
```bat
docker run -d ^
  --name neo4j-gds ^
  -p 7474:7474 -p 7687:7687 ^
  -v neo4j_data:/data ^
  -v "%cd%\Week3\neo4j-import:/import" ^
  -e NEO4J_AUTH=neo4j/s3cureP@ssword ^
  -e NEO4J_PLUGINS="[\"graph-data-science\"]" ^
  -e NEO4J_dbms_security_procedures_unrestricted=gds.* ^
  --memory=4g ^
  neo4j:latest
```

**ทางเลือก B — Docker Compose** (จากโฟลเดอร์ `Week3/config/`)
```powershell
docker compose -f Week3/config/docker-compose.yml up -d
```
> compose ไฟล์จะ mount `Week3/neo4j-import` ให้ container เห็นเป็น `/import` ตรงกับทางเลือก A

## 2) เปิด Browser UI
```
http://localhost:7474/
```
Login: `neo4j` / `s3cureP@ssword`

## 3) วางไฟล์ CSV
คัดลอก `Week3/data/edges_rows.csv` → `Week3/neo4j-import/edges_rows.csv`
(หรือใช้คำสั่ง: `copy Week3\data\edges_rows.csv Week3\neo4j-import\`)

## 4) Import ผ่าน Cypher
สคีมาของ `edges_rows.csv` คือ `id, person_a, person_b, relation`
```cypher
LOAD CSV WITH HEADERS FROM 'file:///edges_rows.csv' AS row
MERGE (a:Person {name: row.person_a})
MERGE (b:Person {name: row.person_b})
CREATE (a)-[:KNOWS {id: toInteger(row.id)}]->(b);
```
ตรวจสอบ:
```cypher
MATCH (p:Person) RETURN count(p);          // 24
MATCH ()-[r:KNOWS]->() RETURN count(r);    // 84
```

## 5) ตรวจ GDS plugin
```cypher
CALL gds.list();
```

## 6) Project graph แบบ undirected
ต้องทำก่อน เพราะ Bridges / Closeness / Eigenvector / Betweenness ใช้ undirected:
```cypher
CALL gds.graph.drop('myGraph', false);
MATCH (s:Person)-[r:KNOWS]->(t:Person)
RETURN gds.graph.project(
  'myGraph', s, t, {},
  { undirectedRelationshipTypes: ['*'] }
);
```

## 7) Centralities (Cypher only)
```cypher
// PageRank
CALL gds.pageRank.stream('myGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC;

// Betweenness
CALL gds.betweenness.stream('myGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC;
```
อ้างอิง: https://neo4j.com/docs/graph-data-science/current/algorithms/centrality/

## 8) Bridges
```cypher
CALL gds.bridges.stream('myGraph')
YIELD from, to, remainingSizes
RETURN gds.util.asNode(from).name AS f,
       gds.util.asNode(to).name   AS t,
       remainingSizes;
```
อ้างอิง: https://neo4j.com/docs/graph-data-science/current/algorithms/bridges/

## 9) Community detection — Louvain
```cypher
// modularity & community count
CALL gds.louvain.stats('myGraph')
YIELD modularity, communityCount, communityDistribution;

// per-node assignment
CALL gds.louvain.stream('myGraph')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).name AS name, communityId
ORDER BY communityId, name;
```
อ้างอิง: https://neo4j.com/docs/graph-data-science/current/algorithms/community/

## 10) Community detection — K-Means (ต้องสร้าง embedding ก่อน)
```cypher
CALL gds.fastRP.mutate('myGraph', {
  embeddingDimension: 128, randomSeed: 42, mutateProperty: 'embedding'
});

CALL gds.kmeans.stream('myGraph', {
  nodeProperty: 'embedding', k: 4, randomSeed: 42,
  computeSilhouette: true, concurrency: 1
})
YIELD nodeId, communityId, silhouette
RETURN gds.util.asNode(nodeId).name AS name, communityId, silhouette
ORDER BY communityId, name;
```

## คำสั่ง Docker ที่มักใช้
```bat
docker ps -a
docker logs neo4j-gds
docker rm -f neo4j-gds
docker volume rm neo4j_data
```
