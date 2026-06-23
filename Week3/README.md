# Week 3 — Neo4j Graph Data Science (GDS)

Week 3 ของ **DADS7201 — Social Network Analysis** ที่ NIDA
ย้ายจากการคำนวณด้วย NetworkX (Week 2) มาใช้ **Neo4j Graph Data Science**
เต็มรูปแบบ: project graph เข้า GDS in-memory store แล้วเรียก algorithm
(Bridges, Centrality, Louvain, K-Means) ผ่าน **Cypher**

> โจทย์: หา bridges + 6 centrality + community detection (K-Means, Louvain)
> บน `edges_rows.csv` (social graph 24 คน) และ Karate Club / IMDB ของ GDS
> **ห้ามใช้ NetworkX** — ทุกอย่างต้องคำนวณผ่าน Neo4j GDS

## สรุปย่อ

- **ข้อมูล:** `data/edges_rows.csv` — 24 คน, 84 ความสัมพันธ์ `KNOWS`
  (กราฟแตกเป็น 2 components: กลุ่มใหญ่ 16 คน + กลุ่มของ Khemasiri/Suphawan 8 คน)
- **ระบบกราฟ:** Neo4j Community + GDS plugin ใน Docker container (ดู [config/readme.md](config/readme.md))
- **อัลกอริทึมหลัก** (ทั้งหมดจาก `gds.*` ใน Cypher):
  - `gds.bridges.stream` — บน projected undirected graph
  - `gds.betweenness.stream`, `gds.closeness.stream`, `gds.degree.stream`,
    `gds.eigenvector.stream`, `gds.pageRank.stream`
  - `gds.louvain.stream` / `gds.louvain.stats`
  - `gds.fastRP.mutate` → `gds.kmeans.stream` (k-means ต้องการ embedding ก่อน)
- **การแสดงผล:** matplotlib สำหรับ static PNG, PyVis สำหรับ interactive HTML
  (circular layout / FastRP 2-dim ขึ้นกับสคริปต์)

## โครงสร้างโฟลเดอร์

```
Week3/
├── config/
│   ├── docker-compose.yml      # Neo4j + GDS plugin
│   └── readme.md               # คู่มือ Lab (CMD + compose + Cypher steps)
├── data/
│   └── edges_rows.csv          # ข้อมูลต้นทาง
├── neo4j-import/               # mount เข้า container path /import (gitignored)
├── scripts/
│   ├── neo4j_utils.py          # driver + GDS client helper
│   ├── import_data.py          # LOAD CSV → Person + KNOWS
│   ├── centrality_gds_only.py  # Bridges + 5 centralities → PNG (6-panel)
│   ├── centrality_pyvis.py     # เดียวกัน → interactive HTML
│   ├── kmeans_community.py     # FastRP + K-Means (k=2..8) → cluster + elbow
│   ├── kmeans_pyvis.py         # เดียวกัน → interactive HTML
│   ├── louvain_community.py    # Louvain vs K-Means side-by-side PNG + NMI/ARI
│   ├── louvain_pyvis.py        # เดียวกัน → interactive HTML
│   ├── karate_kmeans.py        # K-Means บน gds.graph.load_karate_club()
│   ├── karate_louvain.py       # Louvain vs K-Means vs Ground Truth (3-panel)
│   ├── 06_imdb_kmeans.py       # K-Means บน gds.graph.load_imdb() (12K nodes)
│   └── check_bridges.py        # quick sanity check ของ bridges
├── outputs/
│   ├── images/   # PNG ทั้งหมด
│   ├── html/     # interactive HTML
│   └── report/   # GDS_Report.docx
├── docs/         # PDF lecture / รูปโจทย์ / report editor (gitignored)
└── legacy/       # โค้ดเก่ายุค NetworkX (gitignored)
```

## วิธีรัน

### 1. Start Neo4j + GDS
```bat
:: Windows CMD — จากโฟลเดอร์ Week3
docker run -d --name neo4j-gds ^
  -p 7474:7474 -p 7687:7687 ^
  -v neo4j_data:/data ^
  -v "%cd%\neo4j-import:/import" ^
  -e NEO4J_AUTH=neo4j/s3cureP@ssword ^
  -e NEO4J_PLUGINS="[\"graph-data-science\"]" ^
  -e NEO4J_dbms_security_procedures_unrestricted=gds.* ^
  --memory=4g neo4j:latest
```
หรือ `docker compose -f config/docker-compose.yml up -d`

### 2. Import + รัน analysis
```powershell
cd Week3
pip install neo4j graphdatascience pyvis matplotlib numpy pandas python-docx

python scripts/import_data.py            # โหลด CSV เข้า Neo4j

python scripts/centrality_gds_only.py    # Bridges + 5 centralities → PNG
python scripts/centrality_pyvis.py       # → HTML
python scripts/kmeans_community.py       # K-Means + silhouette elbow
python scripts/louvain_community.py      # Louvain vs K-Means + NMI/ARI
python scripts/karate_louvain.py         # Louvain บน Karate Club
```

## ผลลัพธ์หลัก

### Bridges (edges_rows.csv)
**0 bridges** — graph มี 2 components ที่แยกอยู่แล้ว (16 + 8) และ edges
ที่เหลือทั้งหมดอยู่ใน cycle (รวมถึง parallel edges จากการเขียน a→b
และ b→a ใน CSV)

### Centrality — Top 1 ของแต่ละตัว
| Metric | Node | Score |
|---|---|---|
| Betweenness | Supitcha Kaewplengsrisakul | 34.6467 |
| Closeness | Suphawan / Khemasiri | 0.7778 |
| Degree | Noppawat | 15 |
| Eigenvector | Noppawat | 0.4484 |
| PageRank | Noppawat | 1.5681 |

→ **Noppawat** = central hub, **Supitcha** = betweenness leader (เชื่อมระหว่างกลุ่ม),
**Suphawan/Khemasiri** = closeness leader (อยู่ใน cluster หนาแน่น)

### Community Detection
| Method | edges_rows.csv | Karate Club (vs 2-faction GT) |
|---|---|---|
| Louvain | 3 communities, modularity = 0.5318 | 5 communities, modularity = 0.3113, **accuracy 88.24%** |
| K-Means (FastRP) | k=4, silhouette = 0.7840 | k=2, silhouette = 0.6830, accuracy 73.53% |
| Agreement | NMI = 0.9091, ARI = 0.8733 | — |

→ Louvain ดีกว่า K-Means บน Karate Club เพราะใช้โครงสร้าง edge โดยตรง
ไม่ผ่าน embedding

## เอกสารอ้างอิง

- Bridges: <https://neo4j.com/docs/graph-data-science/current/algorithms/bridges/>
- Betweenness: <https://neo4j.com/docs/graph-data-science/current/algorithms/betweenness-centrality/>
- Centrality (ภาพรวม): <https://neo4j.com/docs/graph-data-science/current/algorithms/centrality/>
- Community detection: <https://neo4j.com/docs/graph-data-science/current/algorithms/community/>
