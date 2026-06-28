# Week 4 — Similarity & Node Embeddings (Neo4j GDS)

Week 4 ของ **DADS7201 — Social Network Analysis** ที่ NIDA — เน้นไปที่
**Similarity** (วัดความเหมือนระหว่าง node) และ **Node Embeddings**
(ฝัง node ลงใน vector space) ซึ่งเป็นพื้นฐานของระบบแนะนำ (recommender),
link prediction, และ node classification

> Slides + notebook ต้นทาง อยู่ใน [`Slide/`](Slide/) (gitignored) และ
> ใช้ **Neo4j Sandbox** (live GDS instance) แทน Docker local

## ภาพรวม 4 หัวข้อหลัก

| # | หัวข้อ | API หลัก | ใช้ทำอะไร |
|---|---|---|---|
| 1 | Similarity Functions | `gds.similarity.*` (Cypher functions) | คำนวณ similarity ระหว่าง 2 array โดยตรง |
| 2 | Node Similarity | `gds.nodeSimilarity.*` | similarity ระหว่าง node ใน bipartite graph (ดู neighbor set) |
| 3 | K-Nearest Neighbors | `gds.knn.*` | สร้าง edge `[:SIMILAR {score}]` จาก node properties |
| 4 | FastRP Node Embedding | `gds.fastRP.*` | ฝัง node → vector ขนาดคงที่ (input ของ ML pipeline) |

End-to-end ที่ Slide เน้น: **FastRP → KNN → product recommendation**
(ดู [`Slide/fastrp_and_knn.ipynb`](Slide/fastrp_and_knn.ipynb) — อ้างอิงจาก
[Neo4j GDS docs](https://neo4j.com/docs/graph-data-science/current/end-to-end-examples/fastrp-knn-example/))

---

## 1. Similarity Functions (`gds.similarity.*`)

Cypher functions แบบ lightweight — **ไม่ต้อง project graph** ใช้คำนวณ
similarity ระหว่าง 2 array ตรง ๆ มี 6 ตัว แบ่งเป็น 2 กลุ่ม

### Categorical (set-based)

| Function | สูตรย่อ | Range | ใช้เมื่อ |
|---|---|---|---|
| `jaccard` | `\|A∩B\| / \|A∪B\|` | [0, 1] | วัด overlap ของ 2 set (binary features) |
| `overlap` | `\|A∩B\| / min(\|A\|, \|B\|)` | [0, 1] | ดูว่า set เล็กถูก "ครอบ" ใน set ใหญ่แค่ไหน |

### Numerical (position-based)

| Function | Range | ใช้เมื่อ |
|---|---|---|
| `cosine` | [-1, 1] | vector embeddings, text, NLP — สนใจทิศทางไม่ใช่ขนาด |
| `pearson` | [-1, 1] | linear correlation (2 ตัวแปรไปทิศทางเดียวกัน?) |
| `euclideanDistance` | [0, ∞) | ระยะทางตรง — ยิ่งน้อย = ยิ่งเหมือน |
| `euclidean` | (0, 1] | normalize ของ euclideanDistance |

### ตัวอย่าง Cypher

```cypher
RETURN gds.similarity.jaccard(
  [1.0, 5.0, 3.0, 6.7],
  [5.0, 2.5, 3.1, 9.0]
) AS jaccardSimilarity;
// → 0.1429

RETURN gds.similarity.cosine(
  [1.0, 5.0, 3.0, 6.7],
  [5.0, 2.5, 3.1, 9.0]
) AS cosineSimilarity;
// → 0.8828
```

### Null handling
- **Categorical** (Jaccard/Overlap): `null` ถูกตัดออกจาก set
- **Numerical** (Cosine/Pearson/Euclidean): `null` ถูกแทนด้วย `0.0`

---

## 2. Node Similarity (`gds.nodeSimilarity.*`)

เทียบ node ใน **bipartite graph** โดยดู neighbor set ของแต่ละ node
ถ้า 2 node แชร์ neighbor มาก ⇒ similar metric: **Jaccard / Overlap / Cosine**
(weighted ก็ได้ ใช้ relationship property)

### Cypher แบบเต็ม (example: Person ↔ Instrument)

```cypher
// 1) project bipartite graph
MATCH (source:Person)
OPTIONAL MATCH (source)-[r:LIKES]->(target:Instrument)
RETURN gds.graph.project(
  'myGraph', source, target,
  { relationshipProperties: r { strength: coalesce(r.strength, 1.0) } }
);

// 2) stream similarity (default Jaccard)
CALL gds.nodeSimilarity.stream('myGraph')
YIELD node1, node2, similarity
RETURN gds.util.asNode(node1).name AS Person1,
       gds.util.asNode(node2).name AS Person2,
       similarity
ORDER BY similarity DESC;

// 3) เก็บผลกลับเข้า Neo4j เป็น :SIMILAR edge
CALL gds.nodeSimilarity.write('myGraph', {
  writeRelationshipType: 'SIMILAR',
  writeProperty: 'score'
});
```

### Parameters สำคัญ

| Parameter | Default | ใช้ทำอะไร |
|---|---|---|
| `similarityMetric` | `JACCARD` | หรือ `OVERLAP`, `COSINE` |
| `topK` | 10 | ผลลัพธ์ top-K ต่อ node |
| `similarityCutoff` | 1e-42 | ตัด pair คะแนนต่ำ |
| `degreeCutoff` | 1 | ตัด node degree ต่ำ |
| `relationshipWeightProperty` | — | ใช้ weight (เปลี่ยนเป็น weighted Jaccard/Overlap) |
| `useComponents` | `false` | `true` = ข้าม cross-component (เร็วขึ้น) |

### Modes: `stream` / `stats` / `mutate` / `write` / `estimate`

> **Key takeaway:** Memory = O(n²) ในเคสแย่ ต้องคุมด้วย `topK` + `similarityCutoff`
> ผลลัพธ์เป็น **directed** relationship เสมอ

---

## 3. K-Nearest Neighbors (`gds.knn.*`)

ต่างจาก `nodeSimilarity` ตรงที่ **ไม่สนใจ relationship เลย** — ใช้
**node properties** วัด similarity (เช่น age, lotteryNumbers, embedding)

### Algorithm (Wei Dong et al.)
1. สุ่ม initial neighbor `k` ตัวต่อ node
2. Iterate: เปรียบเทียบ neighbor-of-neighbor → swap ถ้าเจอที่ดีกว่า
3. Stop เมื่อ change < `deltaThreshold` หรือถึง `maxIterations`
4. Output: `(:Node)-[:SIMILAR {score}]->(:Node)`

**Scaling: quasi-linear** (ไม่ใช่ O(n²) แบบ nodeSimilarity)

### Metric ตามชนิด property

| Property type | Available metrics |
|---|---|
| Scalar number | `DEFAULT` |
| List of Integer | `JACCARD` (default), `OVERLAP` |
| List of Float | `COSINE` (default), `EUCLIDEAN`, `PEARSON` |

ใช้หลาย property พร้อมกัน → similarity = **mean ของแต่ละ metric**

### ตัวอย่าง Cypher (single + multi-property)

```cypher
// disconnected nodes, similarity จาก age อย่างเดียว
CALL gds.knn.stream('myGraph', {
  topK: 1,
  nodeProperties: ['age'],
  randomSeed: 1337,
  concurrency: 1,
  sampleRate: 1.0,
  deltaThreshold: 0.0
})
YIELD node1, node2, similarity
RETURN gds.util.asNode(node1).name AS P1,
       gds.util.asNode(node2).name AS P2,
       similarity
ORDER BY similarity DESC;

// multi-property: คะแนนเฉลี่ยจาก embedding/age/lotteryNumbers
CALL gds.knn.stream('myGraph', {
  topK: 1,
  nodeProperties: [
    {embedding: 'EUCLIDEAN'},
    'age',
    {lotteryNumbers: 'OVERLAP'}
  ],
  randomSeed: 1337, concurrency: 1, sampleRate: 1.0, deltaThreshold: 0.0
})
YIELD node1, node2, similarity ...;
```

### Trade-off `sampleRate`

| Value | Accuracy | Memory | Speed |
|---|---|---|---|
| `1.0` | Highest | Highest | Slowest |
| `0.5` (default) | Balanced | Moderate | Moderate |
| `< 0.5` | Lower (อาจพลาด neighbor) | Lower | Faster |

> **Deterministic mode:** `{ concurrency: 1, randomSeed: <int>, sampleRate: 1.0, deltaThreshold: 0.0 }`

---

## 4. FastRP Node Embedding (`gds.fastRP.*`)

**Fast Random Projection** — ฝัง node ลงใน vector space เพื่อให้ ML ใช้
ต่อได้ (KNN, link prediction, classification) อ้างอิงจาก
[Chen et al. 2019](https://arxiv.org/pdf/1908.11512) + Johnson–Lindenstrauss lemma

### หลักการ
1. **Initialization** — กระจาย sparse random vector (`±1`) ให้ทุก node
2. **Iterative neighborhood averaging** — แต่ละรอบ `i = 1..k`:
   ```
   v_i(n) = average of v_{i-1}(neighbors of n)
   ```
3. **Final embedding** — weighted sum:
   ```
   emb(n) = α·R(n) + Σ wᵢ · vᵢ(n)
   ```
   - α = `nodeSelfInfluence`
   - wᵢ = `iterationWeights[i]`

**Properties:**
- **Linear time** ใน |E| → scale ได้ดี
- **Multi-scale** — รอบแรกจับ local, รอบหลังจับ global
- Output: dense vector ขนาด `embeddingDimension`

### Cypher (3 รูปแบบ: plain / weighted / property-aware)

```cypher
// project undirected graph + เก็บ age + weight
MATCH (s:Person)-[r:KNOWS]->(t:Person)
RETURN gds.graph.project(
  'persons', s, t,
  {
    sourceNodeProperties: s { .age },
    targetNodeProperties: t { .age },
    relationshipProperties: r { .weight }
  },
  { undirectedRelationshipTypes: ['*'] }
);

// (A) plain FastRP
CALL gds.fastRP.stream('persons',
  { embeddingDimension: 4, randomSeed: 42 }
) YIELD nodeId, embedding
RETURN gds.util.asNode(nodeId).name AS person, embedding;

// (B) weighted FastRP — ใช้ weight ของ relationship
CALL gds.fastRP.stream('persons', {
  embeddingDimension: 4, randomSeed: 42,
  relationshipWeightProperty: 'weight'
}) YIELD nodeId, embedding ...;

// (C) property-aware FastRP — รวม age เข้าไปใน embedding
CALL gds.fastRP.stream('persons', {
  embeddingDimension: 2, propertyRatio: 1.0,
  featureProperties: ['age'], iterationWeights: [1.0],
  randomSeed: 42
}) YIELD nodeId, embedding ...;
```

### Tuning Tips
1. **`embeddingDimension`** — 128-1024, ทั่วไปใช้ **256**
2. **`iterationWeights`** — `[1, 1, 1]` หรือ `[0, 1, 1]` สำหรับ 2-3 hops
3. **`propertyRatio` + `featureProperties`** — เปิดเมื่อ node property มีข้อมูล
4. **`nodeSelfInfluence`** — เปิดเมื่อ graph มี isolated nodes
5. **`randomSeed`** — set เสมอเพื่อให้ reproducible
6. **ประเมินจาก downstream task** (link prediction / classification) ไม่ใช่ดู embedding ตรง ๆ

---

## End-to-end: Product Recommendation (FastRP → KNN)

ตัวอย่างใน [`Slide/fastrp_and_knn.ipynb`](Slide/fastrp_and_knn.ipynb) — graph
มี 6 `Person` + 7 `Product` กับ `[:BUYS {amount}]` แล้วต้องการแนะนำสินค้าใหม่
ให้ลูกค้าแต่ละคน

```python
from graphdatascience import GraphDataScience
gds = GraphDataScience(NEO4J_URI, auth=(USER, PASSWORD))

# 1) Project bipartite graph (Person ↔ Product)
G, _ = gds.graph.project(
    "purchases",
    ["Person", "Product"],
    {"BUYS": {"orientation": "UNDIRECTED", "properties": "amount"}}
)

# 2) FastRP embeddings → mutate
gds.fast_rp.mutate(G,
    mutate_property="embedding",
    embedding_dimension=4,
    relationship_weight_property="amount",
    iteration_weights=[0.8, 1, 1, 1],
    random_seed=42
)

# 3) KNN บน embedding → เขียน (:Person)-[:SIMILAR {score}]->(:Person)
gds.knn.write(G,
    top_k=2,
    node_properties=["embedding"],
    write_relationship_type="SIMILAR",
    write_property="score",
    random_seed=42, concurrency=1, sample_rate=1.0, delta_threshold=0.0
)

# 4) แนะนำสินค้า — ของที่ "คนคล้ายเรา" ซื้อ แต่เรายังไม่ซื้อ
gds.run_cypher("""
  MATCH (:Person {name: 'Annie'})-[:BUYS]->(p1:Product)
  WITH collect(p1) AS products
  MATCH (:Person {name: 'Matt'})-[:BUYS]->(p2:Product)
  WHERE NOT p2 IN products
  RETURN p2.name AS recommendation
""")
// → "Kale"  (Annie ซื้อ Kale, Matt ไม่ซื้อ → แนะนำ Kale ให้ Matt)
```

> **ทำไมต้อง FastRP ก่อน KNN?** เพราะ KNN ทำงานบน node property (vector)
> ไม่ใช่ edge การฝัง graph topology → embedding ทำให้ KNN ใช้ประโยชน์จาก
> โครงสร้างกราฟได้ทางอ้อม

---

## เปรียบเทียบ 3 อัลกอริทึม similarity

| | Node Similarity | KNN | nodeSimilarity vs KNN |
|---|---|---|---|
| **Input** | Bipartite graph (edges) | Node properties | edge vs property |
| **Metric** | Jaccard / Overlap / Cosine | DEFAULT / JACCARD / OVERLAP / COSINE / EUCLIDEAN / PEARSON | KNN รองรับมากกว่า |
| **Scaling** | O(n²) worst-case | Quasi-linear | KNN ดีกว่ามากเมื่อ graph ใหญ่ |
| **ต้องมี edge?** | ✅ ใช้ neighbor | ❌ ใช้ property | KNN ใช้กับ disconnected nodes ได้ |
| **Use case** | "ใครชอบของเหมือนกัน" | "ใครมี profile ใกล้กัน" | ตามชนิดข้อมูล |

## เอกสารอ้างอิง

- [Similarity Functions](https://neo4j.com/docs/graph-data-science/current/alpha-algorithms/similarity-functions/)
- [Node Similarity](https://neo4j.com/docs/graph-data-science/current/algorithms/node-similarity/)
- [KNN](https://neo4j.com/docs/graph-data-science/current/algorithms/knn/)
- [FastRP](https://neo4j.com/docs/graph-data-science/current/machine-learning/node-embeddings/fastrp/)
- [End-to-end: FastRP + KNN tutorial](https://neo4j.com/docs/graph-data-science/current/end-to-end-examples/fastrp-knn-example/)
- Chen et al., _Fast and Accurate Network Embeddings via Very Sparse Random Projection_ (2019) — <https://arxiv.org/pdf/1908.11512>
- Wei Dong et al., _Efficient k-nearest neighbor graph construction for generic similarity measures_

## หมายเหตุ

- โฟลเดอร์ [`Slide/`](Slide/) (PDF slides + notebooks + Sandbox credentials)
  อยู่ใน `.gitignore` ไม่ commit ขึ้น repo
- Neo4j Sandbox ที่ใช้ใน lab เป็น **shared cloud instance** ของ Neo4j —
  ถ้าจะรันเอง ให้ตั้ง Docker + GDS ตาม [`../Week3/config/readme.md`](../Week3/config/readme.md)
