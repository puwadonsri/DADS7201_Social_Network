# Week 5 — Node Embeddings (from scratch → DeepWalk → GCN)

Week 5 ของ **DADS7201 — Social Network Analysis** ที่ NIDA — เรียน
**Node Embeddings**: แปลง node ในกราฟให้เป็น vector มิติต่ำ ที่ *ยัง
รักษาโครงสร้างของกราฟไว้* เพื่อป้อนต่อให้ ML task (link prediction,
node classification, clustering, visualisation)

> Slides + notebook ต้นทาง อยู่ใน [`Slide/`](Slide/) (gitignored)

## เส้นทางที่ Slide พาไป

```
    Random walk           Softmax + SGD           PCA / Cosine
Graph ─────────► N_R(u) ─────────────► z_u ∈ R^d ────────────► visualise + evaluate
                                       │
                                       ├── DeepWalk (uniform walk + SkipGram)
                                       ├── Node2Vec (biased walk: p, q)
                                       └── GCN (message passing, PyG)
```

3 ก้อนหลักที่คาบนี้ครอบคลุม:

| # | หัวข้อ | ทำอะไร | Reference |
|---|---|---|---|
| 1 | **Encoder-only จาก scratch** | Random walk → softmax NLL → SGD → PCA + cosine similarity | [`Slide/Lab1_inClass.ipynb`](Slide/Lab1_inClass.ipynb) |
| 2 | **DeepWalk / Node2Vec** | Truncated random walk + SkipGram, biased walk (BFS/DFS) | [`Slide/02-nodeemb.pdf`](Slide/02-nodeemb.pdf) |
| 3 | **GCN preview (PyG)** | Message passing บน Karate Club, 3-layer GCN + linear head | [`Slide/CS224W_Colab_0.ipynb`](Slide/CS224W_Colab_0.ipynb), [`Slide/CS224W_Colab_1_2024_25.ipynb`](Slide/CS224W_Colab_1_2024_25.ipynb) |

---

## 1. ทำไมต้อง embedding? — ปัญหาของ one-hot / adjacency

| Representation | ปัญหา |
|---|---|
| One-hot | มิติ = |V|, sparse, ไม่มีความหมายเชิงเรขาคณิต (dot product = 0 เสมอ ยกเว้นตัวเอง) |
| Adjacency row | ไวต่อ noise, ไม่ generalise, feature ยังไม่เชิงความหมาย |
| **Learned embedding** | มิติต่ำคงที่, node ที่ "ใกล้กันในกราฟ" ⇒ vector ก็ใกล้กัน, ป้อน ML ได้ตรง ๆ |

**เป้าหมาย:** เรียน encoder `ENC(u) = z_u ∈ R^d` โดย

$$
\text{similarity}(u, v) \approx z_u^\top z_v
$$

---

## 2. Encoder-only: build from scratch (Lab1_inClass)

Notebook ใน Slide สร้างกราฟเล็ก 10 node / 12 edge แล้วเรียน embedding
โดย **ไม่ใช้ library สำเร็จรูป** — เห็นทุกขั้นตอน

### ขั้นตอน

**(1) สุ่ม walk 2 step จากทุก node → `N_R(u)`**

```python
def perform_random_walks(graph, steps=2):
    node_walks = {}
    for start in graph.nodes():
        walk, cur = [start], start
        for _ in range(steps):
            nbrs = list(graph.neighbors(cur))
            if not nbrs: break
            cur = random.choice(nbrs)
            walk.append(cur)
        node_walks[start] = walk
    return node_walks

# ตัวอย่างผลลัพธ์:
# Node 0: [0, 2, 7]   Node 1: [1, 4, 9]   Node 2: [2, 6, 7] ...
```

**(2) Loss = Negative Log-Likelihood + Softmax**

$$
P(v \mid u) = \frac{\exp(z_u \cdot z_v)}{\sum_{k \in V} \exp(z_u \cdot z_k)}, \qquad
J = -\sum_{u \in V} \sum_{v \in N_R(u)} \log P(v \mid u)
$$

**(3) SGD update per node**

$$
\frac{\partial J}{\partial z_u} = \sum_{k \in V}\bigl(P(k \mid u) - \mathbb{1}[k = v]\bigr) \cdot z_k
$$

```python
# 4-dim embedding, 10,000 iter, lr=0.01, epsilon=1e-4
optimized, final_loss = train_sgd(G, NR_u, node_embeddings,
                                  lr=0.01, iterations=10000, epsilon=0.0001)
# → Loss: 73.93 (init) → 40.75 (converge @ iter 306)
```

**(4) Visualise ด้วย PCA → 2D**

```python
from sklearn.decomposition import PCA
embeddings_2d = PCA(n_components=2).fit_transform(embedding_matrix)
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=500, c='orange')
```

**(5) ประเมินด้วย Top-1 Neighbor Reconstruction (Cosine similarity)**

| Node | ใกล้สุด | มี edge จริงไหม? |
|---:|:---:|:---:|
| 0 | 7 | Not connected (แต่ 0→2→7 ผ่าน walk) |
| 1 | 4 | Connected |
| 2 | 6 | Connected (sim=0.996) |
| 3 | 8 | Connected |
| 6 | 2 | Connected (sim=0.996) |

> **Key takeaway:** node ที่ปรากฏใน walk เดียวกันบ่อย ๆ ได้ embedding ใกล้กัน
> แม้ไม่มี edge ตรง ๆ — เพราะ softmax NLL บังคับให้ `z_u·z_v` สูงเมื่อ `v ∈ N_R(u)`

---

## 3. DeepWalk & Node2Vec — จาก scratch → ของจริง

### DeepWalk (Perozzi et al. 2014)
1. สุ่ม **truncated random walk** ความยาว `L` จำนวน `γ` รอบต่อ node
2. มอง walk แต่ละอันเป็น "ประโยค" ป้อนให้ **SkipGram** (word2vec)
3. ได้ embedding node = word vector

### Node2Vec (Grover & Leskovec 2016)
เพิ่ม **biased walk** ผ่านพารามิเตอร์ 2 ตัว

| Param | ทำอะไร | ตั้งค่าแบบไหน |
|---|---|---|
| `p` (return) | ควบคุมโอกาสกลับ node เดิม | สูง = ไม่ย้อน (สำรวจใหม่) |
| `q` (in-out) | ควบคุม BFS vs DFS | `q > 1` → BFS-like (local, structural equivalence)<br>`q < 1` → DFS-like (global, homophily) |

```python
from node2vec import Node2Vec

# BFS-like: จับ structural role
model_bfs = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200,
                     p=1, q=2.0, workers=4).fit(window=10, min_count=1)

# DFS-like: จับ community / homophily
model_dfs = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200,
                     p=1, q=0.5, workers=4).fit(window=10, min_count=1)

model_bfs.wv.most_similar("0", topn=5)
```

> **BFS vs DFS ต่างกันตรงไหน:**
> BFS → *"node นี้เล่นบทอะไร"* (hub, bridge, leaf)
> DFS → *"node นี้อยู่ community ไหน"*

---

## 4. Link Prediction ด้วย embedding

ใช้ embedding เป็น edge feature (concat) → train classifier

```python
def edge_feature(u, v):
    return np.concatenate([model.wv[str(u)], model.wv[str(v)]])

pos = list(G.edges())
neg = [(u, v) for u, v in itertools.combinations(G.nodes(), 2)
       if not G.has_edge(u, v)][:len(pos)]

X = np.array([edge_feature(u, v) for u, v in pos + neg])
y = np.array([1]*len(pos) + [0]*len(neg))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
clf = LogisticRegression().fit(X_tr, y_tr)
print(f"AUC: {roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]):.3f}")
```

**Alternative operators** สำหรับ edge feature (แทน concat):

| Operator | สูตร | ใช้เมื่อ |
|---|---|---|
| Hadamard | `z_u ⊙ z_v` | มัน default ของ Node2Vec paper |
| Average | `(z_u + z_v) / 2` | symmetric, undirected |
| L1 / L2 | `|z_u − z_v|` | ระยะทางเชิงเวกเตอร์ |

---

## 5. GCN preview (CS224W Colab 0 → 1)

Colab 0 พาไปแตะ **PyTorch Geometric** บน **Zachary's Karate Club** —
34 node / 4 community — แล้วสร้าง 3-layer GCN

```python
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(dataset.num_features, 4)   # 34 → 4
        self.conv2 = GCNConv(4, 4)                       # 4  → 4
        self.conv3 = GCNConv(4, 2)                       # 4  → 2  (2D สำหรับ visualise)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).tanh()
        h = self.conv2(h, edge_index).tanh()
        h = self.conv3(h, edge_index).tanh()   # final embedding space
        return self.classifier(h), h
```

- **1 layer = aggregate จาก 1-hop neighbour** → 3 layer = 3-hop receptive field
- ก่อน train ยัง เห็น community structure แล้ว (inductive bias ของ GCN)
- Semi-supervised: รู้ label แค่ **4 node** (1 ต่อ community) → propagate ผ่าน graph

### Colab 1 pipeline (การบ้าน 7 ข้อ)

| Q | โจทย์ | เฉลย (Karate Club) |
|---:|---|---|
| 1 | Average degree | 5 |
| 2 | Avg clustering coef | 0.57 |
| 3 | PageRank ของ node 0 หลัง 1 iter (β=0.8) | 0.13 |
| 4 | Closeness centrality ของ node 5 | 0.01 |
| 5 | Edge list → `torch.LongTensor` [2, 78] | sum = 2535 |
| 6 | Negative sampling — เช็คไม่ตรงกับ edge ใน G | — |
| 7 | Train `nn.Embedding(34, 16)` ผ่าน link prediction | acc = 1.0 @ epoch 50 |

**Q7 objective** — dot product + sigmoid + BCE loss, SGD lr=0.1

```python
optimizer = SGD(emb.parameters(), lr=0.1, momentum=0.9)
loss_fn = nn.BCELoss()

for epoch in range(500):
    optimizer.zero_grad()
    node_emb = emb(train_edge)              # [2, E, 16]
    dot = (node_emb[0] * node_emb[1]).sum(-1)
    pred = torch.sigmoid(dot)
    loss = loss_fn(pred, train_label)       # 1=positive, 0=negative
    loss.backward()
    optimizer.step()
```

---

## Method comparison — เลือกอันไหน?

| Method | Scale | ต้องใช้ label ไหม? | จับ homophily? | จับ structural role? | หมายเหตุ |
|---|---|---|---|---|---|
| Random walk + softmax (from scratch) | เล็ก (< 100 node) | ไม่ | ✅ | ✅ (บ้าง) | เอาไว้เรียน ไม่ใช้จริง |
| **DeepWalk** | กลาง | ไม่ | ✅ | ⚠️ | uniform walk = biased ไปทาง homophily |
| **Node2Vec** | กลาง | ไม่ | ✅ (q<1) | ✅ (q>1) | ปรับผ่าน p, q |
| **FastRP** (Week 4) | ใหญ่มาก | ไม่ | ✅ | ⚠️ | linear ใน \|E\|, deploy บน Neo4j GDS ได้เลย |
| **GCN** | กลาง–ใหญ่ | ✅ (semi-sup ได้) | ✅ | ✅ | ต้องมี node feature, train เต็มรูป |

---

## ความรู้พื้นฐานที่ควรมี

- Graph basics: node, edge, adjacency, degree, neighbour
- **Week 4** — FastRP, Node Similarity, KNN (embedding บน Neo4j GDS)
- Softmax + cross-entropy + SGD (สำหรับส่วน from-scratch)
- PyTorch basics (สำหรับ Colab 0 / 1)
