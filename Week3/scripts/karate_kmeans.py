from graphdatascience import GraphDataScience
import matplotlib.pyplot as plt
import numpy as np

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "s3cureP@ssword"

gds = GraphDataScience(URI, auth=(USER, PASSWORD), database="neo4j")

# --- 1. Load Karate Club ---
print("==> Loading Karate Club...")
try:
    gds.run_cypher("CALL gds.graph.drop('karate', false)")
except:
    pass
G = gds.graph.load_karate_club("karate", undirected=False)
print(f"    Nodes: {G.node_count()}, Edges: {G.relationship_count()}")

# Node IDs are 0..33 (GDS internal IDs for this graph)
# Zachary's Karate Club known ground truth:
# Mr.Hi faction (0): nodes 0..8,10..13,16,17,19,21
# Officer faction (1): nodes 9,14,15,18,20,22..33
gt = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:1,
      10:0,11:0,12:0,13:0,14:1,15:1,16:0,17:0,
      18:1,19:0,20:1,21:0,22:1,23:1,24:1,25:1,
      26:1,27:1,28:1,29:1,30:1,31:1,32:1,33:1}

# --- 2. FastRP embeddings ---
print("==> Generating FastRP embeddings...")
gds.fastRP.mutate(G, embeddingDimension=128, randomSeed=42, mutateProperty="embedding")

# --- 3. K-Means for k=2..8 ---
print("==> Finding optimal k...")
k_results = []
for k in range(2, 9):
    result = gds.kmeans.stream(G, nodeProperty="embedding", k=k, randomSeed=42,
                                computeSilhouette=True, concurrency=1)
    sil = result["silhouette"].dropna()
    avg_sil = sil.mean()
    sizes = result.groupby("communityId").size().tolist()
    print(f"    k={k}: silhouette={avg_sil:.4f}, sizes={sizes}")
    k_results.append((k, avg_sil, result))

best_k, best_sil, best_df = max(k_results, key=lambda x: x[1])
print(f"\n==> Optimal k = {best_k} (silhouette={best_sil:.4f})")

# --- 4. Accuracy vs ground truth ---
print(f"\n==> K-Means (k={best_k}) vs ground truth:")
mapping = {}  # map kmeans cluster -> ground truth (majority vote)
for _, r in best_df.iterrows():
    nid = int(r["nodeId"])
    pred = r["communityId"]
    actual = gt.get(nid, -1)
    mapping.setdefault(pred, []).append(actual)

# Find best alignment: assign each k-means cluster to majority ground truth
cluster_to_gt = {}
for cid, labels in mapping.items():
    majority = max(set(labels), key=labels.count)
    cluster_to_gt[cid] = majority
    print(f"    Cluster {cid}: {len(labels)} nodes -> GT faction {majority}")

correct = 0
for _, r in best_df.iterrows():
    nid = int(r["nodeId"])
    pred = cluster_to_gt[r["communityId"]]
    actual = gt.get(nid, -1)
    if pred == actual:
        correct += 1
    m = "+" if pred == actual else "-"
    print(f"    node {nid:2d}: kmeans={r['communityId']}->GT={pred}, actual={actual} [{m}]")

acc = correct / len(best_df)
print(f"\n    Accuracy: {correct}/{len(best_df)} = {acc:.2%}")

# --- 5. 2D embedding ---
print("\n==> Generating 2D embedding...")
try:
    gds.run_cypher("CALL gds.graph.drop('karate2d', false)")
except:
    pass
G2 = gds.graph.load_karate_club("karate2d", undirected=False)
gds.fastRP.mutate(G2, embeddingDimension=2, randomSeed=42, mutateProperty="emb2d")

# Get 2D coords by streaming node properties
emb2d = gds.graph.streamNodeProperties(G2, ["emb2d"])
print("emb2d columns:", emb2d.columns.tolist())
print("emb2d head:", emb2d.head())
# The column might be named differently
prop_col = [c for c in emb2d.columns if c != "nodeId"][0]
emb2d["x"] = emb2d[prop_col].apply(lambda v: v[0])
emb2d["y"] = emb2d[prop_col].apply(lambda v: v[1])
emb2d["kmeans"] = emb2d["nodeId"].map(dict(zip(best_df["nodeId"], best_df["communityId"])))

# --- 6. Plot ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
cmap_k = plt.cm.Set1(np.linspace(0, 1, best_k))

for ax, title, col_key, cmap in [
    (axes[0], f"K-Means (k={best_k}, sil={best_sil:.3f})", "kmeans", cmap_k),
    (axes[1], "Ground Truth (green=Mr.Hi, orange=Officer)", None, None)
]:
    for _, r in emb2d.iterrows():
        nid = int(r["nodeId"])
        if col_key == "kmeans":
            c = cmap[int(r["kmeans"])]
        else:
            cid = gt.get(nid, -1)
            c = "#66c2a5" if cid == 0 else "#fc8d62"
        ax.scatter(r["x"], r["y"], c=c, s=500, edgecolors="black", linewidths=0.5, zorder=5)
        ax.text(r["x"], r["y"], str(nid), fontsize=9, ha="center", va="center",
                fontweight="bold", zorder=6)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("FastRP dim-1")
    ax.set_ylabel("FastRP dim-2")

import os
out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "images")
os.makedirs(out_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "karate_kmeans.png"), dpi=200, bbox_inches="tight")
print(f"==> Saved: {os.path.join(out_dir, 'karate_kmeans.png')}")

# --- 7. Elbow ---
fig2, ax2 = plt.subplots(figsize=(10, 6))
ks = [x[0] for x in k_results]
sils = [x[1] for x in k_results]
ax2.plot(ks, sils, "bo-", linewidth=2, markersize=8)
ax2.axvline(best_k, color="red", linestyle="--", alpha=0.5, label=f"best k={best_k}")
ax2.set_xlabel("k")
ax2.set_ylabel("Avg Silhouette")
ax2.set_title("Karate Club: K-Means Silhouette vs k")
ax2.grid(True, alpha=0.3)
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "karate_elbow.png"), dpi=200, bbox_inches="tight")
print(f"==> Saved: {os.path.join(out_dir, 'karate_elbow.png')}")

gds.close()
print("Done.")
