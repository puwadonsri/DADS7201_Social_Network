from neo4j_utils import get_gds, output_path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("=" * 60)
print("IMDB Dataset — K-Means Community Detection")
print("=" * 60)

gds = get_gds()

# --- 1. Load IMDB ---
print("\n==> Loading IMDB dataset...")
try:
    gds.run_cypher("CALL gds.graph.drop('imdb', false)")
except:
    pass
G = gds.graph.load_imdb("imdb", undirected=True)
print(f"    Nodes: {G.node_count():,}")
print(f"    Edges: {G.relationship_count():,}")
print(f"    Labels: {G.node_labels()}")
print(f"    Rel types: {G.relationship_types()}")
print(f"    Node properties: {G.node_properties()}")

# Count by label
counts = gds.run_cypher("""
    CALL gds.graph.nodeProperties.stream('imdb', ['plot_keywords'])
    YIELD nodeId, nodeLabels
    RETURN nodeLabels[0] AS label, count(*) AS cnt
    ORDER BY cnt DESC
""")
print("Node counts by label:")
for _, r in counts.iterrows():
    print(f"    {r['label']}: {r['cnt']:,}")

# --- 2. Generate FastRP embeddings (128-dim) ---
print("\n==> Generating FastRP embeddings (128-dim)...")
gds.fastRP.mutate(G, embeddingDimension=128, randomSeed=42, mutateProperty="embedding")

# --- 3. K-Means (without silhouette in loop, too slow on 12K nodes) ---
print("==> Running K-Means for k=2..8 (no silhouette)...")
k_results = []
for k in range(2, 9):
    result = gds.kmeans.stream(G, nodeProperty="embedding", k=k,
                                randomSeed=42, concurrency=1)
    sizes = result.groupby("communityId").size().tolist()
    print(f"    k={k}: sizes={sizes}")
    k_results.append((k, result))

# Pick best k: most balanced cluster sizes (lower std/size mean = better)
print("\n==> Selecting best k by cluster balance...")
best_k = None
best_balance = -1
for k, result in k_results:
    sizes = result.groupby("communityId").size()
    balance = sizes.mean() / sizes.std() if sizes.std() > 0 else 0
    print(f"    k={k}: sizes={sizes.tolist()}, balance={balance:.2f}")
    if balance > best_balance:
        best_balance = balance
        best_k = k

print(f"\n==> Selected k = {best_k} (balance={best_balance:.2f})")

# --- 4. Store best clustering in in-memory graph (mutate) ---
print(f"\n==> Storing k={best_k} clustering in graph...")
gds.kmeans.mutate(G, nodeProperty="embedding", k=best_k, randomSeed=42,
                   concurrency=1, mutateProperty="kmeans")

# --- 5. Add 2D FastRP embeddings to same graph ---
print("\n==> Generating 2D FastRP embeddings for visualization...")
gds.fastRP.mutate(G, embeddingDimension=2, randomSeed=42, mutateProperty="emb2d")

# --- 6. Evaluate against genre (Movie nodes only) ---
print("\n==> Streaming node properties for evaluation...")
kmeans_df = gds.graph.streamNodeProperty(G, "kmeans")
genre_df = gds.graph.streamNodeProperty(G, "genre")
print(f"    kmeans columns: {kmeans_df.columns.tolist()}, shape={kmeans_df.shape}")
print(f"    genre columns:  {genre_df.columns.tolist()}, shape={genre_df.shape}")
if len(genre_df) > 0:
    print(f"    genre values sample:\n{genre_df['genre'].value_counts().head(10)}")

labels_df = kmeans_df.merge(genre_df, on="nodeId", how="inner")
# Get node labels via Cypher
nodelabel_df = gds.run_cypher("""
    CALL gds.graph.nodeProperties.stream('imdb', 'plot_keywords')
    YIELD nodeId, nodeLabels
    RETURN nodeId, nodeLabels[0] AS label
""")
labels_df = labels_df.merge(nodelabel_df, on="nodeId", how="left")
labels_df = labels_df[(labels_df["label"] == "Movie") & (labels_df["genre"].notna())]

if len(labels_df) > 0:
    print(f"\n==> Ground truth evaluation (Movie nodes with genre):")
    print(f"    Total: {len(labels_df)}")

    mapping = {}
    for _, r in labels_df.iterrows():
        cid = int(r["kmeans"])
        g = r["genre"]
        mapping.setdefault(cid, []).append(g)

    cluster_to_genre = {}
    for cid, genres in mapping.items():
        cluster_to_genre[cid] = max(set(genres), key=genres.count)

    correct = 0
    for _, r in labels_df.iterrows():
        pred = cluster_to_genre[int(r["kmeans"])]
        if pred == r["genre"]:
            correct += 1

    acc = correct / len(labels_df)
    print(f"    Accuracy: {correct}/{len(labels_df)} = {acc:.2%}")

    for g in [0, 1, 2]:
        cluster_names = [str(c) for c, g2 in cluster_to_genre.items() if g2 == g]
        print(f"    Genre {g} -> cluster(s): {', '.join(cluster_names)}")
else:
    print("\n    No Movie nodes with genre found")

# --- 7. 2D embedding for visualization ---
print("\n==> Streaming node properties for visualization...")
emb2d_df = gds.graph.streamNodeProperty(G, "emb2d")
kmeans_viz_df = gds.graph.streamNodeProperty(G, "kmeans")
viz_df = emb2d_df.merge(kmeans_viz_df, on="nodeId")
viz_df["x"] = viz_df["emb2d"].apply(lambda v: v[0])
viz_df["y"] = viz_df["emb2d"].apply(lambda v: v[1])

label_map = gds.run_cypher("""
    CALL gds.graph.nodeProperties.stream('imdb', 'plot_keywords')
    YIELD nodeId, nodeLabels
    RETURN nodeId, nodeLabels[0] AS label
""")
viz_df = viz_df.merge(label_map, on="nodeId", how="left")

# --- 7. Plot ---
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle(f"IMDB K-Means (k={best_k}, balance={best_balance:.2f})", fontsize=16)

plot_df = viz_df.sample(min(2000, len(viz_df)), random_state=42) if len(viz_df) > 2000 else viz_df

ax = axes[0]
cmap_k = plt.cm.Set1(np.linspace(0, 1, max(2, best_k)))
for _, r in plot_df.iterrows():
    ax.scatter(r["x"], r["y"], c=[cmap_k[int(r["kmeans"]) % len(cmap_k)]], s=15, alpha=0.5)
ax.set_title(f"K-Means clusters (sampled {len(plot_df)})", fontsize=13)

ax = axes[1]
label_colors = {"Movie": "#e41a1c", "Actor": "#377eb8", "Director": "#4daf4a", "UnclassifiedMovie": "#984ea3"}
for label in ["Movie", "Actor", "Director", "UnclassifiedMovie"]:
    subset = plot_df[plot_df["label"] == label]
    if len(subset) > 0:
        ax.scatter(subset["x"], subset["y"], c=label_colors[label], s=15, alpha=0.5, label=label)
ax.set_title("Node labels", fontsize=13)
ax.legend(markerscale=3)

for ax in axes:
    ax.set_xlabel("FastRP dim-1")
    ax.set_ylabel("FastRP dim-2")

plt.tight_layout()
img_path = output_path("images", "imdb_kmeans.png")
plt.savefig(img_path, dpi=150, bbox_inches="tight")
print(f"\n==> Saved: {img_path}")

# --- 8. Summary ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Dataset: IMDB ({G.node_count():,} nodes, {G.relationship_count():,} edges)")
print(f"Method: FastRP (128-dim) -> K-Means")
print(f"Optimal k: {best_k}")
if len(labels_df) > 0:
    print(f"Genre prediction accuracy (Movie): {acc:.2%}")

gds.close()
print("Done.")
