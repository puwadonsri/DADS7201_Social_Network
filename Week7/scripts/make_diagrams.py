"""
Generate 6 illustrative PNGs for the Week7 README.

Nothing here retrains the model — the loss curve and split counts come
straight from the notebook's Colab run so the picture matches the
numbers cited in the README. The bipartite / neighbor / architecture
diagrams are hand-drawn mock-ups to explain the concept, NOT the actual
610 × 9,742 MovieLens graph (that would render as a black blob).
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "images")
os.makedirs(OUT, exist_ok=True)

TEAL = "#66c2a5"
ORANGE = "#fc8d62"
PURPLE = "#8da0cb"
YELLOW = "#ffd92f"
PINK = "#e78ac3"
GRAY = "#b3b3b3"


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="#fafbfc")
    plt.close(fig)
    print(f"saved {p}")


# ------------------------------------------------------------------
# 1. Bipartite user ↔ movie mock — labels + genre one-hot chip
# ------------------------------------------------------------------
def bipartite_graph():
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#fafbfc")
    users = [f"u{i}" for i in range(1, 6)]
    movies = ["Toy Story", "Jumanji", "Heat", "Casino", "GoldenEye"]
    edges = [(0, 0), (0, 1), (1, 0), (1, 4), (2, 2), (2, 3),
             (3, 2), (3, 3), (3, 4), (4, 1), (4, 4)]
    y_users, y_movies = 1.0, 0.0
    x_users = np.linspace(0.15, 0.85, len(users))
    x_movies = np.linspace(0.10, 0.90, len(movies))

    # edges
    for u, m in edges:
        ax.plot([x_users[u], x_movies[m]], [y_users, y_movies],
                color=GRAY, lw=1.3, alpha=0.65, zorder=1)

    # user nodes
    for x, u in zip(x_users, users):
        ax.scatter(x, y_users, s=1400, color=TEAL, edgecolors="#2a2a2a",
                   linewidths=0.9, zorder=3)
        ax.text(x, y_users, u, ha="center", va="center",
                fontsize=11, fontweight="bold", zorder=4)

    # movie nodes (with 4-slot genre chip)
    genres = ["Adv", "Chd", "Fnt", "Cmy"]
    movie_genres = [
        [1, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0],
        [0, 0, 0, 0], [0, 0, 0, 0],
    ]
    for x, m, g in zip(x_movies, movies, movie_genres):
        ax.scatter(x, y_movies, s=1600, color=ORANGE, edgecolors="#2a2a2a",
                   linewidths=0.9, zorder=3)
        ax.text(x, y_movies, m[:5], ha="center", va="center",
                fontsize=8.5, fontweight="bold", zorder=4)
        # genre chip below
        for i, (name, val) in enumerate(zip(genres, g)):
            r = Rectangle((x - 0.045 + i * 0.023, -0.18), 0.02, 0.04,
                          facecolor="#333" if val else "#eee",
                          edgecolor="#666", lw=0.4)
            ax.add_patch(r)
            ax.text(x - 0.035 + i * 0.023, -0.22, name,
                    ha="center", va="top", fontsize=6, color="#555")

    # labels
    ax.text(-0.04, y_users, "user\n(node_id only)", ha="right", va="center",
            fontsize=10, style="italic", color="#444")
    ax.text(-0.04, y_movies, "movie\n(20-dim genre)", ha="right", va="center",
            fontsize=10, style="italic", color="#444")
    ax.text(0.5, 0.55, "rates", ha="center", va="center",
            fontsize=10, style="italic", color=GRAY,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY, lw=0.6))
    ax.set_title("MovieLens as a heterogeneous bipartite graph",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_xlim(-0.15, 1.05)
    ax.set_ylim(-0.3, 1.3)
    ax.axis("off")

    fig.text(0.5, 0.02,
             "Real graph: 610 users × 9,742 movies × 100,836 rate edges "
             "(diagram shows a hand-picked 5×5 subset).",
             ha="center", fontsize=9.5, color="#666", style="italic")
    save(fig, "01_bipartite.png")


# ------------------------------------------------------------------
# 2. RandomLinkSplit composition — exact numbers from notebook
# ------------------------------------------------------------------
def split_composition():
    """
    Show two views side by side:
      (top)    positive-edge partition of the original 100,836 rate edges
      (bottom) supervision labels (positives + fixed 2:1 negatives) per split
    Numbers come straight from the notebook.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 5.6), facecolor="#fafbfc",
        gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.85},
    )

    # ---- (a) Positive-edge partition ----
    total = 100836
    train_mp = 56469
    train_sup = 24201
    val_pos = 10083
    test_pos = total - train_mp - train_sup - val_pos    # 10,083

    parts = [
        (train_mp,  "#a6d854", f"train MP · {train_mp:,}"),
        (train_sup, "#8da0cb", f"train sup+ · {train_sup:,}"),
        (val_pos,   "#c86bbf", f"val sup+ · {val_pos:,}"),
        (test_pos,  "#f2ad54", f"test sup+ · {test_pos:,}"),
    ]
    left = 0
    for value, color, label in parts:
        ax1.barh(0, value, height=0.5, left=left, color=color,
                 edgecolor="#333", linewidth=0.4)
        ax1.text(left + value / 2, 0, label, ha="center", va="center",
                 fontsize=9, fontweight="bold", color="#111")
        left += value

    ax1.set_yticks([0])
    ax1.set_yticklabels([f"100,836 rate edges"], fontsize=10)
    ax1.set_xlabel("edges", fontsize=9.5)
    ax1.set_xlim(0, total * 1.02)
    ax1.set_title(
        "(a) Positive edges — how the 100,836 rate edges are partitioned",
        fontsize=12, fontweight="bold", pad=8, loc="left",
    )
    ax1.grid(axis="x", linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)

    # ---- (b) Supervision labels per split ----
    val_neg = 20166
    test_neg = 20166
    rows = [
        ("train", [(train_sup, "#8da0cb", f"positives · {train_sup:,}"),
                   (0,          "#e78ac3", "negatives on-the-fly (2:1)")]),
        ("val",   [(val_pos,   "#8da0cb", f"positives · {val_pos:,}"),
                   (val_neg,   "#e78ac3", f"negatives · {val_neg:,}")]),
        ("test",  [(test_pos,  "#8da0cb", f"positives · {test_pos:,}"),
                   (test_neg,  "#e78ac3", f"negatives · {test_neg:,}")]),
    ]
    ys = [2, 1, 0]
    for y, (name, segs) in zip(ys, rows):
        left = 0
        for value, color, label in segs:
            if value == 0:
                ax2.text(train_sup + 500, y, label,
                         ha="left", va="center", fontsize=9,
                         style="italic", color="#a83b8a")
                continue
            ax2.barh(y, value, height=0.5, left=left, color=color,
                     edgecolor="#333", linewidth=0.4)
            ax2.text(left + value / 2, y, label, ha="center", va="center",
                     fontsize=9, fontweight="bold", color="#111")
            left += value
    ax2.set_yticks(ys)
    ax2.set_yticklabels(["train", "val", "test"], fontsize=10)
    ax2.set_xlabel("supervision labels", fontsize=9.5)
    ax2.set_xlim(0, 35000)
    ax2.set_title(
        "(b) Supervision labels — fixed 2:1 negatives on val/test, "
        "on-the-fly negatives during training",
        fontsize=12, fontweight="bold", pad=8, loc="left",
    )
    ax2.grid(axis="x", linestyle="--", alpha=0.3)
    ax2.set_axisbelow(True)

    handles = [
        mpatches.Patch(color="#a6d854", label="message-passing"),
        mpatches.Patch(color="#8da0cb", label="supervision positives"),
        mpatches.Patch(color="#e78ac3", label="supervision negatives"),
    ]
    ax2.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)

    fig.suptitle("RandomLinkSplit  (num_val=0.1, num_test=0.1, disjoint_train_ratio=0.3, neg_sampling_ratio=2.0)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.5, -0.02,
             "disjoint_train_ratio = 0.3  → 30 % of train edges are removed from message passing "
             "and become supervision labels only, preventing label-leakage into the GNN's hidden state.",
             ha="center", fontsize=9.5, color="#666", style="italic")
    save(fig, "02_split.png")


# ------------------------------------------------------------------
# 3. LinkNeighborLoader — 2-hop sampling illustration
# ------------------------------------------------------------------
def neighbor_sampling():
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#fafbfc")

    # positions
    seed_u = (0.15, 0.55)
    seed_m = (0.85, 0.55)

    # 1st hop
    h1_movies = [(0.35, 0.85), (0.35, 0.65), (0.35, 0.45), (0.35, 0.25)]  # from user
    h1_users = [(0.65, 0.85), (0.65, 0.70), (0.65, 0.30), (0.65, 0.15)]   # from movie

    # 2nd hop
    h2_users = [(0.50, 0.95), (0.50, 0.05)]

    # seed edge (highlighted)
    ax.plot([seed_u[0], seed_m[0]], [seed_u[1], seed_m[1]],
            color="#d62728", lw=3.0, alpha=0.9, zorder=4)
    ax.text(0.5, 0.58, "seed edge  (edge_label_index)",
            ha="center", va="bottom", fontsize=9, color="#d62728",
            fontweight="bold")

    # 1st-hop edges
    for m in h1_movies:
        ax.plot([seed_u[0], m[0]], [seed_u[1], m[1]],
                color="#5aa96a", lw=1.4, alpha=0.85, zorder=2)
    for u in h1_users:
        ax.plot([seed_m[0], u[0]], [seed_m[1], u[1]],
                color="#5aa96a", lw=1.4, alpha=0.85, zorder=2)

    # 2nd-hop edges (dashed / lighter)
    for h2 in h2_users:
        for m in h1_movies[:2]:
            ax.plot([m[0], h2[0]], [m[1], h2[1]],
                    color=PURPLE, lw=1.0, alpha=0.6,
                    linestyle=(0, (3, 2)), zorder=1)

    # draw nodes
    def draw(pos, color, label, size=1100):
        ax.scatter(*pos, s=size, color=color, edgecolors="#2a2a2a",
                   linewidths=0.9, zorder=5)
        ax.text(pos[0], pos[1], label, ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=6)

    draw(seed_u, TEAL, "u★", size=1500)
    draw(seed_m, ORANGE, "m★", size=1500)
    for i, p in enumerate(h1_movies):
        draw(p, ORANGE, f"m{i+1}", 900)
    for i, p in enumerate(h1_users):
        draw(p, TEAL, f"u{i+1}", 900)
    for i, p in enumerate(h2_users):
        draw(p, TEAL, f"u{i+5}", 700)

    # legends
    legend_items = [
        mpatches.Patch(color="#d62728", label="seed edge (predict this)"),
        mpatches.Patch(color="#5aa96a", label="1st-hop neighbors  (num_neighbors[0] = 20)"),
        mpatches.Patch(color=PURPLE, label="2nd-hop neighbors  (num_neighbors[1] = 10)"),
    ]
    ax.legend(handles=legend_items, loc="lower center",
              bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=9)

    ax.set_title(
        "LinkNeighborLoader — 2-hop subgraph around each seed edge",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.axis("off")
    save(fig, "03_neighbor_sampling.png")


# ------------------------------------------------------------------
# 4. Model architecture flow
# ------------------------------------------------------------------
def model_architecture():
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="#fafbfc")

    def box(x, y, w, h, label, color, edge="#2a2a2a", fs=10):
        p = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
            fc=color, ec=edge, lw=1.2,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="#111", wrap=True)

    def arrow(a, b, style="->"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style,
                                     lw=1.4, color="#333",
                                     mutation_scale=14))

    # Input row (left column)
    box(0.02, 0.78, 0.20, 0.14, "user.node_id\n[610]", TEAL, fs=9.5)
    box(0.02, 0.55, 0.20, 0.14, "movie.x\n[9742, 20]  (genre)", ORANGE, fs=9.5)
    box(0.02, 0.32, 0.20, 0.14, "movie.node_id\n[9742]", ORANGE, fs=9.5)

    # 1st transform
    box(0.28, 0.78, 0.20, 0.14, "Embedding(610, 64)", "#e2f0e6", fs=9.5)
    box(0.28, 0.55, 0.20, 0.14, "Linear(20 → 64)", "#fde3d8", fs=9.5)
    box(0.28, 0.32, 0.20, 0.14, "Embedding(9742, 64)", "#fde3d8", fs=9.5)

    # Merge box
    box(0.53, 0.55, 0.14, 0.14, "movie feat\n= Linear + Emb", "#fff2cc", fs=9)
    arrow((0.48, 0.62), (0.53, 0.62))
    arrow((0.48, 0.39), (0.53, 0.55 + 0.14 / 2), style="->")

    # GNN box
    box(0.72, 0.60, 0.24, 0.30,
        "to_hetero(\n  SAGEConv(64→64) · ReLU\n  · SAGEConv(64→64)\n)",
        "#dfe6f0", fs=10)
    arrow((0.48, 0.85), (0.72, 0.83))
    arrow((0.67, 0.62), (0.72, 0.72))

    # Classifier
    box(0.72, 0.30, 0.24, 0.16,
        "Classifier:\ndot(user_h, movie_h)",
        "#fce6ec", fs=10)
    arrow((0.84, 0.60), (0.84, 0.46))

    # Output
    box(0.72, 0.09, 0.24, 0.14, "logit\n(one per candidate edge)",
        "#ffe1e6", fs=10)
    arrow((0.84, 0.30), (0.84, 0.23))

    # Loss
    box(0.30, 0.09, 0.36, 0.14,
        "BCEWithLogitsLoss  ·  Adam(lr=1e-3)  ·  5 epochs",
        "#eef1f4", fs=10)
    arrow((0.72, 0.16), (0.66, 0.16))

    # Section labels
    ax.text(0.12, 0.99, "Inputs", ha="center", va="top",
            fontsize=11, fontweight="bold", color="#555")
    ax.text(0.38, 0.99, "Embedding layer", ha="center", va="top",
            fontsize=11, fontweight="bold", color="#555")
    ax.text(0.60, 0.99, "Merge", ha="center", va="top",
            fontsize=11, fontweight="bold", color="#555")
    ax.text(0.84, 0.99, "Heterogeneous GNN", ha="center", va="top",
            fontsize=11, fontweight="bold", color="#555")

    ax.set_title("Model — heterogeneous GraphSAGE + dot-product classifier",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.axis("off")

    fig.text(0.5, -0.02,
             "to_hetero(...) auto-duplicates the SAGEConv layers per edge type "
             "((user,rates,movie) and (movie,rev_rates,user)).",
             ha="center", fontsize=9.5, color="#666", style="italic")
    save(fig, "04_architecture.png")


# ------------------------------------------------------------------
# 5. Training loss curve (real numbers)
# ------------------------------------------------------------------
def training_loss():
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor="#fafbfc")
    epochs = [1, 2, 3, 4, 5]
    loss = [0.4360, 0.3466, 0.3261, 0.3109, 0.2987]

    ax.plot(epochs, loss, "o-", color="#3a83c3", lw=2.2,
            markersize=9, markerfacecolor="white", markeredgewidth=2)
    for e, l in zip(epochs, loss):
        ax.text(e, l + 0.008, f"{l:.4f}", ha="center", va="bottom",
                fontsize=9.5, color="#111", fontweight="bold")

    ax.fill_between(epochs, loss, [min(loss) - 0.03] * len(epochs),
                    color="#3a83c3", alpha=0.08)
    ax.axhline(loss[-1], color="#5aa96a", linestyle="--", alpha=0.55,
               label=f"final = {loss[-1]:.4f}")

    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("BCE loss (train)", fontsize=11)
    ax.set_title("Training loss — 5 epochs, batch 128",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(min(loss) - 0.03, max(loss) + 0.04)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    fig.text(0.5, -0.02,
             "Loss drops 31.5 % from epoch 1 → 5.  "
             "Final validation AUC on 30,249 held-out edges = 0.9309.",
             ha="center", fontsize=9.5, color="#666", style="italic")
    save(fig, "05_training_loss.png")


# ------------------------------------------------------------------
# 6. AUC = 0.9309 — synthetic ROC curve consistent with that AUC
# ------------------------------------------------------------------
def roc_curve():
    """
    Draw an ROC curve whose area matches the reported AUC (0.9309).
    We can't recover the true curve without model outputs — this is a
    visualisation of what "AUC = 0.9309" means (curve pulled toward the
    top-left corner).  The tpr formula tpr = fpr^gamma with gamma = 1 - AUC
    gives roughly the right area, then we scale to match exactly.
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.4), facecolor="#fafbfc")

    target_auc = 0.9309
    fpr = np.linspace(0, 1, 400)
    # gamma solves ∫₀¹ fpr^γ dfpr = 1/(γ+1) = target_auc
    gamma = 1.0 / target_auc - 1.0
    tpr = fpr ** gamma

    ax.plot(fpr, tpr, color="#3a83c3", lw=2.6,
            label=f"heterogeneous GraphSAGE   AUC = {target_auc:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#3a83c3")
    ax.plot([0, 1], [0, 1], color="#888", ls="--", lw=1.2,
            label="random baseline           AUC = 0.5000")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC on the validation set", fontsize=13,
                 fontweight="bold", pad=10)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_aspect("equal")

    fig.text(0.5, -0.02,
             "Curve is a smooth reconstruction that matches the notebook's "
             "reported AUC — actual model outputs weren't retained.",
             ha="center", fontsize=9, color="#666", style="italic")
    save(fig, "06_roc.png")


if __name__ == "__main__":
    bipartite_graph()
    split_composition()
    neighbor_sampling()
    model_architecture()
    training_loss()
    roc_curve()
    print("\nAll 6 diagrams saved to", OUT)
