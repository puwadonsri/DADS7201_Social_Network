"""
Tiny visualisation helpers — no NetworkX.

- fr_layout(nodes, edges) → Fruchterman-Reingold force-directed layout
  using vectorised numpy. Fine for graphs with up to a few hundred nodes.
- draw_edges / draw_nodes / draw_hull → opinionated matplotlib drawing
  defaults that look better than ax.plot/ax.scatter out of the box.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.spatial import ConvexHull


# Curated palette (ColorBrewer Set2 + extras) — high contrast, colourblind-safe
PALETTE = [
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
]


def _connected_components(nodes, edges):
    """Return list of sets (one per connected component) using union-find."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    groups: dict[str, list[str]] = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return [set(g) for g in groups.values()]


def _fr_single(nodes, edges, iterations: int, seed: int, k_scale: float):
    """FR layout for a single (connected) component, returns pos in [-1, 1]^2."""
    rng = np.random.default_rng(seed)
    n = len(nodes)
    idx = {name: i for i, name in enumerate(nodes)}

    seen = set()
    pairs = []
    for a, b in edges:
        if a not in idx or b not in idx or a == b:
            continue
        key = (min(idx[a], idx[b]), max(idx[a], idx[b]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    e = np.asarray(pairs, dtype=int) if pairs else np.empty((0, 2), dtype=int)

    pos = rng.uniform(0.0, 1.0, (n, 2))
    if n == 1:
        return {nodes[0]: (0.0, 0.0)}

    W = 1.0
    k = k_scale * np.sqrt(W * W / n)
    temp = W / 10.0
    dt = temp / (iterations + 1)
    eps = 1e-6

    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, np.inf)
        dist = np.clip(dist, eps, None)
        rep = (k * k) / dist
        disp = (delta / dist[..., None]) * rep[..., None]
        disp = disp.sum(axis=1)

        if len(e):
            d = pos[e[:, 0]] - pos[e[:, 1]]
            dl = np.clip(np.linalg.norm(d, axis=1), eps, None)
            f = (dl * dl) / k
            unit = d / dl[:, None]
            np.add.at(disp, e[:, 0], -unit * f[:, None])
            np.add.at(disp, e[:, 1], +unit * f[:, None])

        disp_norm = np.clip(np.linalg.norm(disp, axis=1), eps, None)
        step = np.minimum(disp_norm, temp)
        pos += (disp / disp_norm[:, None]) * step[:, None]
        temp = max(temp - dt, 0.0)

    # centre each component to fill [-1, 1]^2
    pos -= (pos.max(axis=0) + pos.min(axis=0)) / 2.0
    half = max(np.abs(pos).max(), eps)
    pos = pos / half
    return {name: (float(pos[idx[name], 0]), float(pos[idx[name], 1])) for name in nodes}


def fr_layout(nodes, edges, iterations: int = 120, seed: int = 42, k_scale: float = 1.0):
    """Fruchterman-Reingold force-directed layout. Vectorised, no NetworkX.

    Connected components are laid out independently (otherwise disconnected
    components would each collapse to a single point because nothing attracts
    them across the gap) and then tiled into the same [-1, 1] canvas.

    Args:
        nodes: ordered list of node names.
        edges: iterable of (a, b) names. Self-loops + duplicates dropped.
        iterations: 100–200 is plenty for ≤ 200 nodes per component.
        seed: RNG seed for reproducibility.
        k_scale: ↑ for more whitespace, ↓ to pack nodes tighter.
    """
    comps = sorted(_connected_components(nodes, edges), key=len, reverse=True)
    if len(comps) <= 1:
        return _fr_single(list(nodes), list(edges), iterations, seed, k_scale)

    # Lay out each component on its own [-1, 1]^2 frame.
    sub_pos = []
    for i, comp in enumerate(comps):
        sub_nodes = [n for n in nodes if n in comp]
        sub_edges = [(a, b) for a, b in edges if a in comp and b in comp]
        sub_pos.append(_fr_single(sub_nodes, sub_edges, iterations, seed + i, k_scale))

    # Tile horizontally. Width is proportional to sqrt(size) so the big
    # component gets more breathing room than the small one. Each component
    # uses the full vertical canvas ([-1, 1] in y) so we don't waste space.
    weights = [max(np.sqrt(len(c)), 0.5) for c in comps]
    total = sum(weights)
    out: dict = {}
    gutter = 0.18
    usable = 2.0 - gutter * (len(comps) - 1)
    cursor = -1.0
    for w, sp in zip(weights, sub_pos):
        width = usable * (w / total)
        x_half = width / 2.0
        cx = cursor + x_half
        for name, (x, y) in sp.items():
            out[name] = (cx + x * x_half, y * 1.0)  # full vertical span
        cursor += width + gutter

    arr = np.array([out[n] for n in nodes])
    arr -= (arr.max(axis=0) + arr.min(axis=0)) / 2.0
    return {n: (float(arr[i, 0]), float(arr[i, 1])) for i, n in enumerate(nodes)}


def draw_edges(ax, pos, edges, *, alpha: float = 0.25, color: str = "#666666", lw: float = 0.7):
    """Draw all edges as thin grey segments (use highlight_edges separately for emphasis)."""
    for a, b in edges:
        if a in pos and b in pos:
            x1, y1 = pos[a]
            x2, y2 = pos[b]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=alpha, zorder=1)


def highlight_edges(ax, pos, edges, *, color: str = "#d62728", lw: float = 3.0, alpha: float = 0.9):
    for a, b in edges:
        if a in pos and b in pos:
            x1, y1 = pos[a]
            x2, y2 = pos[b]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=alpha, zorder=3)


def draw_nodes(ax, pos, names, *, sizes, colors, edgecolor="#222222", edge_lw=0.6, alpha=0.95):
    """Plot nodes as filled circles ordered by size (largest on top)."""
    arr_names = list(names)
    arr_pos = np.array([pos[n] for n in arr_names])
    sizes = np.asarray(sizes, dtype=float)
    if isinstance(colors, str):
        colors = [colors] * len(arr_names)
    # draw smallest first so highlights pop
    order = np.argsort(sizes)
    ax.scatter(
        arr_pos[order, 0], arr_pos[order, 1],
        s=sizes[order], c=[colors[i] for i in order],
        edgecolors=edgecolor, linewidths=edge_lw,
        alpha=alpha, zorder=4,
    )


def annotate_top(ax, pos, names, scores, *, top_k: int = 3, fontsize: int = 11, color: str = "#111"):
    """Bold-label the top-k nodes; everyone else gets a small label."""
    paired = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
    top_set = {n for n, _ in paired[:top_k]}
    for name in names:
        x, y = pos[name]
        if name in top_set:
            ax.annotate(
                name, (x, y), xytext=(0, 12), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#666666", lw=0.5, alpha=0.9),
                zorder=10,
            )
        else:
            ax.text(x, y - 0.045, name, ha="center", va="top",
                    fontsize=7, color="#444", alpha=0.85, zorder=5)


def draw_hull(ax, points, *, color: str, alpha: float = 0.12, lw: float = 1.4):
    """Draw a filled convex hull around a set of 2D points (community boundary)."""
    pts = np.asarray(points)
    if len(pts) < 3:
        # for 1–2 points, draw a soft circle instead
        if len(pts) == 0:
            return
        cx, cy = pts.mean(axis=0)
        r = 0.08 if len(pts) == 1 else max(np.linalg.norm(pts[1] - pts[0]) / 2 + 0.05, 0.08)
        circle = plt.Circle((cx, cy), r, color=color, alpha=alpha, zorder=0, lw=lw, ec=color)
        ax.add_patch(circle)
        return
    try:
        hull = ConvexHull(pts)
    except Exception:
        return
    # expand hull slightly so nodes sit inside it
    cx, cy = pts.mean(axis=0)
    poly = pts[hull.vertices]
    poly = np.array([(cx + (px - cx) * 1.15, cy + (py - cy) * 1.15) for px, py in poly])
    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha, zorder=0)
    ax.plot(
        np.append(poly[:, 0], poly[0, 0]),
        np.append(poly[:, 1], poly[0, 1]),
        color=color, lw=lw, alpha=min(alpha * 4, 0.6), zorder=0,
    )


def style_axes(ax, title: str, subtitle: str | None = None):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16, color="#222")
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=10, color="#666", style="italic")
    ax.axis("off")
    ax.set_aspect("equal")
    pad = 0.18
    ax.set_xlim(-1 - pad, 1 + pad)
    ax.set_ylim(-1 - pad, 1 + pad)


def colormap_for(metric: str):
    """Perceptually uniform colormaps per metric type."""
    return {
        "Betweenness": plt.cm.viridis,
        "Closeness": plt.cm.cividis,
        "Degree": plt.cm.plasma,
        "Eigenvector": plt.cm.magma,
        "PageRank": plt.cm.inferno,
    }.get(metric, plt.cm.viridis)
