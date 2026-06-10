"""
SET50 - Stakeholders Social Network
DADS7201 / Week 1 - Ch1 Introduction (Ekarat Rattagan)

Builds a bipartite graph where one set of nodes is SET50 listed companies
and the other set is their top-5 major shareholders. Edges are weighted
by ownership percentage. Outputs:

  - set50_network.png          : static matplotlib figure (sector-coloured)
  - set50_network.html         : interactive pyvis figure
  - set50_metrics.csv          : per-node centrality metrics
  - set50_top_hubs.txt         : ranked summary of biggest stakeholder hubs
"""

from __future__ import annotations
from pathlib import Path
import math

import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pyvis.network import Network


HERE = Path(__file__).parent
CSV = HERE / "set50_stakeholders.csv"


def _configure_thai_font() -> str:
    """Pick a font that supports Thai glyphs.

    Prefers a bundled `fonts/Sarabun-Regular.ttf` (registered directly with
    matplotlib so it works on every platform). Falls back to system fonts
    if the bundle isn't present.
    """
    bundled = HERE / "fonts" / "Sarabun-Regular.ttf"
    if bundled.exists():
        matplotlib.font_manager.fontManager.addfont(str(bundled))
        plt.rcParams["font.family"] = ["Sarabun", "DejaVu Sans"]
        plt.rcParams["font.sans-serif"] = (
            ["Sarabun"] + plt.rcParams["font.sans-serif"]
        )
        return "Sarabun"

    candidates = [
        "Tahoma", "Leelawadee UI", "Leelawadee",
        "Sukhumvit Set", "Krungthep",
        "TH Sarabun New", "TH SarabunPSK",
        "Sarabun", "Noto Sans Thai", "Norasi", "Loma",
    ]
    installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    chosen = next((n for n in candidates if n in installed), "DejaVu Sans")
    plt.rcParams["font.family"] = [chosen, "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = [chosen] + plt.rcParams["font.sans-serif"]
    return chosen


THAI_FONT = _configure_thai_font()


# Official SET sector palette (matches SET50 H1 2026 industry classification).
SECTOR_COLOR = {
    "Banking":                                "#1f77b4",
    "Finance & Securities":                   "#aec7e8",
    "Insurance":                              "#f7b6d2",
    "Energy & Utilities":                     "#ff7f0e",
    "Petrochemicals & Chemicals":             "#c49c94",
    "Commerce":                               "#2ca02c",
    "Food & Beverage":                        "#bcbd22",
    "Tourism & Leisure":                      "#ff9896",
    "Health Care Services":                   "#98df8a",
    "Information & Communication Technology": "#9467bd",
    "Electronic Components":                  "#8c564b",
    "Property Development":                   "#e377c2",
    "Construction Materials":                 "#7f7f7f",
    "Packaging":                              "#c5b0d5",
    "Transportation & Logistics":             "#17becf",
}
STAKEHOLDER_COLOR = "#d62728"


# ------------------------------------------------------------------ load --
def load_edges() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["company"] = df["company"].str.strip()
    df["stakeholder"] = df["stakeholder"].str.strip()
    return df


# ----------------------------------------------------------------- graph --
def build_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    companies = df[["company", "company_name", "sector"]].drop_duplicates()
    for _, r in companies.iterrows():
        G.add_node(
            r["company"],
            kind="company",
            label=r["company"],
            name=r["company_name"],
            sector=r["sector"],
            color=SECTOR_COLOR.get(r["sector"], "#999999"),
        )

    for s in df["stakeholder"].unique():
        G.add_node(
            s,
            kind="stakeholder",
            label=s,
            color=STAKEHOLDER_COLOR,
        )

    for _, r in df.iterrows():
        G.add_edge(r["company"], r["stakeholder"], weight=float(r["pct"]))

    return G


# --------------------------------------------------------------- metrics --
def compute_metrics(G: nx.Graph) -> pd.DataFrame:
    deg = dict(G.degree())
    wdeg = dict(G.degree(weight="weight"))
    # Centrality measures useful for SNA Ch1.
    btw = nx.betweenness_centrality(G, weight=None)
    # Eigenvector centrality requires a connected graph. With real-world
    # data the bipartite SET50 graph can split into several components
    # (e.g. a company whose top-5 holders are all unique to it). Compute
    # eigenvector on the giant component only; nodes in smaller
    # components get NaN.
    if nx.is_connected(G):
        eig = nx.eigenvector_centrality_numpy(G)
    else:
        giant = max(nx.connected_components(G), key=len)
        sub = G.subgraph(giant)
        eig_sub = nx.eigenvector_centrality_numpy(sub)
        eig = {n: eig_sub.get(n, float("nan")) for n in G.nodes}

    rows = []
    for n, d in G.nodes(data=True):
        rows.append({
            "node": n,
            "kind": d["kind"],
            "sector": d.get("sector", ""),
            "degree": deg[n],
            "weighted_degree": round(wdeg[n], 2),
            "betweenness": round(btw[n], 5),
            "eigenvector": round(eig[n], 5),
        })
    return pd.DataFrame(rows).sort_values(
        ["kind", "degree", "weighted_degree"], ascending=[True, False, False]
    )


# -------------------------------------------------------------- static viz --
def _short_label(s: str, max_chars: int = 24) -> str:
    """Truncate a long node label for static-figure display only."""
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def _layout_components(G: nx.Graph) -> dict:
    """Spring-layout per connected component, with the giant one centred
    and any small components tucked into the right margin (dynamic offset
    based on the giant's bounding box). Tuned for readability — high `k`
    and many iterations push leaves outward so hubs don't overlap them.
    """
    comps = sorted(nx.connected_components(G), key=len, reverse=True)

    giant_sub = G.subgraph(comps[0])
    n_giant = max(giant_sub.number_of_nodes(), 1)
    pos = nx.spring_layout(
        giant_sub,
        k=3.0 / math.sqrt(n_giant),   # strong repulsion -> wider spread
        iterations=500,
        seed=42,
        scale=2.0,
    )
    if len(comps) == 1:
        return pos

    # Anchor extra components to the right of the giant's bounding box
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_right = max(xs) + 0.6
    y_mid = (max(ys) + min(ys)) / 2
    y_off = y_mid + 0.8
    for comp in comps[1:]:
        sub = G.subgraph(comp)
        sub_pos = nx.spring_layout(sub, k=0.5, iterations=300,
                                   seed=42, scale=0.55)
        for n, (x, y) in sub_pos.items():
            pos[n] = (x_right + x, y_off + y)
        y_off -= 1.6
    return pos


def draw_static(G: nx.Graph, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(28, 26))

    # Layout: spring per component so disconnected clusters don't drift.
    pos = _layout_components(G)

    companies     = [n for n, d in G.nodes(data=True) if d["kind"] == "company"]
    stakeholders  = [n for n, d in G.nodes(data=True) if d["kind"] == "stakeholder"]

    deg = dict(G.degree())

    # Edges — thinner + more transparent so they don't crowd the hubs
    weights = [G.edges[e]["weight"] for e in G.edges]
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#cccccc", alpha=0.35,
        width=[0.3 + 0.025 * w for w in weights],
    )

    # Company nodes: coloured by sector. Degree caps at 5 (top-5 holders)
    # so a flat-ish linear size is fine.
    nx.draw_networkx_nodes(
        G, pos, nodelist=companies, ax=ax,
        node_size=[420 + 40 * deg[n] for n in companies],
        node_color=[G.nodes[n]["color"] for n in companies],
        edgecolors="black", linewidths=0.6, alpha=0.95,
    )

    # Stakeholder nodes: red, log-scaled by degree so hubs like Thai NVDR
    # (deg=45) don't drown leaves (deg=1) under a giant blob.
    nx.draw_networkx_nodes(
        G, pos, nodelist=stakeholders, ax=ax,
        node_size=[260 + 320 * math.log(1 + deg[n]) for n in stakeholders],
        node_color=STAKEHOLDER_COLOR,
        edgecolors="black", linewidths=0.6, alpha=0.80,
        node_shape="s",
    )

    # Labels: companies always; stakeholders only if they're hubs (deg >= 3)
    company_labels    = {n: n for n in companies}
    stakeholder_label = {n: _short_label(n) for n in stakeholders if deg[n] >= 3}
    nx.draw_networkx_labels(G, pos, company_labels, font_size=8,
                            font_weight="bold", font_family=THAI_FONT, ax=ax)
    nx.draw_networkx_labels(G, pos, stakeholder_label, font_size=8,
                            font_color="#330000", font_family=THAI_FONT, ax=ax)

    # Legend
    legend = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=c, markeredgecolor="black",
               markersize=10, label=s)
        for s, c in SECTOR_COLOR.items()
    ]
    legend.append(
        Line2D([0], [0], marker="s", linestyle="",
               markerfacecolor=STAKEHOLDER_COLOR, markeredgecolor="black",
               markersize=10, label="Stakeholder")
    )
    ax.legend(handles=legend, loc="lower left", fontsize=10, frameon=True)

    ax.set_title(
        "SET50 Companies and their Top-5 Major Shareholders\n"
        "Bipartite social network — node size = degree, edge width = ownership %",
        fontsize=15,
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------- interactive --
def draw_interactive(G: nx.Graph, out: Path) -> None:
    net = Network(height="900px", width="100%",
                  bgcolor="#ffffff", font_color="#222",
                  notebook=False, directed=False, cdn_resources="in_line")

    deg = dict(G.degree())
    for n, d in G.nodes(data=True):
        if d["kind"] == "company":
            title = (f"<b>{n}</b> — {d.get('name','')}<br>"
                     f"Sector: {d.get('sector','')}<br>"
                     f"Degree: {deg[n]}")
            net.add_node(n, label=n, title=title, color=d["color"],
                         shape="dot", size=14 + 2 * deg[n], group=d.get("sector", ""))
        else:
            title = (f"<b>{n}</b><br>Stakeholder<br>"
                     f"Holds in {deg[n]} SET50 companies")
            net.add_node(n, label=n, title=title, color=STAKEHOLDER_COLOR,
                         shape="square", size=12 + 2 * deg[n], group="Stakeholder")

    for u, v, data in G.edges(data=True):
        w = data["weight"]
        net.add_edge(u, v, value=w, title=f"{w:.2f}% ownership",
                     color="#bbbbbb")

    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -18000,
          "centralGravity": 0.25,
          "springLength": 130,
          "springConstant": 0.04,
          "damping": 0.4,
          "avoidOverlap": 0.4
        },
        "minVelocity": 0.5,
        "solver": "barnesHut"
      },
      "interaction": { "hover": true, "tooltipDelay": 100 }
    }
    """)
    # pyvis save_graph uses locale encoding (cp874 on Thai Windows) and
    # chokes on characters like ©. Write the HTML ourselves as UTF-8.
    net.generate_html(name=str(out), notebook=False)
    out.write_text(net.html, encoding="utf-8")


# ----------------------------------------------------------------- main --
def main() -> None:
    df = load_edges()
    G = build_graph(df)

    print(f"Loaded {len(df)} edges / "
          f"{df['company'].nunique()} companies / "
          f"{df['stakeholder'].nunique()} unique stakeholders")
    print(f"Graph: |V|={G.number_of_nodes()}  |E|={G.number_of_edges()}  "
          f"density={nx.density(G):.4f}")
    print(f"Connected: {nx.is_connected(G)}  "
          f"components={nx.number_connected_components(G)}")
    if nx.is_connected(G):
        print(f"Diameter: {nx.diameter(G)}  "
              f"Avg shortest path: {nx.average_shortest_path_length(G):.3f}")

    metrics = compute_metrics(G)
    metrics.to_csv(HERE / "set50_metrics.csv", index=False)

    # Top stakeholder hubs
    hubs = (metrics[metrics["kind"] == "stakeholder"]
            .sort_values("degree", ascending=False)
            .head(15))
    with open(HERE / "set50_top_hubs.txt", "w", encoding="utf-8") as fh:
        fh.write("Top stakeholder hubs (by number of SET50 companies they hold)\n")
        fh.write("=" * 70 + "\n")
        for _, r in hubs.iterrows():
            fh.write(f"{r['degree']:>3}  {r['node']}\n")

    print("\nTop-10 stakeholder hubs (degree = number of SET50 holdings):")
    print(hubs[["node", "degree", "weighted_degree"]].head(10).to_string(index=False))

    draw_static(G, HERE / "set50_network.png")
    print(f"\nSaved static PNG     -> set50_network.png")

    draw_interactive(G, HERE / "set50_network.html")
    print(f"Saved interactive    -> set50_network.html")
    print(f"Saved metrics CSV    -> set50_metrics.csv")
    print(f"Saved top-hubs list  -> set50_top_hubs.txt")


if __name__ == "__main__":
    main()
