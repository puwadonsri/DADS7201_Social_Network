"""
SET50 - Stakeholders Social Network (Streamlit)
DADS7201 / Week 1 - Ch1 Introduction
"""
from __future__ import annotations
import math
from pathlib import Path

import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network


HERE = Path(__file__).parent
CSV = HERE / "set50_stakeholders.csv"


def _configure_thai_font() -> str:
    """Pick a font that supports Thai glyphs.

    Preference order:
      1. **Bundled** `fonts/Sarabun-Regular.ttf` in the repo — works on
         every platform (Windows, macOS, Linux, Streamlit Cloud) because
         we register it with matplotlib directly. This is the only way
         to be sure Thai renders on the deployed app, where system-font
         discovery via `fonts-thai-tlwg` is unreliable.
      2. **System** fonts (Tahoma on Windows, TLWG on Linux, etc.) as
         a fallback for environments without the bundle.

    Returns the chosen font name so callers can pass it explicitly to
    `nx.draw_networkx_labels`, which otherwise overrides our rcParams.
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
        "Tahoma",             # Windows
        "Leelawadee UI", "Leelawadee",
        "Sukhumvit Set", "Krungthep",                    # macOS
        "TH Sarabun New", "TH SarabunPSK",               # Common Thai
        "Sarabun", "Noto Sans Thai", "Norasi", "Loma",   # Linux (fonts-tlwg)
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

# Stakeholders that appear in many SET50 companies and tend to dominate
# the layout. Matches the names as scraped from set.or.th (Thai + English).
UBIQUITOUS = {
    "บริษัท  ไทยเอ็นวีดีอาร์ จำกัด",                       # Thai NVDR
    "สำนักงานประกันสังคม",                                  # Social Security Office
    "กองทุนรวม  วายุภักษ์หนึ่ง",                            # Vayupak Fund 1
    "SOUTH EAST ASIA UK (TYPE C) NOMINEES LIMITED",
    "STATE STREET EUROPE LIMITED",
}

SET_VERIFY_URL = (
    "https://www.set.or.th/en/market/product/stock/quote/"
    "{ticker}/major-shareholders"
)


def set_url(ticker: str) -> str:
    """Build the official SET 'Major Shareholders' page URL for a ticker."""
    return SET_VERIFY_URL.format(ticker=ticker)


# ---------------------------------------------------------------- page --
st.set_page_config(
    page_title="SET50 Stakeholder Network",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------------- data --
@st.cache_data
def load_edges() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["company"] = df["company"].str.strip()
    df["stakeholder"] = df["stakeholder"].str.strip()
    return df


def filter_edges(df: pd.DataFrame,
                 sectors: list[str],
                 min_pct: float,
                 hide_ubiquitous: bool) -> pd.DataFrame:
    out = df.copy()
    if sectors:
        out = out[out["sector"].isin(sectors)]
    out = out[out["pct"] >= min_pct]
    if hide_ubiquitous:
        out = out[~out["stakeholder"].isin(UBIQUITOUS)]
    return out


# --------------------------------------------------------------- graph --
def build_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, r in df[["company", "company_name", "sector"]].drop_duplicates().iterrows():
        G.add_node(r["company"], kind="company",
                   name=r["company_name"], sector=r["sector"],
                   color=SECTOR_COLOR.get(r["sector"], "#999999"))
    for s in df["stakeholder"].unique():
        G.add_node(s, kind="stakeholder", color=STAKEHOLDER_COLOR)
    for _, r in df.iterrows():
        G.add_edge(r["company"], r["stakeholder"], weight=float(r["pct"]))
    return G


@st.cache_data
def compute_metrics(_G: nx.Graph) -> pd.DataFrame:
    deg = dict(_G.degree())
    wdeg = dict(_G.degree(weight="weight"))
    # Eigenvector centrality only defined on a connected graph; with real
    # data the SET50 bipartite often splits into several components, so
    # compute on the giant component and leave outliers as NaN.
    if nx.is_connected(_G):
        try:
            eig = nx.eigenvector_centrality_numpy(_G)
        except Exception:
            eig = {n: float("nan") for n in _G.nodes}
    else:
        try:
            giant = max(nx.connected_components(_G), key=len)
            eig_sub = nx.eigenvector_centrality_numpy(_G.subgraph(giant))
            eig = {n: eig_sub.get(n, float("nan")) for n in _G.nodes}
        except Exception:
            eig = {n: float("nan") for n in _G.nodes}
    btw = nx.betweenness_centrality(_G)
    rows = []
    for n, d in _G.nodes(data=True):
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


# ----------------------------------------------------------- static viz --
def _short_label(s: str, max_chars: int = 24) -> str:
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def _layout_components(G: nx.Graph) -> dict:
    """Spring layout per connected component, giant centred + small ones offset.

    Uses high `k` and many iterations to push leaves outward so hubs don't
    overlap them. Small components are anchored to the right of the giant's
    bounding box dynamically.
    """
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    giant_sub = G.subgraph(comps[0])
    n_giant = max(giant_sub.number_of_nodes(), 1)
    pos = nx.spring_layout(
        giant_sub,
        k=3.0 / math.sqrt(n_giant),
        iterations=500,
        seed=42,
        scale=2.0,
    )
    if len(comps) == 1:
        return pos

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


def draw_static(G: nx.Graph):
    fig, ax = plt.subplots(figsize=(18, 16))
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No nodes after filtering", ha="center", va="center")
        ax.set_axis_off()
        return fig

    pos = _layout_components(G)

    companies    = [n for n, d in G.nodes(data=True) if d["kind"] == "company"]
    stakeholders = [n for n, d in G.nodes(data=True) if d["kind"] == "stakeholder"]
    deg = dict(G.degree())

    weights = [G.edges[e]["weight"] for e in G.edges]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cccccc", alpha=0.35,
                           width=[0.3 + 0.025 * w for w in weights])

    nx.draw_networkx_nodes(
        G, pos, nodelist=companies, ax=ax,
        node_size=[420 + 40 * deg[n] for n in companies],
        node_color=[G.nodes[n]["color"] for n in companies],
        edgecolors="black", linewidths=0.6, alpha=0.95,
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=stakeholders, ax=ax,
        node_size=[260 + 320 * math.log(1 + deg[n]) for n in stakeholders],
        node_color=STAKEHOLDER_COLOR,
        edgecolors="black", linewidths=0.6, alpha=0.80, node_shape="s",
    )

    nx.draw_networkx_labels(G, pos, {n: n for n in companies},
                            font_size=8, font_weight="bold",
                            font_family=THAI_FONT, ax=ax)
    nx.draw_networkx_labels(G, pos,
                            {n: _short_label(n) for n in stakeholders if deg[n] >= 3},
                            font_size=8, font_color="#330000",
                            font_family=THAI_FONT, ax=ax)

    legend = [Line2D([0], [0], marker="o", linestyle="",
                     markerfacecolor=c, markeredgecolor="black",
                     markersize=10, label=s)
              for s, c in SECTOR_COLOR.items()]
    legend.append(Line2D([0], [0], marker="s", linestyle="",
                         markerfacecolor=STAKEHOLDER_COLOR,
                         markeredgecolor="black", markersize=10,
                         label="Stakeholder"))
    ax.legend(handles=legend, loc="lower left", fontsize=9, frameon=True)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


# ------------------------------------------------------------- pyvis viz --
def build_pyvis_html(G: nx.Graph, show_all_stakeholder_labels: bool = False) -> str:
    net = Network(height="780px", width="100%",
                  bgcolor="#ffffff", font_color="#222",
                  notebook=False, directed=False, cdn_resources="in_line")
    deg = dict(G.degree())
    for n, d in G.nodes(data=True):
        if d["kind"] == "company":
            title = (f"<b>{n}</b> — {d.get('name','')}<br>"
                     f"Sector: {d.get('sector','')}<br>"
                     f"Degree: {deg[n]}")
            # Always show company ticker label, bold, with white stroke
            net.add_node(
                n, label=n, title=title, color=d["color"],
                shape="dot", size=14 + 2 * deg[n],
                group=d.get("sector", ""),
                font={"size": 14, "face": "arial", "bold": True,
                      "color": "#111111",
                      "strokeWidth": 4, "strokeColor": "#ffffff"},
            )
        else:
            is_hub = deg[n] >= 3 or show_all_stakeholder_labels
            title = (f"<b>{n}</b><br>Stakeholder<br>"
                     f"Holds in {deg[n]} companies")
            # Hub stakeholders: bold dark label. Non-hubs: tiny grey label
            # (so they don't clutter but you can still spot them on hover).
            font_cfg = (
                {"size": 14, "face": "arial", "bold": True,
                 "color": "#330000",
                 "strokeWidth": 4, "strokeColor": "#ffffff"}
                if is_hub else
                {"size": 9, "face": "arial",
                 "color": "#888888",
                 "strokeWidth": 2, "strokeColor": "#ffffff"}
            )
            net.add_node(
                n, label=n, title=title, color=STAKEHOLDER_COLOR,
                shape="square", size=12 + 2 * deg[n],
                group="Stakeholder",
                font=font_cfg,
            )
    for u, v, data in G.edges(data=True):
        w = data["weight"]
        net.add_edge(u, v, value=w,
                     title=f"{w:.2f}% ownership", color="#bbbbbb")

    net.set_options("""
    {
      "nodes": {
        "font": { "size": 14, "face": "arial", "strokeWidth": 3, "strokeColor": "#ffffff" }
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -22000,
          "centralGravity": 0.2,
          "springLength": 160,
          "springConstant": 0.04,
          "damping": 0.45,
          "avoidOverlap": 0.6
        },
        "minVelocity": 0.5,
        "solver": "barnesHut"
      },
      "interaction": { "hover": true, "tooltipDelay": 100 }
    }
    """)
    return net.generate_html(notebook=False)


# ================================================================ UI ==

st.title("📈 SET50 Stakeholder Social Network")
st.caption(
    "DADS7201 · Social Network Analysis · Week 1 — bipartite graph of "
    "SET50-listed companies (H1 2026 constituents) and their top-5 major shareholders."
)

with st.expander("ℹ️ Data sources & caveats", expanded=False):
    st.markdown("""
**Company list — SET50 / SET50FF Index Constituents, H1 2026**
(period: 1 Jan – 30 Jun 2026, updated 15 Dec 2025).
Source: *Stock Exchange of Thailand* — official PDF.
✅ All 50 tickers and sector classifications match the official list verbatim.

**Shareholder data — scraped live from set.or.th via Selenium**
(see `set50_companies.py`). For each ticker we open
`set.or.th/th/market/product/stock/quote/{symbol}/major-shareholders`
in a real Chrome session — which bypasses Imperva Incapsula's JS
challenge — and read the top-5 rows of the Major Shareholders table.
- Names are returned **in Thai** (as published by SET); custodians and
  foreign nominees keep their original English/SGX naming.
- Percentages reflect the **latest book-closing date** SET has on file
  per ticker — there is no global "as of" date, so figures across
  tickers may correspond to slightly different snapshots.

**Verify any company** via the 🔗 link in the **📄 Data** tab — it
opens that ticker's official Major Shareholders page on set.or.th.

To refresh the data, re-run `python set50_companies.py` (needs Chrome +
chromedriver). The app reads `set50_stakeholders.csv` at startup, no
code changes needed.
""")

df_all = load_edges()

# -------- sidebar
st.sidebar.header("Filters")

all_sectors = sorted(df_all["sector"].unique().tolist())
sectors = st.sidebar.multiselect(
    "Sector", options=all_sectors, default=all_sectors,
    help="Filter SET50 companies by SET industry group."
)

min_pct = st.sidebar.slider(
    "Minimum ownership %", min_value=0.0, max_value=20.0,
    value=0.0, step=0.5,
    help="Keep edges where the stakeholder owns at least this much.",
)

hide_ubiquitous = st.sidebar.checkbox(
    "Hide ubiquitous holders",
    value=False,
    help=("Drop Thai NVDR, SSO, SEA UK Nominees, State Street. "
          "These appear almost everywhere and dominate the layout."),
)

show_all_labels = st.sidebar.checkbox(
    "Show all stakeholder labels (interactive)",
    value=False,
    help=("On the interactive graph, show every stakeholder name. "
          "Default: only hub stakeholders (degree ≥ 3) get a bold label "
          "and the rest are dimmed — same rule as the static figure."),
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data:** representative dataset built from publicly disclosed "
    "ownership structures (56-1 reports). Percentages are illustrative — "
    "verify before academic citation."
)

# -------- filter + build
df = filter_edges(df_all, sectors, min_pct, hide_ubiquitous)
if df.empty:
    st.warning("No edges left after filtering. Loosen the filters.")
    st.stop()
G = build_graph(df)

# -------- top metrics row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Companies", df["company"].nunique())
c2.metric("Stakeholders", df["stakeholder"].nunique())
c3.metric("Edges", G.number_of_edges())
c4.metric("Density", f"{nx.density(G):.4f}")
c5.metric("Components", nx.number_connected_components(G))

# -------- tabs
tab_net, tab_static, tab_metrics, tab_data, tab_about = st.tabs(
    ["🕸 Interactive", "🖼 Static", "📊 Metrics", "📄 Data", "ℹ️ About"]
)

with tab_net:
    st.markdown(
        "Drag nodes to explore. Hover for details. "
        "Red squares = stakeholders; coloured circles = SET50 companies (by sector)."
    )
    html = build_pyvis_html(G, show_all_stakeholder_labels=show_all_labels)
    components.html(html, height=820, scrolling=True)

with tab_static:
    st.markdown("Spring-layout snapshot (matplotlib).")
    fig = draw_static(G)
    st.pyplot(fig, width="stretch")

with tab_metrics:
    metrics = compute_metrics(G)

    st.subheader("Top stakeholder hubs")
    st.caption(
        "Degree = number of SET50 companies the stakeholder appears in. "
        "Weighted degree = sum of ownership percentages across those companies."
    )
    hubs = (metrics[metrics["kind"] == "stakeholder"]
            .sort_values("degree", ascending=False)
            .head(15))
    st.dataframe(hubs[["node", "degree", "weighted_degree",
                       "betweenness", "eigenvector"]],
                 width="stretch", hide_index=True)

    st.subheader("Company centrality")
    st.caption(
        "Higher betweenness = the company sits on more shareholder-shareholder paths. "
        "Click the **Verify** link to open the official SET 'Major Shareholders' page "
        "for the latest book-closing-date figures."
    )
    comp = (metrics[metrics["kind"] == "company"]
            .sort_values("betweenness", ascending=False)
            .head(15)
            .assign(verify=lambda d: d["node"].map(set_url))
            .rename(columns={"node": "company"}))
    st.dataframe(
        comp[["company", "sector", "degree", "weighted_degree",
              "betweenness", "eigenvector", "verify"]],
        width="stretch", hide_index=True,
        column_config={
            "verify": st.column_config.LinkColumn(
                "Verify on SET",
                display_text="🔗 set.or.th",
                help="Open the official Major Shareholders page on set.or.th",
            ),
        },
    )

    st.download_button(
        "⬇ Download all metrics (CSV)",
        data=metrics.to_csv(index=False).encode("utf-8"),
        file_name="set50_metrics.csv",
        mime="text/csv",
    )

with tab_data:
    st.subheader("Verify on the Stock Exchange of Thailand")
    st.caption(
        "Click 🔗 next to any company to open its official **Major Shareholders** "
        "page on set.or.th — that is the canonical source for the latest "
        "book-closing-date ownership percentages."
    )
    companies_df = (
        df[["company", "company_name", "sector"]]
        .drop_duplicates()
        .sort_values("company")
        .assign(verify=lambda d: d["company"].map(set_url))
        .reset_index(drop=True)
    )
    st.dataframe(
        companies_df,
        width="stretch", hide_index=True,
        column_config={
            "verify": st.column_config.LinkColumn(
                "Verify on SET",
                display_text="🔗 set.or.th",
                help="Open the official Major Shareholders page on set.or.th",
            ),
        },
    )

    st.subheader("Filtered edge list")
    edges_with_link = df.assign(verify=lambda d: d["company"].map(set_url))
    st.dataframe(
        edges_with_link,
        width="stretch", hide_index=True,
        column_config={
            "verify": st.column_config.LinkColumn(
                "Verify",
                display_text="🔗",
                help="Open this company's Major Shareholders page on set.or.th",
            ),
        },
    )
    st.download_button(
        "⬇ Download filtered edges (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="set50_stakeholders_filtered.csv",
        mime="text/csv",
    )

with tab_about:
    st.markdown("""
### About this app

A bipartite **social network** for the DADS7201 SNA course (Week 1) showing
relationships between **SET50** listed companies (H1 2026 constituents) and
their **top-5 major shareholders**.

**Graph model**
- **Nodes** — 50 companies (coloured by official SET industry classification) + unique stakeholders (red squares)
- **Edges** — undirected; weight = ownership percentage
- **Structure** — bipartite (companies on one side, stakeholders on the other)

**Why graphs?** As the Ch1 PDF notes, asking "which SET50 companies share
controlling families?" is a multi-join headache in SQL but a one-hop graph
traversal — exactly the use case that motivates SNA.

**Sources**
- *Company list:* SET50 / SET50FF Index Constituents, H1 2026 (Stock Exchange of Thailand)
- *Shareholders:* scraped from set.or.th via Selenium (`set50_companies.py`).
  See "Data sources & caveats" expander at the top of the app.

**Source code:** `set50_companies.py` (scraper), `build_set50_network.py`
(standalone PNG/HTML build), `streamlit_app.py` (this UI).
""")
