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


# Centrality definitions — Thai + English explanations shown in the
# Centrality tab. Order is the display order in the radio selector.
CENTRALITY_DEFS = {
    "Degree": {
        "en": ("Number of direct neighbours. A pure local measure — "
               "how many edges touch this node."),
        "th": ("จำนวน edge ที่ออกจาก node นี้โดยตรง — "
               "วัด \"การมีเพื่อน\" แบบไม่สนใจว่าเพื่อนเป็นใคร"),
        "col": "degree",
    },
    "Closeness": {
        "en": ("Inverse of the average shortest-path distance to all "
               "reachable nodes. High = sits near the centre of the graph."),
        "th": ("ส่วนกลับของระยะทางเฉลี่ยไปยัง node อื่นทุกตัวที่เดินถึง — "
               "ค่าสูง = อยู่ \"ใจกลาง\" กราฟ"),
        "col": "closeness",
    },
    "Betweenness": {
        "en": ("Fraction of all shortest paths that pass through this node. "
               "High = acts as a bridge / gatekeeper."),
        "th": ("สัดส่วนของเส้นทางสั้นสุดทั้งหมดที่ \"ผ่าน\" node นี้ — "
               "ค่าสูง = เป็น สะพาน/ผู้รักษาประตู ของกราฟ"),
        "col": "betweenness",
    },
    "Eigenvector": {
        "en": ("Recursive importance: connected to nodes that are themselves "
               "important. Computed on the giant connected component only."),
        "th": ("ความสำคัญแบบ recursive: เชื่อมกับ node \"คนสำคัญ\" คนอื่น "
               "(คำนวณบน giant component เท่านั้น — DELTA cluster จะเป็น NaN)"),
        "col": "eigenvector",
    },
    "Katz": {
        "en": ("Like eigenvector but every node gets a base prestige, so it "
               "works on disconnected graphs."),
        "th": ("คล้าย Eigenvector แต่ทุก node มีค่าฐานเริ่มต้น "
               "ทำให้คำนวณได้แม้กราฟไม่เชื่อมต่อ"),
        "col": "katz",
    },
    "PageRank": {
        "en": ("Steady-state probability of a random walker (with damping) "
               "landing on this node. Google's original ranking idea."),
        "th": ("ความน่าจะเป็นที่ random walker (มี damping) "
               "จะมาหยุดอยู่ที่ node นี้ในระยะยาว — algorithm ตั้งต้นของ Google"),
        "col": "pagerank",
    },
}


@st.cache_data
def compute_all_centralities(_G: nx.Graph) -> pd.DataFrame:
    """Compute six centrality measures for every node.

    Disconnected-graph handling:
    - Degree, Closeness, Betweenness, Katz, PageRank — work on the full
      graph (closeness uses the Wasserman-Faust correction).
    - Eigenvector — computed on the giant connected component only.
      Nodes in other components get NaN.
    """
    nodes = list(_G.nodes)
    deg_raw = dict(_G.degree())
    wdeg = dict(_G.degree(weight="weight"))

    # Normalised degree centrality (NetworkX convention).
    deg_c = nx.degree_centrality(_G)
    # Closeness with Wasserman-Faust correction → handles disconnected.
    clo = nx.closeness_centrality(_G, wf_improved=True)
    btw = nx.betweenness_centrality(_G)

    # Eigenvector — giant component only.
    if nx.is_connected(_G):
        sub = _G
    else:
        sub = _G.subgraph(max(nx.connected_components(_G), key=len))
    try:
        eig_sub = nx.eigenvector_centrality_numpy(sub)
        eig = {n: eig_sub.get(n, float("nan")) for n in nodes}
    except Exception:
        eig = {n: float("nan") for n in nodes}

    # Katz needs alpha < 1 / lambda_max — pick a safe alpha.
    try:
        import numpy as np
        A = nx.adjacency_matrix(_G).astype(float).todense()
        lam = float(np.max(np.abs(np.linalg.eigvals(A))).real)
        alpha = 1.0 / (lam + 1.0)
        katz = nx.katz_centrality_numpy(_G, alpha=alpha, beta=1.0)
    except Exception:
        katz = {n: float("nan") for n in nodes}

    pr = nx.pagerank(_G, alpha=0.85)

    rows = []
    for n, d in _G.nodes(data=True):
        rows.append({
            "node": n,
            "kind": d["kind"],
            "sector": d.get("sector", ""),
            "degree_raw": deg_raw[n],
            "weighted_degree": round(wdeg[n], 2),
            "degree": round(deg_c[n], 5),
            "closeness": round(clo[n], 5),
            "betweenness": round(btw[n], 5),
            "eigenvector": (round(eig[n], 5)
                            if not pd.isna(eig[n]) else float("nan")),
            "katz": round(katz[n], 5),
            "pagerank": round(pr[n], 5),
        })
    return pd.DataFrame(rows)


# Backwards-compatible alias for any code that still references compute_metrics.
compute_metrics = compute_all_centralities


# ----------------------------------------------------------- static viz --
def _short_label(s: str, max_chars: int = 24) -> str:
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def _layout_components(G: nx.Graph) -> dict:
    """Spring layout per connected component, giant centred + small ones offset.

    Uses very high `k` + `scale` so leaves push out wide and labels don't
    overlap. Small components are anchored to the right of the giant's
    bounding box dynamically.
    """
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    giant_sub = G.subgraph(comps[0])
    n_giant = max(giant_sub.number_of_nodes(), 1)
    pos = nx.spring_layout(
        giant_sub,
        k=6.0 / math.sqrt(n_giant),
        iterations=800,
        seed=42,
        scale=4.0,
    )
    if len(comps) == 1:
        return pos

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_right = max(xs) + 1.2
    y_mid = (max(ys) + min(ys)) / 2
    y_off = y_mid + 1.5
    for comp in comps[1:]:
        sub = G.subgraph(comp)
        sub_pos = nx.spring_layout(sub, k=0.7, iterations=400,
                                   seed=42, scale=0.9)
        for n, (x, y) in sub_pos.items():
            pos[n] = (x_right + x, y_off + y)
        y_off -= 2.6
    return pos


def draw_static(G: nx.Graph):
    fig, ax = plt.subplots(figsize=(22, 18))
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No nodes after filtering", ha="center", va="center")
        ax.set_axis_off()
        return fig

    pos = _layout_components(G)

    companies    = [n for n, d in G.nodes(data=True) if d["kind"] == "company"]
    stakeholders = [n for n, d in G.nodes(data=True) if d["kind"] == "stakeholder"]
    deg = dict(G.degree())

    weights = [G.edges[e]["weight"] for e in G.edges]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#888888", alpha=0.45,
                           width=[0.25 + 0.02 * w for w in weights])

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
tab_net, tab_static, tab_centrality, tab_data, tab_about = st.tabs(
    ["🕸 Interactive", "🖼 Static", "🏆 Centrality", "📄 Data", "ℹ️ About"]
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

with tab_centrality:
    metrics = compute_all_centralities(G)

    # ---- Controls -----------------------------------------------------
    ctl1, ctl2, ctl3 = st.columns([2, 1, 1])
    with ctl1:
        measure_name = st.radio(
            "Centrality measure",
            options=list(CENTRALITY_DEFS.keys()),
            horizontal=True,
            help="Each measure ranks nodes by a different definition of 'importance'.",
        )
    with ctl2:
        combine_kinds = st.checkbox(
            "Combine Companies + Stakeholders",
            value=False,
            help=("Default: rank Companies and Stakeholders separately "
                  "(recommended — companies cap at degree=5 because we "
                  "only scrape top-5 holders). Tick to rank all 170 nodes "
                  "in one list."),
        )
    with ctl3:
        top_n = st.slider("Top N", min_value=5, max_value=30, value=15, step=5)

    info = CENTRALITY_DEFS[measure_name]
    col = info["col"]

    # ---- Definition box ----------------------------------------------
    st.info(
        f"**📐 {measure_name} centrality**\n\n"
        f"**EN:**  *{info['en']}*\n\n"
        f"**TH:**  {info['th']}"
    )

    # ---- Helper: render one ranked table ------------------------------
    def render_ranking(df_sub: pd.DataFrame, title: str, *,
                       show_verify: bool, show_sector: bool):
        ranked = (df_sub.sort_values(col, ascending=False, na_position="last")
                  .head(top_n)
                  .reset_index(drop=True))
        ranked.insert(0, "rank", ranked.index + 1)
        ranked["score"] = ranked[col]

        cols_show = ["rank", "node"]
        if show_sector:
            cols_show.append("sector")
        cols_show += ["degree_raw", "weighted_degree", "score"]

        col_cfg = {
            "rank": st.column_config.NumberColumn("#", width="small"),
            "node": st.column_config.TextColumn("Node"),
            "degree_raw": st.column_config.NumberColumn(
                "Degree", help="Raw degree (number of neighbours)",
                width="small"),
            "weighted_degree": st.column_config.NumberColumn(
                "W-Degree", help="Sum of ownership % on adjacent edges",
                format="%.1f", width="small"),
            "score": st.column_config.ProgressColumn(
                f"{measure_name} score",
                help=f"{measure_name} centrality value",
                format="%.4f",
                min_value=0.0,
                max_value=float(ranked["score"].max()) if not ranked["score"].isna().all() else 1.0,
            ),
        }
        if show_sector:
            col_cfg["sector"] = st.column_config.TextColumn("Sector")
        if show_verify:
            ranked = ranked.assign(verify=ranked["node"].map(set_url))
            cols_show.append("verify")
            col_cfg["verify"] = st.column_config.LinkColumn(
                "Verify on SET",
                display_text="🔗 set.or.th",
                help="Open the official Major Shareholders page on set.or.th",
            )

        st.markdown(f"**{title}**")
        st.dataframe(ranked[cols_show], width="stretch",
                     hide_index=True, column_config=col_cfg)

    # ---- Main ranking display ----------------------------------------
    if combine_kinds:
        render_ranking(
            metrics,
            title=f"Top {top_n} nodes — {measure_name}  "
                  f"(Companies + Stakeholders combined)",
            show_verify=False,
            show_sector=False,
        )
        st.caption("ℹ️  Mixed ranking — Stakeholders usually dominate "
                   "degree-based measures because Companies are capped at degree = 5.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            render_ranking(
                metrics[metrics["kind"] == "company"],
                title=f"🏢 Top {top_n} Companies — {measure_name}",
                show_verify=True,
                show_sector=True,
            )
        with col_b:
            render_ranking(
                metrics[metrics["kind"] == "stakeholder"],
                title=f"🟥 Top {top_n} Stakeholders — {measure_name}",
                show_verify=False,
                show_sector=False,
            )

    # ---- Cross-measure comparison ------------------------------------
    st.markdown("---")
    with st.expander("🔄 Cross-measure ranking — who ranks high in *every* measure?",
                     expanded=False):
        st.caption(
            "For each centrality, we rank nodes 1, 2, 3, ... "
            "Cells show the rank of that node in that measure "
            "(green = good rank, red = poor). Look for rows that are "
            "**all green** — those are the 'consensus hubs' that matter "
            "no matter which definition of importance you pick."
        )
        kind_choice = st.radio(
            "Show ranks for:",
            ["🏢 Companies", "🟥 Stakeholders"],
            horizontal=True,
            key="cross_kind",
        )
        target_kind = "company" if kind_choice.startswith("🏢") else "stakeholder"

        measures = list(CENTRALITY_DEFS.keys())
        cols_map = [CENTRALITY_DEFS[m]["col"] for m in measures]

        sub = metrics[metrics["kind"] == target_kind].copy()
        for m, c in zip(measures, cols_map):
            sub[f"r_{m}"] = sub[c].rank(ascending=False,
                                        method="min",
                                        na_option="bottom").astype("Int64")

        # Union of top-10 across all measures
        keep_idx = set()
        for m in measures:
            keep_idx.update(sub.nsmallest(10, f"r_{m}").index)
        cross = sub.loc[list(keep_idx)].copy()
        cross["best_rank"] = cross[[f"r_{m}" for m in measures]].min(axis=1)
        cross = cross.sort_values("best_rank")

        display_cols = ["node"] + [f"r_{m}" for m in measures]
        styled = cross[display_cols].rename(
            columns={f"r_{m}": m for m in measures} | {"node": "Node"}
        )

        col_cfg2 = {"Node": st.column_config.TextColumn("Node")}
        for m in measures:
            col_cfg2[m] = st.column_config.NumberColumn(
                m, format="%d",
                help=f"Rank in {m} centrality (1 = best)",
                width="small",
            )
        st.dataframe(styled, width="stretch", hide_index=True,
                     column_config=col_cfg2)

        st.caption("💡  Nodes in the top rows score top-10 across most "
                   "measures — they are robustly central regardless of "
                   "which definition you trust.")

    # ---- Download ----------------------------------------------------
    st.download_button(
        "⬇ Download all centralities (CSV)",
        data=metrics.to_csv(index=False).encode("utf-8"),
        file_name="set50_centralities.csv",
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
