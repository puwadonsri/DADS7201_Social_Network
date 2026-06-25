"""
HW2 — MemeTracker Domain Graph (Streamlit dashboard).

Reads a pre-computed snapshot (outputs/snapshots/hw2_quotes.json) that was
produced by running scripts/hw2_analyze_quotes.py against a local Neo4j +
GDS container. Renders 4 tabs:

  1. Network — interactive PyVis force-directed graph, coloured by the
     selected metric or by Louvain community.
  2. Top-N — ranked table + Plotly bar chart of the selected metric.
  3. Communities — Louvain community drill-down.
  4. Cypher — the exact Cypher / GDS queries used to compute each result.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

HERE = Path(__file__).parent
SNAPSHOT = HERE / "outputs" / "snapshots" / "hw2_quotes.json"

METRICS = ["degree", "betweenness", "closeness", "eigenvector", "pagerank"]

PALETTE = [
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
]


# ---------- Data ----------

@st.cache_data(show_spinner=False)
def load_snapshot() -> dict:
    with open(SNAPSHOT, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def nodes_df(_snap: dict) -> pd.DataFrame:
    return pd.DataFrame(_snap["nodes"])


@st.cache_data(show_spinner=False)
def edges_df(_snap: dict) -> pd.DataFrame:
    return pd.DataFrame(_snap["edges"])


# ---------- PyVis ----------

def build_pyvis(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    color_by: str,            # "community" or one of METRICS
    show_bridges: bool,
    top_label_n: int,
    height_px: int = 720,
) -> str:
    """Return the HTML string for a PyVis network coloured by the chosen mode."""
    net = Network(height=f"{height_px}px", width="100%", directed=False,
                  notebook=False, bgcolor="#fafbfc")
    net.set_options("""
    {
      "nodes": {"font": {"size": 12, "face": "Tahoma"},
                "borderWidth": 1, "borderWidthSelected": 2},
      "edges": {"smooth": {"type": "continuous"}},
      "physics": {"solver": "forceAtlas2Based",
                  "forceAtlas2Based": {"gravitationalConstant": -45,
                                        "springLength": 130,
                                        "avoidOverlap": 0.4}},
      "interaction": {"hover": true, "tooltipDelay": 0}
    }
    """)

    # community colour map (always used for community mode, also as fallback)
    comm_ids = sorted(nodes["community"].unique())
    comm_color = {cid: PALETTE[i % len(PALETTE)] for i, cid in enumerate(comm_ids)}

    if color_by == "community":
        node_color = {row["name"]: comm_color[row["community"]] for _, row in nodes.iterrows()}
    else:
        vals = nodes[color_by].astype(float)
        vmin, vmax = float(vals.min()), float(vals.max())
        rng = max(vmax - vmin, 1e-9)
        # OrRd-ish ramp; light → strong red as score increases
        node_color = {}
        for _, row in nodes.iterrows():
            t = (float(row[color_by]) - vmin) / rng
            r = int(254 - t * 70)
            g = int(232 - t * 200)
            b = int(200 - t * 200)
            node_color[row["name"]] = f"rgb({r},{g},{b})"

    # node size: emphasise the chosen metric (or degree when colouring by community)
    size_metric = color_by if color_by in METRICS else "degree"
    size_vals = nodes[size_metric].astype(float)
    smin, smax = float(size_vals.min()), float(size_vals.max())
    srng = max(smax - smin, 1e-9)

    # only top-N nodes get a visible label; everyone else uses an empty label
    top_names = set(
        nodes.sort_values(size_metric, ascending=False).head(top_label_n)["name"]
    )

    for _, row in nodes.iterrows():
        size = 12 + 35 * ((float(row[size_metric]) - smin) / srng)
        tooltip = (
            f"<b>{row['name']}</b><br>"
            f"posts: {int(row.get('post_count', 0)):,}<br>"
            f"community: {int(row['community'])}<br>"
            + "<br>".join(f"{m.capitalize()}: {row[m]:.4f}" for m in METRICS)
        )
        net.add_node(
            row["name"],
            label=row["name"] if row["name"] in top_names else " ",
            title=tooltip,
            size=size,
            color=node_color[row["name"]],
        )

    for _, row in edges.iterrows():
        is_bridge = bool(row.get("is_bridge", False))
        color = "#d62728" if (show_bridges and is_bridge) else "#cccccc"
        width = 3 if (show_bridges and is_bridge) else 1
        net.add_edge(row["src"], row["dst"], color=color, width=width,
                     title=f"weight: {int(row['weight'])}")

    return net.generate_html(notebook=False)


# ---------- App ----------

st.set_page_config(
    page_title="HW2 — MemeTracker Domain Graph",
    layout="wide",
    initial_sidebar_state="expanded",
)

snap = load_snapshot()
nodes_all = nodes_df(snap)
edges_all = edges_df(snap)
meta = snap["meta"]

st.title("HW2 — MemeTracker Domain Graph")
st.caption(
    f"Stanford SNAP `quotes_2009-04.txt` ({meta['raw_size_gb']} GB) — "
    f"{meta['n_posts_streamed']:,} posts, {meta['n_links_streamed']:,} outbound links streamed; "
    f"aggregated to **top {meta['top_k']} domains** + weight ≥ {meta['min_weight']}, "
    f"largest connected component = **{meta['n_nodes_plotted']} nodes / {meta['n_edges_plotted']} edges**. "
    "All metrics computed via `gds.*` in Cypher."
)

# Top-line metric cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Bridges", meta["n_bridges"])
c2.metric("Louvain communities", meta["louvain_communities"])
c3.metric("Modularity", f"{meta['louvain_modularity']:.3f}")
c4.metric("Nodes / Edges", f"{meta['n_nodes_plotted']} / {meta['n_edges_plotted']}")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    color_mode = st.selectbox(
        "Colour nodes by",
        ["community"] + METRICS,
        index=0,
        help="`community` = Louvain group; otherwise nodes are coloured by the metric score.",
    )
    show_bridges = st.checkbox("Highlight bridge edges (red)", value=True)
    top_label_n = st.slider("Label top-N nodes", min_value=3, max_value=30, value=10)

    st.divider()
    st.subheader("Filter")
    communities_sorted = sorted(nodes_all["community"].unique())
    chosen_comms = st.multiselect(
        "Show only Louvain communities",
        options=communities_sorted,
        default=communities_sorted,
        help="Hide other communities (still kept in the global metric ranking).",
    )

filtered_nodes = nodes_all[nodes_all["community"].isin(chosen_comms)].reset_index(drop=True)
keep = set(filtered_nodes["name"])
filtered_edges = edges_all[edges_all["src"].isin(keep) & edges_all["dst"].isin(keep)].reset_index(drop=True)

tab_net, tab_top, tab_comm, tab_cypher = st.tabs(
    ["🌐 Network", "📊 Top-N", "🧩 Communities", "🧪 Cypher"]
)

# ---- Network tab ----
with tab_net:
    if len(filtered_nodes) == 0:
        st.warning("ไม่มี node ตามที่เลือก — ลองเลือก Louvain community เพิ่ม")
    else:
        html = build_pyvis(
            filtered_nodes, filtered_edges,
            color_by=color_mode,
            show_bridges=show_bridges,
            top_label_n=top_label_n,
        )
        components.html(html, height=740, scrolling=False)
        st.caption(
            f"Showing {len(filtered_nodes)} nodes / {len(filtered_edges)} edges. "
            "Drag a node, hover for the full metric breakdown. "
            "Force-directed physics is computed in your browser."
        )

# ---- Top-N tab ----
with tab_top:
    col_metric, col_n = st.columns([1, 3])
    metric = col_metric.selectbox("Metric", METRICS, index=0,
                                  format_func=str.capitalize, key="topn_metric")
    top_n = col_n.slider("Top N", 5, 50, 15, key="topn_n")

    ranked = (
        nodes_all[["name", "community", "post_count", metric]]
        .sort_values(metric, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    ranked.index = range(1, len(ranked) + 1)
    ranked.index.name = "rank"

    fig = px.bar(
        ranked.reset_index(),
        x=metric, y="name",
        orientation="h",
        color="community",
        color_continuous_scale=px.colors.qualitative.Set2,
        title=f"Top {top_n} domains by {metric}",
        labels={"name": "domain", metric: metric.capitalize()},
        height=max(420, 22 * top_n),
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        ranked.style.format({metric: "{:.4f}", "post_count": "{:,}"}),
        use_container_width=True,
    )

# ---- Communities tab ----
with tab_comm:
    comm_sizes = (
        nodes_all.groupby("community")
        .agg(size=("name", "count"),
             total_degree=("degree", "sum"),
             top_pagerank=("pagerank", "max"))
        .sort_values("size", ascending=False)
        .reset_index()
    )

    st.subheader("Community overview")
    st.dataframe(
        comm_sizes.style.format({"top_pagerank": "{:.3f}", "total_degree": "{:.0f}"}),
        use_container_width=True,
    )

    chosen = st.selectbox(
        "Drill into community",
        options=comm_sizes["community"].tolist(),
        format_func=lambda c: f"Community {c} ({int(comm_sizes[comm_sizes.community == c]['size'].iloc[0])} domains)",
    )
    members = (
        nodes_all[nodes_all["community"] == chosen]
        [["name", "post_count", "degree", "betweenness", "pagerank", "eigenvector"]]
        .sort_values("pagerank", ascending=False)
        .reset_index(drop=True)
    )

    st.subheader(f"Community {chosen} — {len(members)} domains")
    st.dataframe(
        members.style.format({
            "post_count": "{:,}",
            "betweenness": "{:.2f}",
            "pagerank": "{:.3f}",
            "eigenvector": "{:.3f}",
        }),
        use_container_width=True,
    )

    # mini-graph just for this community
    sub_nodes = nodes_all[nodes_all["community"] == chosen].reset_index(drop=True)
    sub_keep = set(sub_nodes["name"])
    sub_edges = edges_all[edges_all["src"].isin(sub_keep) & edges_all["dst"].isin(sub_keep)].reset_index(drop=True)
    if len(sub_edges) > 0:
        st.caption(f"Sub-graph (within-community edges only — {len(sub_edges)} edges)")
        html = build_pyvis(
            sub_nodes, sub_edges,
            color_by="community",
            show_bridges=False,
            top_label_n=max(3, min(15, len(sub_nodes))),
            height_px=500,
        )
        components.html(html, height=520, scrolling=False)
    else:
        st.info("Community นี้ไม่มี within-community edge — เป็นกลุ่มเล็กที่เชื่อมแบบ cross-community เท่านั้น")

# ---- Cypher tab ----
with tab_cypher:
    st.markdown(
        "ทุก metric บนหน้านี้ **คำนวณจาก Cypher / Neo4j GDS** บน local Neo4j + GDS plugin "
        "(ดู [`config/readme.md`](https://github.com/puwadonsri/DADS7201_Social_Network/blob/main/Week3/config/readme.md)). "
        "ผลถูก export เป็น JSON snapshot เพื่อให้ Streamlit Cloud ใช้งานได้โดยไม่ต้องต่อ Neo4j จริง"
    )

    cypher = snap["cypher"]
    order = [
        ("0. LOAD CSV — nodes", "load_nodes"),
        ("1. LOAD CSV — edges", "load_edges"),
        ("2. Project undirected weighted graph", "project"),
        ("3. Bridges", "bridges"),
        ("4. Betweenness", "betweenness"),
        ("5. Closeness", "closeness"),
        ("6. Degree", "degree"),
        ("7. Eigenvector", "eigenvector"),
        ("8. PageRank", "pagerank"),
        ("9. Louvain", "louvain"),
    ]
    for title, key in order:
        with st.expander(title, expanded=(key == "project")):
            st.code(cypher[key], language="cypher")

st.divider()
st.caption(
    "Source: [DADS7201_Social_Network/Week3](https://github.com/puwadonsri/DADS7201_Social_Network/tree/main/Week3) · "
    "Dataset: [Stanford MemeTracker](https://snap.stanford.edu/data/memetracker9.html)"
)
