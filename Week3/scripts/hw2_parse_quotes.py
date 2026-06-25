"""
Stream-parse Stanford MemeTracker quotes_2009-04.txt (~10.9 GB) into a
domain-level link graph small enough for Neo4j + GDS.

Format of input (one record per blank-line block):
    P <post URL>
    T <timestamp>
    Q <quote line>          (may repeat)
    L <link URL>            (may repeat)

We build a directed graph at the *domain* level — every L line emits an edge
(src_domain, dst_domain) where src_domain comes from the most recent P line.

Two streaming passes:
  1) Count P-domain frequencies → pick TOP_K most active domains.
  2) Re-stream and aggregate edges, keeping only ones where both endpoints
     fall in TOP_K. Filter by MIN_WEIGHT, then write to CSV.

Output: Week3/homework/quotes_domain_edges.csv  (src,dst,weight)
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INPUT = os.path.join(ROOT, "homework", "quotes_2009-04.txt")
EDGES_OUT = os.path.join(ROOT, "homework", "quotes_domain_edges.csv")
NODES_OUT = os.path.join(ROOT, "homework", "quotes_domains.csv")

TOP_K = 200
MIN_WEIGHT = 5

# domains we skip — non-bloggy CDN / image / shortener noise
SKIP_SUFFIXES = (
    ".bp.blogspot.com",
    ".photobucket.com",
    ".flickr.com",
    ".tinypic.com",
    "i.imgur.com",
)


def domain_of(url: str) -> str | None:
    url = url.strip()
    if not url:
        return None
    if "://" in url:
        url = url.split("://", 1)[1]
    domain = url.split("/", 1)[0].lower()
    domain = domain.split("?", 1)[0]
    domain = domain.split(":", 1)[0]
    if domain.startswith("www."):
        domain = domain[4:]
    if not domain or " " in domain or "." not in domain:
        return None
    if any(domain.endswith(s) for s in SKIP_SUFFIXES):
        return None
    return domain


def stream(path: str, top_k_domains: set[str] | None = None, log_every: int = 5_000_000):
    """Single pass over the file.

    Pass 1 (top_k_domains is None):  counts P-domain frequencies only.
    Pass 2 (top_k_domains is set):   aggregates edges in that filter.
    """
    domain_count: Counter[str] = Counter()
    edge_count: Counter[tuple[str, str]] = Counter()
    current_p: str | None = None
    n_posts = n_links = 0
    start = time.time()

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if not line:
                continue
            tag = line[0]
            if tag == "P":
                d = domain_of(line[2:])
                if d is None:
                    current_p = None
                    continue
                n_posts += 1
                if top_k_domains is None:
                    domain_count[d] += 1
                    current_p = d
                else:
                    current_p = d if d in top_k_domains else None
            elif tag == "L" and current_p is not None:
                d = domain_of(line[2:])
                if d is None or d == current_p:
                    continue
                if top_k_domains is not None and d not in top_k_domains:
                    continue
                edge_count[(current_p, d)] += 1
                n_links += 1
            if i % log_every == 0:
                rate = i / max(time.time() - start, 1e-3)
                print(f"  line {i:>12,}  P={n_posts:>10,}  L={n_links:>10,}  "
                      f"({rate:,.0f} l/s)", flush=True)
    elapsed = time.time() - start
    print(f"  done in {elapsed:.1f}s — {n_posts:,} P, {n_links:,} L")
    return domain_count, edge_count


def main():
    if not os.path.exists(INPUT):
        sys.exit(f"missing input: {INPUT}")
    size_gb = os.path.getsize(INPUT) / (1024 ** 3)
    print(f"Input: {INPUT}  ({size_gb:.2f} GB)")
    print(f"TOP_K = {TOP_K}, MIN_WEIGHT = {MIN_WEIGHT}")

    print("\n[Pass 1] counting P-domain frequencies...")
    dc, _ = stream(INPUT, top_k_domains=None)
    print(f"  unique P-domains: {len(dc):,}")
    top_pairs = dc.most_common(TOP_K)
    top_k_set = {d for d, _ in top_pairs}
    print(f"  top 10 by post count: {top_pairs[:10]}")

    print(f"\n[Pass 2] extracting edges within top-{TOP_K}...")
    _, ec = stream(INPUT, top_k_domains=top_k_set)
    print(f"  unique edges: {len(ec):,}")

    filtered = [(s, t, w) for (s, t), w in ec.items() if w >= MIN_WEIGHT]
    filtered.sort(key=lambda x: -x[2])
    print(f"  after weight >= {MIN_WEIGHT}: {len(filtered):,} edges")

    # Domain degree (deg_in + deg_out) so the importer can size nodes nicely.
    deg: Counter[str] = Counter()
    for s, t, w in filtered:
        deg[s] += w
        deg[t] += w

    os.makedirs(os.path.dirname(EDGES_OUT), exist_ok=True)
    with open(EDGES_OUT, "w", encoding="utf-8") as f:
        f.write("src,dst,weight\n")
        for s, t, w in filtered:
            f.write(f"{s},{t},{w}\n")
    with open(NODES_OUT, "w", encoding="utf-8") as f:
        f.write("domain,post_count,degree\n")
        for d in sorted(top_k_set, key=lambda x: -deg.get(x, 0)):
            f.write(f"{d},{dc[d]},{deg.get(d, 0)}\n")
    print(f"\nSaved edges: {EDGES_OUT}")
    print(f"Saved nodes: {NODES_OUT}")


if __name__ == "__main__":
    main()
