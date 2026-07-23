"""
09 - Temporal link prediction on the Pantip SET50 co-mention graph.

Predicts which stock pairs (currently without a co-mention edge) will
acquire one in the next event window. Compares 5 methods spanning
Weeks 3, 5, 7:

    1. Jaccard coefficient           (Week 3)
    2. Adamic-Adar                   (Week 3)
    3. Preferential attachment       (Week 3)
    4. Node2Vec + logistic regression (Week 5)
    5. Homogeneous GraphSAGE          (Week 7)

Data (from ../Midterm/data/):
    posts_raw.csv     1,374 scraped Pantip topics
    stocks.csv        SET50 universe
    event_anchor.json T = 2026-05-28  (peak posting day)

Temporal split (event-anchored):
    train  = co-mentions in [t_min,        T - 20)
    val    = co-mentions in [T - 20,       T)     -- first-time only
    test   = co-mentions in [T,            T + 20) -- first-time only

Outputs (./output/):
    lp_baselines.csv                per-method summary metrics + bootstrap CI
    lp_gnn.csv                      GraphSAGE detail
    lp_topk_predicted_edges.csv     top-30 unseen pairs, tagged with Midterm bucket
    lp_roc.png/.svg                 test-set ROC (5 curves)
    lp_precision_at_k.png/.svg      test-set Precision@k bar chart
"""
from __future__ import annotations

import io
import json
import re
import sys
import warnings
from itertools import combinations
from pathlib import Path

import torch  # keep BEFORE transformers/pyg on Windows
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

MIDTERM_DIR = Path('../Midterm')
DATA_DIR    = MIDTERM_DIR / 'data'
FIN_OUT     = Path('output')
FIN_OUT.mkdir(exist_ok=True)

RNG_SEED = 42
NEG_RATIO_TRAIN = 2
NEG_RATIO_EVAL  = 1
N_BOOT = 500

WINDOW_DAYS = 20   # each phase length

# ============================================================
# 1. Load Midterm data + regex-extract mentions on the RAW corpus
#    (posts.csv is only 96 rows after event-anchor filter; for link
#    prediction we want as many training edges as possible.)
# ============================================================
def load_universe():
    stocks = pd.read_csv(DATA_DIR / 'stocks.csv')
    return stocks['Symbol'].tolist(), stocks


def build_ticker_regex(symbols):
    pattern = '|'.join(sorted(symbols, key=len, reverse=True))
    return re.compile(r'(?<![A-Za-z0-9])(' + pattern + r')(?![A-Za-z0-9])')


def extract_all_mentions():
    """Reads posts_raw.csv (~1,374 rows), applies SET50 regex, keeps
    posts with >= 2 mentions. Returns DataFrame:
       PostID, Date (datetime), Mentions (list[str])"""
    raw = pd.read_csv(DATA_DIR / 'posts_raw.csv')
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw['Title'] = raw['Title'].fillna('')
    raw['Text']  = raw['Text'].fillna('')
    raw['_combined'] = (raw['Title'] + '  ' + raw['Text']).str.upper()

    symbols, _ = load_universe()
    ticker_re = build_ticker_regex(symbols)

    def _extract(t):
        return sorted(set(ticker_re.findall(t))) if isinstance(t, str) else []

    raw['Mentions']  = raw['_combined'].apply(_extract)
    raw['NumMentions'] = raw['Mentions'].apply(len)
    kept = raw[raw['NumMentions'] >= 2][['PostID', 'Date', 'Mentions']].copy()
    return kept.sort_values('Date').reset_index(drop=True)


# ============================================================
# 2. Temporal split
# ============================================================
def build_temporal_split(mentions_df, event_date, window_days=WINDOW_DAYS):
    """Return (train_edges, val_first, test_first, universe, split_meta).

    train_edges: dict[frozenset] -> weight (co-mention count)
    val_first / test_first: set of frozensets, edges that FIRST appear
                            in that phase (not already in earlier phases)
    """
    event = pd.to_datetime(event_date)
    train_end = event - pd.Timedelta(days=window_days)
    val_end   = event
    test_end  = event + pd.Timedelta(days=window_days)

    train_posts = mentions_df[mentions_df['Date'] < train_end]
    val_posts   = mentions_df[(mentions_df['Date'] >= train_end) & (mentions_df['Date'] < val_end)]
    test_posts  = mentions_df[(mentions_df['Date'] >= val_end)   & (mentions_df['Date'] < test_end)]

    def _edges(df):
        edges = {}
        for _, row in df.iterrows():
            for a, b in combinations(row['Mentions'], 2):
                if a == b: continue
                edges[frozenset((a, b))] = edges.get(frozenset((a, b)), 0) + 1
        return edges

    train_edges = _edges(train_posts)
    val_edges   = _edges(val_posts)
    test_edges  = _edges(test_posts)

    val_first  = set(val_edges)  - set(train_edges)
    test_first = set(test_edges) - set(train_edges) - set(val_edges)

    universe, _ = load_universe()
    universe = sorted(set(universe))

    split_meta = {
        'event_date':   str(event.date()),
        'train_end':    str(train_end.date()),
        'val_end':      str(val_end.date()),
        'test_end':     str(test_end.date()),
        'n_train_posts': int(len(train_posts)),
        'n_val_posts':   int(len(val_posts)),
        'n_test_posts':  int(len(test_posts)),
        'n_train_edges': len(train_edges),
        'n_val_positive_first_seen':  len(val_first),
        'n_test_positive_first_seen': len(test_first),
    }
    return train_edges, val_first, test_first, universe, split_meta


def sample_negatives(train_edges, positives, universe, ratio, rng):
    """Sample negative pairs (not in train_edges and not in positives)
    at the given ratio."""
    all_pairs = set()
    universe_list = list(universe)
    positive_set = set(positives)
    excluded = set(train_edges) | positive_set

    n_neg_target = int(ratio * len(positives))
    max_pairs = len(universe_list) * (len(universe_list) - 1) // 2
    n_neg = min(n_neg_target, max_pairs - len(excluded))

    while len(all_pairs) < n_neg:
        i, j = rng.integers(0, len(universe_list), size=2)
        if i == j: continue
        pair = frozenset((universe_list[i], universe_list[j]))
        if pair in excluded or pair in all_pairs:
            continue
        all_pairs.add(pair)
    return all_pairs


# ============================================================
# 3. Heuristic scorers (Week 3)
# ============================================================
def _pair_to_tuple(pair):
    a, b = tuple(pair)
    return a, b


def score_jaccard(G, pairs):
    result = {}
    ebunch = [_pair_to_tuple(p) for p in pairs]
    for u, v, p in nx.jaccard_coefficient(G, ebunch):
        result[frozenset((u, v))] = p
    return result


def score_adamic_adar(G, pairs):
    result = {}
    ebunch = [_pair_to_tuple(p) for p in pairs]
    for u, v, p in nx.adamic_adar_index(G, ebunch):
        result[frozenset((u, v))] = p
    return result


def score_preferential(G, pairs):
    result = {}
    ebunch = [_pair_to_tuple(p) for p in pairs]
    for u, v, p in nx.preferential_attachment(G, ebunch):
        result[frozenset((u, v))] = p
    return result


# ============================================================
# 4. Node2Vec + logistic (Week 5)
# ============================================================
def train_node2vec(G, dim=64, walks=15, walk_len=40, window=8, p=1, q=1,
                   seed=RNG_SEED):
    """Generate biased random walks (p=q=1 = DeepWalk) and train
    gensim Word2Vec (skip-gram)."""
    from gensim.models import Word2Vec

    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())

    walks_list = []
    for _ in range(walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            for _ in range(walk_len - 1):
                cur = walk[-1]
                nbrs = list(G.neighbors(cur))
                if not nbrs:
                    break
                walk.append(nbrs[rng.integers(0, len(nbrs))])
            walks_list.append(walk)

    model = Word2Vec(
        sentences=walks_list,
        vector_size=dim,
        window=window,
        min_count=0,
        sg=1,
        workers=1,
        epochs=10,
        seed=seed,
    )
    return {n: model.wv[n] for n in G.nodes() if n in model.wv}


def hadamard(z_u, z_v):
    return z_u * z_v


def train_node2vec_classifier(embeddings, train_pos, train_neg, seed=RNG_SEED):
    Xtr, ytr = [], []
    for e in train_pos:
        u, v = tuple(e)
        if u in embeddings and v in embeddings:
            Xtr.append(hadamard(embeddings[u], embeddings[v]))
            ytr.append(1)
    for e in train_neg:
        u, v = tuple(e)
        if u in embeddings and v in embeddings:
            Xtr.append(hadamard(embeddings[u], embeddings[v]))
            ytr.append(0)
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(Xtr, ytr)
    return lr


def score_node2vec(embeddings, classifier, pairs):
    scores = {}
    for e in pairs:
        u, v = tuple(e)
        if u in embeddings and v in embeddings:
            x = hadamard(embeddings[u], embeddings[v]).reshape(1, -1)
            scores[e] = float(classifier.predict_proba(x)[0, 1])
        else:
            scores[e] = 0.0
    return scores


# ============================================================
# 5. Homogeneous GraphSAGE (Week 7 style, simplified to 1 node type)
# ============================================================
def build_pyg_data(G, universe):
    """Convert to torch_geometric.data.Data. Node feature = one-hot
    degree bucket (or ID one-hot); use identity here as we have no
    external node features and 50 nodes is tiny."""
    from torch_geometric.data import Data
    node2id = {n: i for i, n in enumerate(universe)}
    N = len(universe)
    edge_list = []
    for u, v in G.edges():
        i, j = node2id[u], node2id[v]
        edge_list.extend([[i, j], [j, i]])   # undirected
    edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list \
        else torch.zeros((2, 0), dtype=torch.long)
    x = torch.eye(N, dtype=torch.float)  # identity features (small graph)
    return Data(x=x, edge_index=edge_index), node2id


class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)

    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        return self.conv2(h, edge_index)

    @staticmethod
    def decode(z, edge_pairs):
        # edge_pairs: [2, E] tensor of index pairs
        return (z[edge_pairs[0]] * z[edge_pairs[1]]).sum(dim=-1)


def train_graphsage(pyg_data, train_pos, train_neg, node2id,
                    hidden=64, epochs=200, lr=1e-3, seed=RNG_SEED):
    torch.manual_seed(seed)
    model = GraphSAGEModel(in_dim=pyg_data.x.size(1), hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def to_pair_tensor(edges):
        idx = []
        for e in edges:
            u, v = tuple(e)
            idx.append([node2id[u], node2id[v]])
        return torch.tensor(idx, dtype=torch.long).t() if idx \
            else torch.zeros((2, 0), dtype=torch.long)

    pos_pairs = to_pair_tensor(train_pos)
    neg_pairs = to_pair_tensor(train_neg)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        z = model.encode(pyg_data.x, pyg_data.edge_index)
        pos_score = model.decode(z, pos_pairs)
        neg_score = model.decode(z, neg_pairs)
        loss = loss_fn(pos_score, torch.ones_like(pos_score)) + \
               loss_fn(neg_score, torch.zeros_like(neg_score))
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(pyg_data.x, pyg_data.edge_index)
    return model, z


def score_graphsage(model, z, pairs, node2id):
    scores = {}
    idx = []
    valid_pairs = []
    for e in pairs:
        u, v = tuple(e)
        if u in node2id and v in node2id:
            idx.append([node2id[u], node2id[v]])
            valid_pairs.append(e)
    if not idx:
        return scores
    idx_t = torch.tensor(idx, dtype=torch.long).t()
    with torch.no_grad():
        logits = model.decode(z, idx_t)
        prob = torch.sigmoid(logits).numpy()
    for p, s in zip(valid_pairs, prob):
        scores[p] = float(s)
    return scores


# ============================================================
# 6. Metrics + bootstrap CI
# ============================================================
def evaluate(y_true, y_score, k_list=(10, 25, 50), n_boot=N_BOOT, seed=RNG_SEED):
    y_true  = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    result = {}

    # AUC
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float('nan')
    result['auc'] = auc

    # Precision@k
    order = np.argsort(-y_score)
    for k in k_list:
        k_eff = min(k, len(y_true))
        top_k = order[:k_eff]
        result[f'precision_at_{k}'] = float(y_true[top_k].mean()) if k_eff else float('nan')

    # MRR
    ranks = []
    for i, is_pos in enumerate(y_true[order]):
        if is_pos:
            ranks.append(1.0 / (i + 1))
    result['mrr'] = float(np.mean(ranks)) if ranks else float('nan')

    # Bootstrap CI on AUC
    boot_aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(set(yt)) < 2:
            continue
        try:
            boot_aucs.append(roc_auc_score(yt, ys))
        except ValueError:
            continue
    if boot_aucs:
        result['auc_ci_low']  = float(np.percentile(boot_aucs, 2.5))
        result['auc_ci_high'] = float(np.percentile(boot_aucs, 97.5))
    else:
        result['auc_ci_low'] = result['auc_ci_high'] = float('nan')
    return result


# ============================================================
# 7. Midterm bucket tagging
# ============================================================
def load_midterm_buckets():
    """Return dict[frozenset] -> 'hype-only' | 'overlooked' | 'both' | None."""
    tags = {}
    hype_only = pd.read_csv(MIDTERM_DIR / 'output' / 'attention_without_correlation.csv')
    overlk    = pd.read_csv(MIDTERM_DIR / 'output' / 'correlation_without_attention.csv')
    validated = pd.read_csv(MIDTERM_DIR / 'output' / 'validated_attention_edges.csv')
    for _, r in hype_only.iterrows():
        tags[frozenset((r['StockA'], r['StockB']))] = 'hype-only'
    for _, r in overlk.iterrows():
        tags[frozenset((r['StockA'], r['StockB']))] = 'overlooked'
    for _, r in validated.iterrows():
        tags[frozenset((r['StockA'], r['StockB']))] = 'both'
    return tags


# ============================================================
# 8. Main pipeline
# ============================================================
def main():
    print('=' * 65)
    print('Path A — Temporal Link Prediction on Pantip SET50 co-mentions')
    print('=' * 65)

    # ---- Load + build ----
    print('\n[1/6] Extracting mentions from raw Pantip corpus...')
    mentions_df = extract_all_mentions()
    print(f'  Posts with ≥2 SET50 mentions: {len(mentions_df)}')

    event_meta = json.load(open(DATA_DIR / 'event_anchor.json', encoding='utf-8'))
    print(f'  Event date: {event_meta["event_date"]}')

    train_edges, val_first, test_first, universe, split_meta = \
        build_temporal_split(mentions_df, event_meta['event_date'])
    print('  Split:', json.dumps(split_meta, indent=2, ensure_ascii=False))

    with open(Path('data') / 'temporal_split.json', 'w', encoding='utf-8') as f:
        json.dump(split_meta, f, indent=2, ensure_ascii=False)

    # ---- Graph for training ----
    G_train = nx.Graph()
    G_train.add_nodes_from(universe)
    for e, w in train_edges.items():
        u, v = tuple(e)
        G_train.add_edge(u, v, weight=w)
    print(f'  Train graph: {G_train.number_of_nodes()} nodes, '
          f'{G_train.number_of_edges()} edges')

    # ---- Sample negatives ----
    rng = np.random.default_rng(RNG_SEED)
    train_pos = set(train_edges)
    train_neg = sample_negatives(train_edges, train_pos, universe,
                                 NEG_RATIO_TRAIN, rng)
    val_neg   = sample_negatives(train_edges, val_first,  universe,
                                 NEG_RATIO_EVAL, rng)
    test_neg  = sample_negatives(train_edges, test_first, universe,
                                 NEG_RATIO_EVAL, rng)
    print(f'  Val: {len(val_first)} positives / {len(val_neg)} negatives')
    print(f'  Test: {len(test_first)} positives / {len(test_neg)} negatives')

    # ---- 5 methods score val + test ----
    def _score_all(scorer, pairs):
        return scorer(G_train, pairs)

    def _run_method(name, val_scorer, test_scorer=None):
        if test_scorer is None:
            test_scorer = val_scorer
        val_scores  = val_scorer(list(val_first) + list(val_neg))
        test_scores = test_scorer(list(test_first) + list(test_neg))
        return val_scores, test_scores

    def _labels_and_scores(pos, neg, scores):
        y_true, y_score = [], []
        for e in pos:
            y_true.append(1); y_score.append(scores.get(e, 0.0))
        for e in neg:
            y_true.append(0); y_score.append(scores.get(e, 0.0))
        return y_true, y_score

    all_rows  = []
    all_curves = {}   # method -> (test y_true, test y_score)

    print('\n[2/6] Scoring heuristic baselines (Jaccard, Adamic-Adar, PA)...')
    for name, scorer in [('Jaccard',              score_jaccard),
                          ('Adamic-Adar',         score_adamic_adar),
                          ('Preferential',        score_preferential)]:
        val_scores  = scorer(G_train, list(val_first)  + list(val_neg))
        test_scores = scorer(G_train, list(test_first) + list(test_neg))
        for split, scores, pos, neg in [
                ('val',  val_scores,  val_first,  val_neg),
                ('test', test_scores, test_first, test_neg)]:
            y_true, y_score = _labels_and_scores(pos, neg, scores)
            m = evaluate(y_true, y_score)
            all_rows.append({'method': name, 'split': split, **m})
        all_curves[name] = _labels_and_scores(test_first, test_neg, test_scores)
        print(f'  {name:15s}  val AUC={all_rows[-2]["auc"]:.3f}  '
              f'test AUC={all_rows[-1]["auc"]:.3f}')

    print('\n[3/6] Training Node2Vec (Week 5)...')
    emb = train_node2vec(G_train, dim=64, walks=15, walk_len=40)
    clf = train_node2vec_classifier(emb, train_pos, train_neg)
    val_scores  = score_node2vec(emb, clf, list(val_first)  + list(val_neg))
    test_scores = score_node2vec(emb, clf, list(test_first) + list(test_neg))
    for split, scores, pos, neg in [
            ('val',  val_scores,  val_first,  val_neg),
            ('test', test_scores, test_first, test_neg)]:
        y_true, y_score = _labels_and_scores(pos, neg, scores)
        m = evaluate(y_true, y_score)
        all_rows.append({'method': 'Node2Vec', 'split': split, **m})
    all_curves['Node2Vec'] = _labels_and_scores(test_first, test_neg, test_scores)
    print(f'  Node2Vec          val AUC={all_rows[-2]["auc"]:.3f}  '
          f'test AUC={all_rows[-1]["auc"]:.3f}')

    print('\n[4/6] Training GraphSAGE (Week 7)...')
    pyg_data, node2id = build_pyg_data(G_train, universe)
    model, z = train_graphsage(pyg_data, train_pos, train_neg, node2id,
                                epochs=250, lr=5e-3)
    val_scores  = score_graphsage(model, z, list(val_first)  + list(val_neg),  node2id)
    test_scores = score_graphsage(model, z, list(test_first) + list(test_neg), node2id)
    for split, scores, pos, neg in [
            ('val',  val_scores,  val_first,  val_neg),
            ('test', test_scores, test_first, test_neg)]:
        y_true, y_score = _labels_and_scores(pos, neg, scores)
        m = evaluate(y_true, y_score)
        all_rows.append({'method': 'GraphSAGE', 'split': split, **m})
    all_curves['GraphSAGE'] = _labels_and_scores(test_first, test_neg, test_scores)
    print(f'  GraphSAGE         val AUC={all_rows[-2]["auc"]:.3f}  '
          f'test AUC={all_rows[-1]["auc"]:.3f}')

    # ---- Save metrics tables ----
    print('\n[5/6] Saving metrics + top-k predicted edges...')
    df = pd.DataFrame(all_rows)
    df.to_csv(FIN_OUT / 'lp_baselines.csv', index=False)
    df[df.method == 'GraphSAGE'].to_csv(FIN_OUT / 'lp_gnn.csv', index=False)

    # Top-k predicted unseen edges (using best test method)
    best_method = df[df.split == 'test'].sort_values('auc', ascending=False).iloc[0]['method']
    print(f'  Best method: {best_method}')

    all_universe_pairs = {
        frozenset((a, b))
        for a, b in combinations(universe, 2)
    }
    unseen_pairs = all_universe_pairs - set(train_edges)

    # Score unseen pairs with best method
    if best_method in ('Jaccard', 'Adamic-Adar', 'Preferential'):
        scorer = {'Jaccard': score_jaccard,
                  'Adamic-Adar': score_adamic_adar,
                  'Preferential': score_preferential}[best_method]
        unseen_scores = scorer(G_train, list(unseen_pairs))
    elif best_method == 'Node2Vec':
        unseen_scores = score_node2vec(emb, clf, list(unseen_pairs))
    else:
        unseen_scores = score_graphsage(model, z, list(unseen_pairs), node2id)

    buckets = load_midterm_buckets()
    top_rows = []
    for pair, score in sorted(unseen_scores.items(), key=lambda x: -x[1])[:30]:
        a, b = sorted(tuple(pair))
        tag = buckets.get(pair, 'new')
        emerged = 'yes' if (pair in val_first or pair in test_first) else 'no'
        top_rows.append({
            'StockA': a, 'StockB': b, 'PredScore': round(score, 4),
            'MidtermBucket': tag, 'EmergedInEval': emerged,
        })
    pd.DataFrame(top_rows).to_csv(FIN_OUT / 'lp_topk_predicted_edges.csv', index=False)

    # ---- Figures ----
    print('\n[6/6] Rendering figures...')

    PAL = {
        'Jaccard':      '#4A6FA5',
        'Adamic-Adar':  '#8E7CC3',
        'Preferential': '#8DA47E',
        'Node2Vec':     '#E27D60',
        'GraphSAGE':    '#C1272D',
    }

    # ROC on test
    plt.figure(figsize=(6.5, 5.2))
    for method, (y_true, y_score) in all_curves.items():
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        if len(set(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, lw=2, color=PAL.get(method, 'gray'),
                 label=f'{method}  (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='#999', lw=1)
    plt.xlabel('False positive rate', fontweight='bold')
    plt.ylabel('True positive rate',  fontweight='bold')
    plt.title('Link prediction ROC (test set)', fontweight='bold', pad=10)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(axis='both', linestyle='--', alpha=0.4)
    plt.gca().set_axisbelow(True)
    for s in ('top', 'right'):
        plt.gca().spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIN_OUT / 'lp_roc.png', dpi=150)
    plt.savefig(FIN_OUT / 'lp_roc.svg')
    plt.close()

    # Precision@k bar chart
    test_df = df[df.split == 'test'].copy()
    k_cols = ['precision_at_10', 'precision_at_25', 'precision_at_50']
    x = np.arange(len(test_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, k in enumerate([10, 25, 50]):
        vals = test_df[f'precision_at_{k}'].values
        ax.bar(x + (i - 1) * width, vals, width,
               label=f'Precision@{k}', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(test_df['method'], rotation=15)
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('Top-k precision on test set', fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIN_OUT / 'lp_precision_at_k.png', dpi=150)
    plt.savefig(FIN_OUT / 'lp_precision_at_k.svg')
    plt.close()

    print('\n=== DONE ===')
    print('Artefacts:')
    for p in sorted(FIN_OUT.iterdir()):
        if p.name.startswith('lp_'):
            print(f'  {p.name}  ({p.stat().st_size // 1024} KB)')


if __name__ == '__main__':
    main()
