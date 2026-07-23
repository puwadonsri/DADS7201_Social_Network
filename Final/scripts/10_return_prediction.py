"""
10 - Return prediction with graph features.

Predict the sign of next-day SET50 return using own-return lags plus
graph / attention / sentiment features derived from the Midterm pipeline.

Feature groups per (stock, day) row:
    * Own-return lags               r_{d-1}, r_{d-5:d-1} mean, std   (baseline B0)
    * Static graph properties        degree centrality, hype hub score,
                                     community one-hot                  (B1)
    * Rolling attention              10-day mention count               (B2)
    * Rolling sentiment              5-day mean sentiment score         (B2)
    * Cross-sectional                daily rank of hype hub score       (B2)

Models:
    B0  logistic regression on lags
    B1  logistic regression on lags + graph
    B2  logistic regression on all features (full)
    B3  XGBoost on all features (non-linear ceiling)

Evaluation: expanding-window time-series CV, 5 folds.
Portfolio: long top decile predicted-up, short bottom decile predicted-down,
daily rebalance. Report Sharpe-proxy annualised.

Inputs (../Midterm/):
    data/posts_raw.csv, stocks.csv, event_anchor.json
    output/hype_hubs.csv, communities.csv, centrality.csv, sentiment (via 02)

Also fetches a wider yfinance return history (~180 days) for training data.

Outputs (./output/):
    rp_features.csv                per (stock, day) feature matrix
    rp_metrics.json                {model: {acc, f1, prec_decile, sharpe_proxy}}
    rp_feature_importance.png/.svg XGBoost gain importance top-15
    rp_baseline_vs_full.png/.svg   bar chart B0..B3 accuracy
    rp_sharpe_curve.png/.svg       cumulative return vs benchmark
"""
from __future__ import annotations

import json
import re
import warnings
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

MIDTERM_DIR = Path('../Midterm')
DATA_DIR    = MIDTERM_DIR / 'data'
MID_OUT     = MIDTERM_DIR / 'output'
FIN_OUT     = Path('output')
FIN_OUT.mkdir(exist_ok=True)

RNG_SEED = 42
HISTORY_DAYS = 180
N_FOLDS = 5


# ============================================================
# 1. Ticker regex reuse
# ============================================================
def load_universe():
    return pd.read_csv(DATA_DIR / 'stocks.csv')['Symbol'].tolist()


def build_ticker_regex(symbols):
    pattern = '|'.join(sorted(symbols, key=len, reverse=True))
    return re.compile(r'(?<![A-Za-z0-9])(' + pattern + r')(?![A-Za-z0-9])')


def extract_all_mentions():
    raw = pd.read_csv(DATA_DIR / 'posts_raw.csv')
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw['Title'] = raw['Title'].fillna('')
    raw['Text']  = raw['Text'].fillna('')
    raw['_combined'] = (raw['Title'] + '  ' + raw['Text']).str.upper()
    symbols = load_universe()
    ticker_re = build_ticker_regex(symbols)
    raw['Mentions']  = raw['_combined'].apply(
        lambda t: sorted(set(ticker_re.findall(t))) if isinstance(t, str) else []
    )
    raw['NumMentions'] = raw['Mentions'].apply(len)
    return raw[raw['NumMentions'] >= 1][['PostID', 'Date', 'Mentions']].copy()


# ============================================================
# 2. Fetch wider price history
# ============================================================
def fetch_returns():
    universe = load_universe()
    end = datetime(2026, 6, 18)
    start = end - timedelta(days=HISTORY_DAYS)
    print(f'  Fetching {len(universe)} SET50 tickers from yfinance...')
    data = yf.download([s + '.BK' for s in universe],
                       start=start, end=end + timedelta(days=1),
                       auto_adjust=True, progress=False)
    close = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
    close.columns = [c.replace('.BK', '') for c in close.columns]
    close = close.dropna(axis=1, how='all')
    returns = close.pct_change().dropna(how='all')
    return returns


# ============================================================
# 3. Feature assembly
# ============================================================
def assemble_features(returns, mentions_df):
    """Build per (stock, day) feature matrix.
    Returns X (DataFrame), y (Series of int in {0, 1})."""
    print('  Assembling features...')
    stocks_df   = pd.read_csv(DATA_DIR / 'stocks.csv')
    hype_hubs   = pd.read_csv(MID_OUT / 'hype_hubs.csv')
    communities = pd.read_csv(MID_OUT / 'communities.csv')
    centrality  = pd.read_csv(MID_OUT / 'centrality.csv')
    try:
        sent = pd.read_csv(DATA_DIR / 'sentiment.csv')
    except FileNotFoundError:
        sent = pd.DataFrame(columns=['PostID', 'Score'])

    # Static per-stock lookup
    hype_score = dict(zip(hype_hubs['Symbol'], hype_hubs['HypeScore']))
    degree_cen = dict(zip(centrality['Symbol'], centrality['DegreeCentrality']))
    community  = dict(zip(communities['Symbol'], communities['Community']))

    # Post-level sentiment merge (score per PostID) -> mention-level
    m = mentions_df.explode('Mentions').rename(columns={'Mentions': 'Symbol'})
    m = m.merge(sent[['PostID', 'Score']], on='PostID', how='left')
    m['Score'] = m['Score'].fillna(0)
    m['Day'] = m['Date'].dt.normalize()

    # Rolling attention count + rolling sentiment per (stock, day)
    daily_mention = (
        m.groupby(['Symbol', 'Day']).size()
         .unstack(fill_value=0).T   # index=day, columns=stock
    )
    daily_sent = (
        m.groupby(['Symbol', 'Day'])['Score'].mean()
         .unstack(fill_value=0).T
    )

    # Reindex to full trading calendar
    calendar = returns.index
    daily_mention = daily_mention.reindex(calendar, method='ffill').fillna(0)
    daily_sent    = daily_sent.reindex(calendar, method='ffill').fillna(0)
    for s in returns.columns:
        if s not in daily_mention.columns:
            daily_mention[s] = 0
        if s not in daily_sent.columns:
            daily_sent[s] = 0
    daily_mention = daily_mention[returns.columns]
    daily_sent    = daily_sent[returns.columns]

    # 10-day rolling mention, 5-day rolling sentiment
    roll_ment = daily_mention.rolling(10, min_periods=1).sum()
    roll_sent = daily_sent.rolling(5, min_periods=1).mean()

    # Build long-format feature matrix
    rows = []
    community_dummies = sorted(set(community.values()))
    comm_to_ix = {c: i for i, c in enumerate(community_dummies)}
    n_communities = len(community_dummies)

    for d_idx in range(5, len(calendar) - 1):
        d = calendar[d_idx]
        d_next = calendar[d_idx + 1]
        cs_ranks = returns.iloc[d_idx].rank(pct=True)   # cross-sectional rank of today's returns
        for s in returns.columns:
            r_next = returns.at[d_next, s]
            if pd.isna(r_next):
                continue
            r_lag1 = returns.at[calendar[d_idx],     s]
            r_lag5 = returns.iloc[d_idx - 4: d_idx + 1][s]
            if pd.isna(r_lag1) or r_lag5.isna().all():
                continue

            comm_vec = [0] * n_communities
            if s in community:
                comm_vec[comm_to_ix[community[s]]] = 1

            row = {
                'Date':          d,
                'Symbol':        s,
                # own return lags -- baseline B0
                'r_lag1':        float(r_lag1),
                'r_lag5_mean':   float(r_lag5.mean()),
                'r_lag5_std':    float(r_lag5.std()),
                'cs_rank':       float(cs_ranks.get(s, 0.5)),
                # static graph B1
                'hype_score':    float(hype_score.get(s, 0)),
                'degree_cen':    float(degree_cen.get(s, 0)),
                # rolling attention + sentiment B2
                'mention_10d':   float(roll_ment.at[d, s]),
                'sentiment_5d':  float(roll_sent.at[d, s]),
                # target
                'y_sign_next':   int(r_next > 0),
                'r_next':        float(r_next),
            }
            # community one-hot B1
            for i, cid in enumerate(community_dummies):
                row[f'comm_{cid.replace(" ", "_")}'] = comm_vec[i]
            rows.append(row)

    X = pd.DataFrame(rows).sort_values(['Date', 'Symbol']).reset_index(drop=True)
    return X


# ============================================================
# 4. Feature slices per model
# ============================================================
LAG_COLS   = ['r_lag1', 'r_lag5_mean', 'r_lag5_std', 'cs_rank']
GRAPH_COLS = ['hype_score', 'degree_cen']
ATTN_COLS  = ['mention_10d', 'sentiment_5d']


def feature_slices(all_columns):
    comm_cols = [c for c in all_columns if c.startswith('comm_')]
    B0 = LAG_COLS
    B1 = LAG_COLS + GRAPH_COLS + comm_cols
    B2 = LAG_COLS + GRAPH_COLS + comm_cols + ATTN_COLS
    return {'B0': B0, 'B1': B1, 'B2': B2, 'B3': B2}


# ============================================================
# 5. Expanding-window CV
# ============================================================
def expanding_window_folds(dates, n_splits=N_FOLDS):
    """Return list of (train_mask, test_mask) tuples on the Date column."""
    uniq = sorted(dates.unique())
    fold_size = max(len(uniq) // (n_splits + 1), 3)
    for k in range(n_splits):
        train_end_ix = fold_size * (k + 1)
        test_end_ix  = min(train_end_ix + fold_size, len(uniq))
        train_days = set(uniq[:train_end_ix])
        test_days  = set(uniq[train_end_ix:test_end_ix])
        if not test_days:
            break
        yield train_days, test_days


# ============================================================
# 6. Model fits
# ============================================================
def fit_predict(model_name, X_train, y_train, X_test):
    if model_name == 'B3':
        m = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RNG_SEED, use_label_encoder=False,
            eval_metric='logloss', verbosity=0,
        )
    else:
        m = LogisticRegression(
            max_iter=2000, C=1.0, random_state=RNG_SEED,
        )
    m.fit(X_train, y_train)
    proba = m.predict_proba(X_test)[:, 1]
    pred  = (proba > 0.5).astype(int)
    return pred, proba, m


# ============================================================
# 7. Portfolio backtest (Sharpe proxy)
# ============================================================
def daily_long_short_pnl(X_test, proba):
    """Long top-decile predicted up, short bottom-decile predicted down,
    equal-weight within each leg. Return daily portfolio return series."""
    df = X_test[['Date', 'Symbol', 'r_next']].copy()
    df['proba'] = proba

    pnl = []
    for day, group in df.groupby('Date'):
        n = len(group)
        if n < 5:
            pnl.append((day, 0.0))
            continue
        k = max(1, int(np.ceil(n * 0.10)))
        top = group.nlargest(k, 'proba')
        bot = group.nsmallest(k, 'proba')
        long_r  = top['r_next'].mean()
        short_r = bot['r_next'].mean()
        pnl.append((day, long_r - short_r))
    return pd.DataFrame(pnl, columns=['Date', 'pnl']).set_index('Date')


def sharpe_proxy(pnl_series, ann=252):
    r = pnl_series.dropna()
    if r.std() == 0 or len(r) < 2:
        return float('nan')
    return float(r.mean() / r.std() * np.sqrt(ann))


# ============================================================
# 8. Bootstrap CI on accuracy
# ============================================================
def bootstrap_metric(y_true, y_pred, n_boot=500, seed=RNG_SEED):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        accs.append((y_true[idx] == y_pred[idx]).mean())
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


# ============================================================
# 9. Main pipeline
# ============================================================
def main():
    print('=' * 65)
    print('Path B — Return Prediction with Graph Features')
    print('=' * 65)

    print('\n[1/5] Loading Midterm outputs + fetching return history...')
    mentions_df = extract_all_mentions()
    returns = fetch_returns()
    print(f'  Returns: {returns.shape[0]} days x {returns.shape[1]} stocks')

    print('\n[2/5] Assembling feature matrix...')
    X = assemble_features(returns, mentions_df)
    print(f'  Feature matrix: {X.shape[0]} rows x {X.shape[1]} columns')
    X.to_csv(FIN_OUT / 'rp_features.csv', index=False)

    slices = feature_slices(X.columns.tolist())
    for k, cols in slices.items():
        print(f'  {k}: {len(cols)} features')

    print(f'\n[3/5] Expanding-window CV ({N_FOLDS} folds)...')

    results = {}
    pnl_by_model = {}
    xgb_last_model = None
    xgb_last_features = None

    for model_name in ['B0', 'B1', 'B2', 'B3']:
        cols = slices[model_name]
        fold_metrics = []
        fold_pnl_series = []

        for k, (train_days, test_days) in enumerate(
            expanding_window_folds(X['Date'], N_FOLDS), 1
        ):
            train = X[X['Date'].isin(train_days)]
            test  = X[X['Date'].isin(test_days)]
            if len(train) < 20 or len(test) < 5:
                continue

            X_train, y_train = train[cols].values, train['y_sign_next'].values
            X_test,  y_test  = test[cols].values,  test['y_sign_next'].values

            pred, proba, m = fit_predict(model_name, X_train, y_train, X_test)
            acc = accuracy_score(y_test, pred)
            f1  = f1_score(y_test, pred, zero_division=0)

            # portfolio
            pnl = daily_long_short_pnl(test.reset_index(drop=True), proba)
            fold_pnl_series.append(pnl)

            fold_metrics.append({
                'fold': k, 'n_train': len(train), 'n_test': len(test),
                'accuracy': acc, 'f1': f1,
            })
            if model_name == 'B3' and k == N_FOLDS:
                xgb_last_model = m
                xgb_last_features = cols

        y_true_all = np.concatenate(
            [X[X['Date'].isin(td)]['y_sign_next'].values
             for _, td in expanding_window_folds(X['Date'], N_FOLDS)
             if X[X['Date'].isin(td)].shape[0] >= 5]
        )
        y_pred_all = []
        for fm, (train_days, test_days) in zip(
            fold_metrics, expanding_window_folds(X['Date'], N_FOLDS)
        ):
            test = X[X['Date'].isin(test_days)]
            if len(test) < 5: continue
            train = X[X['Date'].isin(train_days)]
            pred, _, _ = fit_predict(
                model_name, train[cols].values, train['y_sign_next'].values,
                test[cols].values,
            )
            y_pred_all.extend(pred.tolist())
        y_pred_all = np.array(y_pred_all)
        y_true_all = y_true_all[:len(y_pred_all)]

        ci = bootstrap_metric(y_true_all, y_pred_all)
        mean_acc = np.mean([f['accuracy'] for f in fold_metrics])
        mean_f1  = np.mean([f['f1']       for f in fold_metrics])

        pnl_all = pd.concat(fold_pnl_series).sort_index() if fold_pnl_series else pd.DataFrame()
        sharpe  = sharpe_proxy(pnl_all['pnl']) if len(pnl_all) else float('nan')

        results[model_name] = {
            'mean_accuracy': float(mean_acc),
            'mean_f1':       float(mean_f1),
            'acc_ci_low':    ci[0],
            'acc_ci_high':   ci[1],
            'sharpe_proxy':  sharpe,
            'n_folds':       len(fold_metrics),
            'fold_detail':   fold_metrics,
        }
        pnl_by_model[model_name] = pnl_all
        print(f'  {model_name}: acc={mean_acc:.3f} '
              f'[{ci[0]:.3f}, {ci[1]:.3f}]  '
              f'F1={mean_f1:.3f}  Sharpe={sharpe:.2f}')

    print('\n[4/5] Saving metrics + feature importance...')
    with open(FIN_OUT / 'rp_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # XGBoost feature importance (last fold model)
    if xgb_last_model is not None:
        importances = xgb_last_model.feature_importances_
        imp_df = (
            pd.DataFrame({'feature': xgb_last_features,
                          'importance': importances})
              .sort_values('importance', ascending=False)
              .head(15)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(imp_df['feature'][::-1], imp_df['importance'][::-1],
                color='#4A6FA5', edgecolor='white')
        ax.set_xlabel('Gain importance', fontweight='bold')
        ax.set_title('XGBoost feature importance (top 15)',
                     fontweight='bold', pad=10)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(FIN_OUT / 'rp_feature_importance.png', dpi=150)
        plt.savefig(FIN_OUT / 'rp_feature_importance.svg')
        plt.close()

    print('\n[5/5] Rendering figures...')

    # Baseline vs full accuracy
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(results.keys())
    accs  = [results[n]['mean_accuracy'] for n in names]
    ci_lo = [results[n]['acc_ci_low']    for n in names]
    ci_hi = [results[n]['acc_ci_high']   for n in names]
    err_lo = [a - lo for a, lo in zip(accs, ci_lo)]
    err_hi = [hi - a for a, hi in zip(accs, ci_hi)]
    colors = ['#A0A0A0', '#7A9CC6', '#4A6FA5', '#C1272D']
    bars = ax.bar(names, accs, color=colors, edgecolor='white',
                  yerr=[err_lo, err_hi], capsize=5)
    ax.axhline(0.5, ls='--', color='#999', lw=1)
    ax.set_ylabel('Directional accuracy', fontweight='bold')
    ax.set_ylim(0.4, max(0.65, max(ci_hi) * 1.05))
    ax.set_title('Baseline vs graph-augmented models\n(mean over 5 CV folds, 95 % CI)',
                 fontweight='bold', pad=10)
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIN_OUT / 'rp_baseline_vs_full.png', dpi=150)
    plt.savefig(FIN_OUT / 'rp_baseline_vs_full.svg')
    plt.close()

    # Sharpe curve (cumulative PnL of best model)
    best = max(results, key=lambda k: results[k]['sharpe_proxy']
               if not np.isnan(results[k]['sharpe_proxy']) else -np.inf)
    pnl = pnl_by_model[best]
    if len(pnl):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        cum = (1 + pnl['pnl']).cumprod()
        ax.plot(cum.index, cum.values, color='#4A6FA5', lw=2,
                label=f'{best} portfolio')
        ax.axhline(1.0, ls='--', color='#999', lw=1, label='no-trade baseline')
        ax.set_ylabel('Cumulative return (multiplier)', fontweight='bold')
        ax.set_title(f'Long top-decile / short bottom-decile portfolio '
                     f'({best})\nSharpe proxy = {results[best]["sharpe_proxy"]:.2f}',
                     fontweight='bold', pad=10)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        plt.tight_layout()
        plt.savefig(FIN_OUT / 'rp_sharpe_curve.png', dpi=150)
        plt.savefig(FIN_OUT / 'rp_sharpe_curve.svg')
        plt.close()

    print('\n=== DONE ===')
    for p in sorted(FIN_OUT.iterdir()):
        if p.name.startswith('rp_'):
            print(f'  {p.name}  ({p.stat().st_size // 1024} KB)')


if __name__ == '__main__':
    main()
