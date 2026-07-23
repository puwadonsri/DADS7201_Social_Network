"""
11 - Fine-tune WangchanBERTa on the 50-post Midterm gold set via 5-fold
cross-validation, and compare against the off-the-shelf baseline.

The Midterm review flagged the classifier as chance-level (accuracy 0.34,
Cohen kappa 0.14, Positive recall 0.00). This script shows what a light,
proof-of-concept fine-tune can recover on the same 50 posts.

Design:
    Base model:  poom-sci/WangchanBERTa-finetuned-sentiment
    CV:          Stratified 5-fold on GoldLabel
    Fine-tune:   3 epochs, batch 4, lr 2e-5, weight decay 0.01
    Metrics:     accuracy, macro F1, Cohen kappa (mean over 5 folds + std)

Inputs:
    ../Midterm/data/sentiment_gold.csv   (50 Claude-labelled posts)
    ../Midterm/data/posts.csv            (post text lookup)
    ../Midterm/data/sentiment.csv        (baseline model predictions)

Outputs (./output/):
    sentiment_finetuned_validation.json     per-fold + aggregate metrics
    sentiment_finetuned_confusion.png/.svg  aggregate confusion matrix
    sentiment_finetuned_predictions.csv     PostID, gold, baseline_pred, ft_pred
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

# torch MUST be imported before transformers on Windows
import torch  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, confusion_matrix,
    precision_recall_fscore_support,
)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

MIDTERM_DIR = Path('../Midterm')
FIN_OUT     = Path('output')
FIN_OUT.mkdir(exist_ok=True)

BASE_MODEL = 'poom-sci/WangchanBERTa-finetuned-sentiment'
LABELS     = ['Negative', 'Neutral', 'Positive']
LABEL2ID   = {l: i for i, l in enumerate(LABELS)}
ID2LABEL   = {i: l for l, i in LABEL2ID.items()}

RNG_SEED   = 42
N_FOLDS    = 5
EPOCHS     = 3
LR         = 2e-5
BATCH_SIZE = 4
MAX_LEN    = 416


# ============================================================
# 1. Load gold + text
# ============================================================
def load_gold_with_text():
    gold  = pd.read_csv(MIDTERM_DIR / 'data' / 'sentiment_gold.csv')
    posts = pd.read_csv(MIDTERM_DIR / 'data' / 'posts.csv')
    df = gold.merge(posts[['PostID', 'Text']], on='PostID', how='left')
    # Some gold PostIDs may exist in the pre-event-anchor corpus but not
    # the anchored posts.csv — fall back to posts_raw for those
    if df['Text'].isna().any():
        raw = pd.read_csv(MIDTERM_DIR / 'data' / 'posts_raw.csv')
        raw['Text'] = raw['Title'].fillna('') + ' | ' + raw['Text'].fillna('')
        raw_lookup = dict(zip(raw['PostID'], raw['Text']))
        df['Text'] = df['Text'].fillna(df['PostID'].map(raw_lookup))
    df = df.dropna(subset=['Text'])
    df['Text'] = df['Text'].str.slice(0, 1000)
    df['GoldID'] = df['GoldLabel'].map(LABEL2ID)
    return df


def load_baseline_predictions():
    sent = pd.read_csv(MIDTERM_DIR / 'data' / 'sentiment.csv')
    return dict(zip(sent['PostID'], sent['Label']))


# ============================================================
# 2. Fine-tune one fold
# ============================================================
def train_one_fold(train_texts, train_labels, val_texts, val_labels, seed):
    """Plain PyTorch training loop -- avoids transformers.Trainer which
    transitively imports torchvision (broken on this Windows setup)."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import Dataset, DataLoader

    class SentDataset(Dataset):
        def __init__(self, texts, labels, tok):
            self.enc = tok(list(texts), truncation=True, padding=True,
                           max_length=MAX_LEN, return_tensors='pt')
            self.labels = torch.tensor(list(labels), dtype=torch.long)
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            return {'input_ids':      self.enc['input_ids'][i],
                    'attention_mask': self.enc['attention_mask'][i],
                    'labels':         self.labels[i]}

    torch.manual_seed(seed)
    np.random.seed(seed)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, ignore_mismatched_sizes=True,
    )
    device = torch.device('cpu')
    model.to(device)

    train_ds = SentDataset(train_texts, train_labels, tok)
    val_ds   = SentDataset(val_texts,   val_labels,   tok)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    model.train()
    for _ in range(EPOCHS):
        for batch in train_dl:
            opt.zero_grad()
            out = model(input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        labels=batch['labels'].to(device))
            out.loss.backward()
            opt.step()

    # Predict val
    model.eval()
    with torch.no_grad():
        out = model(input_ids=val_ds.enc['input_ids'].to(device),
                    attention_mask=val_ds.enc['attention_mask'].to(device))
        pred_ids = out.logits.argmax(dim=-1).cpu().numpy()
    return [ID2LABEL[int(i)] for i in pred_ids]


# ============================================================
# 3. Main: 5-fold CV
# ============================================================
def main():
    print('=' * 65)
    print('Path C — Sentiment classifier fine-tune (5-fold CV on 50 gold)')
    print('=' * 65)

    df = load_gold_with_text()
    print(f'\n[1/4] Loaded {len(df)} gold posts with text')
    print('  Gold label distribution:')
    print(df['GoldLabel'].value_counts().to_string())

    baseline_pred = load_baseline_predictions()
    df['BaselineLabel'] = df['PostID'].map(baseline_pred).fillna('Neutral')

    print(f'\n[2/4] Training {N_FOLDS} folds x {EPOCHS} epochs each...')
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
    fold_metrics = []
    all_ft_predictions = {}   # PostID -> ft label

    for fold, (tr_idx, val_idx) in enumerate(skf.split(df['Text'], df['GoldID']), 1):
        tr, val = df.iloc[tr_idx], df.iloc[val_idx]
        print(f'  Fold {fold}: {len(tr)} train / {len(val)} val ...', end=' ')
        preds = train_one_fold(
            tr['Text'].values, tr['GoldID'].values,
            val['Text'].values, val['GoldID'].values,
            seed=RNG_SEED + fold,
        )
        gold = val['GoldLabel'].values
        acc  = accuracy_score(gold, preds)
        f1   = f1_score(gold, preds, average='macro', zero_division=0)
        kap  = cohen_kappa_score(gold, preds)
        fold_metrics.append({'fold': fold, 'n_val': len(val),
                             'accuracy': acc, 'macro_f1': f1, 'kappa': kap})
        for pid, p in zip(val['PostID'].values, preds):
            all_ft_predictions[pid] = p
        print(f'acc={acc:.2f}  F1={f1:.2f}  κ={kap:.2f}')

    print('\n[3/4] Aggregating results...')
    mean_acc  = np.mean([m['accuracy'] for m in fold_metrics])
    std_acc   = np.std([m['accuracy'] for m in fold_metrics])
    mean_f1   = np.mean([m['macro_f1'] for m in fold_metrics])
    mean_kap  = np.mean([m['kappa']    for m in fold_metrics])

    # Aggregate confusion matrix
    df['FinetunedLabel'] = df['PostID'].map(all_ft_predictions)
    cm_ft = confusion_matrix(df['GoldLabel'], df['FinetunedLabel'], labels=LABELS)
    cm_baseline = confusion_matrix(df['GoldLabel'], df['BaselineLabel'], labels=LABELS)

    baseline_acc = accuracy_score(df['GoldLabel'], df['BaselineLabel'])
    baseline_f1  = f1_score(df['GoldLabel'], df['BaselineLabel'],
                            average='macro', zero_division=0)
    baseline_kap = cohen_kappa_score(df['GoldLabel'], df['BaselineLabel'])

    ft_acc = accuracy_score(df['GoldLabel'], df['FinetunedLabel'])
    ft_f1  = f1_score(df['GoldLabel'], df['FinetunedLabel'],
                      average='macro', zero_division=0)
    ft_kap = cohen_kappa_score(df['GoldLabel'], df['FinetunedLabel'])

    per_class_ft = precision_recall_fscore_support(
        df['GoldLabel'], df['FinetunedLabel'],
        labels=LABELS, zero_division=0,
    )
    per_class_baseline = precision_recall_fscore_support(
        df['GoldLabel'], df['BaselineLabel'],
        labels=LABELS, zero_division=0,
    )
    per_class = {}
    for i, lab in enumerate(LABELS):
        per_class[lab] = {
            'baseline':  {'precision': float(per_class_baseline[0][i]),
                          'recall':    float(per_class_baseline[1][i]),
                          'f1':        float(per_class_baseline[2][i]),
                          'support':   int(per_class_baseline[3][i])},
            'finetuned': {'precision': float(per_class_ft[0][i]),
                          'recall':    float(per_class_ft[1][i]),
                          'f1':        float(per_class_ft[2][i]),
                          'support':   int(per_class_ft[3][i])},
        }

    results = {
        'n_gold_posts':   len(df),
        'n_folds':        N_FOLDS,
        'epochs_per_fold': EPOCHS,
        'baseline': {
            'accuracy': float(baseline_acc),
            'macro_f1': float(baseline_f1),
            'kappa':    float(baseline_kap),
        },
        'finetuned_cv_mean': {
            'accuracy_mean': float(mean_acc),
            'accuracy_std':  float(std_acc),
            'macro_f1_mean': float(mean_f1),
            'kappa_mean':    float(mean_kap),
        },
        'finetuned_pooled': {
            'accuracy': float(ft_acc),
            'macro_f1': float(ft_f1),
            'kappa':    float(ft_kap),
        },
        'per_class_metrics':  per_class,
        'confusion_finetuned': {gold: {model: int(v) for model, v in
                                        zip(LABELS, cm_ft[i])}
                                 for i, gold in enumerate(LABELS)},
        'confusion_baseline':  {gold: {model: int(v) for model, v in
                                        zip(LABELS, cm_baseline[i])}
                                 for i, gold in enumerate(LABELS)},
        'fold_detail': fold_metrics,
    }
    with open(FIN_OUT / 'sentiment_finetuned_validation.json', 'w',
              encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    df[['PostID', 'GoldLabel', 'BaselineLabel', 'FinetunedLabel']].to_csv(
        FIN_OUT / 'sentiment_finetuned_predictions.csv', index=False)

    print(f'\n  Baseline (WangchanBERTa off-the-shelf):')
    print(f'    accuracy={baseline_acc:.3f}  F1={baseline_f1:.3f}  '
          f'kappa={baseline_kap:.3f}')
    print(f'  Fine-tuned (5-fold CV, pooled predictions):')
    print(f'    accuracy={ft_acc:.3f}  F1={ft_f1:.3f}  kappa={ft_kap:.3f}')
    print(f'    per-fold acc: {[round(m["accuracy"], 2) for m in fold_metrics]}')

    print('\n[4/4] Rendering confusion matrices...')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cm, title in zip(
        axes, [cm_baseline, cm_ft],
        ['Baseline WangchanBERTa', f'Fine-tuned (5-fold CV, n={len(df)})'],
    ):
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black',
                        fontweight='bold', fontsize=12)
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS)
        ax.set_yticklabels(LABELS)
        ax.set_xlabel('Model prediction', fontweight='bold')
        ax.set_ylabel('Gold label', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=8)
        plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(FIN_OUT / 'sentiment_finetuned_confusion.png', dpi=150)
    plt.savefig(FIN_OUT / 'sentiment_finetuned_confusion.svg')
    plt.close()

    print('\n=== DONE ===')
    for p in sorted(FIN_OUT.iterdir()):
        if 'sentiment' in p.name:
            print(f'  {p.name}  ({p.stat().st_size // 1024} KB)')


if __name__ == '__main__':
    main()
