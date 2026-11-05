"""Train every model on the temporal train split and evaluate on the held-out future.

Usage:
    python -m ml.evaluate                 # full run, writes reports/results.md + results.json
    python -m ml.evaluate --tune          # also grid-search MF / ALS on a validation split

Metrics
-------
RMSE (rating prediction), over every held-out rating:
    sqrt( mean_{(u,i) in test} (r_ui - r_hat_ui)^2 )

Top-N ranking, for a sample of test users, ranking every item the user has not rated in
train. Relevant items Rel_u = held-out items rated >= 4.
    Recall@K = |top_K(u) ∩ Rel_u| / min(K, |Rel_u|)
    NDCG@K   = DCG@K / IDCG@K,  DCG@K = sum_{r=1..K} rel_r / log2(r + 1)
where IDCG@K is the DCG of a perfect ranking (all relevant items first).
Coverage@K: fraction of the catalog that appears in at least one user's top K.
"""
import argparse
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # implicit's own threading does the work

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ml import config
from ml.models import BiasBaseline, BiasedMF, ImplicitALS, ItemKNN, Popularity


def load_split():
    d = config.PROCESSED_DIR
    train = pd.read_parquet(d / "train.parquet")
    test = pd.read_parquet(d / "test.parquet")
    n_users = int(max(train.u.max(), test.u.max())) + 1
    n_items = int(max(train.i.max(), test.i.max())) + 1
    X = sp.csr_matrix((train.rating.to_numpy(), (train.u.to_numpy(), train.i.to_numpy())),
                      shape=(n_users, n_items), dtype=np.float32)
    return train, test, X


def validation_split(train):
    """Carve a validation set out of train the same way test was carved: per-user last 10%."""
    tr = train.sort_values(["u", "ts", "i"], kind="mergesort").reset_index(drop=True)
    pos = tr.groupby("u").cumcount().to_numpy()
    n = tr.groupby("u")["i"].transform("size").to_numpy()
    is_val = pos >= n - np.maximum(1, (n * 0.1).astype(int))
    return tr[~is_val], tr[is_val]


def to_csr(df, shape):
    return sp.csr_matrix((df.rating.to_numpy(), (df.u.to_numpy(), df.i.to_numpy())),
                         shape=shape, dtype=np.float32)


def rmse(model, test):
    pred = np.clip(model.predict(test.u.to_numpy(), test.i.to_numpy()), 0.5, 5.0)
    return float(np.sqrt(np.mean((pred - test.rating.to_numpy()) ** 2)))


def ranking_metrics(model, X, test, users, k=config.EVAL_K, batch=1000):
    rel = test[test.rating >= config.RELEVANT_THRESHOLD]
    rel_sets = rel.groupby("u")["i"].apply(lambda s: set(s.to_numpy())).to_dict()
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    recalls, ndcgs, seen = [], [], np.zeros(X.shape[1], dtype=bool)
    for start in range(0, len(users), batch):
        ub = users[start:start + batch]
        scores = np.array(model.score_users(ub, X), dtype=np.float32, copy=True)
        Xb = X[ub]
        scores[Xb.nonzero()] = -np.inf  # never recommend what the user already rated
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        top = np.take_along_axis(top, order, axis=1)
        seen[top.ravel()] = True
        for u, items in zip(ub, top):
            relevant = rel_sets[u]
            hits = np.fromiter((i in relevant for i in items), dtype=bool, count=k)
            recalls.append(hits.sum() / min(k, len(relevant)))
            idcg = discounts[: min(k, len(relevant))].sum()
            ndcgs.append((hits * discounts).sum() / idcg)
    return {f"recall@{k}": float(np.mean(recalls)), f"ndcg@{k}": float(np.mean(ndcgs)),
            f"coverage@{k}": float(seen.mean())}


def sample_eval_users(test, n, seed=config.SEED):
    has_rel = test.loc[test.rating >= config.RELEVANT_THRESHOLD, "u"].unique()
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(has_rel, size=min(n, len(has_rel)), replace=False))


def tune(train, X_shape, parts=("mf", "ials")):
    """Small grid search on the validation split (test is never touched while tuning)."""
    tr, val = validation_split(train)
    Xtr = to_csr(tr, X_shape)
    users = sample_eval_users(val, 5000)
    out = {}
    if "mf" in parts:
        for reg in (0.02, 0.05, 0.1):
            m = BiasedMF(factors=64, reg=reg, iterations=8, verbose=False).fit(Xtr)
            out[f"mf reg={reg}"] = rmse(m, val)
            print(f"  MF reg={reg}: val RMSE {out[f'mf reg={reg}']:.4f}")
    if "ials" in parts:
        for weight in ("rating", "binary"):
            for alpha in (0.25, 0.5, 1.0, 2.0):
                m = ImplicitALS(factors=128, alpha=alpha, reg=0.05, weight=weight, iterations=12).fit(Xtr)
                r = ranking_metrics(m, Xtr, val, users)
                out[f"ials weight={weight} alpha={alpha}"] = r
                print(f"  iALS weight={weight} alpha={alpha}: val NDCG@10 {r['ndcg@10']:.4f} "
                      f"Recall@10 {r['recall@10']:.4f}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", nargs="*", choices=["mf", "ials"], default=None,
                        help="grid-search on a validation split first (default: both)")
    parser.add_argument("--mf-reg", type=float, default=0.05)
    parser.add_argument("--ials-alpha", type=float, default=4.0)
    parser.add_argument("--ials-reg", type=float, default=0.05)
    parser.add_argument("--ials-weight", choices=["rating", "binary"], default="rating")
    args = parser.parse_args()

    t0 = time.time()
    train, test, X = load_split()
    print(f"train {X.nnz:,} ratings, test {len(test):,}, {X.shape[0]:,} users x {X.shape[1]:,} items")
    if args.tune is not None:
        print("tuning on validation split ...")
        tuning = tune(train, X.shape, args.tune or ("mf", "ials"))
    users = sample_eval_users(test, config.EVAL_USERS)

    models = {
        "Popularity (most rated)": Popularity("count"),
        "Popularity (damped mean)": Popularity("damped"),
        "Bias baseline (mu + b_u + b_i)": BiasBaseline(iterations=5, verbose=False),
        "Item-kNN (centered cosine)": ItemKNN(),
        "Biased MF (ALS, k=64)": BiasedMF(factors=64, reg=args.mf_reg, iterations=10),
        "Implicit ALS (k=128)": ImplicitALS(factors=128, alpha=args.ials_alpha, reg=args.ials_reg,
                                             weight=args.ials_weight),
    }
    rows, fitted = [], {}
    for name, model in models.items():
        t = time.time()
        print(f"fitting {name} ...")
        model.fit(X)
        fit_s = time.time() - t
        row = {"model": name, "fit_seconds": round(fit_s, 1)}
        if hasattr(model, "predict"):
            row["rmse"] = rmse(model, test)
        row.update(ranking_metrics(model, X, test, users))
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        rows.append(row)
        fitted[name] = model

    config.REPORTS_DIR.mkdir(exist_ok=True)
    meta = {
        "dataset": "MovieLens 32M",
        "train_ratings": int(X.nnz), "test_ratings": int(len(test)),
        "users": int(X.shape[0]), "items": int(X.shape[1]),
        "eval_users": int(len(users)), "k": config.EVAL_K,
        "relevant_threshold": config.RELEVANT_THRESHOLD,
        "params": {"mf_reg": args.mf_reg, "ials_alpha": args.ials_alpha, "ials_reg": args.ials_reg,
                   "ials_weight": args.ials_weight},
    }
    if args.tune is not None:
        meta["tuning"] = tuning
    (config.REPORTS_DIR / "results.json").write_text(json.dumps({"meta": meta, "results": rows}, indent=2))
    write_markdown(rows, meta)
    print(f"total {time.time() - t0:.0f}s")
    return fitted


def write_markdown(rows, meta):
    k = meta["k"]
    lines = [
        f"Evaluated on {meta['dataset']}: {meta['train_ratings']:,} train / "
        f"{meta['test_ratings']:,} test ratings, {meta['users']:,} users, {meta['items']:,} movies. "
        f"Ranking metrics over {meta['eval_users']:,} sampled users; relevant = held-out rating "
        f">= {meta['relevant_threshold']:g}.",
        "",
        f"| Model | RMSE | Recall@{k} | NDCG@{k} | Coverage@{k} | Fit time |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        rmse_s = f"{r['rmse']:.4f}" if "rmse" in r else "–"
        lines.append(f"| {r['model']} | {rmse_s} | {r[f'recall@{k}']:.4f} | {r[f'ndcg@{k}']:.4f} | "
                     f"{r[f'coverage@{k}']:.1%} | {r['fit_seconds']:.0f}s |")
    (config.REPORTS_DIR / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
