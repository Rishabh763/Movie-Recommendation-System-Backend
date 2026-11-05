"""Retrain the chosen models on ALL ratings (train + test) and write serving artifacts.

Evaluation tells us which configuration to ship; once chosen, the production model should
learn from every rating we have. Only the SERVE_TOP_ITEMS most-rated movies are exported,
which keeps the artifacts a few MB and the API well under Render's 512 MB.

Usage:
    python -m ml.export --ials-alpha 4 --ials-reg 0.05 --mf-reg 0.05

Artifacts (artifacts/, committed):
    items.parquet        movieId, title, year, genres, tmdbId, n_ratings, mean_rating, damped_mean
    ials_items.npy       (N, k) float16  implicit-ALS item factors y_i
    ials_gram.npy        (k, k) float32  Y^T Y over ALL training items (needed for exact fold-in)
    mf_items.npy         (N, k) float16  explicit MF item factors q_i
    mf_bias.npy          (N,)   float32  item biases b_i
    model.json           hyperparameters + mu, so the API's fold-in matches training exactly
"""
import argparse
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ml import config
from ml.models import BiasedMF, ImplicitALS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ials-alpha", type=float, default=4.0)
    p.add_argument("--ials-reg", type=float, default=0.05)
    p.add_argument("--ials-factors", type=int, default=128)
    p.add_argument("--mf-reg", type=float, default=0.05)
    p.add_argument("--mf-factors", type=int, default=64)
    args = p.parse_args()

    t0 = time.time()
    d = config.PROCESSED_DIR
    ratings = pd.concat([pd.read_parquet(d / "train.parquet"), pd.read_parquet(d / "test.parquet")])
    items = pd.read_parquet(d / "items.parquet")
    n_users, n_items = int(ratings.u.max()) + 1, len(items)
    X = sp.csr_matrix((ratings.rating.to_numpy(), (ratings.u.to_numpy(), ratings.i.to_numpy())),
                      shape=(n_users, n_items), dtype=np.float32)
    print(f"full data: {X.nnz:,} ratings")

    ials = ImplicitALS(factors=args.ials_factors, alpha=args.ials_alpha, reg=args.ials_reg).fit(X)
    mf = BiasedMF(factors=args.mf_factors, reg=args.mf_reg, iterations=10, verbose=False).fit(X)

    # Item stats on the full data
    n = np.diff(X.tocsc().indptr)
    sums = np.asarray(X.sum(axis=0)).ravel()
    mu = float(X.data.mean())
    items["n_ratings"] = n.astype(np.int32)
    items["mean_rating"] = (sums / np.maximum(n, 1)).astype(np.float32)
    items["damped_mean"] = ((sums + 50 * mu) / (n + 50)).astype(np.float32)

    keep = np.sort(np.argsort(-n, kind="stable")[: config.SERVE_TOP_ITEMS])
    served = items.iloc[keep].reset_index(drop=True)
    served = served[["movieId", "title", "year", "genres", "tmdbId",
                     "n_ratings", "mean_rating", "damped_mean"]]

    out = config.ARTIFACTS_DIR
    out.mkdir(exist_ok=True)
    served.to_parquet(out / "items.parquet", index=False)
    np.save(out / "ials_items.npy", ials.V[keep].astype(np.float16))
    np.save(out / "ials_gram.npy", (ials.V.T @ ials.V).astype(np.float32))
    np.save(out / "mf_items.npy", mf.Q[keep].astype(np.float16))
    np.save(out / "mf_bias.npy", mf.bi[keep].astype(np.float32))
    meta = {
        "dataset": "MovieLens 32M",
        "trained_on_ratings": int(X.nnz),
        "served_items": int(len(keep)),
        "ials": {"factors": args.ials_factors, "alpha": args.ials_alpha, "reg": args.ials_reg,
                 "confidence": "alpha * clip(rating - 2, 0.5)"},
        "mf": {"factors": args.mf_factors, "reg": mf.reg, "reg_b": mf.reg_b, "mu": mf.mu},
    }
    (out / "model.json").write_text(json.dumps(meta, indent=2))
    sizes = {f.name: f"{f.stat().st_size / 1e6:.1f} MB" for f in sorted(out.iterdir())}
    print(f"wrote {sizes} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
