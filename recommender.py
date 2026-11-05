"""Online recommender: loads the exported artifacts and serves recommendations with numpy only.

No retraining happens here. A website user's ratings are "folded in" against the fixed
item factors with the same closed-form least-squares step ALS uses during training:

Implicit ALS (ranking). With confidence c_i = alpha * w_i for each rated item, the user
vector x minimizing  sum_i c_ui (p_ui - x.y_i)^2 + reg |x|^2  over ALL items is
    x = (Y^T Y + reg I + sum_{i rated} (c_i - 1) y_i y_i^T)^{-1}  sum_{i rated} c_i y_i
Y^T Y is precomputed once (artifacts/ials_gram.npy), so each request costs one k x k solve.

Explicit MF (predicted star rating). Ridge regression for [p_u, b_u] on target r - mu - b_i,
exactly as in ml/models.py::fold_in_user.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
LIKED_THRESHOLD = 3.5      # ratings at or above this count as "liked" for explanations
HYBRID_MAX_RATINGS = 4     # below this many ratings, blend in genre-based content scores
SORTS = {"popular", "rating", "year", "title"}
# MovieLens stores "Dark Knight, The" and "Lives of Others, The (Das Leben der Anderen)".
_ARTICLE_RE = re.compile(
    r"^(?P<base>.+?), (?P<art>The|A|An|Les|La|Le|L'|Il|Das|Der|Die|El|Los|Las)(?P<rest> \(.*\))?$")


def display_title(title):
    m = _ARTICLE_RE.match(title)
    if not m:
        return title
    sep = "" if m["art"].endswith("'") else " "
    return f"{m['art']}{sep}{m['base']}{m['rest'] or ''}"


def confidence_weight(ratings):
    """Must match ml/models.py::ImplicitALS._confidence."""
    return np.clip(np.asarray(ratings, dtype=np.float64) - 2.0, 0.5, None)


class Recommender:
    def __init__(self, artifacts=None):
        artifacts = Path(artifacts or os.getenv("ARTIFACTS_DIR") or ARTIFACTS)
        meta = json.loads((artifacts / "model.json").read_text())
        self.meta = meta
        items = pd.read_parquet(artifacts / "items.parquet")
        items["year"] = items["year"].astype("float").fillna(0).astype(int)
        items["tmdbId"] = items["tmdbId"].astype("float")
        items["title"] = items["title"].map(display_title)
        self.items = items
        self.movie_ids = items["movieId"].to_numpy()
        self.index = {int(m): k for k, m in enumerate(self.movie_ids)}
        self.title_lower = items["title"].str.lower().to_numpy()
        self.genres = [g.split("|") if g else [] for g in items["genres"]]
        self.all_genres = sorted({g for gs in self.genres for g in gs})
        self.popular_order = np.argsort(-items["n_ratings"].to_numpy(), kind="stable")

        # Implicit ALS
        self.Y = np.load(artifacts / "ials_items.npy").astype(np.float32)
        self.gram = np.load(artifacts / "ials_gram.npy").astype(np.float64)
        self.alpha = meta["ials"]["alpha"]
        self.reg = meta["ials"]["reg"]
        norms = np.linalg.norm(self.Y, axis=1, keepdims=True)
        self.Y_unit = self.Y / np.maximum(norms, 1e-9)

        # Explicit MF
        self.Q = np.load(artifacts / "mf_items.npy").astype(np.float32)
        self.bi = np.load(artifacts / "mf_bias.npy").astype(np.float32)
        self.mu = meta["mf"]["mu"]
        self.mf_reg, self.mf_reg_b = meta["mf"]["reg"], meta["mf"]["reg_b"]

        # Genre matrix for the content-based cold-start blend
        self.genre_index = {g: j for j, g in enumerate(self.all_genres)}
        G = np.zeros((len(items), len(self.all_genres)), dtype=np.float32)
        for k, gs in enumerate(self.genres):
            for g in gs:
                G[k, self.genre_index[g]] = 1.0
        self.G_unit = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-9)

    # ---------------------------------------------------------------- catalog
    def movie(self, k, score=None):
        row = self.items.iloc[k]
        tmdb = row["tmdbId"]
        out = {
            "id": int(row["movieId"]),
            "title": row["title"],
            "year": int(row["year"]) or None,
            "genres": self.genres[k],
            "tmdbId": None if np.isnan(tmdb) else int(tmdb),
            "avgRating": round(float(row["mean_rating"]), 2),
            "numRatings": int(row["n_ratings"]),
        }
        if score is not None:
            out["score"] = round(float(score), 4)
        return out

    def get(self, movie_id):
        k = self.index.get(int(movie_id))
        return None if k is None else self.movie(k)

    def get_many(self, movie_ids):
        return [self.movie(self.index[int(m)]) for m in movie_ids if int(m) in self.index]

    def search(self, q="", genre=None, sort="popular", page=1, page_size=24):
        mask = np.ones(len(self.items), dtype=bool)
        if q:
            ql = q.lower().strip()
            mask &= np.fromiter((ql in t for t in self.title_lower), dtype=bool, count=len(mask))
        if genre:
            mask &= np.fromiter((genre in gs for gs in self.genres), dtype=bool, count=len(mask))
        idx = np.where(mask)[0]
        if sort == "rating":
            idx = idx[np.argsort(-self.items["damped_mean"].to_numpy()[idx], kind="stable")]
        elif sort == "year":
            idx = idx[np.argsort(-self.items["year"].to_numpy()[idx], kind="stable")]
        elif sort == "title":
            idx = idx[np.argsort(self.title_lower[idx], kind="stable")]
        else:  # popular
            idx = idx[np.argsort(-self.items["n_ratings"].to_numpy()[idx], kind="stable")]
        start = (page - 1) * page_size
        return {"total": int(len(idx)), "page": page, "pageSize": page_size,
                "results": [self.movie(k) for k in idx[start:start + page_size]]}

    def similar(self, movie_id, n=12):
        k = self.index.get(int(movie_id))
        if k is None:
            return []
        sims = self.Y_unit @ self.Y_unit[k]
        sims[k] = -np.inf
        top = np.argpartition(-sims, n)[:n]
        top = top[np.argsort(-sims[top])]
        return [self.movie(j, sims[j]) for j in top]

    # ---------------------------------------------------------------- fold-in
    def fold_in_ials(self, idx, ratings):
        Yr = self.Y[idx].astype(np.float64)
        c = self.alpha * confidence_weight(ratings)
        A = self.gram + self.reg * np.eye(self.gram.shape[0]) + (Yr.T * (c - 1.0)) @ Yr
        b = Yr.T @ c
        return np.linalg.solve(A, b)

    def fold_in_mf(self, idx, ratings):
        Qr = self.Q[idx].astype(np.float64)
        n, k = Qr.shape
        X = np.hstack([Qr, np.ones((n, 1))])
        y = np.asarray(ratings, dtype=np.float64) - self.mu - self.bi[idx]
        A = X.T @ X + n * np.diag(np.r_[np.full(k, self.mf_reg), self.mf_reg_b])
        w = np.linalg.solve(A, X.T @ y)
        return w[:k], w[k]

    # ---------------------------------------------------------------- recommend
    def recommend(self, user_ratings, n=20, exclude=()):
        """user_ratings: {movieId: rating}. Returns (strategy, [movie dicts])."""
        pairs = [(self.index[int(m)], float(r)) for m, r in user_ratings.items()
                 if int(m) in self.index and r is not None]
        blocked = {self.index[int(m)] for m in exclude if int(m) in self.index}
        blocked |= {k for k, _ in pairs}

        if not pairs:
            order = [k for k in self.popular_order[: n + len(blocked)] if k not in blocked][:n]
            return "popular", [self.movie(k) for k in order]

        idx = np.array([k for k, _ in pairs])
        r = np.array([v for _, v in pairs])
        x = self.fold_in_ials(idx, r)
        scores = self.Y @ x
        strategy = "implicit_als"

        if len(pairs) < HYBRID_MAX_RATINGS:
            # Very little signal: blend in a genre profile (liked genres up, disliked down).
            profile = ((r - 3.0)[:, None] * self.G_unit[idx]).sum(axis=0)
            content = self.G_unit @ profile
            z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
            pop = np.log1p(self.items["n_ratings"].to_numpy())
            scores = 0.5 * z(scores) + 0.3 * z(content) + 0.2 * z(pop)
            strategy = "hybrid"

        scores = scores.astype(np.float64)
        scores[list(blocked)] = -np.inf
        top = np.argpartition(-scores, n)[:n]
        top = top[np.argsort(-scores[top])]

        p, bu = self.fold_in_mf(idx, r)
        predicted = np.clip(self.mu + bu + self.bi[top] + self.Q[top] @ p, 0.5, 5.0)

        liked = idx[r >= LIKED_THRESHOLD]
        sims = self.Y_unit[top] @ self.Y_unit[liked].T if len(liked) else None
        out = []
        for row, k in enumerate(top):
            m = self.movie(k, scores[k])
            m["predictedRating"] = round(float(predicted[row]) * 2) / 2
            if sims is not None:
                j = int(np.argmax(sims[row]))
                if sims[row, j] > 0.2:
                    src = liked[j]
                    m["because"] = {"id": int(self.movie_ids[src]), "title": self.items["title"].iloc[src]}
            out.append(m)
        return strategy, out
