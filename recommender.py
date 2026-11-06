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

# How much each kind of signal steers the ranking, like an e-commerce funnel
# (purchase > add to cart > view). Each kind gets its own fold-in user vector; their
# catalog scores are z-normalised and mixed with these shares, so 20 casual page views
# can never outweigh a handful of ratings just by count. Missing kinds are dropped and
# the remaining shares rescaled to sum to 1.
SIGNAL_SHARES = {"rated": 0.7, "bookmarked": 0.2, "viewed": 0.1}

# Confidence of a single item *within* its kind, in confidence_weight(rating) units
# (5-star = 3.0, 4-star = 2.0, 3-star = 1.0, <= 2.5-star = 0.5).
BOOKMARK_WEIGHT = 1.5
VIEW_WEIGHT = 0.4
MAX_VIEWS = 100            # only the most recent page views are used
HYBRID_MIN_WEIGHT = 6.0    # below this total confidence (~4 good ratings), blend in genres
# Genre-profile preference per signal: rating r contributes (r - 3); bookmark/view are mild likes.
BOOKMARK_PREF, VIEW_PREF = 1.0, 0.5
# When explaining a pick, prefer stronger signals for equally similar source movies.
REASON_PRIORITY = {"rated": 1.0, "bookmarked": 0.9, "viewed": 0.8}
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
        # Fixed-width unicode array so substring search runs in C (np.char.find), not Python.
        self.title_lower = items["title"].str.lower().to_numpy().astype(str)
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
        self.G_bool = G > 0

        # Every sort order is computed once; a search just filters it with a boolean mask.
        self.sort_orders = {
            "popular": self.popular_order,
            "rating": np.argsort(-items["damped_mean"].to_numpy(), kind="stable"),
            "year": np.argsort(-items["year"].to_numpy(), kind="stable"),
            "title": np.argsort(self.title_lower, kind="stable"),
        }

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
            mask &= np.char.find(self.title_lower, q.lower().strip()) >= 0
        if genre:
            mask &= self.G_bool[:, self.genre_index[genre]]
        order = self.sort_orders.get(sort, self.popular_order)
        idx = order[mask[order]]
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
        return self.fold_in_ials_weighted(idx, confidence_weight(ratings))

    def fold_in_ials_weighted(self, idx, weights):
        """Fold-in with arbitrary per-item signal weights (ratings, bookmarks, views)."""
        Yr = self.Y[idx].astype(np.float64)
        c = self.alpha * np.asarray(weights, dtype=np.float64)
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
    def collect_signals(self, ratings=None, bookmarks=(), views=None):
        """Merge a user's interactions into one entry per movie, strongest signal wins.

        Returns parallel arrays: catalog index, confidence weight, genre preference, kind.
        """
        signals = {}
        recent_views = sorted((views or {}).items(), key=lambda kv: kv[1], reverse=True)[:MAX_VIEWS]
        for m, _ in recent_views:
            if int(m) in self.index:
                signals[self.index[int(m)]] = (VIEW_WEIGHT, VIEW_PREF, "viewed")
        for m in bookmarks or ():
            if int(m) in self.index:
                signals[self.index[int(m)]] = (BOOKMARK_WEIGHT, BOOKMARK_PREF, "bookmarked")
        for m, r in (ratings or {}).items():
            if int(m) in self.index and r is not None:
                r = float(r)
                signals[self.index[int(m)]] = (float(confidence_weight([r])[0]), r - 3.0, "rated")
        idx = np.fromiter(signals.keys(), dtype=np.int64, count=len(signals))
        w = np.array([v[0] for v in signals.values()])
        pref = np.array([v[1] for v in signals.values()])
        kinds = np.array([v[2] for v in signals.values()])
        return idx, w, pref, kinds

    def recommend(self, user_ratings=None, n=20, exclude=(), bookmarks=(), views=None):
        """Personalised picks from ratings ({movieId: stars}), bookmarks ([movieId]) and
        page views ({movieId: timestamp}). Returns (strategy, [movie dicts], shares).

        Ratings, bookmarks and views steer the ranking with SIGNAL_SHARES (70/20/10);
        everything the user interacted with is excluded from the results.
        """
        idx, w, pref, kinds = self.collect_signals(user_ratings, bookmarks, views)
        blocked = {self.index[int(m)] for m in exclude if int(m) in self.index} | set(idx.tolist())

        if len(idx) == 0:
            order = [k for k in self.popular_order[: n + len(blocked)] if k not in blocked][:n]
            return "popular", [self.movie(k) for k in order], {}

        scores, shares = self.blend_signal_scores(idx, w, kinds)
        strategy = "implicit_als"

        if w.sum() < HYBRID_MIN_WEIGHT:
            # Little signal so far: blend in a genre profile (liked genres up, disliked down).
            profile = (pref[:, None] * self.G_unit[idx]).sum(axis=0)
            content = self.G_unit @ profile
            z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
            pop = np.log1p(self.items["n_ratings"].to_numpy())
            scores = 0.5 * scores + 0.3 * z(content) + 0.2 * z(pop)
            strategy = "hybrid"

        scores = scores.astype(np.float64)
        scores[list(blocked)] = -np.inf
        top = np.argpartition(-scores, n)[:n]
        top = top[np.argsort(-scores[top])]

        # Predicted stars come from the explicit MF model, so they need actual ratings.
        rated = kinds == "rated"
        predicted = None
        if rated.any():
            stars = pref[rated] + 3.0
            p, bu = self.fold_in_mf(idx[rated], stars)
            predicted = np.clip(self.mu + bu + self.bi[top] + self.Q[top] @ p, 0.5, 5.0)

        # Explanation sources: liked ratings, bookmarks, views.
        src_mask = (~rated) | (pref + 3.0 >= LIKED_THRESHOLD)
        src_idx, src_kind = idx[src_mask], kinds[src_mask]
        sims = None
        if len(src_idx):
            priority = np.array([REASON_PRIORITY[k] for k in src_kind])
            sims = (self.Y_unit[top] @ self.Y_unit[src_idx].T) * priority[None, :]

        out = []
        for row, k in enumerate(top):
            m = self.movie(k, scores[k])
            m["predictedRating"] = None if predicted is None else round(float(predicted[row]) * 2) / 2
            if sims is not None:
                j = int(np.argmax(sims[row]))
                if sims[row, j] > 0.2:
                    src = src_idx[j]
                    m["because"] = {"id": int(self.movie_ids[src]),
                                    "title": self.items["title"].iloc[src],
                                    "reason": str(src_kind[j])}
            out.append(m)
        return strategy, out, shares

    def blend_signal_scores(self, idx, w, kinds):
        """One fold-in per signal kind, z-scored over the catalog, mixed by SIGNAL_SHARES.

        Returns (scores, shares actually used). z-scoring puts the three score vectors on
        the same scale, so the shares are what decides each kind's influence.
        """
        present = {k: v for k, v in SIGNAL_SHARES.items() if (kinds == k).any()}
        total = sum(present.values())
        shares = {k: v / total for k, v in present.items()}
        scores = np.zeros(len(self.Y))
        for kind, share in shares.items():
            mask = kinds == kind
            s = self.Y @ self.fold_in_ials_weighted(idx[mask], w[mask])
            scores += share * (s - s.mean()) / (s.std() + 1e-9)
        return scores, {k: round(v, 4) for k, v in shares.items()}
