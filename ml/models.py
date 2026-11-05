"""Recommendation models used offline (training / evaluation) and online (fold-in).

Notation: R is the user x item rating matrix, Omega the set of observed (u, i) pairs,
mu the global mean rating, b_u / b_i user and item biases, p_u / q_i latent factors.

All models expose:
    score_users(users, X_train) -> dense (len(users), n_items) array, higher = better
Models that predict ratings also expose:
    predict(u, i) -> predicted rating for arrays of (u, i) pairs
"""
import numpy as np
import scipy.sparse as sp
from numba import njit, prange


# --------------------------------------------------------------------------------------
# Popularity baselines
# --------------------------------------------------------------------------------------
class Popularity:
    """Non-personalized baseline: every user gets the same ranking.

    mode="count": rank by number of ratings (strongest simple ranking baseline).
    mode="damped": rank by a Bayesian (damped) mean,
        score_i = (n_i * mean_i + m * mu) / (n_i + m)
    which shrinks items with few ratings toward the global mean, so one 5-star rating
    can't put an obscure title at #1. m acts as a prior "pseudo-count" of mu-ratings.
    """

    def __init__(self, mode="count", m=50):
        self.mode, self.m = mode, m

    def fit(self, X):
        n = np.diff(X.tocsc().indptr).astype(np.float64)
        sums = np.asarray(X.sum(axis=0)).ravel()
        if self.mode == "count":
            self.scores = n
        else:
            mu = X.data.mean()
            self.scores = (sums + self.m * mu) / (n + self.m)
        return self

    def score_users(self, users, X):
        return np.broadcast_to(self.scores, (len(users), len(self.scores))).astype(np.float32)


# --------------------------------------------------------------------------------------
# Biased matrix factorization trained with ALS on observed ratings only
# --------------------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _als_half_step(indptr, indices, ratings, other_f, other_b, mu, reg, reg_b):
    """Solve every row of one side while the other side is fixed.

    For a user u with rated items I_u, minimizing
        sum_{i in I_u} (r_ui - mu - b_i - b_u - p_u . q_i)^2 + reg*n_u*|p_u|^2 + reg_b*n_u*b_u^2
    over w = [p_u, b_u] is ridge regression with design X = [Q_{I_u}, 1] and target
    y = r - mu - b_i, so the closed form is
        w = (X^T X + n_u * diag(reg,...,reg, reg_b))^{-1} X^T y.
    Scaling the penalty by n_u is ALS-WR (Zhou et al., 2008). The item step is symmetric.
    """
    n_rows = len(indptr) - 1
    k = other_f.shape[1]
    out_f = np.zeros((n_rows, k), dtype=np.float32)
    out_b = np.zeros(n_rows, dtype=np.float32)
    for row in prange(n_rows):
        start, end = indptr[row], indptr[row + 1]
        n = end - start
        if n == 0:
            continue
        idx = indices[start:end]
        X = np.empty((n, k + 1))
        X[:, :k] = other_f[idx]
        X[:, k] = 1.0
        y = ratings[start:end] - mu - other_b[idx]
        A = X.T @ X
        for j in range(k):
            A[j, j] += reg * n
        A[k, k] += reg_b * n
        w = np.linalg.solve(A, X.T @ y)
        out_f[row] = w[:k]
        out_b[row] = w[k]
    return out_f, out_b


class BiasedMF:
    """r_hat(u, i) = mu + b_u + b_i + p_u . q_i, fit by alternating least squares.

    Unlike the old zero-filled TruncatedSVD, the loss only sums over observed ratings,
    so "not rated" is never treated as "rated 0".
    """

    def __init__(self, factors=64, reg=0.05, reg_b=0.01, iterations=10, seed=42, verbose=True):
        self.k, self.reg, self.reg_b = factors, reg, reg_b
        self.iterations, self.seed, self.verbose = iterations, seed, verbose

    def fit(self, X):
        X = X.tocsr().astype(np.float32)
        Xt = X.T.tocsr()
        n_users, n_items = X.shape
        rng = np.random.default_rng(self.seed)
        self.mu = float(X.data.mean())
        self.P = np.zeros((n_users, self.k), dtype=np.float32)
        self.bu = np.zeros(n_users, dtype=np.float32)
        self.Q = (rng.standard_normal((n_items, self.k)) * 0.01).astype(np.float32)
        self.bi = np.zeros(n_items, dtype=np.float32)
        for it in range(self.iterations):
            self.P, self.bu = _als_half_step(X.indptr, X.indices, X.data, self.Q, self.bi,
                                             self.mu, self.reg, self.reg_b)
            self.Q, self.bi = _als_half_step(Xt.indptr, Xt.indices, Xt.data, self.P, self.bu,
                                             self.mu, self.reg, self.reg_b)
            if self.verbose:
                pred = self.predict(*_pairs(X))
                print(f"  ALS iter {it + 1}: train RMSE {np.sqrt(np.mean((pred - X.data) ** 2)):.4f}")
        return self

    def predict(self, u, i, chunk=1_000_000):
        # Chunked: gathering P[u], Q[i] for 25M pairs at once would need ~6 GB.
        u, i = np.asarray(u), np.asarray(i)
        out = np.empty(len(u), dtype=np.float32)
        for s in range(0, len(u), chunk):
            uu, ii = u[s:s + chunk], i[s:s + chunk]
            out[s:s + chunk] = (self.mu + self.bu[uu] + self.bi[ii]
                                + np.einsum("nk,nk->n", self.P[uu], self.Q[ii]))
        return out

    def score_users(self, users, X):
        return (self.P[users] @ self.Q.T + self.bi[None, :]).astype(np.float32)


class BiasBaseline(BiasedMF):
    """mu + b_u + b_i only (BiasedMF with zero factors). Reference point for RMSE."""

    def __init__(self, **kw):
        super().__init__(factors=0, **kw)

    def score_users(self, users, X):
        return np.broadcast_to(self.bi, (len(users), len(self.bi))).astype(np.float32)


def fold_in_user(item_idx, ratings, Q, bi, mu, reg=0.05, reg_b=0.01):
    """Online version of one ALS user step: a brand-new user's (p_u, b_u) from their ratings.

    Same closed form as _als_half_step, so the API never retrains: given the items a user
    rated on the website, it solves a (k+1)x(k+1) linear system and scores the catalog.
    """
    item_idx = np.asarray(item_idx)
    n, k = len(item_idx), Q.shape[1]
    X = np.hstack([Q[item_idx].astype(np.float64), np.ones((n, 1))])
    y = np.asarray(ratings, dtype=np.float64) - mu - bi[item_idx]
    A = X.T @ X + n * np.diag(np.r_[np.full(k, reg), reg_b])
    w = np.linalg.solve(A, X.T @ y)
    return w[:k].astype(np.float32), float(w[k])


# --------------------------------------------------------------------------------------
# Implicit-feedback ALS (Hu, Koren & Volinsky, 2008) for top-N ranking
# --------------------------------------------------------------------------------------
class ImplicitALS:
    """Treat every rating as a positive interaction with confidence c_ui = 1 + alpha * w_ui.

    Minimizes sum_{all u,i} c_ui (p_ui - x_u . y_i)^2 + reg(|x_u|^2 + |y_i|^2) with
    p_ui = 1 if u rated i else 0. Because it also learns from *which* movies people chose
    to watch (not just the score), it is usually much stronger at top-N ranking than
    rating-prediction MF.
    """

    def __init__(self, factors=128, reg=0.05, alpha=2.0, iterations=15, weight="rating", seed=42):
        self.k, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.weight, self.seed = iterations, weight, seed

    def _confidence(self, X):
        C = X.tocsr().astype(np.float32, copy=True)
        if self.weight == "binary":
            C.data[:] = 1.0
        else:  # weight above-neutral ratings more; every rating still counts as a positive
            C.data = np.clip(C.data - 2.0, 0.5, None)
        return C

    def fit(self, X):
        from implicit.cpu.als import AlternatingLeastSquares
        self.model = AlternatingLeastSquares(
            factors=self.k, regularization=self.reg, alpha=self.alpha,
            iterations=self.iterations, random_state=self.seed, calculate_training_loss=False,
        )
        self.model.fit(self._confidence(X), show_progress=False)
        self.U = np.asarray(self.model.user_factors)
        self.V = np.asarray(self.model.item_factors)
        return self

    def score_users(self, users, X):
        return (self.U[users] @ self.V.T).astype(np.float32)


# --------------------------------------------------------------------------------------
# Item-based kNN
# --------------------------------------------------------------------------------------
class ItemKNN:
    """Item-item collaborative filtering with mean-centered cosine and shrinkage.

    1. Center each rating by its user's mean, so a harsh and a generous rater with the
       same taste produce similar vectors.
    2. sim(i, j) = cos(x_i, x_j) * n_ij / (n_ij + shrink), where n_ij is the number of
       users who rated both; the shrink term discounts similarities supported by few users.
    3. Keep the top `neighbors` per item.
    4. score(u, j) = sum over neighbors i of j that u rated of sim(i, j) * (r_ui - mean_u),
       plus a small popularity prior to break ties for users with little overlap.
    """

    def __init__(self, neighbors=50, shrink=100.0, max_user_history=500, block=2000):
        self.neighbors, self.shrink = neighbors, shrink
        self.max_user_history, self.block = max_user_history, block

    @staticmethod
    def centered(X):
        X = X.tocsr().astype(np.float32, copy=True)
        counts = np.diff(X.indptr)
        means = np.asarray(X.sum(axis=1)).ravel() / np.maximum(counts, 1)
        X.data -= np.repeat(means, counts)
        return X

    def _cap_history(self, X):
        """For heavy users keep a random `max_user_history` of their ratings when computing
        similarities. Similarity cost is O(sum_u n_u^2); a few users with 10k+ ratings would
        otherwise dominate both the runtime and the similarities themselves."""
        rng = np.random.default_rng(0)
        X = X.tocsr()
        keep = np.ones(X.nnz, dtype=bool)
        counts = np.diff(X.indptr)
        for u in np.where(counts > self.max_user_history)[0]:
            s, e = X.indptr[u], X.indptr[u + 1]
            drop = rng.choice(e - s, size=(e - s) - self.max_user_history, replace=False)
            keep[s + drop] = False
        coo = X.tocoo()
        return sp.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=X.shape)

    def fit(self, X):
        Xc = self.centered(self._cap_history(X)).tocsc()
        B = Xc.copy()
        B.data[:] = 1.0
        norms = np.sqrt(np.asarray(Xc.multiply(Xc).sum(axis=0)).ravel()) + 1e-9
        n_items = X.shape[1]
        XcT, BT = Xc.T.tocsr(), B.T.tocsr()
        rows, cols, vals = [], [], []
        for start in range(0, n_items, self.block):
            stop = min(start + self.block, n_items)
            dot = (XcT[start:stop] @ Xc).toarray()
            co = (BT[start:stop] @ B).toarray()
            sim = dot / (norms[start:stop, None] * norms[None, :]) * (co / (co + self.shrink))
            sim[np.arange(stop - start), np.arange(start, stop)] = 0.0
            top = np.argpartition(-sim, self.neighbors, axis=1)[:, : self.neighbors]
            rows.append(np.repeat(np.arange(start, stop), self.neighbors))
            cols.append(top.ravel())
            vals.append(np.take_along_axis(sim, top, axis=1).ravel())
        S = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n_items, n_items), dtype=np.float32)
        S.data[S.data < 0] = 0.0  # only positive neighbours contribute
        S.eliminate_zeros()
        self.S = S
        self.pop = np.log1p(np.diff(X.tocsc().indptr)).astype(np.float32)
        self.pop /= self.pop.max()
        return self

    def score_users(self, users, X):
        Xc = self.centered(X[users])
        # Users whose ratings are all equal have zero centered vectors; fall back to raw.
        raw = X[users].tocsr().astype(np.float32)
        scores = (Xc @ self.S).toarray() + 0.1 * (raw.sign() @ self.S).toarray()
        return scores + 1e-3 * self.pop[None, :]


def _pairs(X):
    coo = X.tocoo()
    return coo.row, coo.col
