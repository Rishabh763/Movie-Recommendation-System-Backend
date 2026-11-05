import numpy as np
import pytest

from recommender import Recommender, confidence_weight, display_title


@pytest.fixture()
def rec(artifacts):
    return Recommender(artifacts)


def test_search_filters_sorts_and_pages(rec):
    first = rec.search(page=1, page_size=10)
    assert first["total"] == 300 and len(first["results"]) == 10
    counts = [m["numRatings"] for m in first["results"]]
    assert counts == sorted(counts, reverse=True)
    assert rec.search(q="star wars")["results"][0]["title"] == "Star Wars"
    action = rec.search(genre="Action", page_size=60)["results"]
    assert action and all("Action" in m["genres"] for m in action)
    years = [m["year"] for m in rec.search(sort="year", page_size=20)["results"]]
    assert years == sorted(years, reverse=True)
    assert rec.search(page=99)["results"] == []


def test_movie_shape_handles_missing_fields(rec):
    m0, m1 = rec.get(10), rec.get(20)
    assert m0["genres"] == [] and m1["tmdbId"] is None
    assert rec.get(999999) is None
    assert [m["id"] for m in rec.get_many([30, 999999, 10])] == [30, 10]


def test_cold_start_returns_popular(rec):
    strategy, recs = rec.recommend({}, n=5)
    assert strategy == "popular"
    assert [m["id"] for m in recs] == [10, 20, 30, 40, 50]


def test_few_ratings_use_hybrid_and_many_use_ials(rec):
    strategy, recs = rec.recommend({30: 5.0, 40: 4.0}, n=10)
    assert strategy == "hybrid" and len(recs) == 10
    many = {int(m): 4.0 for m in rec.movie_ids[:8]}
    strategy, recs = rec.recommend(many, n=10)
    assert strategy == "implicit_als"
    assert not {m["id"] for m in recs} & set(many)  # never recommend what was rated


def test_recommendations_have_predictions_and_explanations(rec):
    ratings = {int(m): 5.0 for m in rec.movie_ids[:6]}
    _, recs = rec.recommend(ratings, n=10)
    assert all(0.5 <= m["predictedRating"] <= 5.0 for m in recs)
    assert all(m["predictedRating"] * 2 == int(m["predictedRating"] * 2) for m in recs)
    explained = [m for m in recs if "because" in m]
    assert explained and all(m["because"]["id"] in ratings for m in explained)


def test_unknown_movies_are_ignored(rec):
    strategy, _ = rec.recommend({123456789: 5.0}, n=5)
    assert strategy == "popular"


def test_similar_excludes_itself_and_is_sorted(rec):
    sims = rec.similar(30, n=5)
    assert len(sims) == 5 and 30 not in [m["id"] for m in sims]
    scores = [m["score"] for m in sims]
    assert scores == sorted(scores, reverse=True)


def test_ials_fold_in_is_the_exact_least_squares_solution(rec):
    """Solve the same weighted least-squares problem by brute force over every item."""
    idx = np.array([3, 7, 11, 19])
    r = np.array([5.0, 4.0, 2.0, 3.5])
    x = rec.fold_in_ials(idx, r)
    Y = rec.Y.astype(np.float64)
    c = np.ones(len(Y))
    p = np.zeros(len(Y))
    c[idx] = rec.alpha * confidence_weight(r)
    p[idx] = 1.0
    # minimize sum_i c_i (p_i - x.y_i)^2 + reg |x|^2  =>  (Y^T C Y + reg I) x = Y^T C p
    A = (Y.T * c) @ Y + rec.reg * np.eye(Y.shape[1])
    expected = np.linalg.solve(A, (Y.T * c) @ p)
    np.testing.assert_allclose(x, expected, rtol=1e-6, atol=1e-8)


def test_ials_fold_in_matches_implicit_library():
    implicit = pytest.importorskip("implicit")
    import scipy.sparse as sp
    from implicit.cpu.als import AlternatingLeastSquares

    rng = np.random.default_rng(1)
    X = sp.random(200, 120, density=0.05, random_state=2, format="csr", dtype=np.float32)
    X.data = rng.integers(1, 11, size=X.nnz).astype(np.float32) / 2  # ratings 0.5..5
    alpha, reg = 4.0, 0.05
    conf = X.copy()
    conf.data = confidence_weight(conf.data).astype(np.float32)
    model = AlternatingLeastSquares(factors=8, regularization=reg, alpha=alpha, iterations=5,
                                    random_state=0)
    model.fit(conf, show_progress=False)

    class Stub(Recommender):
        def __init__(self):
            self.Y = np.asarray(model.item_factors)
            self.gram = self.Y.T.astype(np.float64) @ self.Y.astype(np.float64)
            self.alpha, self.reg = alpha, reg

    stub = Stub()
    row = X[5]
    ours = stub.fold_in_ials(row.indices, row.data)
    theirs = model.recalculate_user(0, conf[5])
    np.testing.assert_allclose(ours, theirs, rtol=1e-3, atol=1e-4)


def test_mf_fold_in_matches_training_code():
    pytest.importorskip("numba")
    from ml.models import fold_in_user

    rng = np.random.default_rng(3)
    Q = rng.standard_normal((50, 6)).astype(np.float32)
    bi = rng.normal(0, 0.3, 50).astype(np.float32)

    class Stub(Recommender):
        def __init__(self):
            self.Q, self.bi, self.mu, self.mf_reg, self.mf_reg_b = Q, bi, 3.5, 0.05, 0.01

    idx, r = np.array([1, 4, 9]), np.array([4.0, 2.5, 5.0])
    p, b = Stub().fold_in_mf(idx, r)
    p2, b2 = fold_in_user(idx, r, Q, bi, 3.5)
    np.testing.assert_allclose(p, p2, rtol=1e-5)
    assert abs(b - b2) < 1e-6


@pytest.mark.parametrize("raw, shown", [
    ("Dark Knight, The", "The Dark Knight"),
    ("Lives of Others, The (Das leben der Anderen)", "The Lives of Others (Das leben der Anderen)"),
    ("Few Good Men, A", "A Few Good Men"),
    ("Homme, L'", "L'Homme"),
    ("Toy Story", "Toy Story"),
    ("Good, the Bad and the Ugly, The", "The Good, the Bad and the Ugly"),
])
def test_display_title_moves_trailing_article(raw, shown):
    assert display_title(raw) == shown
