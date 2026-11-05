"""Shared fixtures: a small synthetic artifact set shaped exactly like ml/export.py output."""
import json

import numpy as np
import pandas as pd
import pytest

GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi"]


def write_artifacts(path, n_items=300, k_ials=16, k_mf=8, seed=0):
    rng = np.random.default_rng(seed)
    movie_ids = np.arange(1, n_items + 1) * 10
    genres = ["|".join(rng.choice(GENRES, size=rng.integers(1, 3), replace=False)) for _ in movie_ids]
    genres[0] = ""  # a movie with no genres listed
    tmdb = pd.array([int(m) + 5 for m in movie_ids], dtype="Int64")
    tmdb[1] = pd.NA  # a movie without a TMDB id
    items = pd.DataFrame({
        "movieId": movie_ids.astype(np.int32),
        "title": [f"Movie {m}" for m in movie_ids],
        "year": pd.array(rng.integers(1950, 2024, size=n_items), dtype="Int32"),
        "genres": genres,
        "tmdbId": tmdb,
        "n_ratings": np.arange(n_items, 0, -1).astype(np.int32) * 10,
        "mean_rating": rng.uniform(2, 4.8, n_items).astype(np.float32),
        "damped_mean": rng.uniform(2.5, 4.5, n_items).astype(np.float32),
    })
    items.loc[2, "title"] = "Star Wars"
    path.mkdir(parents=True, exist_ok=True)
    items.to_parquet(path / "items.parquet", index=False)
    Y = rng.standard_normal((n_items, k_ials)).astype(np.float16)
    np.save(path / "ials_items.npy", Y)
    np.save(path / "ials_gram.npy", (Y.astype(np.float32).T @ Y.astype(np.float32)))
    np.save(path / "mf_items.npy", (rng.standard_normal((n_items, k_mf)) * 0.3).astype(np.float16))
    np.save(path / "mf_bias.npy", rng.normal(0, 0.3, n_items).astype(np.float32))
    (path / "model.json").write_text(json.dumps({
        "dataset": "synthetic",
        "ials": {"factors": k_ials, "alpha": 4.0, "reg": 0.05},
        "mf": {"factors": k_mf, "reg": 0.05, "reg_b": 0.01, "mu": 3.5},
    }))
    return items


@pytest.fixture()
def artifacts(tmp_path):
    path = tmp_path / "artifacts"
    write_artifacts(path)
    return path
