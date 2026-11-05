"""Convert raw MovieLens CSVs into compact Parquet files and a per-user temporal split.

Usage:
    python -m ml.prepare --raw /path/to/ml-32m

Outputs (data/processed/, gitignored):
    train.parquet, test.parquet  columns: u (int32), i (int32), rating (float32), ts (int32)
    users.npy                    u -> original MovieLens userId
    items.parquet                i -> movieId, title, year, genres, tmdbId, tags, n_ratings, mean_rating
"""
import argparse
import re
import time

import numpy as np
import pandas as pd

from ml import config


def load_ratings(raw):
    return pd.read_csv(
        raw / "ratings.csv",
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    ).rename(columns={"timestamp": "ts"})


def filter_sparse(df):
    """Drop rare items, then users left with too few ratings (one pass of each)."""
    item_counts = df["movieId"].value_counts()
    df = df[df["movieId"].isin(item_counts[item_counts >= config.MIN_ITEM_RATINGS].index)]
    user_counts = df["userId"].value_counts()
    return df[df["userId"].isin(user_counts[user_counts >= config.MIN_USER_RATINGS].index)]


def temporal_split(df):
    """Per user: sort by time, the last TEST_FRACTION of ratings (at least 1) become test.

    Splitting by time rather than at random means a model is always asked to predict a
    user's future from their past, which is what it has to do in production.
    """
    df = df.sort_values(["u", "ts", "i"], kind="mergesort").reset_index(drop=True)
    pos = df.groupby("u").cumcount().to_numpy()
    n = df.groupby("u")["i"].transform("size").to_numpy()
    n_test = np.maximum(1, np.floor(n * config.TEST_FRACTION)).astype(np.int64)
    is_test = pos >= (n - n_test)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def build_items(raw, item_ids, train):
    movies = pd.read_csv(raw / "movies.csv", dtype={"movieId": "int32"})
    links = pd.read_csv(raw / "links.csv", dtype={"movieId": "int32"})
    movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")

    year = movies["title"].str.extract(YEAR_RE)[0]
    movies["year"] = pd.to_numeric(year, errors="coerce").astype("Int32")
    movies["clean_title"] = movies["title"].str.replace(YEAR_RE, "", regex=True).str.strip()
    movies["genres"] = movies["genres"].replace("(no genres listed)", "")
    movies["tmdbId"] = movies["tmdbId"].astype("Int64")

    tags = pd.read_csv(raw / "tags.csv", usecols=["movieId", "tag"], dtype={"movieId": "int32"})
    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    tag_counts = tags.groupby(["movieId", "tag"]).size().rename("n").reset_index()
    tag_counts = tag_counts[tag_counts["n"] >= 2].sort_values(["movieId", "n"], ascending=[True, False])
    top_tags = tag_counts.groupby("movieId")["tag"].apply(lambda s: "|".join(s.head(20)))

    stats = train.groupby("i")["rating"].agg(n_ratings="size", mean_rating="mean")

    items = pd.DataFrame({"i": np.arange(len(item_ids), dtype=np.int32), "movieId": item_ids})
    items = items.merge(movies, on="movieId", how="left")
    items["tags"] = items["movieId"].map(top_tags).fillna("")
    items = items.join(stats, on="i")
    items["n_ratings"] = items["n_ratings"].fillna(0).astype("int32")
    items["mean_rating"] = items["mean_rating"].astype("float32")
    return items[["i", "movieId", "clean_title", "year", "genres", "tmdbId", "tags",
                  "n_ratings", "mean_rating"]].rename(columns={"clean_title": "title"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="folder containing ratings.csv, movies.csv, ...")
    args = parser.parse_args()
    from pathlib import Path
    raw = Path(args.raw)
    out = config.PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = load_ratings(raw)
    print(f"raw: {len(df):,} ratings, {df.userId.nunique():,} users, {df.movieId.nunique():,} movies")
    df = filter_sparse(df)

    user_ids, u = np.unique(df["userId"].to_numpy(), return_inverse=True)
    item_ids, i = np.unique(df["movieId"].to_numpy(), return_inverse=True)
    df = pd.DataFrame({
        "u": u.astype(np.int32),
        "i": i.astype(np.int32),
        "rating": df["rating"].to_numpy(),
        "ts": df["ts"].to_numpy().astype(np.int32),
    })
    print(f"filtered: {len(df):,} ratings, {len(user_ids):,} users, {len(item_ids):,} movies")

    train, test = temporal_split(df)
    print(f"split: train {len(train):,}, test {len(test):,}")

    train.to_parquet(out / "train.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    np.save(out / "users.npy", user_ids.astype(np.int32))
    build_items(raw, item_ids.astype(np.int32), train).to_parquet(out / "items.parquet", index=False)
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
