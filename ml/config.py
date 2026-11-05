"""Shared paths and hyperparameters for the offline training pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"  # gitignored: derived from MovieLens, not redistributable
ARTIFACTS_DIR = ROOT / "artifacts"            # small files the API loads at startup
REPORTS_DIR = ROOT / "reports"                # evaluation results (committed)

# Filtering: drop very rare items and users with too little history to split.
MIN_ITEM_RATINGS = 20
MIN_USER_RATINGS = 10

# Per-user temporal split: each user's most recent 20% of ratings go to test.
TEST_FRACTION = 0.2

# Ranking evaluation
EVAL_K = 10
RELEVANT_THRESHOLD = 4.0   # a held-out rating >= 4 counts as "relevant"
EVAL_USERS = 20_000        # sampled test users for ranking metrics
SEED = 42

# Serving: only the most-rated movies are shipped to the API (Render free tier = 512 MB RAM)
SERVE_TOP_ITEMS = 10_000
