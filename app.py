from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore, auth as fb_auth
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from flask_cors import CORS
from dotenv import load_dotenv
import os
import base64
import hmac
import json

# Load env
load_dotenv()

DEFAULT_TOP_N = 8
MAX_TOP_N = 20
DEFAULT_ORIGINS = (
    "https://movie-recommendation-system-alpha-nine.vercel.app,"
    "http://localhost:5173"
)


def load_firebase_credentials():
    """Load the service account from FIREBASE_KEY_B64 (preferred) or FIREBASE_KEY_PATH.

    The decoded key is parsed in memory, so it never touches the disk.
    """
    key_b64 = os.getenv("FIREBASE_KEY_B64")
    if key_b64:
        return credentials.Certificate(json.loads(base64.b64decode(key_b64)))
    key_path = os.getenv("FIREBASE_KEY_PATH")
    if key_path and os.path.exists(key_path):
        return credentials.Certificate(key_path)
    raise RuntimeError("Set FIREBASE_KEY_B64 (or FIREBASE_KEY_PATH for local dev).")


firebase_admin.initialize_app(load_firebase_credentials())
db = firestore.client()

# Load movies
with open('data.json', 'r') as f:
    movies = json.load(f)

movie_id_to_title = {m['id']: m['title'] for m in movies}

# App
app = Flask(__name__)
allowed_origins = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()
]
CORS(app, origins=allowed_origins)

latest_ratings_df = pd.DataFrame(columns=["userId", "movieId", "rating"])
last_refreshed_at = None


def parse_user_ratings(user_id, data):
    """Turn one Firestore user document into rating rows, skipping malformed entries."""
    rows = []
    for movie_id_str, rating in (data or {}).get('ratings', {}).items():
        try:
            rows.append({
                "userId": user_id,
                "movieId": int(movie_id_str),
                "rating": float(rating),
            })
        except (ValueError, TypeError):
            continue
    return rows


def fetch_ratings():
    global latest_ratings_df, last_refreshed_at
    ratings = []
    for doc in db.collection('users').stream():
        ratings.extend(parse_user_ratings(doc.id, doc.to_dict()))
    latest_ratings_df = pd.DataFrame(ratings, columns=["userId", "movieId", "rating"])
    last_refreshed_at = pd.Timestamp.now().isoformat()
    print(f"[{last_refreshed_at}] Ratings refreshed. Total: {len(latest_ratings_df)}")


def ratings_with_fresh_user(uid):
    """Snapshot of all ratings, with this user's rows re-read from Firestore.

    The full table is only rebuilt by /refresh (every 6 hours), so without this a
    user who just rated a movie would not see it reflected. One document read per
    request keeps the requesting user's ratings current.
    """
    df = latest_ratings_df
    df = df[df["userId"] != uid]
    snap = db.collection('users').document(uid).get()
    fresh = parse_user_ratings(uid, snap.to_dict() if snap.exists else None)
    if fresh:
        df = pd.concat([df, pd.DataFrame(fresh)], ignore_index=True)
    return df


def get_verified_uid():
    """Return the uid from a valid Firebase ID token in the Authorization header.

    Returns (uid, None) on success or (None, (json, status)) on failure.
    """
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None, (jsonify({"error": "Missing bearer token"}), 401)
    try:
        decoded = fb_auth.verify_id_token(header[len("Bearer "):].strip())
    except fb_auth.CertificateFetchError:
        return None, (jsonify({"error": "Could not verify token right now"}), 503)
    except (ValueError, fb_auth.InvalidIdTokenError, fb_auth.UserDisabledError):
        return None, (jsonify({"error": "Invalid or expired token"}), 401)
    return decoded["uid"], None


def parse_top_n():
    try:
        top_n = int(request.args.get("top", DEFAULT_TOP_N))
    except (TypeError, ValueError):
        top_n = DEFAULT_TOP_N
    return max(1, min(top_n, MAX_TOP_N))


def to_payload(movie_ids):
    return [
        {"id": int(mid), "title": movie_id_to_title.get(int(mid), f"Movie {mid}")}
        for mid in movie_ids
    ]


# Initial fetch
fetch_ratings()


@app.route("/")
def home():
    return "🎬 Flask Movie Recommendation API is running!"


@app.route("/refresh", methods=['POST'])
def manual_refresh():
    secret = os.getenv("REFRESH_SECRET")
    if not secret:
        return jsonify({"error": "Refresh is disabled: REFRESH_SECRET not set"}), 503
    provided = request.headers.get("X-Refresh-Token", "")
    if not hmac.compare_digest(provided.encode(), secret.encode()):
        return jsonify({"error": "Forbidden"}), 403
    fetch_ratings()
    return jsonify({
        "status": "Ratings refreshed",
        "lastRefreshedAt": last_refreshed_at
    })


@app.route("/recommendations", methods=['GET'])
def get_recommendations():
    uid, err = get_verified_uid()
    if err:
        return err

    model_type = request.args.get("model", "svd").lower()
    if model_type not in ("svd", "cosine"):
        return jsonify({"error": "Invalid model. Use 'svd' or 'cosine'."}), 400
    top_n = parse_top_n()

    ratings_df = ratings_with_fresh_user(uid)
    if ratings_df.empty:
        return jsonify({"error": "No ratings available"}), 503

    user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    if uid not in user_item_matrix.index:
        top_movies = ratings_df.groupby("movieId")["rating"].mean().sort_values(ascending=False).head(top_n).index
        return jsonify({
            "uid": uid,
            "model": "cold_start",
            "recommendations": to_payload(top_movies),
            "lastRefreshedAt": last_refreshed_at
        })

    if model_type == "cosine":
        similarity = cosine_similarity(user_item_matrix)
        similarity_df = pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
        similar_users = similarity_df[uid].drop(uid)

        weighted_scores = {}
        for movie in user_item_matrix.columns:
            if user_item_matrix.loc[uid, movie] > 0:
                continue
            total_sim = 0
            weighted_sum = 0
            for other_user, sim in similar_users.items():
                rating = user_item_matrix.loc[other_user, movie]
                if rating > 0:
                    weighted_sum += sim * rating
                    total_sim += sim
            if total_sim > 0:
                weighted_scores[movie] = weighted_sum / total_sim

        top_recs = sorted(weighted_scores, key=weighted_scores.get, reverse=True)[:top_n]

    else:  # svd
        matrix = user_item_matrix.values
        svd = TruncatedSVD(n_components=min(10, matrix.shape[1] - 1), random_state=42)
        reduced = svd.fit_transform(matrix)
        reconstructed = np.dot(reduced, svd.components_)
        recon_df = pd.DataFrame(reconstructed, index=user_item_matrix.index, columns=user_item_matrix.columns)

        user_row = recon_df.loc[uid]
        already_rated = user_item_matrix.loc[uid]
        unrated = user_row[already_rated == 0]
        top_recs = unrated.sort_values(ascending=False).head(top_n).index.tolist()

    return jsonify({
        "uid": uid,
        "model": model_type,
        "recommendations": to_payload(top_recs),
        "lastRefreshedAt": last_refreshed_at
    })


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")
