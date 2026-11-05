from flask import Flask, request, jsonify, redirect
import firebase_admin
from firebase_admin import credentials, firestore, auth as fb_auth
from flask_cors import CORS
from dotenv import load_dotenv
import os
import base64
import json

from recommender import Recommender, SORTS
from tmdb import TMDB

load_dotenv()

DEFAULT_TOP_N = 20
MAX_TOP_N = 50
MAX_PAGE_SIZE = 60
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

recommender = Recommender()
tmdb = TMDB(db=db)

app = Flask(__name__)
allowed_origins = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()
]
CORS(app, origins=allowed_origins)


def get_verified_uid():
    """Return (uid, None) for a valid Firebase ID token, else (None, error response)."""
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


def int_arg(name, default, lo, hi):
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(value, hi))


def user_ratings(uid):
    """{movieId: rating} saved by the frontend in users/{uid}.mlRatings."""
    snap = db.collection("users").document(uid).get()
    raw = (snap.to_dict() or {}).get("mlRatings", {}) if snap.exists else {}
    out = {}
    for movie_id, rating in raw.items():
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            continue
        if 0.5 <= rating <= 5.0:
            out[int(movie_id)] = rating
    return out


@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "dataset": recommender.meta["dataset"],
        "movies": len(recommender.items),
        "model": "implicit ALS + biased MF (fold-in)",
        "posters": tmdb.enabled,
    })


@app.route("/genres")
def genres():
    return jsonify(recommender.all_genres)


@app.route("/movies")
def movies():
    ids = request.args.get("ids")
    if ids:
        try:
            wanted = [int(x) for x in ids.split(",") if x.strip()][:MAX_PAGE_SIZE]
        except ValueError:
            return jsonify({"error": "ids must be comma-separated integers"}), 400
        return jsonify({"results": recommender.get_many(wanted)})
    sort = request.args.get("sort", "popular")
    if sort not in SORTS:
        return jsonify({"error": f"sort must be one of {sorted(SORTS)}"}), 400
    genre = request.args.get("genre") or None
    if genre and genre not in recommender.genre_index:
        return jsonify({"error": "unknown genre"}), 400
    return jsonify(recommender.search(
        q=request.args.get("q", "")[:100],
        genre=genre,
        sort=sort,
        page=int_arg("page", 1, 1, 10_000),
        page_size=int_arg("pageSize", 24, 1, MAX_PAGE_SIZE),
    ))


@app.route("/movies/<int:movie_id>")
def movie_detail(movie_id):
    movie = recommender.get(movie_id)
    if movie is None:
        return jsonify({"error": "Movie not found"}), 404
    if movie["tmdbId"]:
        info = tmdb.details(movie["tmdbId"])
        movie["overview"] = info.get("overview")
    movie["similar"] = recommender.similar(movie_id, n=int_arg("similar", 12, 1, 30))
    return jsonify(movie)


@app.route("/poster/<int:tmdb_id>")
def poster(tmdb_id):
    url = tmdb.poster_url(tmdb_id, request.args.get("size", "w342"))
    if not url:
        return jsonify({"error": "No poster"}), 404
    res = redirect(url, code=302)
    res.headers["Cache-Control"] = "public, max-age=604800"  # browsers reuse it for a week
    return res


@app.route("/recommendations")
def recommendations():
    uid, err = get_verified_uid()
    if err:
        return err
    ratings = user_ratings(uid)
    strategy, recs = recommender.recommend(ratings, n=int_arg("top", DEFAULT_TOP_N, 1, MAX_TOP_N))
    return jsonify({
        "uid": uid,
        "strategy": strategy,
        "basedOnRatings": len(ratings),
        "recommendations": recs,
    })


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")
