from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from flask_cors import CORS
from dotenv import load_dotenv
import os
import base64
import tempfile
import json

# Load .env locally for dev
load_dotenv()

# Read and decode the base64 key
firebase_key_b64 = os.getenv("FIREBASE_KEY_B64")
if not firebase_key_b64:
    raise Exception("FIREBASE_KEY_B64 environment variable not set.")

decoded_key = base64.b64decode(firebase_key_b64)

# Write to temp file
with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
    temp_file.write(decoded_key)
    cred_path = temp_file.name

# Initialize Firebase with the decoded key
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load movie metadata
with open('data.json', 'r') as f:
    movies = json.load(f)

movie_id_to_title = {m['id']: m['title'] for m in movies}

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "🎬 Flask Movie Recommendation API is running!"

@app.route("/recommendations", methods=['GET'])
def get_recommendations():
    uid = request.args.get("uid")
    model_type = request.args.get("model", "svd").lower()
    top_n = int(request.args.get("top", 5))

    if not uid:
        return jsonify({"error": "Missing uid parameter"}), 400

    # Fetch ratings
    ratings_docs = list(db.collection('users').stream())
    ratings = []
    for doc in ratings_docs:
        data = doc.to_dict()
        user_id = doc.id
        if 'ratings' in data:
            for movie_id_str, rating in data['ratings'].items():
                try:
                    movie_id = int(movie_id_str)
                    rating_val = float(rating)
                    ratings.append({
                        "userId": user_id,
                        "movieId": movie_id,
                        "rating": rating_val
                    })
                except (ValueError, TypeError):
                    continue

    ratings_df = pd.DataFrame(ratings)
    if ratings_df.empty:
        return jsonify({"error": "No ratings available in database"}), 500

    # User-item matrix
    user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    # Cold start
    if uid not in user_item_matrix.index:
        top_movies = ratings_df.groupby("movieId")["rating"].mean().sort_values(ascending=False).head(top_n).index
        return jsonify({
            "uid": uid,
            "model": "cold_start",
            "recommendations": [movie_id_to_title.get(mid, f"Movie {mid}") for mid in top_movies]
        })

    # Cosine similarity
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

    # SVD
    elif model_type == "svd":
        matrix = user_item_matrix.values
        svd = TruncatedSVD(n_components=min(10, matrix.shape[1] - 1), random_state=42)
        reduced = svd.fit_transform(matrix)
        reconstructed = np.dot(reduced, svd.components_)
        recon_df = pd.DataFrame(reconstructed, index=user_item_matrix.index, columns=user_item_matrix.columns)

        user_row = recon_df.loc[uid]
        already_rated = user_item_matrix.loc[uid]
        unrated = user_row[already_rated == 0]
        top_recs = unrated.sort_values(ascending=False).head(top_n).index.tolist()

    else:
        return jsonify({"error": "Invalid model. Use 'svd' or 'cosine'."}), 400

    return jsonify({
        "uid": uid,
        "model": model_type,
        "recommendations": [movie_id_to_title.get(mid, f"Movie {mid}") for mid in top_recs]
    })

@app.route("/ratings_dump", methods=['GET'])
def ratings_dump():
    docs = list(db.collection('users').stream())
    out = []
    for doc in docs:
        d = doc.to_dict()
        d["uid"] = doc.id
        out.append(d)
    return jsonify(out)

if __name__ == "__main__":
    app.run(debug=True)
