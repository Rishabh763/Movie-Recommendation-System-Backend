from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from flask_cors import CORS




# Initialize Flask app
app = Flask(__name__)
CORS(app) 


# Initialize Firebase Admin SDK
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load movie metadata
import json
with open('data.json', 'r') as f:
    movies = json.load(f)

movie_id_to_title = {m['id']: m['title'] for m in movies}

@app.route("/")
def home():
    return "🎬 Flask Movie Recommendation API is running!"



# Dummy movie ID to title map — replace with your actual mapping
movie_id_to_title = {
    1: "Beyond Earth",
    2: "Bottom Gear",
    3: "Undiscovered Cities",
    4: "1998",
    5: "Dark Side of the Moon",
    6: "The Great Lands",
    7: "The Diary",
    8: "Earth’s Untouched",
    9: "No Land Beyond",
    10: "During the Hunt",
    11: "Autosport the Series",
    12: "Same Answer II",
    13: "Below Echo",
    14: "The Rockies",
    15: "Relentless",
    16: "Community of Ours",
    17: "Van Life",
    18: "The Heiress",
    19: "Off the Track",
    20: "Whispering Hill",
    21: "112",
    22: "Lone Heart",
    23: "Production Line",
    24: "Dogs",
    25: "Asia in 24 Days",
    26: "The Tasty Tour",
    27: "Darker",
    28: "Unresolved Cases",
    29: "Mission: Saturn"
}


@app.route("/recommendations", methods=['GET'])
def get_recommendations():
    uid = request.args.get("uid")
    model_type = request.args.get("model", "svd").lower()
    top_n = int(request.args.get("top", 5))

    if not uid:
        return jsonify({"error": "Missing uid parameter"}), 400

    # Fetch all ratings from Firestore
    ratings_ref = db.collection('users')
    print("Fetched docs:")
    ratings_docs = list(ratings_ref.stream())  # ✅ Materialize once

    ratings = []
    for doc in ratings_docs:
        data = doc.to_dict()
        print(f"{doc.id}: {data}")  # 👈 Debug print

        user_id = doc.id  # UID is the doc ID
        if 'ratings' in data:
            for movie_id_str, rating in data['ratings'].items():
                try:
                    movie_id = int(movie_id_str)
                    rating_val = float(rating)  # Ensure numeric type
                    ratings.append({
                        "userId": user_id,
                        "movieId": movie_id,
                        "rating": rating_val
                    })
                except (ValueError, TypeError) as e:
                    print(f"⚠️ Skipped rating: {movie_id_str}={rating} — {e}")


    ratings_df = pd.DataFrame(ratings)
    print(ratings_df)

    if ratings_df.empty:
        return jsonify({"error": "No ratings available in database"}), 500

    # Create user-item matrix
    user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    # 🔥 Cold start
    if uid not in user_item_matrix.index:
        print("Cold start for user:", uid)
        top_movies = ratings_df.groupby("movieId")["rating"].mean().sort_values(ascending=False).head(top_n).index
        return jsonify({
            "uid": uid,
            "model": "cold_start",
            "recommendations": [movie_id_to_title.get(mid, f"Movie {mid}") for mid in top_movies]
        })

    # ================= MODEL LOGIC =================
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

    elif model_type == "svd":
        matrix = user_item_matrix.values
        svd = TruncatedSVD(n_components=min(10, matrix.shape[1]-1), random_state=42)
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
    ratings_ref = db.collection('users')
    docs = list(ratings_ref.stream())

    out = []
    for doc in docs:
        d = doc.to_dict()
        d["uid"] = doc.id
        out.append(d)
    return jsonify(out)

if __name__ == "__main__":
    app.run(debug=True)
