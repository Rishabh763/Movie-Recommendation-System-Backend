# Movie Recommendation System: Backend

Flask API that serves personalized movie recommendations for the
[frontend](https://github.com/Rishabh763/Movie-Recommendation-System). Ratings live in
Firebase Firestore; the API builds a user-item matrix from them and scores unseen movies
with one of two collaborative-filtering models.

## Models

| `model` | Method |
|---|---|
| `cosine` | User-user collaborative filtering: cosine similarity between users' rating vectors, then a similarity-weighted average of neighbours' ratings for each unseen movie |
| `svd` | Truncated SVD (rank ≤ 10) of the user-item matrix; unseen movies ranked by their reconstructed score |
| `cold_start` | Returned automatically when the user has no ratings: movies ranked by mean rating |

## API

### `GET /recommendations`

Requires a Firebase ID token: `Authorization: Bearer <idToken>`. The user is taken from
the verified token, never from the query string.

| Param | Default | Notes |
|---|---|---|
| `model` | `svd` | `svd` or `cosine` |
| `top` | `8` | Clamped to 1–20 |

```json
{
  "uid": "…",
  "model": "svd",
  "recommendations": [{ "id": 3, "title": "Undiscovered Cities" }],
  "lastRefreshedAt": "2026-11-05T06:00:00"
}
```

Errors: `401` missing/invalid token, `400` unknown model, `503` no ratings yet.

### `POST /refresh`

Rebuilds the in-memory ratings table from Firestore. Requires header
`X-Refresh-Token: <REFRESH_SECRET>`. Called every 6 hours by
`.github/workflows/refresh.yml`. The requesting user's own ratings are always re-read on
each `/recommendations` call, so new ratings show up immediately.

## Environment variables

| Name | Required | Purpose |
|---|---|---|
| `FIREBASE_KEY_B64` | yes (prod) | Base64-encoded service account JSON. Parsed in memory, never written to disk |
| `FIREBASE_KEY_PATH` | local dev alternative | Path to the service account JSON |
| `REFRESH_SECRET` | yes | Shared secret for `/refresh`; also set as a GitHub Actions secret |
| `ALLOWED_ORIGINS` | no | Comma-separated CORS origins. Defaults to the Vercel URL and `http://localhost:5173` |

## Run locally

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# .env: FIREBASE_KEY_PATH=serviceAccountKey.json and REFRESH_SECRET=<anything>
python app.py
```

## Deployment

Hosted on Render (free tier, so the first request after idle takes 30–60 s to wake the
service). `render.yaml` documents the service configuration.
