"""API tests with Firestore, Firebase Auth and TMDB mocked out (no network, no real key).

Run: pytest -q
"""
import importlib
import sys
from unittest import mock

import pytest

USERS = {
    "alice": {"mlRatings": {"30": 5, "40": 4.5, "50": 4, "60": 5, "70": 1}},
    "bob": {"mlRatings": {"30": 5}, "ratings": {"1": 5}},  # old-format field is ignored
    "carol": {"mlRatings": {"30": "junk", "40": 99}},      # invalid values are dropped
}
TOKENS = {"tok-alice": "alice", "tok-bob": "bob", "tok-carol": "carol", "tok-new": "newbie"}


class FakeDoc:
    def __init__(self, data):
        self._data = data
        self.exists = data is not None

    def to_dict(self):
        return None if self._data is None else dict(self._data)


class FakeDocRef:
    def __init__(self, store, key):
        self.store, self.key = store, key

    def get(self):
        return FakeDoc(self.store.get(self.key))

    def set(self, data):
        self.store[self.key] = data


class FakeDB:
    def __init__(self):
        self.collections = {"users": USERS, "tmdb_cache": {}}

    def collection(self, name):
        store = self.collections.setdefault(name, {})
        return mock.Mock(document=lambda key: FakeDocRef(store, key))


@pytest.fixture()
def app_module(monkeypatch, artifacts):
    monkeypatch.setenv("FIREBASE_KEY_B64", "e30=")  # base64 of "{}"
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts))
    monkeypatch.delenv("TMDB_READ_TOKEN", raising=False)
    import firebase_admin
    from firebase_admin import auth as fb_auth

    def fake_verify(token):
        if token not in TOKENS:
            raise fb_auth.InvalidIdTokenError("bad token")
        return {"uid": TOKENS[token]}

    with mock.patch("firebase_admin.credentials.Certificate"), \
         mock.patch.object(firebase_admin, "initialize_app"), \
         mock.patch("firebase_admin.firestore.client", return_value=FakeDB()), \
         mock.patch.object(fb_auth, "verify_id_token", side_effect=fake_verify):
        sys.modules.pop("app", None)
        yield importlib.import_module("app")


@pytest.fixture()
def client(app_module):
    return app_module.app.test_client()


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_health(client):
    body = client.get("/").get_json()
    assert body["status"] == "ok" and body["movies"] == 300 and body["posters"] is False


def test_old_endpoints_are_gone(client):
    assert client.get("/ratings_dump").status_code == 404
    assert client.post("/refresh").status_code == 404


def test_movies_search_and_validation(client):
    body = client.get("/movies?q=star&pageSize=5").get_json()
    assert body["results"][0]["title"] == "Star Wars"
    assert client.get("/movies?sort=bogus").status_code == 400
    assert client.get("/movies?genre=Nope").status_code == 400
    assert len(client.get("/movies?pageSize=1000").get_json()["results"]) == 60
    assert client.get("/movies?page=abc").status_code == 200


def test_movies_by_ids(client):
    body = client.get("/movies?ids=30,10,999999").get_json()
    assert [m["id"] for m in body["results"]] == [30, 10]
    assert client.get("/movies?ids=1,x").status_code == 400


def test_genres(client):
    assert "Action" in client.get("/genres").get_json()


def test_movie_detail_with_tmdb_overview(client, app_module):
    app_module.tmdb.details = lambda tmdb_id: {"poster_path": "/p.jpg", "overview": "A film."}
    body = client.get("/movies/30").get_json()
    assert body["overview"] == "A film." and len(body["similar"]) == 12
    assert client.get("/movies/123456").status_code == 404


def test_poster_redirects_and_caches(client, app_module):
    app_module.tmdb.details = lambda tmdb_id: {"poster_path": "/p.jpg"}
    res = client.get("/poster/35?size=w500")
    assert res.status_code == 302
    assert res.headers["Location"] == "https://image.tmdb.org/t/p/w500/p.jpg"
    assert "max-age" in res.headers["Cache-Control"]
    app_module.tmdb.details = lambda tmdb_id: {"poster_path": None}
    assert client.get("/poster/35").status_code == 404


def test_recommendations_require_token(client):
    assert client.get("/recommendations").status_code == 401
    assert client.get("/recommendations", headers=auth("forged")).status_code == 401


def test_recommendations_use_saved_ml_ratings(client):
    body = client.get("/recommendations?top=8", headers=auth("tok-alice")).get_json()
    assert body["uid"] == "alice" and body["basedOnRatings"] == 5
    assert body["strategy"] == "implicit_als" and len(body["recommendations"]) == 8
    assert not {m["id"] for m in body["recommendations"]} & {30, 40, 50, 60, 70}


def test_recommendation_strategies_by_history(client):
    assert client.get("/recommendations", headers=auth("tok-bob")).get_json()["strategy"] == "hybrid"
    assert client.get("/recommendations", headers=auth("tok-new")).get_json()["strategy"] == "popular"
    carol = client.get("/recommendations", headers=auth("tok-carol")).get_json()
    assert carol["basedOnRatings"] == 0 and carol["strategy"] == "popular"


def test_cors_only_allows_known_origins(client):
    good = client.get("/", headers={"Origin": "http://localhost:5173"})
    bad = client.get("/", headers={"Origin": "https://evil.example"})
    assert good.headers.get("Access-Control-Allow-Origin") == "http://localhost:5173"
    assert "Access-Control-Allow-Origin" not in bad.headers


def test_tmdb_cache_layers():
    from tmdb import TMDB

    db = FakeDB()
    session = mock.Mock()
    session.get.return_value = mock.Mock(
        status_code=200, json=lambda: {"poster_path": "/x.jpg", "overview": "o", "extra": 1})
    t = TMDB(db=db, token="t", session=session)
    assert t.poster_url(5, "w185") == "https://image.tmdb.org/t/p/w185/x.jpg"
    assert db.collections["tmdb_cache"]["5"]["poster_path"] == "/x.jpg"
    t.poster_url(5)
    TMDB(db=db, token="t", session=session).details(5)  # new process: served from Firestore
    assert session.get.call_count == 1
    assert TMDB(db=None, token=None).details(7)["poster_path"] is None  # no token: no call
