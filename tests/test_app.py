"""API tests with Firestore and Firebase Auth mocked out (no network, no real key).

Run: pytest -q
"""
import importlib
import sys
from unittest import mock

import pytest

# Fake Firestore data: three users with overlapping tastes.
USERS = {
    "alice": {"ratings": {"1": 5, "2": 4, "3": 1}},
    "bob": {"ratings": {"1": 5, "2": 5, "4": 4}},
    "carol": {"ratings": {"2": 1, "3": 5, "5": 4}},
    "dave": {"ratings": {}},
}
TOKENS = {"tok-alice": "alice", "tok-dave": "dave", "tok-new": "newbie"}


class FakeDoc:
    def __init__(self, uid, data):
        self.id = uid
        self._data = data
        self.exists = data is not None

    def to_dict(self):
        return None if self._data is None else dict(self._data)


class FakeCollection:
    def stream(self):
        return [FakeDoc(uid, data) for uid, data in USERS.items()]

    def document(self, uid):
        doc = FakeDoc(uid, USERS.get(uid))
        return mock.Mock(get=mock.Mock(return_value=doc))


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("FIREBASE_KEY_B64", "e30=")  # base64 of "{}"
    monkeypatch.setenv("REFRESH_SECRET", "s3cret")
    import firebase_admin
    from firebase_admin import auth as fb_auth

    def fake_verify(token):
        if token not in TOKENS:
            raise fb_auth.InvalidIdTokenError("bad token")
        return {"uid": TOKENS[token]}

    with mock.patch("firebase_admin.credentials.Certificate"), \
         mock.patch.object(firebase_admin, "initialize_app"), \
         mock.patch("firebase_admin.firestore.client") as fs_client, \
         mock.patch.object(fb_auth, "verify_id_token", side_effect=fake_verify):
        fs_client.return_value.collection.return_value = FakeCollection()
        sys.modules.pop("app", None)
        app_module = importlib.import_module("app")
        yield app_module.app.test_client()


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_ratings_dump_removed(client):
    assert client.get("/ratings_dump").status_code == 404


def test_recommendations_require_token(client):
    assert client.get("/recommendations?uid=alice").status_code == 401
    assert client.get("/recommendations", headers=auth("forged")).status_code == 401


def test_uid_comes_from_token_not_query(client):
    res = client.get("/recommendations?uid=bob&model=svd", headers=auth("tok-alice"))
    assert res.status_code == 200
    assert res.get_json()["uid"] == "alice"


@pytest.mark.parametrize("model", ["svd", "cosine"])
def test_recommendations_return_ids_and_skip_rated(client, model):
    res = client.get(f"/recommendations?model={model}", headers=auth("tok-alice"))
    body = res.get_json()
    assert res.status_code == 200 and body["model"] == model
    ids = [r["id"] for r in body["recommendations"]]
    assert ids and not set(ids) & {1, 2, 3}  # never recommend what alice rated
    assert all(isinstance(r["title"], str) for r in body["recommendations"])


def test_top_param_is_respected_and_clamped(client):
    one = client.get("/recommendations?model=svd&top=1", headers=auth("tok-alice")).get_json()
    assert len(one["recommendations"]) == 1
    junk = client.get("/recommendations?model=svd&top=abc", headers=auth("tok-alice"))
    assert junk.status_code == 200


def test_invalid_model(client):
    assert client.get("/recommendations?model=knn", headers=auth("tok-alice")).status_code == 400


def test_cold_start_for_users_without_ratings(client):
    for tok in ("tok-dave", "tok-new"):
        body = client.get("/recommendations", headers=auth(tok)).get_json()
        assert body["model"] == "cold_start" and body["recommendations"]


def test_fresh_ratings_are_used_without_refresh(client):
    USERS["alice"]["ratings"]["4"] = 2  # alice rates movie 4 after the last refresh
    try:
        body = client.get("/recommendations?model=svd", headers=auth("tok-alice")).get_json()
        assert 4 not in [r["id"] for r in body["recommendations"]]
    finally:
        del USERS["alice"]["ratings"]["4"]


def test_refresh_is_protected(client):
    assert client.get("/refresh").status_code == 405
    assert client.post("/refresh").status_code == 403
    assert client.post("/refresh", headers={"X-Refresh-Token": "wrong"}).status_code == 403
    assert client.post("/refresh", headers={"X-Refresh-Token": "s3cret"}).status_code == 200


def test_cors_only_allows_known_origins(client):
    good = client.get("/", headers={"Origin": "http://localhost:5173"})
    bad = client.get("/", headers={"Origin": "https://evil.example"})
    assert good.headers.get("Access-Control-Allow-Origin") == "http://localhost:5173"
    assert "Access-Control-Allow-Origin" not in bad.headers
