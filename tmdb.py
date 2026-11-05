"""TMDB metadata (poster path, overview) with a two-level cache.

Lookup order: in-process dict -> Firestore `tmdb_cache/{tmdbId}` -> TMDB API.
Firestore keeps the cache across Render restarts (the free tier sleeps when idle), so each
movie is fetched from TMDB at most once. Clients can't read that collection: Firestore
rules deny everything outside users/{uid}; the Admin SDK used here bypasses rules.

This product uses the TMDB API but is not endorsed or certified by TMDB.
"""
import os
import threading

import requests

API = "https://api.themoviedb.org/3/movie/{id}"
IMAGE_BASE = "https://image.tmdb.org/t/p/{size}{path}"
POSTER_SIZES = {"w185", "w342", "w500", "w780"}
_MISSING = {"poster_path": None, "overview": None, "backdrop_path": None}


class TMDB:
    def __init__(self, db=None, token=None, session=None):
        self.db = db
        self.token = token or os.getenv("TMDB_READ_TOKEN")
        self.session = session or requests.Session()
        self._cache = {}
        self._lock = threading.Lock()

    @property
    def enabled(self):
        return bool(self.token)

    def _fetch(self, tmdb_id):
        res = self.session.get(
            API.format(id=tmdb_id),
            headers={"Authorization": f"Bearer {self.token}", "accept": "application/json"},
            timeout=5,
        )
        if res.status_code == 404:
            return dict(_MISSING)
        res.raise_for_status()
        data = res.json()
        return {k: data.get(k) for k in _MISSING}

    def details(self, tmdb_id):
        tmdb_id = int(tmdb_id)
        with self._lock:
            if tmdb_id in self._cache:
                return self._cache[tmdb_id]
        info = None
        doc_ref = self.db.collection("tmdb_cache").document(str(tmdb_id)) if self.db else None
        if doc_ref is not None:
            snap = doc_ref.get()
            if snap.exists:
                info = snap.to_dict()
        if info is None:
            if not self.enabled:
                return dict(_MISSING)
            try:
                info = self._fetch(tmdb_id)
            except requests.RequestException:
                return dict(_MISSING)  # transient: don't cache failures
            if doc_ref is not None:
                doc_ref.set(info)
        with self._lock:
            self._cache[tmdb_id] = info
        return info

    def poster_url(self, tmdb_id, size="w342"):
        path = self.details(tmdb_id).get("poster_path")
        if not path:
            return None
        return IMAGE_BASE.format(size=size if size in POSTER_SIZES else "w342", path=path)
