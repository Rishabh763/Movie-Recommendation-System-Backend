"""TMDB metadata (poster path, overview) with a two-level cache.

Resolution order for a MovieLens title:
  1. /movie/{tmdbId}                    the id from MovieLens links.csv
  2. /tv/{tmdbId}                       MovieLens uses TMDB *TV* ids for miniseries
                                        (Planet Earth, Band of Brothers); accepted only if
                                        the show's name matches the title
  3. /search/multi?query=title          for missing or stale ids; accepted only if the
                                        title matches and the year is within one year

Lookup order: in-process dict -> Firestore `tmdb_cache_v2/{movieId}` -> TMDB API.
Firestore keeps the cache across Render restarts (the free tier sleeps when idle), so each
movie is resolved at most once. Clients can't read that collection: Firestore rules deny
everything outside users/{uid}; the Admin SDK used here bypasses rules.

This product uses the TMDB API but is not endorsed or certified by TMDB.
"""
import os
import re
import threading

import requests

API = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/{size}{path}"
POSTER_SIZES = {"w185", "w342", "w500", "w780"}
CACHE_COLLECTION = "tmdb_cache_v2"
_MISSING = {"poster_path": None, "overview": None}


def _norm(title):
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def titles_match(a, b):
    """Loose match: equal after normalising, or one contains the other (subtitles, a.k.a.)."""
    a, b = _norm(a), _norm(b)
    return bool(a and b) and (a == b or a in b or b in a)


def _year(date):
    return int(date[:4]) if date and date[:4].isdigit() else None


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

    def _get(self, path, **params):
        res = self.session.get(
            f"{API}{path}",
            params=params or None,
            headers={"Authorization": f"Bearer {self.token}", "accept": "application/json"},
            timeout=5,
        )
        if res.status_code == 404:
            return None
        res.raise_for_status()
        return res.json()

    def _resolve(self, tmdb_id, title, year):
        if tmdb_id:
            data = self._get(f"/movie/{tmdb_id}")
            if data and data.get("poster_path"):
                return {"poster_path": data["poster_path"], "overview": data.get("overview")}
            data = self._get(f"/tv/{tmdb_id}")
            if data and data.get("poster_path") and titles_match(data.get("name"), title):
                return {"poster_path": data["poster_path"], "overview": data.get("overview")}
        if title:
            data = self._get("/search/multi", query=title, include_adult="false")
            for hit in (data or {}).get("results", [])[:10]:
                if hit.get("media_type") not in ("movie", "tv") or not hit.get("poster_path"):
                    continue
                name = hit.get("title") or hit.get("name")
                hit_year = _year(hit.get("release_date") or hit.get("first_air_date"))
                if titles_match(name, title) and (not year or not hit_year or abs(hit_year - year) <= 1):
                    return {"poster_path": hit["poster_path"], "overview": hit.get("overview")}
        return dict(_MISSING)

    def details(self, movie_id, tmdb_id=None, title=None, year=None):
        """Poster path and overview for a MovieLens movie (cached per movieId)."""
        key = str(int(movie_id))
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        info = None
        doc_ref = self.db.collection(CACHE_COLLECTION).document(key) if self.db else None
        if doc_ref is not None:
            snap = doc_ref.get()
            if snap.exists:
                info = snap.to_dict()
        if info is None:
            if not self.enabled:
                return dict(_MISSING)
            try:
                info = self._resolve(tmdb_id, title, year)
            except requests.RequestException:
                return dict(_MISSING)  # transient (timeout, rate limit): don't cache failures
            if doc_ref is not None:
                doc_ref.set(info)
        with self._lock:
            self._cache[key] = info
        return info

    def poster_url(self, movie_id, size="w342", **movie):
        path = self.details(movie_id, **movie).get("poster_path")
        if not path:
            return None
        return IMAGE_BASE.format(size=size if size in POSTER_SIZES else "w342", path=path)
