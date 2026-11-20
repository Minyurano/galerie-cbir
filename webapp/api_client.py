"""
Client de l'API REST : toutes les communications de l'application web avec le
service de descripteurs passent par ce module (requêtes HTTP + gestion d'erreurs).
"""
from __future__ import annotations

from pathlib import Path

import requests

import config


class APIError(RuntimeError):
    """Erreur renvoyée par l'API ou API injoignable."""


class CBIRClient:
    def __init__(self, base_url: str = config.API_URL, timeout: float = 120):
        self.base = base_url.rstrip("/") + "/api"
        self.timeout = timeout
        self.http = requests.Session()

    # ----------------------------------------------------------------- outils
    def _call(self, method: str, path: str, **kw):
        try:
            r = self.http.request(method, self.base + path, timeout=kw.pop("timeout", self.timeout), **kw)
        except requests.RequestException as exc:
            raise APIError(f"API de descripteurs injoignable ({self.base}). Est-elle lancée ? ({exc.__class__.__name__})")
        if r.status_code >= 400:
            try:
                msg = r.json().get("message", r.text)
            except ValueError:
                msg = r.text
            raise APIError(f"Erreur API {r.status_code} : {msg}")
        return r.json()

    @staticmethod
    def _files(paths: list[Path], field: str = "images"):
        return [(field, (p.name, p.read_bytes(), "application/octet-stream")) for p in paths]

    # ----------------------------------------------------------- descripteurs
    def health(self):
        return self._call("GET", "/health", timeout=3)

    def rf_methods(self):
        return self._call("GET", "/rf-methods", timeout=5)

    def descriptors(self, paths: list[Path], types: str = "all", details: bool = False):
        suffix = "" if types == "all" else f"/{types}"
        if details:
            suffix += "?details=1"
        return self._call("POST", f"/descriptors{suffix}", files=self._files(paths))

    # --------------------------------------------------------------- index
    def index_file(self, collection: str, image_id: int, path: Path, label: str | None = None):
        data = {"image_id": str(image_id)}
        if label is not None:
            data["label"] = label
        return self._call("POST", f"/collections/{collection}/images", data=data,
                          files=self._files([path], "image"))

    def index_paths(self, collection: str, items: list[dict]):
        return self._call("POST", f"/collections/{collection}/images", json={"items": items}, timeout=1800)

    def unindex(self, collection: str, image_id: int):
        try:
            return self._call("DELETE", f"/collections/{collection}/images/{image_id}")
        except APIError:
            return None   # déjà absente de l'index : pas bloquant

    def collection_info(self, collection: str):
        return self._call("GET", f"/collections/{collection}", timeout=5)

    # ------------------------------------------------------------ recherche
    def search_by_id(self, collection: str, image_id: int, scope: int):
        return self._call("POST", f"/collections/{collection}/search",
                          json={"image_id": str(image_id), "scope": scope})

    def search_by_file(self, collection: str, path: Path, scope: int):
        return self._call("POST", f"/collections/{collection}/search", data={"scope": str(scope)},
                          files=self._files([path], "image"))

    def feedback(self, collection: str, session_id: str, method: str,
                 relevant: list[int], non_relevant: list[int]):
        return self._call("POST", f"/collections/{collection}/feedback",
                          json={"session_id": session_id, "method": method,
                                "relevant": [str(i) for i in relevant],
                                "non_relevant": [str(i) for i in non_relevant]})


api = CBIRClient()
