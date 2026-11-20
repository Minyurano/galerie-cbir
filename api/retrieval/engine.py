"""
Moteur de recherche : recherche simple et recherche avec retour de pertinence.

Chaque recherche ouvre une SESSION (identifiant aléatoire) qui mémorise la
requête, les poids et les jugements. Les sessions vivent en mémoire et
expirent après SESSION_TTL secondes d'inactivité.
"""
from __future__ import annotations

import secrets
import threading
import time

import numpy as np

from .distance import GROUP_NAMES
from .index import Collection
from .relevance_feedback import METHODS, RFState, apply_feedback, new_state, rank

SESSION_TTL = 2 * 3600


class SessionNotFound(KeyError):
    pass


class SearchEngine:
    def __init__(self):
        self._sessions: dict[str, tuple[str, RFState, float]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- sessions
    def _purge(self):
        now = time.time()
        for sid in [s for s, (_, _, t) in self._sessions.items() if now - t > SESSION_TTL]:
            del self._sessions[sid]

    def _open(self, collection: str, state: RFState) -> str:
        sid = secrets.token_urlsafe(16)
        with self._lock:
            self._purge()
            self._sessions[sid] = (collection, state, time.time())
        return sid

    def _get(self, sid: str, collection: str) -> RFState:
        with self._lock:
            item = self._sessions.get(sid)
            if item is None or item[0] != collection:
                raise SessionNotFound(sid)
            self._sessions[sid] = (item[0], item[1], time.time())
            return item[1]

    # ------------------------------------------------------------- recherche
    @staticmethod
    def _results(col: Collection, state: RFState, order: np.ndarray, d: np.ndarray, scope: int):
        top = order[:scope]
        state.last_retrieved = top.tolist()
        return [
            {"rank": r + 1, "image_id": col.ids[i], "distance": round(float(d[i]), 5),
             "label": col.labels[i]}
            for r, i in enumerate(top)
        ]

    @staticmethod
    def _weights(state: RFState) -> dict:
        from ..descriptors.extractor import GROUP_SLICES
        return {
            "groups": {g: round(float(state.e[j]), 4) for j, g in enumerate(GROUP_NAMES)},
            "families": {
                "couleur": round(float(state.e[:3].sum()), 4),
                "texture": round(float(state.e[3:].sum()), 4),
            },
            "components_mean": {g: round(float(state.w[GROUP_SLICES[g]].mean()), 4) for g in GROUP_NAMES},
            "components_max": {g: round(float(state.w[GROUP_SLICES[g]].max()), 4) for g in GROUP_NAMES},
        }

    def search(self, col: Collection, query_vector: np.ndarray | None = None,
               image_id: str | None = None, scope: int = 25) -> dict:
        """Recherche simple (poids égaux, distance euclidienne normalisée)."""
        if len(col) == 0:
            raise ValueError("La collection est vide : indexez d'abord des images.")
        Xn = col.normalized()
        q_index = None
        if image_id is not None:
            q_index = col.position(image_id)
            if q_index is None:
                raise KeyError(f"Image {image_id} absente de l'index.")
            q0 = Xn[q_index]
        else:
            q0 = col.normalizer()(np.asarray(query_vector, dtype=np.float32)[None, :])[0]
        scope = int(max(1, min(scope, len(col))))
        state = new_state(q0, scope, q_index)
        if q_index is not None:
            state.relevant.add(q_index)        # la requête est pertinente par définition
        order, d = rank(Xn, state)
        results = self._results(col, state, order, d, scope)
        sid = self._open(col.name, state)
        return {"session_id": sid, "iteration": 0, "method": "simple", "scope": scope,
                "query_image_id": image_id, "results": results, "weights": self._weights(state)}

    def feedback(self, col: Collection, session_id: str, method: str,
                 relevant_ids: list[str], non_relevant_ids: list[str]) -> dict:
        """Une itération de retour de pertinence."""
        state = self._get(session_id, col.name)
        Xn = col.normalized()
        rel = [p for p in (col.position(i) for i in relevant_ids) if p is not None]
        non = [p for p in (col.position(i) for i in non_relevant_ids) if p is not None]
        if not rel and not non:
            raise ValueError("Indiquez au moins une image pertinente ou non pertinente.")
        apply_feedback(Xn, state, method, rel, non)
        order, d = rank(Xn, state)
        results = self._results(col, state, order, d, state.scope)
        return {"session_id": session_id, "iteration": state.iteration, "method": method,
                "method_name": METHODS[method]["name"], "scope": state.scope,
                "relevant_count": len(state.relevant), "non_relevant_count": len(state.non_relevant),
                "results": results, "weights": self._weights(state)}
