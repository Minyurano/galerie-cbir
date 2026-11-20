"""
Index des descripteurs : une « collection » par utilisateur de la galerie.

Chaque collection est stockée sur disque dans data/index/<collection>/ :
    vectors.npy   matrice (N, 181) des descripteurs bruts
    meta.json     identifiants des images et étiquettes facultatives

Les descripteurs sont calculés UNE seule fois (à l'ajout de l'image). La
recherche compare ensuite la requête à la matrice en mémoire : pour 10 000
images, une recherche exhaustive prend quelques millisecondes, une structure
d'index approchée (FAISS, HNSW...) n'est donc pas nécessaire à cette échelle.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import numpy as np

from ..descriptors.extractor import FEATURE_DIM
from .distance import Normalizer

_SAFE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


class Collection:
    def __init__(self, name: str, root: Path):
        if not _SAFE.match(name):
            raise ValueError("Nom de collection invalide.")
        self.name = name
        self.dir = root / name
        self.lock = threading.RLock()
        self.ids: list[str] = []
        self.labels: list[str | None] = []
        self.X = np.zeros((0, FEATURE_DIM), dtype=np.float32)
        self._pos: dict[str, int] = {}
        self._norm: Normalizer | None = None
        self._Xn: np.ndarray | None = None
        self._load()

    # ------------------------------------------------------------------ disque
    def _load(self):
        meta, vec = self.dir / "meta.json", self.dir / "vectors.npy"
        if meta.exists() and vec.exists():
            m = json.loads(meta.read_text(encoding="utf-8"))
            self.ids, self.labels = m["ids"], m["labels"]
            self.X = np.load(vec).astype(np.float32)
            self._pos = {i: k for k, i in enumerate(self.ids)}

    def save(self):
        with self.lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            np.save(self.dir / "vectors.npy", self.X)
            (self.dir / "meta.json").write_text(
                json.dumps({"ids": self.ids, "labels": self.labels}), encoding="utf-8")

    # ------------------------------------------------------------ modification
    def add(self, image_id: str, vector: np.ndarray, label: str | None = None, save: bool = True):
        with self.lock:
            v = np.asarray(vector, dtype=np.float32).reshape(1, FEATURE_DIM)
            if image_id in self._pos:                     # mise à jour
                self.X[self._pos[image_id]] = v[0]
                self.labels[self._pos[image_id]] = label
            else:
                self._pos[image_id] = len(self.ids)
                self.ids.append(image_id)
                self.labels.append(label)
                self.X = np.vstack([self.X, v])
            self._invalidate()
            if save:
                self.save()

    def add_many(self, image_ids: list[str], vectors: np.ndarray, labels: list[str | None]):
        """Ajout groupé (une seule copie de la matrice, une seule sauvegarde)."""
        with self.lock:
            new_rows, new_ids, new_labels = [], [], []
            for iid, v, lab in zip(image_ids, vectors, labels):
                if iid in self._pos:
                    self.X[self._pos[iid]] = v
                    self.labels[self._pos[iid]] = lab
                else:
                    self._pos[iid] = len(self.ids) + len(new_ids)
                    new_ids.append(iid)
                    new_labels.append(lab)
                    new_rows.append(np.asarray(v, dtype=np.float32))
            if new_rows:
                self.X = np.vstack([self.X, np.stack(new_rows)])
                self.ids.extend(new_ids)
                self.labels.extend(new_labels)
            self._invalidate()
            self.save()

    def remove(self, image_id: str) -> bool:
        with self.lock:
            if image_id not in self._pos:
                return False
            k = self._pos.pop(image_id)
            self.X = np.delete(self.X, k, axis=0)
            del self.ids[k]
            del self.labels[k]
            self._pos = {i: j for j, i in enumerate(self.ids)}
            self._invalidate()
            self.save()
            return True

    def _invalidate(self):
        self._norm, self._Xn = None, None

    # ---------------------------------------------------------------- lecture
    def __len__(self):
        return len(self.ids)

    def position(self, image_id: str) -> int | None:
        return self._pos.get(image_id)

    def normalizer(self) -> Normalizer:
        with self.lock:
            if self._norm is None:
                self._norm = Normalizer().fit(self.X) if len(self) else None
            return self._norm

    def normalized(self) -> np.ndarray:
        """Matrice normalisée (mise en cache tant que la collection ne change pas)."""
        with self.lock:
            if self._Xn is None:
                n = self.normalizer()
                self._Xn = n(self.X).astype(np.float32) if n else self.X.copy()
            return self._Xn


class IndexStore:
    """Accès aux collections (chargées à la demande, une seule fois)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cols: dict[str, Collection] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> Collection:
        with self._lock:
            if name not in self._cols:
                self._cols[name] = Collection(name, self.root)
            return self._cols[name]

    def names(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())
